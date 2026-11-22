//! Raon-1-nano forward pass on candle, mirroring the released `modeling_raon1_nano.py`:
//! 24 decoder layers on a 4-stream Raon-mHC residual — 18 RDA (gated delta rule with per-key-
//! channel decay and a write gate) + 6 NoPE gated MLA — a SwiGLU MLP per layer, the per-layer
//! embedding (PLE) added after each MLP, and a final RMSNorm over the stream mean.
//!
//! Activations are f32. Every GEMM goes through [`crate::weights::Linear`], i.e. the quantized
//! blocks resident on the device (CUDA / ROCm / Vulkan / Metal / CPU).

use crate::weights::{Linear, Weights};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::{rms_norm, sigmoid, softmax_last_dim};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Opt-in phase profiler (`RAON_PROFILE=1`): accumulated milliseconds per phase.
pub mod profile {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;
    pub static ON: AtomicBool = AtomicBool::new(false);
    pub static ACC: Mutex<[f64; 6]> = Mutex::new([0.0; 6]);
    pub const NAMES: [&str; 6] = ["mhc_route", "rda", "mla", "mlp", "ple_gate", "mix"];
    pub fn enabled() -> bool {
        ON.load(Ordering::Relaxed)
    }
    pub fn add(i: usize, t: Instant) {
        if enabled() {
            ACC.lock().unwrap()[i] += t.elapsed().as_secs_f64() * 1e3;
        }
    }
    pub fn take() -> Vec<(&'static str, f64)> {
        let mut a = ACC.lock().unwrap();
        let out = NAMES.iter().zip(a.iter()).map(|(n, v)| (*n, *v)).collect();
        *a = [0.0; 6];
        out
    }
}
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    pub heads: usize,
    pub inter: usize,
    pub eps: f64,
    pub mla_eps: f64,
    pub sel_eps: f64,
    pub post_factor: f64,
    pub streams: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub qk_nope: usize,
    pub qk_pe: usize,
    pub v_dim: usize,
    pub rda_heads: usize,
    pub rda_head_dim: usize,
    pub conv_k: usize,
    pub gate_lower_bound: Option<f64>,
    pub ple_dim: usize,
    pub rda_layer: Vec<bool>,
    pub eos: Vec<u32>,
}

impl Config {
    pub fn from_json(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s)?;
        let u = |k: &str, d: usize| v[k].as_u64().map(|x| x as usize).unwrap_or(d);
        let f = |k: &str, d: f64| v[k].as_f64().unwrap_or(d);
        let layers = u("num_hidden_layers", 24);
        let rda = &v["rda_config"];
        let rda_layers: Vec<usize> = rda["rda_layers"]
            .as_array()
            .context("rda_config.rda_layers")?
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let head_dim = u("head_dim", 128);
        Ok(Config {
            hidden: u("hidden_size", 1536),
            layers,
            vocab: u("vocab_size", 259840),
            heads: u("num_attention_heads", 32),
            inter: u("intermediate_size", 6144),
            eps: f("rms_norm_eps", 1e-6),
            mla_eps: f("mla_latent_norm_eps", 1e-6),
            sel_eps: v["raon_mhc_selector_eps"].as_f64().unwrap_or(1e-5),
            post_factor: v["raon_mhc_post_factor"].as_f64().unwrap_or(2.0),
            streams: v["raon_mhc_streams"].as_u64().map(|x| x as usize).unwrap_or(4),
            q_lora: u("q_lora_rank", 512),
            kv_lora: u("kv_lora_rank", 512),
            qk_nope: u("qk_nope_head_dim", 128),
            qk_pe: u("qk_rope_head_dim", 64),
            v_dim: u("v_head_dim", 128),
            rda_heads: rda["num_heads"].as_u64().map(|x| x as usize).unwrap_or(16),
            rda_head_dim: rda["head_dim"].as_u64().map(|x| x as usize).unwrap_or(head_dim),
            conv_k: rda["conv_kernel_size"].as_u64().map(|x| x as usize).unwrap_or(4),
            gate_lower_bound: if rda.get("gate_lower_bound").is_some_and(|g| g.is_null()) {
                None
            } else {
                Some(rda["gate_lower_bound"].as_f64().unwrap_or(-5.0))
            },
            ple_dim: u("per_layer_embedding_dim", 256),
            rda_layer: (0..layers).map(|i| rda_layers.contains(&(i + 1))).collect(),
            eos: vec![259202, 259200],
        })
    }
}

fn rms(x: &Tensor, w: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let xc = x.contiguous()?;
    if !xc.device().is_cpu()
        && crate::device::accel_op_ok(xc.device(), crate::device::AccelOp::RmsNorm)
    {
        if let Ok(y) = candle_core::decoder_ops::rmsnorm_fused(&xc, w, eps as f32, false) {
            return Ok(y);
        }
    }
    rms_norm(&xc, w, eps as f32)
}

// ---------------------------------------------------------------------------------------------
// Raon-mHC
// ---------------------------------------------------------------------------------------------

/// Helmert basis `Q ∈ R^{n×(n-1)}`, `QᵀQ = I`, `Qᵀ1 = 0`.
fn helmert(n: usize) -> Vec<f32> {
    let mut q = vec![0f32; n * (n - 1)];
    for k in 1..n {
        let s = ((k * (k + 1)) as f64).sqrt();
        for i in 0..k {
            q[i * (n - 1) + k - 1] = (1.0 / s) as f32;
        }
        q[k * (n - 1) + k - 1] = (-(k as f64) / s) as f32;
    }
    q
}

struct MhcConsts {
    q: Tensor,     // [n, n-1]
    q_t: Tensor,   // [n-1, n] precomputed transpose
    b: Tensor,     // [n, n]  (I + 11ᵀ/n)/2
    inv_b: Tensor, // [n, n]  1 / b precomputed (avoids 48x/token broadcast_div)
    eye: Tensor,   // [n, n]
}

struct Mhc {
    sel: Tensor, // [2n + (n-1)², n·C]
    alpha_pre: f64,
    alpha_post: f64,
    alpha_z: f64,
    b_pre: Tensor,
    b_post: Tensor,
    b_z: Tensor,
    gamma: f64,
    /// Precomputed `(1 - gamma) * eye` `[n, n]` (avoids scaling `eye` 48x per token).
    eye_res: Tensor,
}

impl Mhc {
    fn load(w: &Weights, p: &str, consts: &MhcConsts) -> Result<Self> {
        let sel = Tensor::cat(
            &[
                w.tensor(&format!("{p}.phi_pre"), None)?,
                w.tensor(&format!("{p}.phi_post"), None)?,
                w.tensor(&format!("{p}.res.phi_z"), None)?,
            ],
            0,
        )?;
        let eta = w.scalar(&format!("{p}.res.eta"))? as f64;
        let gamma = 1.0 / (1.0 + (-eta).exp());
        let eye_res = (&consts.eye * (1.0 - gamma))?;
        Ok(Self {
            sel: sel.t()?.contiguous()?,
            alpha_pre: w.scalar(&format!("{p}.alpha_pre"))? as f64,
            alpha_post: w.scalar(&format!("{p}.alpha_post"))? as f64,
            alpha_z: w.scalar(&format!("{p}.res.alpha_z"))? as f64,
            b_pre: w.tensor(&format!("{p}.b_pre"), None)?,
            b_post: w.tensor(&format!("{p}.b_post"), None)?,
            b_z: w.tensor(&format!("{p}.res.b_z"), None)?,
            gamma,
            eye_res,
        })
    }

    /// `stream [T, n, C]` → (`h_pre [T, n]`, `h_post [T, n]`, `h_res [T, n, n]`).
    fn route(&self, stream: &Tensor, k: &MhcConsts, cfg: &Config) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let (t, n, c) = stream.dims3()?;
        let flat = stream.reshape((t, n * c))?;
        let inv = flat.sqr()?.mean_keepdim(1)?.affine(1.0, cfg.sel_eps)?.sqrt()?.recip()?;
        let z = flat.matmul(&self.sel)?.broadcast_mul(&inv)?;
        let h_pre = sigmoid(&z.narrow(1, 0, n)?.affine(self.alpha_pre, 0.0)?.broadcast_add(&self.b_pre)?)?;
        let h_post = (sigmoid(&z.narrow(1, n, n)?.affine(self.alpha_post, 0.0)?.broadcast_add(&self.b_post)?)? * cfg.post_factor)?;
        let m = n - 1;
        let tz = z
            .narrow(1, 2 * n, m * m)?
            .affine(self.alpha_z, 0.0)?
            .broadcast_add(&self.b_z)?;
        // e = Q · tz · Qᵀ -> compute tz · Qᵀ first as 2D matmul (zero broadcast of Qᵀ),
        // and for single-token decode (t == 1) avoid broadcasting Q as well.
        let tz_qt = tz.reshape((t * m, m))?.matmul(&k.q_t)?;
        let e = if t == 1 {
            k.q.matmul(&tz_qt)?.unsqueeze(0)?
        } else {
            let tz_qt = tz_qt.reshape((t, m, n))?;
            k.q.unsqueeze(0)?.broadcast_as((t, n, m))?.contiguous()?.matmul(&tz_qt)?
        };
        let rho = e.neg()?.broadcast_mul(&k.inv_b)?.reshape((t, n * n))?.max_keepdim(1)?.relu()?; // [T, 1]
        let phi = e.broadcast_div(&(rho.affine(1.0, 1.0)?.unsqueeze(2)?))?.broadcast_add(&k.b)?;
        let h_res = (phi * self.gamma)?.broadcast_add(&self.eye_res)?;
        Ok((h_pre, h_post, h_res))
    }
}

fn aggregate(stream: &Tensor, h_pre: &Tensor) -> candle_core::Result<Tensor> {
    // [T, 1, n] @ [T, n, C] -> [T, 1, C] -> [T, C] without allocating a [T, n, C] temporary.
    h_pre.unsqueeze(1)?.matmul(&stream.contiguous()?)?.squeeze(1)
}

fn mix(stream: &Tensor, branch: &Tensor, h_post: &Tensor, h_res: &Tensor) -> candle_core::Result<Tensor> {
    h_res
        .matmul(&stream.contiguous()?)?
        .broadcast_add(&h_post.unsqueeze(2)?.broadcast_mul(&branch.unsqueeze(1)?)?)
}

// ---------------------------------------------------------------------------------------------
// RDA
// ---------------------------------------------------------------------------------------------

/// Recurrent state of one RDA layer: `[H, V, K]` f32, kept where the scan runs.
#[derive(Clone)]
pub enum RecState {
    Empty,
    Host(Vec<f32>),
    Dev(Tensor),
}

#[derive(Clone)]
pub struct RdaState {
    /// `[3·proj, K-1]` raw (pre-conv) q|k|v rows.
    pub conv: Option<Tensor>,
    pub rec: RecState,
}

struct Rda {
    q: Linear,
    k: Linear,
    v: Linear,
    g: Linear,
    b: Linear,
    w: Linear,
    fa: Linear,
    fb: Linear,
    conv: [Tensor; 3], // [C, K]
    a_exp: Tensor,     // [1, H, 1]
    dt_bias: Tensor,   // [H, D]
    o_norm: Tensor,
    o: Linear,
}

/// Token-serial V-first delta rule on the host (zero-alloc SIMD AVX2+FMA / rayon over heads).
#[allow(clippy::too_many_arguments)]
fn host_scan(q: &[f32], k: &[f32], v: &[f32], la: &[f32], beta: &[f32], state: &mut [f32], t: usize, h: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0f32; t * h * d];
    raon_cpu_kernels::rda_host_scan_f32(q, k, v, la, beta, state, t, h, d, &mut out);
    out
}

fn l2norm(x: &Tensor) -> candle_core::Result<Tensor> {
    // torch F.normalize: x / max(‖x‖₂, eps)
    let n = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.maximum(1e-6)?;
    x.broadcast_div(&n)
}

fn short_conv(x: &Tensor, w: &Tensor, hist: Option<Tensor>, k: usize) -> candle_core::Result<(Tensor, Tensor)> {
    let (t, c) = x.dims2()?;
    let sig = x.t()?.contiguous()?; // [C, T]
    let hist = match hist {
        Some(h) => h,
        None => Tensor::zeros((c, k - 1), DType::F32, x.device())?,
    };
    let padded = Tensor::cat(&[&hist, &sig], 1)?; // [C, T+K-1]
    let mut out = padded.narrow(1, 0, t)?.broadcast_mul(&w.narrow(1, 0, 1)?)?;
    for j in 1..k {
        out = (out + padded.narrow(1, j, t)?.broadcast_mul(&w.narrow(1, j, 1)?)?)?;
    }
    let new_hist = padded.narrow(1, t, k - 1)?.contiguous()?;
    Ok((out.silu()?.t()?.contiguous()?, new_hist))
}

impl Rda {
    fn load(w: &Weights, p: &str, cfg: &Config) -> Result<Self> {
        let h = cfg.rda_heads;
        let a: Vec<f32> = w.host_f32(&format!("{p}.A_log"))?.iter().map(|x| x.exp()).collect();
        Ok(Self {
            q: w.linear(&format!("{p}.q_proj.weight"))?,
            k: w.linear(&format!("{p}.k_proj.weight"))?,
            v: w.linear(&format!("{p}.v_proj.weight"))?,
            g: w.linear(&format!("{p}.g_proj.weight"))?,
            b: w.linear(&format!("{p}.b_proj.weight"))?,
            w: w.linear(&format!("{p}.rda_w_proj.weight"))?,
            fa: w.linear(&format!("{p}.f_a_proj.weight"))?,
            fb: w.linear(&format!("{p}.f_b_proj.weight"))?,
            conv: [
                w.tensor(&format!("{p}.q_conv1d.weight"), None)?,
                w.tensor(&format!("{p}.k_conv1d.weight"), None)?,
                w.tensor(&format!("{p}.v_conv1d.weight"), None)?,
            ],
            a_exp: Tensor::from_vec(a, (1, h, 1), &w.device)?,
            dt_bias: w.tensor(&format!("{p}.dt_bias"), Some(&[h, cfg.rda_head_dim]))?,
            o_norm: w.tensor(&format!("{p}.o_norm.weight"), None)?,
            o: w.linear(&format!("{p}.o_proj.weight"))?,
        })
    }

    fn forward(&self, x: &Tensor, st: &mut RdaState, cfg: &Config) -> Result<Tensor> {
        let (t, _) = x.dims2()?;
        let (h, d) = (cfg.rda_heads, cfg.rda_head_dim);
        let proj = h * d;
        let (q, k, v) = (self.q.forward(x)?, self.k.forward(x)?, self.v.forward(x)?);
        let gate = self.g.forward(x)?;
        let beta_raw = self.b.forward(x)?;
        let delta = self.w.forward(x)?;
        let decay_raw = self.fb.forward(&self.fa.forward(x)?)?;

        let hist = |i: usize| st.conv.as_ref().map(|c| c.narrow(0, i * proj, proj)).transpose();
        let (q, qh) = short_conv(&q, &self.conv[0], hist(0)?, cfg.conv_k)?;
        let (k, kh) = short_conv(&k, &self.conv[1], hist(1)?, cfg.conv_k)?;
        let (v, vh) = short_conv(&v, &self.conv[2], hist(2)?, cfg.conv_k)?;
        st.conv = Some(Tensor::cat(&[qh, kh, vh], 0)?);

        let q = (l2norm(&q.reshape((t, h, d))?)? * (d as f64).powf(-0.5))?;
        let k = l2norm(&k.reshape((t, h, d))?)?;
        let scaled = decay_raw
            .reshape((t, h, d))?
            .broadcast_add(&self.dt_bias)?
            .broadcast_mul(&self.a_exp)?;
        let log_alpha = match cfg.gate_lower_bound {
            Some(lb) => (sigmoid(&scaled)? * lb)?,
            // Numerically stable -softplus(x) = -(max(x, 0) + log1p(exp(-|x|))) avoiding exp overflow for x > 88.
            None => (scaled.relu()? + scaled.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?)?.neg()?,
        };
        let beta = sigmoid(&beta_raw)?; // [T, H]
                                        // write gate sigmoid(b + δ) / erase gate sigmoid(b), realised as a value scale
        let ratio = sigmoid(&(&beta_raw + &delta)?)?.div(&beta)?;
        let v = v.reshape((t, h, d))?.broadcast_mul(&ratio.unsqueeze(2)?)?;

        let out = if t == 1 && !x.device().is_cpu() {
            self.device_step(&q, &k, &v, &log_alpha, &beta, st, h, d)?
        } else {
            let mut s = match std::mem::replace(&mut st.rec, RecState::Empty) {
                RecState::Empty => vec![0f32; h * d * d],
                RecState::Host(s) => s,
                RecState::Dev(t) => t.flatten_all()?.to_vec1::<f32>()?,
            };
            let f = |x: &Tensor| x.flatten_all().and_then(|x| x.to_vec1::<f32>());
            let o = host_scan(&f(&q)?, &f(&k)?, &f(&v)?, &f(&log_alpha)?, &f(&beta)?, &mut s, t, h, d);
            st.rec = if x.device().is_cpu() {
                RecState::Host(s)
            } else {
                RecState::Dev(Tensor::from_vec(s, (h, d, d), x.device())?)
            };
            Tensor::from_vec(o, (t, h, d), x.device())?
        };
        let normed = rms(&out, &self.o_norm, cfg.eps)?;
        let gated = normed.mul(&sigmoid(&gate.reshape((t, h, d))?)?)?;
        Ok(self.o.forward(&gated.reshape((t, proj))?)?)
    }

    /// One decode token entirely on the device (no host round trip).
    #[allow(clippy::too_many_arguments)]
    fn device_step(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        la: &Tensor,
        beta: &Tensor,
        st: &mut RdaState,
        h: usize,
        d: usize,
    ) -> Result<Tensor> {
        let dev = q.device();
        let s = match std::mem::replace(&mut st.rec, RecState::Empty) {
            RecState::Empty => Tensor::zeros((h, d, d), DType::F32, dev)?,
            RecState::Host(s) => Tensor::from_vec(s, (h, d, d), dev)?,
            RecState::Dev(t) => t,
        };
        let (q, k, v, la) = (
            q.reshape((h, d, 1))?,
            k.reshape((h, d, 1))?,
            v.reshape((h, d, 1))?,
            la.reshape((h, 1, d))?,
        );
        let s = s.broadcast_mul(&la.exp()?)?; // decay along K
        let pred = s.matmul(&k)?; // [H, V, 1]
        let err = (v - pred)?.broadcast_mul(&beta.reshape((h, 1, 1))?)?;
        let s = (s + err.matmul(&k.transpose(1, 2)?)?)?;
        let out = s.matmul(&q)?.reshape((1, h, d))?;
        st.rec = RecState::Dev(s);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// MLA (NoPE, gated) with CoW pre-allocated KV cache + CPU 64-key streaming MLA attention
// ---------------------------------------------------------------------------------------------

/// Pre-allocated doubling-capacity KV cache with Copy-on-Write (`Arc::strong_count == 1`) safety.
///
/// Cloning `State` from a cached `prefix_state` shares the backing buffers via `Arc`; the first
/// `append` on a cloned state copies into a freshly owned buffer so the shared prefix is NEVER
/// mutated, and every subsequent decode token appends in O(T) via `slice_set` (zero O(L²) cat
/// copies and zero-copy transposed K view `k_view.transpose(1, 2)` accepted directly by cuBLAS /
/// rocBLAS / Vulkan / CPU without maintaining a duplicate `k_t` buffer).
#[derive(Clone)]
pub struct KvState {
    pub k: Option<Tensor>,   // [H, cap, dq]
    pub v: Option<Tensor>,   // [H, cap, dv]
    pub len: usize,
    cap: usize,
    planned_cap: usize,
    share: Arc<()>,
}

impl Default for KvState {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl KvState {
    /// Creates an empty KV state that reserves at least `planned_cap` tokens on its first
    /// allocation, avoiding power-of-two reallocations when prompt + `max_new` is known upfront.
    pub fn with_capacity(planned_cap: usize) -> Self {
        Self {
            k: None,
            v: None,
            len: 0,
            cap: 0,
            planned_cap,
            share: Arc::new(()),
        }
    }

    pub fn plan_capacity(&mut self, planned_cap: usize) {
        if planned_cap > self.planned_cap {
            self.planned_cap = planned_cap;
        }
    }

    /// Appends `new_k [H, T, dq]` and `new_v [H, T, dv]`, returning zero-copy valid views
    /// `(k [H, L, dq], v [H, L, dv], k_t Option<[H, dq, L]>)`.
    fn append(
        &mut self,
        new_k: &Tensor,
        new_v: &Tensor,
        store_kt: bool,
    ) -> candle_core::Result<(Tensor, Tensor, Option<Tensor>)> {
        let (h, t, dq) = new_k.dims3()?;
        let dv = new_v.dim(2)?;
        let needed = self.len + t;
        let can_reuse = self.k.is_some()
            && self.v.is_some()
            && Arc::strong_count(&self.share) == 1
            && needed <= self.cap;

        if can_reuse {
            let k_buf = self.k.as_ref().unwrap();
            let v_buf = self.v.as_ref().unwrap();
            k_buf.slice_set(new_k, 1, self.len)?;
            v_buf.slice_set(new_v, 1, self.len)?;
            self.len = needed;
        } else {
            let new_cap = needed.max(self.planned_cap).next_power_of_two().max(256);
            let dev = new_k.device();
            let dtype = new_k.dtype();
            let k_buf = Tensor::zeros((h, new_cap, dq), dtype, dev)?;
            let v_buf = Tensor::zeros((h, new_cap, dv), dtype, dev)?;
            if self.len > 0 {
                if let (Some(old_k), Some(old_v)) = (&self.k, &self.v) {
                    let prev_k = old_k.narrow(1, 0, self.len)?.contiguous()?;
                    let prev_v = old_v.narrow(1, 0, self.len)?.contiguous()?;
                    k_buf.slice_set(&prev_k, 1, 0)?;
                    v_buf.slice_set(&prev_v, 1, 0)?;
                }
            }
            k_buf.slice_set(new_k, 1, self.len)?;
            v_buf.slice_set(new_v, 1, self.len)?;
            self.k = Some(k_buf);
            self.v = Some(v_buf);
            self.cap = new_cap;
            self.len = needed;
            self.share = Arc::new(());
        }

        let k_view = self.k.as_ref().unwrap().narrow(1, 0, self.len)?;
        let v_view = self.v.as_ref().unwrap().narrow(1, 0, self.len)?;
        // `k_view.transpose(1, 2)` has shape [H, dq, L] with stride [cap*dq, 1, dq] — zero-copy
        // CUBLAS_OP_T / rocblas_operation_transpose without a duplicate `[H, dq, cap]` buffer.
        let kt_view = if store_kt {
            Some(k_view.transpose(1, 2)?)
        } else {
            None
        };
        Ok((k_view, v_view, kt_view))
    }
}

fn cpu_f32_slice_if_zero_offset<'a>(
    storage: &'a candle_core::Storage,
    layout: &candle_core::Layout,
) -> Option<&'a [f32]> {
    if layout.start_offset() != 0 || !layout.is_contiguous() {
        return None;
    }
    match storage {
        candle_core::Storage::Cpu(candle_core::CpuStorage::F32(s)) => {
            s.get(..layout.shape().elem_count())
        }
        _ => None,
    }
}

struct Mla {
    q_a: Linear,
    q_a_norm: Tensor,
    q_b: Linear,
    kv_a: Linear,
    kv_a_norm: Tensor,
    kv_b: Linear,
    g: Linear,
    o: Linear,
    mask_cache: Mutex<HashMap<(usize, usize), Tensor>>,
}

impl Mla {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            q_a: w.linear(&format!("{p}.q_a_proj.weight"))?,
            q_a_norm: w.tensor(&format!("{p}.q_a_layernorm.weight"), None)?,
            q_b: w.linear(&format!("{p}.q_b_proj.weight"))?,
            kv_a: w.linear(&format!("{p}.kv_a_proj_with_mqa.weight"))?,
            kv_a_norm: w.tensor(&format!("{p}.kv_a_layernorm.weight"), None)?,
            kv_b: w.linear(&format!("{p}.kv_b_proj.weight"))?,
            g: w.linear(&format!("{p}.g_proj.weight"))?,
            o: w.linear(&format!("{p}.o_proj.weight"))?,
            mask_cache: Mutex::new(HashMap::new()),
        })
    }

    fn causal_mask(&self, t: usize, l: usize, dev: &Device) -> candle_core::Result<Tensor> {
        if let Some(m) = self.mask_cache.lock().unwrap_or_else(|e| e.into_inner()).get(&(t, l)) {
            return Ok(m.clone());
        }
        let off = l - t;
        let mask: Vec<f32> = (0..t)
            .flat_map(|i| (0..l).map(move |j| if j > i + off { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let tensor = Tensor::from_vec(mask, (1, t, l), dev)?;
        let mut cache = self.mask_cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len() >= 32 {
            cache.clear();
        }
        cache.insert((t, l), tensor.clone());
        Ok(tensor)
    }

    fn forward(&self, x: &Tensor, st: &mut KvState, cfg: &Config) -> Result<Tensor> {
        let (t, _) = x.dims2()?;
        let (h, dn, dp, dv) = (cfg.heads, cfg.qk_nope, cfg.qk_pe, cfg.v_dim);
        let dq = dn + dp;
        let q = self
            .q_b
            .forward(&rms(&self.q_a.forward(x)?, &self.q_a_norm, cfg.mla_eps)?)?
            .reshape((t, h, dq))?
            .transpose(0, 1)?
            .contiguous()?;
        let kv_full = self.kv_a.forward(x)?;
        let lat = kv_full.narrow(1, 0, cfg.kv_lora)?;
        let pe = kv_full.narrow(1, cfg.kv_lora, dp)?;
        let kv = self
            .kv_b
            .forward(&rms(&lat, &self.kv_a_norm, cfg.mla_eps)?)?
            .reshape((t, h, dn + dv))?;
        let new_k = Tensor::cat(&[kv.narrow(2, 0, dn)?, pe.unsqueeze(1)?.broadcast_as((t, h, dp))?.contiguous()?], 2)?
            .transpose(0, 1)?
            .contiguous()?;
        let new_v = kv.narrow(2, dn, dv)?.transpose(0, 1)?.contiguous()?;
        let on_gpu = !x.device().is_cpu();
        let (key, val, key_t) = st.append(&new_k, &new_v, on_gpu)?;
        let l = st.len;
        let scale = (dq as f64).powf(-0.5);

        let ctx = if !on_gpu && q.dtype() == DType::F32 {
            // CPU F32: zero-copy strided 64-key streaming MLA attention directly over the
            // pre-allocated [H, cap, dq] / [H, cap, dv] KV buffers (zero KV copy, zero score matrix).
            let mut out = vec![0.0f32; t * h * dv];
            let ran_strided = if let (Some(k_buf), Some(v_buf)) = (st.k.as_ref(), st.v.as_ref()) {
                let (qs, ql) = q.storage_and_layout();
                let (ks, kl) = k_buf.storage_and_layout();
                let (vs, vl) = v_buf.storage_and_layout();
                if let (Some(q_sl), Some(k_sl), Some(v_sl)) = (
                    cpu_f32_slice_if_zero_offset(&qs, ql),
                    cpu_f32_slice_if_zero_offset(&ks, kl),
                    cpu_f32_slice_if_zero_offset(&vs, vl),
                ) {
                    raon_cpu_kernels::stream_attention_mla_hsd_strided_f32(
                        q_sl,
                        k_sl,
                        v_sl,
                        h,
                        t,
                        l,
                        dq,
                        dv,
                        st.cap * dq,
                        st.cap * dv,
                        scale as f32,
                        t > 1,
                        l - t,
                        &mut out,
                    );
                    true
                } else {
                    false
                }
            } else {
                false
            };
            if !ran_strided {
                let qv = q.flatten_all()?.to_vec1::<f32>()?;
                let kv_vec = key.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
                let vv = val.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
                raon_cpu_kernels::stream_attention_mla_hsd_f32(
                    &qv,
                    &kv_vec,
                    &vv,
                    h,
                    t,
                    l,
                    dq,
                    dv,
                    scale as f32,
                    t > 1,
                    l - t,
                    &mut out,
                );
            }
            Tensor::from_vec(out, (t, h * dv), x.device())?
        } else {
            let kt = match key_t {
                Some(kt) => kt,
                None => key.t()?.contiguous()?,
            };
            let mut scores = (q.matmul(&kt)? * scale)?; // [H, T, L]
            if t > 1 {
                let mask = self.causal_mask(t, l, x.device())?;
                scores = scores.broadcast_add(&mask)?;
            }
            let ctx = softmax_last_dim(&scores)?.matmul(&val)?; // [H, T, dv]
            ctx.transpose(0, 1)?.contiguous()?.reshape((t, h * dv))?
        };
        let ctx = ctx.mul(&sigmoid(&self.g.forward(x)?)?)?;
        Ok(self.o.forward(&ctx)?)
    }
}

// ---------------------------------------------------------------------------------------------
// Layer / model
// ---------------------------------------------------------------------------------------------

enum Attn {
    Rda(Box<Rda>),
    Mla(Box<Mla>),
}

#[derive(Clone)]
pub enum LayerState {
    Rda(RdaState),
    Mla(KvState),
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            gate: w.linear(&format!("{p}.gate_proj.weight"))?,
            up: w.linear(&format!("{p}.up_proj.weight"))?,
            down: w.linear(&format!("{p}.down_proj.weight"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = if let Some(fused) = self.gate.fused_gate_up_silu(&self.up, x)? {
            fused
        } else {
            let g = self.gate.forward(x)?;
            let u = self.up.forward(x)?;
            if !x.device().is_cpu()
                && crate::device::accel_op_ok(x.device(), crate::device::AccelOp::DecoderOps)
            {
                candle_core::decoder_ops::silu_mul(&g, &u).or_else(|_| g.silu()?.mul(&u))?
            } else {
                g.silu()?.mul(&u)?
            }
        };
        Ok(self.down.forward(&a)?)
    }
}

struct PleGate {
    gate: Linear,
    proj: Linear,
    norm: Tensor,
}

struct PleTimer(Instant);
impl Drop for PleTimer {
    fn drop(&mut self) {
        profile::add(4, self.0);
    }
}

struct Layer {
    in_norm: Tensor,
    post_norm: Tensor,
    attn: Attn,
    attn_mhc: Mhc,
    mlp_mhc: Mhc,
    mlp: Mlp,
    ple: Option<PleGate>,
}

impl Layer {
    fn load(w: &Weights, i: usize, cfg: &Config, rda: bool, consts: &MhcConsts) -> Result<Self> {
        let p = format!("model.layers.{i}");
        let ple = if w.has(&format!("{p}.per_layer_input_gate.gate")) {
            Some(PleGate {
                gate: w.linear(&format!("{p}.per_layer_input_gate.gate"))?,
                proj: w.linear(&format!("{p}.per_layer_input_gate.projection"))?,
                norm: w.tensor(&format!("{p}.per_layer_input_gate.output_norm.weight"), None)?,
            })
        } else {
            None
        };
        Ok(Self {
            in_norm: w.tensor(&format!("{p}.input_layernorm.weight"), None)?,
            post_norm: w.tensor(&format!("{p}.post_attention_layernorm.weight"), None)?,
            attn: if rda {
                Attn::Rda(Box::new(Rda::load(w, &format!("{p}.self_attn"), cfg)?))
            } else {
                Attn::Mla(Box::new(Mla::load(w, &format!("{p}.self_attn"))?))
            },
            attn_mhc: Mhc::load(w, &format!("{p}.self_attn_raon_mhc"), consts)?,
            mlp_mhc: Mhc::load(w, &format!("{p}.mlp_raon_mhc"), consts)?,
            mlp: Mlp::load(w, &format!("{p}.mlp"))?,
            ple,
        })
    }

    fn forward(&self, stream: &Tensor, st: &mut LayerState, ple_in: Option<&Tensor>, k: &MhcConsts, cfg: &Config) -> Result<Tensor> {
        let t = Instant::now();
        let (h_pre, h_post, h_res) = self.attn_mhc.route(stream, k, cfg)?;
        let x = rms(&aggregate(stream, &h_pre)?, &self.in_norm, cfg.eps)?;
        profile::add(0, t);
        let t = Instant::now();
        let out = match (&self.attn, st) {
            (Attn::Rda(a), LayerState::Rda(s)) => {
                let o = a.forward(&x, s, cfg)?;
                profile::add(1, t);
                o
            }
            (Attn::Mla(a), LayerState::Mla(s)) => {
                let o = a.forward(&x, s, cfg)?;
                profile::add(2, t);
                o
            }
            _ => anyhow::bail!("layer state does not match the layer type"),
        };
        let t = Instant::now();
        let stream = mix(stream, &out, &h_post, &h_res)?;
        profile::add(5, t);
        let t = Instant::now();
        let (h_pre, h_post, h_res) = self.mlp_mhc.route(&stream, k, cfg)?;
        let x = rms(&aggregate(&stream, &h_pre)?, &self.post_norm, cfg.eps)?;
        profile::add(0, t);
        let t = Instant::now();
        let out = self.mlp.forward(&x)?;
        profile::add(3, t);
        let t = Instant::now();
        let mut stream = mix(&stream, &out, &h_post, &h_res)?;
        profile::add(5, t);
        let t = Instant::now();
        let _g = PleTimer(t);
        if let (Some(p), Some(li)) = (&self.ple, ple_in) {
            let hidden = stream.mean(1)?;
            let gated = p.gate.forward(&hidden)?.gelu()?.mul(li)?;
            let delta = rms(&p.proj.forward(&gated)?, &p.norm, cfg.eps)?;
            stream = stream.broadcast_add(&delta.unsqueeze(1)?)?;
        }
        Ok(stream)
    }
}

#[derive(Clone)]
pub struct State {
    pub layers: Vec<LayerState>,
    pub pos: usize,
}

impl State {
    /// Reserves planned total sequence capacity (`prompt + max_new`) on all MLA KV layers so the
    /// next allocation sizes buffers once without doubling reallocs.
    pub fn plan_capacity(&mut self, planned_cap: usize) {
        for layer in &mut self.layers {
            if let LayerState::Mla(kv) = layer {
                kv.plan_capacity(planned_cap);
            }
        }
    }
}

pub struct Model {
    pub cfg: Config,
    pub w: Weights,
    layers: Vec<Layer>,
    norm: Tensor,
    ple_proj: Option<Linear>,
    ple_norm: Option<Tensor>,
    consts: MhcConsts,
}

impl Model {
    pub fn load(mut w: Weights) -> Result<Self> {
        let cfg = Config::from_json(&w.config_json)?;
        let n = cfg.streams;
        let dev = w.device.clone();
        let eye: Vec<f32> = (0..n * n).map(|i| if i / n == i % n { 1.0 } else { 0.0 }).collect();
        let b: Vec<f32> = eye.iter().map(|e| 0.5 * (e + 1.0 / n as f32)).collect();
        let q = Tensor::from_vec(helmert(n), (n, n - 1), &dev)?;
        let q_t = q.t()?.contiguous()?;
        let b_tensor = Tensor::from_vec(b, (n, n), &dev)?;
        let inv_b = b_tensor.recip()?;
        let consts = MhcConsts {
            q,
            q_t,
            b: b_tensor,
            inv_b,
            eye: Tensor::from_vec(eye, (n, n), &dev)?,
        };
        let layers = (0..cfg.layers)
            .map(|i| Layer::load(&w, i, &cfg, cfg.rda_layer[i], &consts).with_context(|| format!("layer {i}")))
            .collect::<Result<Vec<_>>>()?;
        let (ple_proj, ple_norm) = if w.has("model.per_layer_inputs.embedding_projection") {
            (
                Some(w.linear("model.per_layer_inputs.embedding_projection")?),
                Some(w.tensor("model.per_layer_inputs.projection_norm.weight", None)?),
            )
        } else {
            (None, None)
        };
        let norm = w.tensor("model.norm.weight", None)?;
        // Weights are on the device now; re-map the sources so only rows gathered later
        // (embeddings, PLE, head scan) stay resident.
        w.release_mapped_pages()?;
        Ok(Self {
            cfg,
            w,
            layers,
            norm,
            ple_proj,
            ple_norm,
            consts,
        })
    }

    pub fn device(&self) -> &Device {
        &self.w.device
    }

    pub fn new_state(&self) -> State {
        self.new_state_with_capacity(0)
    }

    pub fn new_state_with_capacity(&self, planned_cap: usize) -> State {
        let layers = self
            .cfg
            .rda_layer
            .iter()
            .map(|&r| {
                if r {
                    LayerState::Rda(RdaState {
                        conv: None,
                        rec: RecState::Empty,
                    })
                } else {
                    LayerState::Mla(KvState::with_capacity(planned_cap))
                }
            })
            .collect();
        State { layers, pos: 0 }
    }

    pub fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        self.w.rows("model.embed_tokens.weight", ids)
    }

    /// `p = (RMSNorm(W_mp e) + √d · T[token]) / √2` → `[T, layers, d]`.
    fn ple_inputs(&self, ids: &[u32], e: &Tensor) -> Result<Option<Tensor>> {
        let (Some(proj), Some(norm)) = (&self.ple_proj, &self.ple_norm) else {
            return Ok(None);
        };
        let (t, l, d) = (ids.len(), self.cfg.layers, self.cfg.ple_dim);
        let projected = rms(&proj.forward(e)?.reshape((t, l, d))?, norm, self.cfg.eps)?;
        let rows = (self.w.ple_rows(ids)?.reshape((t, l, d))? * (d as f64).sqrt())?;
        Ok(Some(((projected + rows)? * std::f64::consts::FRAC_1_SQRT_2)?))
    }

    /// Run `ids` through the model, updating `st`; returns the final normed hidden states `[T, C]`.
    pub fn forward(&self, ids: &[u32], st: &mut State) -> Result<Tensor> {
        let e = self.embed(ids)?;
        self.forward_embeds(ids, &e, st)
    }

    pub fn forward_embeds(&self, ids: &[u32], e: &Tensor, st: &mut State) -> Result<Tensor> {
        let (t, c) = e.dims2()?;
        let ple = self.ple_inputs(ids, e)?;
        let mut stream = e.unsqueeze(1)?.broadcast_as((t, self.cfg.streams, c))?.contiguous()?;
        for (i, layer) in self.layers.iter().enumerate() {
            let li = ple.as_ref().map(|p| p.narrow(1, i, 1).and_then(|x| x.squeeze(1))).transpose()?;
            stream = layer
                .forward(&stream, &mut st.layers[i], li.as_ref(), &self.consts, &self.cfg)
                .with_context(|| format!("layer {i}"))?;
        }
        st.pos += t;
        Ok(rms(&stream.mean(1)?, &self.norm, self.cfg.eps)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helmert_is_orthonormal_and_orthogonal_to_ones() {
        let n = 4;
        let q = helmert(n);
        for a in 0..n - 1 {
            let col_sum: f32 = (0..n).map(|i| q[i * (n - 1) + a]).sum();
            assert!(col_sum.abs() < 1e-6);
            for b in 0..n - 1 {
                let dot: f32 = (0..n).map(|i| q[i * (n - 1) + a] * q[i * (n - 1) + b]).sum();
                assert!((dot - if a == b { 1.0 } else { 0.0 }).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn host_scan_matches_device_step_on_cpu_tensors() {
        // Run one token through host_scan and through the tensor formulation; they must agree.
        let (h, d) = (2usize, 4usize);
        let mk = |s: u64| {
            (0..h * d)
                .map(|i| (((i as u64 * 2654435761 + s) % 1000) as f32 / 1000.0) - 0.5)
                .collect::<Vec<f32>>()
        };
        let (q, k, v, la) = (mk(1), mk(2), mk(3), mk(4).iter().map(|x| -x.abs()).collect::<Vec<f32>>());
        let beta = vec![0.3f32, 0.8];
        let mut s0: Vec<f32> = (0..h * d * d).map(|i| (i as f32 * 0.01).sin()).collect();
        let dev = Device::Cpu;
        let st = Tensor::from_vec(s0.clone(), (h, d, d), &dev).unwrap();
        let out = host_scan(&q, &k, &v, &la, &beta, &mut s0, 1, h, d);
        let t = |x: &Vec<f32>, s: (usize, usize, usize)| Tensor::from_vec(x.clone(), s, &dev).unwrap();
        let s1 = st.broadcast_mul(&t(&la, (h, 1, d)).exp().unwrap()).unwrap();
        let pred = s1.matmul(&t(&k, (h, d, 1))).unwrap();
        let err = (t(&v, (h, d, 1)) - pred)
            .unwrap()
            .broadcast_mul(&Tensor::from_vec(beta.clone(), (h, 1, 1), &dev).unwrap())
            .unwrap();
        let s2 = (s1 + err.matmul(&t(&k, (h, 1, d))).unwrap()).unwrap();
        let o = s2
            .matmul(&t(&q, (h, d, 1)))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (a, b) in out.iter().zip(&o) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        let s2v = s2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in s0.iter().zip(&s2v) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn kv_state_cow_append_and_kt_view_match_naive_cat_on_active_device() -> Result<()> {
        let dev = crate::device::select_device(crate::device::Backend::Auto, 0)?;
        let (h, dq, dv) = (4usize, 192usize, 128usize);
        let mk = |n: usize, seed: u64| -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        };
        let k0 = Tensor::from_vec(mk(h * 5 * dq, 1), (h, 5, dq), &dev)?;
        let v0 = Tensor::from_vec(mk(h * 5 * dv, 2), (h, 5, dv), &dev)?;
        let k1 = Tensor::from_vec(mk(h * 1 * dq, 3), (h, 1, dq), &dev)?;
        let v1 = Tensor::from_vec(mk(h * 1 * dv, 4), (h, 1, dv), &dev)?;
        let k2 = Tensor::from_vec(mk(h * 1 * dq, 5), (h, 1, dq), &dev)?;
        let v2 = Tensor::from_vec(mk(h * 1 * dv, 6), (h, 1, dv), &dev)?;
        let q = Tensor::from_vec(mk(h * 1 * dq, 7), (h, 1, dq), &dev)?;

        let mut prefix = KvState::default();
        prefix.append(&k0, &v0, true)?;
        let prefix_snap = prefix.k.as_ref().unwrap().narrow(1, 0, prefix.len)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;

        // Cloned prefix must be isolated via Copy-on-Write (`Arc::strong_count == 1`).
        let mut req1 = prefix.clone();
        let (k_b1, v_b1, kt_b1) = req1.append(&k1, &v1, true)?;
        assert_eq!(req1.len, 6);
        let (k_b1_2, v_b1_2, kt_b1_2) = req1.append(&k2, &v2, true)?;
        assert_eq!(req1.len, 7);

        let prefix_after = prefix.k.as_ref().unwrap().narrow(1, 0, prefix.len)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(prefix_snap, prefix_after, "CoW violated: shared prefix was mutated!");

        // Incremental k_t view matmul must match naive Tensor::cat + transpose + matmul.
        let naive_k = Tensor::cat(&[&k0, &k1, &k2], 1)?;
        let naive_v = Tensor::cat(&[&v0, &v1, &v2], 1)?;
        let scale = (dq as f64).powf(-0.5);
        let want = softmax_last_dim(&(q.matmul(&naive_k.t()?.contiguous()?)? * scale)?)?.matmul(&naive_v)?;
        let got = softmax_last_dim(&(q.matmul(&kt_b1_2.unwrap())? * scale)?)?.matmul(&v_b1_2)?;
        let diff = (&want.to_device(&Device::Cpu)? - &got.to_device(&Device::Cpu)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-4, "KvState k_t matmul diff {diff}");
        let _ = (k_b1, v_b1, kt_b1, k_b1_2);
        if !dev.is_cpu() {
            crate::device::trim_idle_gpu_pool(&dev);
        }
        Ok(())
    }

    #[test]
    fn kv_state_with_capacity_preallocates_without_doubling() -> Result<()> {
        let dev = Device::Cpu;
        let (h, dq, dv) = (2usize, 16usize, 8usize);
        let mut kv = KvState::with_capacity(300);
        let k0 = Tensor::zeros((h, 4, dq), DType::F32, &dev)?;
        let v0 = Tensor::zeros((h, 4, dv), DType::F32, &dev)?;
        kv.append(&k0, &v0, false)?;
        assert_eq!(kv.cap, 512, "planned_cap=300 should pre-allocate next_power_of_two(300)=512");
        assert_eq!(kv.len, 4);
        Ok(())
    }

    #[test]
    fn aggregate_matmul_matches_broadcast_mul_sum() -> Result<()> {
        let dev = Device::Cpu;
        let stream = Tensor::from_vec((0..3 * 4 * 16).map(|i| (i as f32) * 0.05 - 1.0).collect(), (3, 4, 16), &dev)?;
        let h_pre = Tensor::from_vec((0..3 * 4).map(|i| 0.1 * (i as f32 + 1.0)).collect(), (3, 4), &dev)?;
        let want = stream.broadcast_mul(&h_pre.unsqueeze(2)?)?.sum(1)?;
        let got = aggregate(&stream, &h_pre)?;
        let diff = (&want - &got)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "aggregate matmul diff {diff}");
        Ok(())
    }
}
