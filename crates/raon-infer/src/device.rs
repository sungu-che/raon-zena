//! Device selection, lifecycle management, atomic file helpers, and per-op `accel_check`.
//!
//! Follows logis-center's `utils/device_utils.rs`:
//! - `Backend::Auto`: CUDA → ROCm → Vulkan → Metal → CPU (non-silent failure logging).
//! - Caches opened GPU devices per `(Backend, ordinal)` so LLM and screen watcher share one context.
//! - Per-`(device_label, AccelOp)` `catch_unwind` verification cache with fault injection.
//! - Metal native exact quantized matmul (`exact_matmul_native` includes Metal on candle@881b3c80).
//! - Idle GPU pool trim (`trim_memory_pool(0)`) + Windows working-set trim + `rocm_exit_now_if_used`.

use anyhow::{bail, Result};
use candle_core::{DType, Device, DeviceLocation, Tensor};
use std::collections::HashMap;
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Auto,
    Cpu,
    Cuda,
    Rocm,
    Vulkan,
    Metal,
}

impl std::str::FromStr for Backend {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "auto" | "gpu" => Backend::Auto,
            "cpu" => Backend::Cpu,
            "cuda" => Backend::Cuda,
            "rocm" | "hip" => Backend::Rocm,
            "vulkan" | "vk" => Backend::Vulkan,
            "metal" => Backend::Metal,
            _ => bail!("unknown device {s} (auto, cpu, cuda, rocm, vulkan, metal)"),
        })
    }
}

/// Backends this binary was built with.
pub fn compiled_backends() -> Vec<&'static str> {
    let mut v = vec!["cpu"];
    if cfg!(feature = "cuda") {
        v.push("cuda");
    }
    if cfg!(feature = "rocm") {
        v.push("rocm");
    }
    if cfg!(feature = "vulkan") {
        v.push("vulkan");
    }
    if cfg!(feature = "metal") {
        v.push("metal");
    }
    v
}

static ROCM_USED: AtomicBool = AtomicBool::new(false);
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);
static DEVICE_CACHE: OnceLock<Mutex<HashMap<(Backend, usize), Device>>> = OnceLock::new();

fn device_cache() -> &'static Mutex<HashMap<(Backend, usize), Device>> {
    DEVICE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Crash-safe atomic file replacement (Windows-safe backup swap).
pub fn atomic_replace(tmp: &Path, target: &Path) -> std::io::Result<()> {
    match std::fs::rename(tmp, target) {
        Ok(()) => Ok(()),
        Err(_) if target.exists() => {
            let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
            let mut bak_os = target.as_os_str().to_os_string();
            bak_os.push(format!(".bak.{}.{seq}", std::process::id()));
            let bak = PathBuf::from(bak_os);
            let _ = std::fs::remove_file(&bak);
            std::fs::rename(target, &bak)?;
            match std::fs::rename(tmp, target) {
                Ok(()) => {
                    let _ = std::fs::remove_file(&bak);
                    Ok(())
                }
                Err(e) => {
                    let _ = std::fs::rename(&bak, target);
                    Err(e)
                }
            }
        }
        Err(e) => Err(e),
    }
}

/// Writes `bytes` to a unique sibling temp file, `sync_all()`s it, and atomically replaces `target`.
pub fn write_file_atomic(target: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let mut tmp_os = target.as_os_str().to_os_string();
    tmp_os.push(format!(".tmp.{}.{seq}", std::process::id()));
    let tmp = PathBuf::from(tmp_os);
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    if let Err(e) = atomic_replace(&tmp, target) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn open_best_vulkan(ordinal: usize) -> Result<Device> {
    if ordinal != 0 {
        return Ok(Device::new_vulkan(ordinal)?);
    }
    let count = candle_core::vulkan_backend::device_count().unwrap_or(0).min(8);
    if count <= 1 {
        return Ok(Device::new_vulkan(0)?);
    }
    // Prefer discrete > integrated > virtual > other > CPU renderers, breaking ties by free VRAM.
    let mut best: Option<((u8, u64), Device)> = None;
    for id in 0..count {
        if let Ok(d) = Device::new_vulkan(id) {
            let tier: u8 = match d.as_vulkan_device().map(|v| v.device_type()) {
                Ok("discrete") => 4,
                Ok("integrated") => 3,
                Ok("virtual") => 2,
                Ok("other") => 1,
                _ => 0,
            };
            let free = d.mem_info().ok().flatten().map_or(0, |(f, _)| f as u64);
            let rank = (tier, free);
            if best.as_ref().is_none_or(|(b, _)| rank > *b) {
                best = Some((rank, d));
            }
        }
    }
    match best {
        Some((_, d)) => Ok(d),
        None => Ok(Device::new_vulkan(0)?),
    }
}

fn try_gpu(b: Backend, ordinal: usize) -> Result<Option<Device>> {
    let _ = ordinal;
    let opened = match b {
        #[cfg(feature = "cuda")]
        Backend::Cuda => Some(Device::new_cuda(ordinal)?),
        #[cfg(feature = "rocm")]
        Backend::Rocm => Some(Device::new_rocm(ordinal)?),
        #[cfg(feature = "vulkan")]
        Backend::Vulkan => Some(open_best_vulkan(ordinal)?),
        #[cfg(feature = "metal")]
        Backend::Metal => Some(Device::new_metal(ordinal)?),
        Backend::Cpu => Some(Device::Cpu),
        _ => None,
    };
    if let Some(ref d) = opened {
        if is_rocm(d) {
            ROCM_USED.store(true, Ordering::Relaxed);
        }
    }
    Ok(opened)
}

pub fn select_device(b: Backend, ordinal: usize) -> Result<Device> {
    if b == Backend::Cpu {
        return Ok(Device::Cpu);
    }
    if let Some(d) = device_cache().lock().unwrap_or_else(|e| e.into_inner()).get(&(b, ordinal)) {
        return Ok(d.clone());
    }
    if b != Backend::Auto {
        return match try_gpu(b, ordinal)? {
            Some(d) => {
                if !d.is_cpu() {
                    device_cache().lock().unwrap_or_else(|e| e.into_inner()).insert((b, ordinal), d.clone());
                }
                Ok(d)
            }
            None => bail!("{b:?} support is not compiled in (built with: {})", compiled_backends().join(", ")),
        };
    }
    for cand in [Backend::Cuda, Backend::Rocm, Backend::Vulkan, Backend::Metal] {
        if let Some(d) = device_cache().lock().unwrap_or_else(|e| e.into_inner()).get(&(cand, ordinal)) {
            let d = d.clone();
            device_cache().lock().unwrap_or_else(|e| e.into_inner()).insert((Backend::Auto, ordinal), d.clone());
            return Ok(d);
        }
        match try_gpu(cand, ordinal) {
            Ok(Some(d)) => {
                let mut cache = device_cache().lock().unwrap_or_else(|e| e.into_inner());
                cache.insert((cand, ordinal), d.clone());
                cache.insert((Backend::Auto, ordinal), d.clone());
                return Ok(d);
            }
            Ok(None) => {}
            Err(e) => eprintln!("[raon-device] {cand:?}:{ordinal} open failed -> trying next backend: {e}"),
        }
    }
    Ok(Device::Cpu)
}

/// Stable ordinal-aware label (`cpu`, `cuda:0`, `rocm:0`, `vulkan:0`, `metal:0`, `opencl:0`).
pub fn device_label(d: &Device) -> String {
    match d.location() {
        DeviceLocation::Cpu => "cpu".into(),
        DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
        DeviceLocation::Rocm { gpu_id } => format!("rocm:{gpu_id}"),
        DeviceLocation::Vulkan { gpu_id } => format!("vulkan:{gpu_id}"),
        DeviceLocation::Metal { gpu_id } => format!("metal:{gpu_id}"),
        DeviceLocation::OpenCl { gpu_id } => format!("opencl:{gpu_id}"),
    }
}

/// GPU ordinal of `d` (0 for CPU).
pub fn device_ordinal(d: &Device) -> usize {
    match d.location() {
        DeviceLocation::Cpu => 0,
        DeviceLocation::Cuda { gpu_id }
        | DeviceLocation::Rocm { gpu_id }
        | DeviceLocation::Vulkan { gpu_id }
        | DeviceLocation::Metal { gpu_id }
        | DeviceLocation::OpenCl { gpu_id } => gpu_id,
    }
}

pub fn describe(d: &Device) -> String {
    match d.location() {
        DeviceLocation::Cpu => format!("cpu ({} threads)", rayon::current_num_threads()),
        DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
        DeviceLocation::Rocm { gpu_id } => {
            #[cfg(feature = "rocm")]
            {
                let name = candle_rocm::device_name(gpu_id).unwrap_or_else(|_| "AMD GPU".into());
                let arch = candle_rocm::device_arch(gpu_id).unwrap_or_default();
                if arch.is_empty() {
                    format!("rocm:{gpu_id} ({name})")
                } else {
                    format!("rocm:{gpu_id} ({name}, {arch})")
                }
            }
            #[cfg(not(feature = "rocm"))]
            {
                format!("rocm:{gpu_id}")
            }
        }
        DeviceLocation::Vulkan { gpu_id } => {
            #[cfg(feature = "vulkan")]
            if let Device::Vulkan(vd) = d {
                return format!("vulkan:{gpu_id} ({})", vd.name());
            }
            format!("vulkan:{gpu_id}")
        }
        DeviceLocation::Metal { gpu_id } => format!("metal:{gpu_id}"),
        DeviceLocation::OpenCl { gpu_id } => format!("opencl:{gpu_id}"),
    }
}

/// Whether `QMatMul::forward_exact` is cheap on this device (native exact kernels without full
/// dequantization every call). Includes CUDA, ROCm (`w8a16.hip`), and Metal (native F32-activation
/// quantized kernel). Vulkan uses `QMatMul::forward` (`GpuQuant` VRAM shaders) instead.
pub fn exact_matmul_native(d: &Device) -> bool {
    d.is_cuda() || is_rocm(d) || d.is_metal()
}

/// Conservatively trims idle GPU memory pools between turns without dropping the CUDA/ROCm hot pool.
///
/// On CUDA and ROCm, trims down to a 256 MiB hot pool only when `pool_idle_bytes() >= 320 MiB`,
/// avoiding allocator pipeline stalls between inference requests.
pub fn trim_idle_gpu_pool(d: &Device) {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    const HIGH_WATER: usize = 320 << 20;
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    const KEEP_HOT: usize = 256 << 20;
    match d {
        #[cfg(feature = "rocm")]
        Device::Rocm(rd) => {
            if rd.pool_idle_bytes().unwrap_or(0) >= HIGH_WATER {
                let _ = rd.trim_memory_pool_to(KEEP_HOT);
            }
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(cd) => {
            if cd.pool_idle_bytes().unwrap_or(0) >= HIGH_WATER {
                let _ = cd.trim_memory_pool(KEEP_HOT);
            }
        }
        #[cfg(feature = "vulkan")]
        Device::Vulkan(vd) => {
            const VK_HIGH_WATER: usize = 64 << 20;
            if vd.pooled_bytes() >= VK_HIGH_WATER {
                let _ = vd.trim_memory_pool();
            }
        }
        _ => {}
    }
}

/// Full teardown drain: synchronizes, releases workspaces, trims pool to 0, and frees ROCm caches.
pub fn drain_gpu_pool(d: &Device) {
    if d.is_cpu() {
        return;
    }
    let _ = d.synchronize_all();
    let _ = d.release_workspaces();
    let _ = d.trim_memory_pool(0);
    #[cfg(feature = "rocm")]
    if let Device::Rocm(rd) = d {
        let _ = rd.release_cached_resources();
    }
}

/// Ordinal-safe VRAM `(free, total)` query directly on `d` without cross-backend context leaks.
pub fn device_mem_info_of(d: &Device) -> Option<(usize, usize)> {
    if d.is_cpu() {
        return None;
    }
    d.mem_info().ok().flatten()
}

/// Drains every distinct GPU device cached by `select_device` (safe to call on idle unload or exit).
pub fn trim_all_opened_gpu_pools() {
    let mut seen = std::collections::HashSet::new();
    let devs: Vec<Device> = device_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .filter(|d| seen.insert(d.location()))
        .cloned()
        .collect();
    for d in devs {
        drain_gpu_pool(&d);
    }
}

#[cfg(windows)]
extern "system" {
    fn GetCurrentProcess() -> isize;
    fn SetProcessWorkingSetSize(process: isize, min_size: usize, max_size: usize) -> i32;
    fn TerminateProcess(process: isize, code: u32) -> i32;
}

/// Hints the OS to trim unused resident pages (Windows working-set trim after idle unload or re-mmap).
pub fn trim_process_working_set() {
    #[cfg(windows)]
    unsafe {
        let _ = SetProcessWorkingSetSize(GetCurrentProcess(), usize::MAX, usize::MAX);
    }
}

/// Prevents Windows `rocblas.dll` process-exit hangs when ROCm was initialized in this process.
/// Set `RAON_ROCM_FAST_EXIT=0` to disable for debugging.
pub fn rocm_exit_now_if_used(code: u32) {
    if !ROCM_USED.load(Ordering::Relaxed)
        || std::env::var("RAON_ROCM_FAST_EXIT").is_ok_and(|v| v.trim() == "0")
    {
        return;
    }
    trim_all_opened_gpu_pools();
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    #[cfg(windows)]
    unsafe {
        TerminateProcess(GetCurrentProcess(), code);
    }
    #[cfg(not(windows))]
    let _ = code;
}

#[allow(unused_variables)]
pub fn is_rocm(d: &Device) -> bool {
    #[cfg(feature = "rocm")]
    {
        d.is_rocm()
    }
    #[cfg(not(feature = "rocm"))]
    {
        false
    }
}

// ---------------------------------------------------------------------------------------------
// Per-(DeviceLocation, AccelOp) catch_unwind verification cache + fault injection
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccelOp {
    Matmul,
    Softmax,
    RmsNorm,
    LayerNorm,
    DecoderOps,
    FusedGateUpSilu,
    LmHeadArgmax,
    QuantQ8,
}

static ACCEL_VERDICTS: OnceLock<Mutex<HashMap<(DeviceLocation, AccelOp), bool>>> = OnceLock::new();
static ACCEL_FAULTS: OnceLock<Mutex<HashMap<AccelOp, bool>>> = OnceLock::new();

fn accel_verdicts() -> &'static Mutex<HashMap<(DeviceLocation, AccelOp), bool>> {
    ACCEL_VERDICTS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn accel_faults() -> &'static Mutex<HashMap<AccelOp, bool>> {
    ACCEL_FAULTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Fault-injection hook for tests: forces `accel_op_ok(_, op)` to fail inside `catch_unwind`
/// and cache a negative verdict until `reset_accel_cache()` is called.
pub fn set_accel_fault(op: AccelOp, fail: bool) {
    accel_faults().lock().unwrap_or_else(|e| e.into_inner()).insert(op, fail);
    if fail {
        accel_verdicts().lock().unwrap_or_else(|e| e.into_inner()).retain(|(_, o), _| *o != op);
    }
}

/// Clears the per-(device, op) verification cache and active fault injections.
pub fn reset_accel_cache() {
    accel_verdicts().lock().unwrap_or_else(|e| e.into_inner()).clear();
    accel_faults().lock().unwrap_or_else(|e| e.into_inner()).clear();
}

fn det_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn rel_err_f32(got: &Tensor, want: &Tensor) -> Result<f32> {
    let g = got.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let w = want.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let d = (&g - &w)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
    let n = w.sqr()?.sum_all()?.to_scalar::<f32>()?;
    let r = (d / n.max(1e-30)).sqrt();
    Ok(if r.is_finite() { r } else { f32::INFINITY })
}

/// Verifies `op` once per `(DeviceLocation, op)` against CPU reference under `catch_unwind`.
pub fn accel_op_ok(device: &Device, op: AccelOp) -> bool {
    if device.is_cpu() {
        return true;
    }
    let key = (device.location(), op);
    if let Some(&ok) = accel_verdicts().lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return ok;
    }
    let ok = match catch_unwind(AssertUnwindSafe(|| {
        if accel_faults()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&op)
            .copied()
            .unwrap_or(false)
        {
            panic!("injected accel fault for {op:?}");
        }
        probe_accel_op(device, op)
    })) {
        Ok(Ok(err)) if err <= 5e-3 => true,
        Ok(Ok(err)) => {
            eprintln!("[RAON-ACCEL] {} {:?}: rel_err={err:.2e} > 5e-3 -> fallback", device_label(device), op);
            false
        }
        Ok(Err(e)) => {
            eprintln!("[RAON-ACCEL] {} {:?}: probe error ({e}) -> fallback", device_label(device), op);
            false
        }
        Err(_) => {
            eprintln!("[RAON-ACCEL] {} {:?}: probe panicked -> fallback", device_label(device), op);
            false
        }
    };
    accel_verdicts().lock().unwrap_or_else(|e| e.into_inner()).insert(key, ok);
    ok
}

fn probe_accel_op(device: &Device, op: AccelOp) -> Result<f32> {
    match op {
        AccelOp::Matmul => {
            let a = Tensor::from_vec(det_vec(16 * 32, 1), (16, 32), &Device::Cpu)?;
            let b = Tensor::from_vec(det_vec(32 * 16, 2), (32, 16), &Device::Cpu)?;
            let want = a.matmul(&b)?;
            let got = a.to_device(device)?.matmul(&b.to_device(device)?)?;
            rel_err_f32(&got, &want)
        }
        AccelOp::Softmax => {
            let a = Tensor::from_vec(det_vec(8 * 64, 3), (8, 64), &Device::Cpu)?;
            let want = candle_nn::ops::softmax_last_dim(&a)?;
            let got = candle_nn::ops::softmax_last_dim(&a.to_device(device)?)?;
            rel_err_f32(&got, &want)
        }
        AccelOp::RmsNorm => {
            let x = Tensor::from_vec(det_vec(4 * 128, 4), (4, 128), &Device::Cpu)?;
            let w = Tensor::from_vec(det_vec(128, 5), 128, &Device::Cpu)?;
            let want = candle_nn::ops::rms_norm(&x, &w, 1e-6)?;
            let got = candle_core::decoder_ops::rmsnorm_fused(
                &x.to_device(device)?,
                &w.to_device(device)?,
                1e-6,
                false,
            )?;
            rel_err_f32(&got, &want)
        }
        AccelOp::LayerNorm => {
            // Includes +1000.0 DC-offset probe (BUG LN-1) against two-pass f64 CPU reference.
            let (rows, h) = (4usize, 128usize);
            let x_vals: Vec<f32> = det_vec(rows * h, 6)
                .into_iter()
                .enumerate()
                .map(|(i, v)| if i / h >= 2 { v * 0.05 + 1000.0 } else { v })
                .collect();
            let w_vals = det_vec(h, 7);
            let b_vals = det_vec(h, 8);
            let mut want_vals = vec![0.0f32; rows * h];
            raon_cpu_kernels::layer_norm(&x_vals, h, &w_vals, Some(&b_vals), 1e-6, &mut want_vals);
            let want = Tensor::from_vec(want_vals, (rows, h), &Device::Cpu)?;
            let x_dev = Tensor::from_vec(x_vals, (rows, h), &Device::Cpu)?.to_device(device)?;
            let w_dev = Tensor::from_vec(w_vals, h, &Device::Cpu)?.to_device(device)?;
            let b_dev = Tensor::from_vec(b_vals, h, &Device::Cpu)?.to_device(device)?;
            let got = candle_nn::ops::layer_norm(&x_dev, &w_dev, &b_dev, 1e-6)?;
            rel_err_f32(&got, &want)
        }
        AccelOp::FusedGateUpSilu => {
            if !(device.is_cuda() || is_rocm(device)) {
                bail!("native fused_gate_up_silu requires CUDA or ROCm");
            }
            use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
            let x = Tensor::from_vec(det_vec(2 * 256, 20), (2, 256), &Device::Cpu)?;
            let wg = Tensor::from_vec(det_vec(64 * 256, 21), (64, 256), &Device::Cpu)?;
            let wu = Tensor::from_vec(det_vec(64 * 256, 22), (64, 256), &Device::Cpu)?;
            let x_dev = x.to_device(device)?;
            let wg_dev = wg.to_device(device)?;
            let wu_dev = wu.to_device(device)?;
            let mut worst = 0f32;
            for dt in [
                GgmlDType::Q8_0,
                GgmlDType::Q4_0,
                GgmlDType::Q4K,
                GgmlDType::Q5K,
                GgmlDType::Q6K,
            ] {
                let g_cpu = QTensor::quantize(&wg, dt)?.dequantize(&Device::Cpu)?;
                let u_cpu = QTensor::quantize(&wu, dt)?.dequantize(&Device::Cpu)?;
                let want = x.matmul(&g_cpu.t()?)?.silu()?.mul(&x.matmul(&u_cpu.t()?)?)?;
                let qg = QMatMul::from_qtensor(QTensor::quantize(&wg_dev, dt)?)?;
                let qu = QMatMul::from_qtensor(QTensor::quantize(&wu_dev, dt)?)?;
                let got = qg.fused_gate_up_silu(&qu, &x_dev)?;
                let err = rel_err_f32(&got, &want)?;
                worst = worst.max(if err <= 1e-2 { err * 0.1 } else { err });
            }
            Ok(worst)
        }
        AccelOp::DecoderOps => {
            let x = Tensor::from_vec(det_vec(4 * 128, 10), (4, 128), &Device::Cpu)?;
            let d = Tensor::from_vec(det_vec(4 * 128, 11), (4, 128), &Device::Cpu)?;
            let w = Tensor::from_vec(det_vec(128, 12), 128, &Device::Cpu)?;
            let (wr, wy) = candle_core::decoder_ops::residual_rmsnorm(&x, Some(&d), &w, 1e-5, false)?;
            let (gr, gy) = candle_core::decoder_ops::residual_rmsnorm(
                &x.to_device(device)?,
                Some(&d.to_device(device)?),
                &w.to_device(device)?,
                1e-5,
                false,
            )?;
            let e1 = rel_err_f32(&gr, &wr)?;
            let e2 = rel_err_f32(&gy, &wy)?;
            let ws = candle_core::decoder_ops::silu_mul(&x, &d)?;
            let gs = candle_core::decoder_ops::silu_mul(&x.to_device(device)?, &d.to_device(device)?)?;
            Ok(e1.max(e2).max(rel_err_f32(&gs, &ws)?))
        }
        AccelOp::LmHeadArgmax => {
            let h = Tensor::from_vec(det_vec(2 * 64, 13), (2, 64), &Device::Cpu)?;
            let w = Tensor::from_vec(det_vec(128 * 64, 14), (128, 64), &Device::Cpu)?;
            let want = candle_core::lm_head::fused_linear_argmax_cs(&h, &w, None, None, None)?;
            let got = candle_core::lm_head::fused_linear_argmax_cs(
                &h.to_device(device)?,
                &w.to_device(device)?,
                None,
                None,
                None,
            )?;
            let (want_ids, _) = want.download_ids_and_mask()?;
            let (got_ids, _) = got.download_ids_and_mask()?;
            if want_ids != got_ids {
                return Ok(f32::INFINITY);
            }
            let e_fused = rel_err_f32(got.max_vals.as_ref().unwrap(), want.max_vals.as_ref().unwrap())?;
            let l = Tensor::from_vec(det_vec(2 * 128, 15), (2, 128), &Device::Cpu)?;
            let want_l = candle_core::lm_head::logits_argmax_cs(&l, None, None, None)?;
            let got_l = candle_core::lm_head::logits_argmax_cs(&l.to_device(device)?, None, None, None)?;
            let (want_l_ids, _) = want_l.download_ids_and_mask()?;
            let (got_l_ids, _) = got_l.download_ids_and_mask()?;
            if want_l_ids != got_l_ids {
                return Ok(f32::INFINITY);
            }
            let e_logits = rel_err_f32(got_l.max_vals.as_ref().unwrap(), want_l.max_vals.as_ref().unwrap())?;
            let mut worst = e_fused.max(e_logits);
            if is_rocm(device) || device.is_cuda() {
                use candle_core::quantized::{GgmlDType, QTensor};
                // M=12 (> 8) exercises chunked CUDA/ROCm Quantized Mode A across Q8_0 / Q4_0 / Q4K / Q5K / Q6K.
                let hq = Tensor::from_vec(det_vec(12 * 256, 18), (12, 256), &Device::Cpu)?;
                let wq = Tensor::from_vec(det_vec(64 * 256, 19), (64, 256), &Device::Cpu)?;
                let hq_dev = hq.to_device(device)?;
                let wq_dev = wq.to_device(device)?;
                for dt in [
                    GgmlDType::Q8_0,
                    GgmlDType::Q4_0,
                    GgmlDType::Q4K,
                    GgmlDType::Q5K,
                    GgmlDType::Q6K,
                ] {
                    let qt_cpu = QTensor::quantize(&wq, dt)?;
                    let want_q = candle_core::lm_head::logits_argmax_cs(&qt_cpu.forward_exact(&hq)?, None, None, None)?;
                    let qt_dev = QTensor::quantize(&wq_dev, dt)?;
                    let got_q = candle_core::lm_head::fused_quant_argmax_cs(&hq_dev, &qt_dev, None, None, None)?;
                    let (got_ids, _) = got_q.download_ids_and_mask()?;
                    if want_q.ids.to_vec1::<u32>()? != got_ids {
                        return Ok(f32::INFINITY);
                    }
                    if let (Some(gv), Some(wv)) = (got_q.max_vals.as_ref(), want_q.max_vals.as_ref()) {
                        worst = worst.max(rel_err_f32(gv, wv)?);
                    }
                }
            }
            Ok(worst)
        }
        AccelOp::QuantQ8 => {
            use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
            use candle_core::Module;
            let a = Tensor::from_vec(det_vec(4 * 256, 16), (4, 256), &Device::Cpu)?;
            let w = Tensor::from_vec(det_vec(64 * 256, 17), (64, 256), &Device::Cpu)?;
            let a_dev = a.to_device(device)?;
            let w_dev = w.to_device(device)?;
            let mut worst = 0f32;
            for dt in [GgmlDType::Q8_0, GgmlDType::Q4_0, GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K] {
                let qt_cpu = QTensor::quantize(&w, dt)?;
                let want = a.matmul(&qt_cpu.dequantize(&Device::Cpu)?.t()?)?;
                let qd = QMatMul::from_qtensor(QTensor::quantize(&w_dev, dt)?)?;
                let got = if exact_matmul_native(device) {
                    qd.forward_exact(&a_dev)?
                } else {
                    qd.forward(&a_dev)?
                };
                let err = rel_err_f32(&got, &want)?;
                let tol = if exact_matmul_native(device) || device.is_vulkan() { 1e-2 } else { 5e-2 };
                worst = worst.max(if err <= tol { err * 0.1 } else { err });
            }
            Ok(worst)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    use candle_core::{Module, Tensor};

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        Ok((a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?)
    }

    /// With a GPU backend compiled in (`--features cuda` …) and a device present, dense and quantized kernels run
    /// on it and agree with the CPU; if `RAON_REQUIRE_GPU=1`, missing GPU fails the test.
    #[test]
    fn the_gpu_backend_runs_and_matches_the_cpu() -> Result<()> {
        let require_gpu = std::env::var("RAON_REQUIRE_GPU")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        let dev = select_device(Backend::Auto, 0)?;
        eprintln!("device: {} (compiled: {})", describe(&dev), compiled_backends().join(", "));
        if dev.is_cpu() {
            assert!(!require_gpu, "RAON_REQUIRE_GPU=1 but no GPU device opened");
            return Ok(());
        }
        #[cfg(feature = "vulkan")]
        let vk_native_before = dev
            .as_vulkan_device()
            .is_ok_and(|v| !v.is_cpu_device())
            .then(candle_core::vulkan_backend::shaders::native_exec_count);

        let a = Tensor::randn(0f32, 1.0, (8, 256), &Device::Cpu)?;
        let w = Tensor::randn(0f32, 0.1, (512, 256), &Device::Cpu)?;
        let cpu = candle_nn::ops::softmax_last_dim(&a.matmul(&w.t()?)?)?;
        let gpu = candle_nn::ops::softmax_last_dim(&a.to_device(&dev)?.matmul(&w.to_device(&dev)?.t()?)?)?;
        let d = max_abs_diff(&cpu, &gpu.to_device(&Device::Cpu)?)?;
        assert!(d < 1e-4, "dense matmul + softmax differ by {d}");
        // Q8_0 and K-quant weights (Q4_0 / Q4K / Q5K / Q6K) through QMatMul — verifies both
        // ROCm w8a16 and Vulkan GpuQuant shaders across all pyramid tiers.
        let a_dev = a.to_device(&dev)?;
        let w_dev = w.to_device(&dev)?;
        for dt in [GgmlDType::Q8_0, GgmlDType::Q4_0, GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K] {
            let q_cpu = QMatMul::from_qtensor(QTensor::quantize(&w, dt)?)?;
            let q_gpu = QMatMul::from_qtensor(QTensor::quantize(&w_dev, dt)?)?;
            let y_cpu = q_cpu.forward(&a)?;
            let y_gpu = q_gpu.forward(&a_dev)?.to_device(&Device::Cpu)?;
            let scale = y_cpu.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let d = max_abs_diff(&y_cpu, &y_gpu)?;
            assert!(d <= 0.05 * scale.max(1e-3), "{dt:?} matmul differs by {d} (scale {scale})");
        }
        #[cfg(feature = "vulkan")]
        if let Some(before) = vk_native_before {
            let after = candle_core::vulkan_backend::shaders::native_exec_count();
            assert!(
                after > before,
                "Vulkan hardware GPU must execute native GpuQuant shaders (before={before}, after={after})"
            );
            eprintln!("Vulkan native GpuQuant dispatches: {}", after - before);
        }
        drain_gpu_pool(&dev);
        Ok(())
    }

    #[test]
    fn accel_check_verifies_all_ops_and_fault_injection() -> Result<()> {
        let require_gpu = std::env::var("RAON_REQUIRE_GPU")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        let dev = select_device(Backend::Auto, 0)?;
        if dev.is_cpu() {
            assert!(!require_gpu, "RAON_REQUIRE_GPU=1 but no GPU device opened");
        }
        reset_accel_cache();
        for op in [
            AccelOp::Matmul,
            AccelOp::Softmax,
            AccelOp::RmsNorm,
            AccelOp::LayerNorm,
            AccelOp::DecoderOps,
            AccelOp::LmHeadArgmax,
            AccelOp::QuantQ8,
        ] {
            assert!(accel_op_ok(&dev, op), "accel_op_ok failed for {op:?} on {}", describe(&dev));
        }
        assert_eq!(
            accel_op_ok(&dev, AccelOp::FusedGateUpSilu),
            dev.is_cpu() || dev.is_cuda() || is_rocm(&dev),
            "FusedGateUpSilu availability mismatch on {}",
            describe(&dev)
        );
        if !dev.is_cpu() {
            let _ = device_mem_info_of(&dev);
        }
        if !dev.is_cpu() {
            reset_accel_cache();
            set_accel_fault(AccelOp::DecoderOps, true);
            assert!(!accel_op_ok(&dev, AccelOp::DecoderOps));
            set_accel_fault(AccelOp::DecoderOps, false);
            assert!(
                !accel_op_ok(&dev, AccelOp::DecoderOps),
                "negative verdict must stay cached after clearing fault until reset_accel_cache()"
            );
            reset_accel_cache();
            assert!(accel_op_ok(&dev, AccelOp::DecoderOps));
            trim_idle_gpu_pool(&dev);
        }
        Ok(())
    }
}
