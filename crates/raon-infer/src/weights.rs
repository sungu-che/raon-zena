//! Weight sources: the pyramid GGUF written by `raon-quant` (production) or the original
//! safetensors (exact BF16 → F32, used to validate the forward pass against the reference).

use anyhow::{bail, Context, Result};
use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul};
use candle_core::{Device, Tensor};
use raon_quant::gguf::{GgmlType, GgufFile};
use raon_quant::imatrix::read_imatrix;
use raon_quant::kernels::dequant_row;
use raon_quant::kquant::{encode, Format};
use raon_quant::pyramid::{self, PyramidConfig, Residency, Tier};
use raon_quant::safetensors::SafeTensors;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Mutex;

pub enum Source {
    Gguf(GgufFile),
    SafeTensors(SafeTensors),
}

pub struct Weights {
    pub src: Source,
    pub device: Device,
    /// Use `QMatMul::forward_exact` (no activation quantization).
    pub exact: bool,
    /// For a GGUF with a sliced vocabulary: the kept ids (ascending).
    pub kept_ids: Option<Vec<u32>>,
    pub config_json: String,
    /// Diagnostics (`RAON_MIX_GGUF` + `RAON_MIX_PATTERN=a,b`): with original safetensors weights,
    /// take the tensors whose name contains one of the patterns from a pyramid GGUF instead
    /// (dequantized) — attributes quantization error to tensor groups.
    mix: Option<(GgufFile, Vec<String>)>,
    /// Load-time pyramid quantization of an original safetensors model (no GGUF file).
    pub quant: Option<LoadQuant>,
}

/// Importance-matrix collection (PRD §3.3): Σx² per input column of every linear layer, over a
/// calibration corpus run through the *original* weights. Written in llama.cpp's legacy
/// `imatrix.dat` format keyed by the original tensor names, which `raon-quant --imatrix` reads.
pub mod imatrix {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    pub static ON: AtomicBool = AtomicBool::new(false);
    pub static ACC: Mutex<BTreeMap<String, (Vec<f64>, u64)>> = Mutex::new(BTreeMap::new());

    pub fn record(name: &str, x: &candle_core::Tensor) -> candle_core::Result<()> {
        if !ON.load(Ordering::Relaxed) {
            return Ok(());
        }
        let cols = x.dim(candle_core::D::Minus1)?;
        let v = x.flatten_all()?.to_vec1::<f32>()?;
        let mut acc = ACC.lock().unwrap();
        let e = acc.entry(name.to_string()).or_insert_with(|| (vec![0f64; cols], 0));
        for row in v.chunks_exact(cols) {
            for (a, &b) in e.0.iter_mut().zip(row) {
                *a += (b as f64) * (b as f64);
            }
            e.1 += 1;
        }
        Ok(())
    }

    /// Write `Σx²/rows` per column (ncall = 1).
    pub fn write(path: &std::path::Path) -> std::io::Result<usize> {
        let acc = ACC.lock().unwrap();
        let mut b = Vec::new();
        b.extend((acc.len() as i32).to_le_bytes());
        for (name, (sum, n)) in acc.iter() {
            b.extend((name.len() as i32).to_le_bytes());
            b.extend(name.as_bytes());
            b.extend(1i32.to_le_bytes());
            b.extend((sum.len() as i32).to_le_bytes());
            for s in sum {
                b.extend(((s / (*n).max(1) as f64) as f32).to_le_bytes());
            }
        }
        crate::device::write_file_atomic(path, &b)?;
        Ok(acc.len())
    }
}

/// Load-time pyramid ("tower") quantization: every matrix the 4-tier plan quantizes is encoded
/// from the mmap'd original safetensors the first and only time it is loaded, and the blocks go
/// straight to the device. Gathered tables (`embed_tokens`, the PLE table) are never quantized:
/// their original BF16 rows are read from the mmap per token.
pub struct LoadQuant {
    formats: HashMap<String, (Tier, Format, Residency)>,
    imatrix: Option<HashMap<String, Vec<f32>>>,
    /// Runtime block cache (`raon_quant::cache`): blocks built once from the originals, mapped
    /// on later runs. Tensors missing from it are quantized in memory.
    cache: Option<GgufFile>,
    pub stats: Mutex<LoadStats>,
}

/// Where load-time quantization keeps its blocks between runs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum BlockCache {
    /// Quantize in memory on every load.
    Off,
    /// `<model dir>/runtime-<preset>-<hash>/`, falling back to the user cache directory.
    #[default]
    Default,
    /// An explicit cache root.
    Dir(std::path::PathBuf),
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LoadStats {
    /// `hit`, `built` or `off` (and the blocks path).
    pub cache: String,
    pub cache_path: Option<String>,
    /// Seconds spent building the cache (0 on a hit).
    pub cache_build_seconds: f64,
    /// Matrices taken from the cache vs quantized in memory.
    pub cached_tensors: usize,
    pub tensors: usize,
    pub params: usize,
    pub bytes: usize,
    /// Wall time inside the encoder / reading+decoding the originals / uploading the blocks.
    pub encode_seconds: f64,
    pub read_seconds: f64,
    pub upload_seconds: f64,
    /// format → (tensors, params, bytes)
    pub by_format: BTreeMap<&'static str, (usize, usize, usize)>,
}

/// A linear layer `y = x Wᵀ` whose weight stays in its stored (quantized) form on the device.
pub struct Linear {
    inner: QMatMul,
    exact: bool,
    pub name: String,
    pub out_features: usize,
    pub in_features: usize,
}

impl Linear {
    pub(crate) fn inner(&self) -> &QMatMul {
        &self.inner
    }

    #[cfg(test)]
    pub(crate) fn from_qmatmul(
        inner: QMatMul,
        exact: bool,
        name: &str,
        out_features: usize,
        in_features: usize,
    ) -> Self {
        Self {
            inner,
            exact,
            name: name.to_string(),
            out_features,
            in_features,
        }
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.contiguous()?;
        imatrix::record(&self.name, &x)?;
        if self.exact {
            self.inner.forward_exact(&x)
        } else {
            candle_core::Module::forward(&self.inner, &x)
        }
    }

    /// Single-rounding fused `silu(x @ gate^T) * (x @ up^T)` for native CUDA/ROCm W8A16 decode
    /// (`rows <= 8`, both weights `QTensor`, `exact`, imatrix disabled, verified via `AccelOp::FusedGateUpSilu`).
    /// Returns `Ok(None)` on unsupported devices/shapes so the caller runs the standard path.
    pub fn fused_gate_up_silu(&self, up: &Self, x: &Tensor) -> candle_core::Result<Option<Tensor>> {
        if !self.exact
            || !up.exact
            || imatrix::ON.load(std::sync::atomic::Ordering::Relaxed)
        {
            return Ok(None);
        }
        let dev = x.device();
        if !(dev.is_cuda() || crate::device::is_rocm(dev)) {
            return Ok(None);
        }
        let rows = x.elem_count() / x.dim(candle_core::D::Minus1)?;
        if rows == 0 || rows > 8 {
            return Ok(None);
        }
        if !matches!((&self.inner, &up.inner), (QMatMul::QTensor(_), QMatMul::QTensor(_))) {
            return Ok(None);
        }
        if !crate::device::accel_op_ok(dev, crate::device::AccelOp::FusedGateUpSilu) {
            return Ok(None);
        }
        let x = x.contiguous()?;
        match self.inner.fused_gate_up_silu(&up.inner, &x) {
            Ok(y) => Ok(Some(y)),
            Err(_) => Ok(None),
        }
    }
}

fn ggml_dtype(t: GgmlType) -> Result<GgmlDType> {
    Ok(match t {
        GgmlType::F32 => GgmlDType::F32,
        GgmlType::F16 => GgmlDType::F16,
        GgmlType::BF16 => GgmlDType::BF16,
        GgmlType::Q8_0 => GgmlDType::Q8_0,
        GgmlType::Q4K => GgmlDType::Q4K,
        GgmlType::Q5K => GgmlDType::Q5K,
        GgmlType::Q6K => GgmlDType::Q6K,
        GgmlType::I32 => bail!("I32 tensors are not weights"),
    })
}

impl Weights {
    pub fn open_gguf(path: &Path, device: Device, exact: bool) -> Result<Self> {
        let g = GgufFile::open(path)?;
        if g.kv_str("general.architecture") != Some("raon1_nano") {
            bail!("{} is not a raon1_nano GGUF", path.display());
        }
        let config_json = g
            .kv_str("raon.source_config")
            .context("GGUF has no raon.source_config")?
            .to_string();
        let kept_ids = if g.tensors.contains_key("raon.vocab.kept_ids") {
            Some(g.i32_values("raon.vocab.kept_ids")?.into_iter().map(|x| x as u32).collect())
        } else {
            None
        };
        Ok(Self {
            src: Source::Gguf(g),
            device,
            exact,
            kept_ids,
            config_json,
            mix: None,
            quant: None,
        })
    }

    pub fn open_safetensors(dir: &Path, device: Device) -> Result<Self> {
        let st = SafeTensors::open_model_dir(dir, &[])?;
        let config_json = std::fs::read_to_string(dir.join("config.json"))?;
        let mix = match (std::env::var("RAON_MIX_GGUF"), std::env::var("RAON_MIX_PATTERN")) {
            (Ok(g), Ok(p)) => {
                let pats: Vec<String> = p.split(',').filter(|x| !x.is_empty()).map(String::from).collect();
                eprintln!("[raon-infer] mixing GGUF tensors matching {pats:?} from {g}");
                Some((GgufFile::open(Path::new(&g))?, pats))
            }
            _ => None,
        };
        Ok(Self {
            src: Source::SafeTensors(st),
            device,
            exact: true,
            kept_ids: None,
            config_json,
            mix,
            quant: None,
        })
    }

    /// Original safetensors + pyramid quantization at load time (see [`LoadQuant`]).
    pub fn open_safetensors_quantized(
        dir: &Path,
        device: Device,
        preset: &str,
        cfg: &PyramidConfig,
        imatrix: Option<&Path>,
        cache: &BlockCache,
    ) -> Result<Self> {
        let mut w = Self::open_safetensors(dir, device)?;
        let Source::SafeTensors(st) = &w.src else { unreachable!() };
        let n_layers = pyramid::n_layers_from_config(dir)?;
        let formats = pyramid::nano_formats(st, cfg, n_layers)?;
        let mut stats = LoadStats {
            cache: "off".into(),
            ..Default::default()
        };
        // logis-center scheme: build the blocks once from the originals, map them afterwards.
        let cache_file = match cache {
            BlockCache::Off => None,
            BlockCache::Default | BlockCache::Dir(_) => {
                let root = match cache {
                    BlockCache::Dir(d) => Some(d.as_path()),
                    _ => None,
                };
                let spec = raon_quant::cache::CacheSpec {
                    model_dir: dir,
                    preset,
                    cfg,
                    imatrix,
                    // A GPU full head needs the Q8_0 embedding table on the device.
                    host_gather: !w.device.is_cpu(),
                    root,
                };
                match raon_quant::cache::prepare(&spec) {
                    Ok(p) => {
                        match &p {
                            raon_quant::cache::Prepared::Hit { .. } => stats.cache = "hit".into(),
                            raon_quant::cache::Prepared::Built { seconds, .. } => {
                                stats.cache = "built".into();
                                stats.cache_build_seconds = *seconds as f64;
                            }
                        }
                        stats.cache_path = Some(p.blocks().display().to_string());
                        Some(GgufFile::open(p.blocks())?)
                    }
                    Err(e) => {
                        eprintln!("[quant-cache] cannot use a block cache ({e:#}); quantizing in memory");
                        None
                    }
                }
            }
        };
        // The imatrix is only needed when something still has to be encoded here.
        let imatrix = if cache_file.is_some() {
            None
        } else {
            imatrix.map(read_imatrix).transpose()?
        };
        w.exact = crate::device::exact_matmul_native(&w.device);
        w.quant = Some(LoadQuant {
            formats,
            imatrix,
            cache: cache_file,
            stats: Mutex::new(stats),
        });
        Ok(w)
    }

    fn cached_linear(&self, q: &LoadQuant, name: &str, shape: &[usize]) -> Result<Option<QMatMul>> {
        let Some(c) = &q.cache else { return Ok(None) };
        let Some(t) = c.tensors.get(name) else { return Ok(None) };
        if t.shape != shape {
            return Ok(None);
        }
        let tm = std::time::Instant::now();
        let qt = qtensor_from_ggml(ggml_dtype(t.ty)?, c.tensor_bytes(name)?, shape.to_vec(), &self.device)?;
        let mut s = q.stats.lock().unwrap();
        s.cached_tensors += 1;
        s.tensors += 1;
        s.params += shape.iter().product::<usize>();
        s.bytes += t.nbytes();
        s.upload_seconds += tm.elapsed().as_secs_f64();
        let e = s.by_format.entry(t.ty.name()).or_default();
        e.0 += 1;
        e.1 += shape.iter().product::<usize>();
        e.2 += t.nbytes();
        Ok(Some(QMatMul::from_qtensor(qt)?))
    }

    /// Drop the pages of the original weights that were touched while quantizing: re-map the
    /// files, so only the rows gathered later (embeddings, PLE) count toward the RSS.
    pub fn release_mapped_pages(&mut self) -> Result<()> {
        match &self.src {
            Source::SafeTensors(st) => {
                let paths = st.paths.clone();
                self.src = Source::SafeTensors(SafeTensors::open(&paths)?);
                // The cached blocks were copied to the device as well.
                if let Some(q) = &mut self.quant {
                    q.cache = None;
                }
            }
            // The device copies of the blocks are made; drop the touched file pages as well.
            Source::Gguf(g) => {
                let path = g.path.clone();
                self.src = Source::Gguf(GgufFile::open(&path)?);
            }
        }
        crate::device::trim_process_working_set();
        Ok(())
    }

    fn quantize_linear(&self, q: &LoadQuant, st: &SafeTensors, name: &str, f: Format, shape: &[usize]) -> Result<QMatMul> {
        let t = std::time::Instant::now();
        let cols = shape[1];
        let data = st.all_f32(name)?;
        let t_read = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let imat = q
            .imatrix
            .as_ref()
            .and_then(|m| m.get(name))
            .filter(|v| v.len() == cols)
            .map(|v| v.as_slice());
        let bytes = encode(&data, cols, f, imat).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
        let t_enc = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let ty = f.ggml_type();
        let qt = qtensor_from_ggml(ggml_dtype(ty)?, &bytes, shape.to_vec(), &self.device)?;
        let mut s = q.stats.lock().unwrap();
        s.tensors += 1;
        s.params += data.len();
        s.bytes += bytes.len();
        s.encode_seconds += t_enc;
        s.read_seconds += t_read;
        s.upload_seconds += t.elapsed().as_secs_f64();
        let e = s.by_format.entry(ty.name()).or_default();
        e.0 += 1;
        e.1 += data.len();
        e.2 += bytes.len();
        Ok(QMatMul::from_qtensor(qt)?)
    }

    fn mixed(&self, name: &str) -> Option<&GgufFile> {
        let (g, pats) = self.mix.as_ref()?;
        pats.iter().any(|p| name.contains(p.as_str())).then_some(g)
    }

    pub fn gguf(&self) -> Option<&GgufFile> {
        match &self.src {
            Source::Gguf(g) => Some(g),
            _ => None,
        }
    }

    pub fn has(&self, name: &str) -> bool {
        match &self.src {
            Source::Gguf(g) => g.tensors.contains_key(name),
            Source::SafeTensors(s) => s.tensors.contains_key(name),
        }
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(match &self.src {
            Source::Gguf(g) => g.tensor(name)?.shape.clone(),
            Source::SafeTensors(s) => s.info(name)?.shape.clone(),
        })
    }

    /// Any tensor, decoded to f32 on the host.
    pub fn host_f32(&self, name: &str) -> Result<Vec<f32>> {
        if let Some(g) = self.mixed(name).filter(|g| g.tensors.contains_key(name)) {
            let t = g.tensor(name)?;
            let mut v = g.f32_values(name)?;
            // Undo zero padding (vision-style padded shapes do not occur in the nano GGUF).
            v.truncate(t.shape.iter().product());
            return Ok(v);
        }
        match &self.src {
            Source::Gguf(g) => g.f32_values(name),
            Source::SafeTensors(s) => s.all_f32(name),
        }
    }

    /// Any tensor as an f32 device tensor with `shape` (defaults to the stored shape).
    pub fn tensor(&self, name: &str, shape: Option<&[usize]>) -> Result<Tensor> {
        let v = self.host_f32(name).with_context(|| format!("load {name}"))?;
        let shape = match shape {
            Some(s) => s.to_vec(),
            None => {
                let s = self.shape(name)?;
                if s.is_empty() {
                    vec![1]
                } else {
                    s
                }
            }
        };
        Ok(Tensor::from_vec(v, shape, &self.device)?)
    }

    pub fn scalar(&self, name: &str) -> Result<f32> {
        Ok(self.host_f32(name)?[0])
    }

    pub fn linear(&self, name: &str) -> Result<Linear> {
        let shape = self.shape(name)?;
        if shape.len() != 2 {
            bail!("{name}: expected a matrix, got {shape:?}");
        }
        let (out_features, in_features) = (shape[0], shape[1]);
        let inner = match &self.src {
            Source::Gguf(g) => {
                let t = g.tensor(name)?;
                match t.ty {
                    GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => QMatMul::Tensor(self.tensor(name, None)?),
                    ty => {
                        let q = qtensor_from_ggml(ggml_dtype(ty)?, g.tensor_bytes(name)?, shape.clone(), &self.device)?;
                        QMatMul::from_qtensor(q)?
                    }
                }
            }
            Source::SafeTensors(st) => match self.quant.as_ref().and_then(|q| q.formats.get(name).map(|e| (q, e.1))) {
                Some((q, f)) => match self.cached_linear(q, name, &shape)? {
                    Some(m) => m,
                    None => self.quantize_linear(q, st, name, f, &shape)?,
                },
                None => QMatMul::Tensor(self.tensor(name, None)?),
            },
        };
        // Dense weights run a plain f32 matmul; `exact` only matters for quantized ones.
        Ok(Linear {
            inner,
            exact: self.exact,
            name: name.to_string(),
            out_features,
            in_features,
        })
    }

    /// Rows of a `[V, cols]` table as an f32 `[ids.len(), cols]` device tensor, gathered from the
    /// mmap (never resident as a whole).
    pub fn rows(&self, name: &str, ids: &[u32]) -> Result<Tensor> {
        if let Some(g) = self.mixed(name) {
            if name == "model.per_layer_inputs.table" {
                return self.gguf_ple_rows(g, ids);
            }
            let (ty, cols) = (g.tensor(name)?.ty, g.tensor(name)?.shape[1]);
            let mut out = vec![0f32; ids.len() * cols];
            for (i, &id) in ids.iter().enumerate() {
                dequant_row(ty, g.row_bytes(name, id as usize)?, &mut out[i * cols..(i + 1) * cols]);
            }
            return Ok(Tensor::from_vec(out, (ids.len(), cols), &self.device)?);
        }
        let cols = *self.shape(name)?.last().unwrap();
        let mut out = vec![0f32; ids.len() * cols];
        match &self.src {
            Source::Gguf(g) => {
                let ty = g.tensor(name)?.ty;
                for (i, &id) in ids.iter().enumerate() {
                    dequant_row(ty, g.row_bytes(name, id as usize)?, &mut out[i * cols..(i + 1) * cols]);
                }
            }
            Source::SafeTensors(s) => {
                out.clear();
                s.gather_f32(name, ids, &mut out)?;
            }
        }
        Ok(Tensor::from_vec(out, (ids.len(), cols), &self.device)?)
    }

    fn gguf_ple_rows(&self, g: &GgufFile, ids: &[u32]) -> Result<Tensor> {
        let kept: Vec<u32> = g.i32_values("raon.vocab.kept_ids")?.into_iter().map(|x| x as u32).collect();
        self.split_ple_rows(g, &kept, ids)
    }

    /// Per-layer-embedding table rows; in a sliced GGUF the table is split into `.kept` / `.cold`.
    pub fn ple_rows(&self, ids: &[u32]) -> Result<Tensor> {
        const T: &str = "model.per_layer_inputs.table";
        match (&self.src, &self.kept_ids) {
            (Source::Gguf(g), Some(kept)) if g.tensors.contains_key("model.per_layer_inputs.table.kept") => {
                self.split_ple_rows(g, kept, ids)
            }
            _ => self.rows(T, ids),
        }
    }

    fn split_ple_rows(&self, g: &GgufFile, kept: &[u32], ids: &[u32]) -> Result<Tensor> {
        {
            {
                let cols = g.tensor("model.per_layer_inputs.table.kept")?.shape[1];
                let mut out = vec![0f32; ids.len() * cols];
                for (i, &id) in ids.iter().enumerate() {
                    let (name, row) = match kept.binary_search(&id) {
                        Ok(r) => ("model.per_layer_inputs.table.kept", r),
                        // Cold rows are the complement in ascending order.
                        Err(rank) => ("model.per_layer_inputs.table.cold", id as usize - rank),
                    };
                    let ty = g.tensor(name)?.ty;
                    dequant_row(ty, g.row_bytes(name, row)?, &mut out[i * cols..(i + 1) * cols]);
                }
                Ok(Tensor::from_vec(out, (ids.len(), cols), &self.device)?)
            }
        }
    }
}
