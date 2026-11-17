//! KRAFTON Raon Zena — overlay client.
//!
//! ```text
//! raon-zena [--synthetic] [--model out/raon-1-nano.zena.gguf | models/KRAFTON/Raon-1-nano]
//!           [--quant-preset zena|prd|q8] [--imatrix out/raon-1-nano.imatrix.dat] [--quant-cache DIR | --no-quant-cache]
//!           [--tokenizer tokenizer.json] [--llm-device auto|cpu|cuda|rocm|vulkan|metal] [--head full|kept|guarded]
//!           [--llm-idle SECS]                       unload Raon-1-nano after SECS without a request (0: keep it)
//!           [--lexicon data/zena_lexicon.json]      dictionary seed (default: the bundled one)
//!           [--lexicon-db DIR | --no-lexicon-db]    the LanceDB dictionary (default: the per-user one) [--learn]
//!           [--screen models/KRAFTON/Raon-VisionEncoder | models/google/siglip2-so400m-patch16-naflex]
//!           [--screen-tokenizer tokenizer.json]     to recompute stale anchors (default: next to the checkpoint)
//!           [--screen-every SECS] [--screen-patches N]
//!           [--screen-pass auto|full|checker|fovea|checker-x|fovea-x] [--screen-format f32|f16|bf16|q8]
//!           [--screen-device auto|cpu|cuda|rocm|vulkan|metal]
//!           [--vision out/raon-ve-vision.pyramid.gguf]   (residency report only)
//!           [--proactive SECS]                      quiet time before Zena speaks up on her own (0: never; default 90)
//!           [--history N]                           exchanges remembered for Lv3 prompts (default 4)
//!           [--lang ko|en|…] [--locale-dir DIR]     Zena's language pack (persona, prompt, cues, verdict lines;
//!                                                   data/locales/<code>.json, more packs from DIR; default ko)
//!           [--no-pd] [--pd-share]                  premium-discount bridge (records, verdict lines; on when
//!                                                   premium-discount-mcp is installed) / share feedback as votes
//!           [--nlp auto|rules|v1]                   natural-language router: v2 with installed models (Stanza +
//!                                                   granite + SDS), v1 + rule analyzer, or v1 (default auto)
//!           [--track procedural|solidcraft|subculture-2d|subculture-3d]   the avatar body to start with
//!           [--vrm model.vrm] [--chibi-kit DIR]     Track B assets (3D VRM body, layered 2D SD parts)
//!           [--no-avatar-download]                  without --vrm / --chibi-kit, do not fetch the avatar catalog's
//!                                                   default model (zena_avatar_mcp::catalog) into the asset cache
//!           [--solidcraft PATH | --no-solidcraft]   SolidCraft CLI for Track A (default: found on PATH / Program Files)
//!           [--no-subculture-mcp]                   build Track B assets in-process instead of over MCP
//!           [--asset-cache DIR] [--font FONT.ttf]
//!           [--snapshot DIR]                        render scripted moments headless to PNGs and exit
//! ```
//! Build with `--features cuda` / `rocm` / `vulkan` / `metal` (+ `flash-attn`) to accelerate the LLM and the screen
//! encoder. Talk to Zena in the overlay's chat box (`Ctrl+Shift+T`, Enter to send, Esc to close) or type a line on
//! stdin (voice input is not wired yet).

mod assets;
mod capture;
mod course;
mod lexicon;
mod llm;
mod overlay;
mod pd_bridge;
mod pipeline;
mod renderers;
mod snapshot;
mod watcher;
#[cfg(windows)]
mod win32;

use anyhow::{bail, Result};
use raon_quant::gguf::GgufFile;
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;
use winit::event_loop::EventLoop;
use zena_core::geometry::ScreenGeometry;

struct Args {
    synthetic: bool,
    model: Option<PathBuf>,
    vision: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    llm_device: raon_infer::device::Backend,
    head: raon_infer::head::HeadMode,
    quant_preset: String,
    imatrix: Option<PathBuf>,
    quant_cache: raon_infer::weights::BlockCache,
    llm_idle: u64,
    lexicon: Option<PathBuf>,
    lexicon_db: Option<PathBuf>,
    learn: bool,
    screen: Option<PathBuf>,
    screen_tokenizer: Option<PathBuf>,
    screen_every: f32,
    screen_patches: usize,
    screen_pass: String,
    screen_format: Option<raon_vision::WeightFormat>,
    screen_device: raon_infer::device::Backend,
    proactive_s: u64,
    lang: String,
    locale_dir: Option<PathBuf>,
    pd: bool,
    pd_share: bool,
    history: usize,
    nlp: pipeline::NlpMode,
    track: zena_core::tools::AvatarTrack,
    vrm: Option<PathBuf>,
    chibi_kit: Option<PathBuf>,
    avatar_download: bool,
    solidcraft: Option<PathBuf>,
    subculture_mcp: bool,
    asset_cache: PathBuf,
    font: Option<PathBuf>,
    snapshot: Option<PathBuf>,
}

fn parse_track(s: &str) -> zena_core::tools::AvatarTrack {
    use zena_core::tools::AvatarTrack as T;
    match s.to_ascii_lowercase().replace('_', "-").as_str() {
        "solidcraft" | "robot" | "a" => T::Solidcraft,
        "subculture-2d" | "2d" | "b1" => T::Subculture2d,
        "subculture-3d" | "3d" | "vrm" | "b2" => T::Subculture3d,
        _ => T::Procedural,
    }
}

fn args() -> Args {
    let mut a = Args {
        synthetic: !cfg!(windows),
        model: None,
        vision: None,
        tokenizer: None,
        llm_device: raon_infer::device::Backend::Auto,
        head: raon_infer::head::HeadMode::Full,
        quant_preset: "zena".into(),
        imatrix: None,
        quant_cache: raon_infer::weights::BlockCache::Default,
        llm_idle: 120,
        lexicon: None,
        lexicon_db: Some(lexicon::default_db()),
        learn: false,
        screen: None,
        screen_tokenizer: None,
        screen_every: 3.0,
        screen_patches: 256,
        screen_pass: "auto".into(),
        screen_format: None,
        screen_device: raon_infer::device::Backend::Auto,
        proactive_s: 90,
        lang: zena_locale::DEFAULT_LANGUAGE.into(),
        locale_dir: None,
        pd: true,
        pd_share: false,
        history: 4,
        nlp: pipeline::NlpMode::Auto,
        track: zena_core::tools::AvatarTrack::Procedural,
        vrm: None,
        chibi_kit: None,
        avatar_download: true,
        solidcraft: zena_avatar_mcp::solidcraft::find_solidcraft(),
        subculture_mcp: true,
        asset_cache: assets::default_cache(),
        font: None,
        snapshot: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        match x.as_str() {
            "--synthetic" => a.synthetic = true,
            "--model" => a.model = it.next().map(PathBuf::from),
            "--vision" => a.vision = it.next().map(PathBuf::from),
            "--tokenizer" => a.tokenizer = it.next().map(PathBuf::from),
            "--llm-device" => a.llm_device = it.next().unwrap_or_default().parse().unwrap_or(raon_infer::device::Backend::Auto),
            "--head" => a.head = it.next().unwrap_or_default().parse().unwrap_or(raon_infer::head::HeadMode::Full),
            "--quant-preset" => a.quant_preset = it.next().unwrap_or_else(|| "zena".into()),
            "--imatrix" => a.imatrix = it.next().map(PathBuf::from),
            "--quant-cache" => {
                a.quant_cache = it
                    .next()
                    .map(|d| raon_infer::weights::BlockCache::Dir(PathBuf::from(d)))
                    .unwrap_or_default()
            }
            "--no-quant-cache" => a.quant_cache = raon_infer::weights::BlockCache::Off,
            "--llm-idle" => a.llm_idle = it.next().and_then(|v| v.parse().ok()).unwrap_or(120),
            "--lexicon" => a.lexicon = it.next().map(PathBuf::from),
            "--lexicon-db" => a.lexicon_db = it.next().map(PathBuf::from),
            "--no-lexicon-db" => a.lexicon_db = None,
            "--learn" => a.learn = true,
            "--screen" => a.screen = it.next().map(PathBuf::from),
            "--screen-tokenizer" => a.screen_tokenizer = it.next().map(PathBuf::from),
            "--screen-every" => a.screen_every = it.next().and_then(|v| v.parse().ok()).unwrap_or(3.0),
            "--screen-patches" => a.screen_patches = it.next().and_then(|v| v.parse().ok()).unwrap_or(256),
            "--screen-pass" => a.screen_pass = it.next().unwrap_or_else(|| "auto".into()),
            "--screen-format" => a.screen_format = it.next().and_then(|v| v.parse().ok()),
            "--screen-device" => a.screen_device = it.next().unwrap_or_default().parse().unwrap_or(raon_infer::device::Backend::Auto),
            "--proactive" => a.proactive_s = it.next().and_then(|v| v.parse().ok()).unwrap_or(90),
            "--history" => a.history = it.next().and_then(|v| v.parse().ok()).unwrap_or(4),
            "--lang" => a.lang = it.next().unwrap_or_else(|| zena_locale::DEFAULT_LANGUAGE.into()),
            "--locale-dir" => a.locale_dir = it.next().map(PathBuf::from),
            "--no-pd" => a.pd = false,
            "--pd-share" => a.pd_share = true,
            "--nlp" => match it.next().unwrap_or_default().parse() {
                Ok(m) => a.nlp = m,
                Err(e) => eprintln!("{e}"),
            },
            "--track" => a.track = parse_track(&it.next().unwrap_or_default()),
            "--vrm" => a.vrm = it.next().map(PathBuf::from),
            "--chibi-kit" => a.chibi_kit = it.next().map(PathBuf::from),
            "--no-avatar-download" => a.avatar_download = false,
            "--solidcraft" => a.solidcraft = it.next().map(PathBuf::from),
            "--no-solidcraft" => a.solidcraft = None,
            "--no-subculture-mcp" => a.subculture_mcp = false,
            "--asset-cache" => a.asset_cache = it.next().map(PathBuf::from).unwrap_or_else(assets::default_cache),
            "--font" => a.font = it.next().map(PathBuf::from),
            "--snapshot" => a.snapshot = it.next().map(PathBuf::from),
            _ => eprintln!("ignoring argument {x}"),
        }
    }
    a
}

/// Map a pyramid GGUF and report what would be resident vs gathered from disk.
fn open_model(path: &std::path::Path) -> Result<GgufFile> {
    let g = GgufFile::open(path)?;
    let meta: std::collections::BTreeMap<String, serde_json_lite::Meta> =
        serde_json_lite::parse(g.kv_str("raon.tensor_meta").unwrap_or("{}"));
    let (mut dev, mut host) = (0usize, 0usize);
    for (n, t) in &g.tensors {
        match meta.get(n).map(|m| m.residency.as_str()) {
            Some("Device") => dev += t.nbytes(),
            _ => host += t.nbytes(),
        }
    }
    eprintln!(
        "{}: arch={} resident {:.0} MiB, disk-gather {:.0} MiB (mmap)",
        path.display(),
        g.kv_str("general.architecture").unwrap_or("?"),
        dev as f64 / 1048576.0,
        host as f64 / 1048576.0
    );
    Ok(g)
}

/// Tiny residency extractor so the client does not need a JSON dependency.
mod serde_json_lite {
    pub struct Meta {
        pub residency: String,
    }
    pub fn parse(s: &str) -> std::collections::BTreeMap<String, Meta> {
        let mut out = std::collections::BTreeMap::new();
        for part in s.split("},\"") {
            let name = part.trim_start_matches("{\"").split('"').next().unwrap_or("").to_string();
            if let Some(i) = part.find("\"residency\":\"") {
                let r = part[i + 13..].split('"').next().unwrap_or("").to_string();
                out.insert(name, Meta { residency: r });
            }
        }
        out
    }
}

fn main() -> Result<()> {
    let a = args();
    // The language pack first: the LLM backend caches the system persona in its prompt prefix when it loads
    if let Some(dir) = &a.locale_dir {
        match zena_locale::load_dir(dir) {
            Ok(codes) => eprintln!("[locale] loaded {codes:?} from {}", dir.display()),
            Err(e) => bail!("language packs in {}: {e}", dir.display()),
        }
    }
    zena_locale::set_language(&a.lang).map_err(anyhow::Error::msg)?;
    if let Some(dir) = &a.snapshot {
        return snapshot::run(dir, a.font.as_deref(), a.vrm.as_deref());
    }
    // PRD §3.5: verify the fused kernels before enabling them.
    let checks = raon_quant::kernels::verify_kernels(1e-3);
    if checks.iter().any(|c| !c.ok) {
        bail!("kernel self-test failed: {checks:?}");
    }
    // Report residency of pre-quantized GGUFs; a safetensors directory is quantized while the
    // LLM backend loads it.
    let _models: Vec<GgufFile> = [a.model.as_ref(), a.vision.as_ref()]
        .into_iter()
        .flatten()
        .filter(|p| p.is_file())
        .map(|p| open_model(p))
        .collect::<Result<_>>()?;

    let (src, screen): (Box<dyn capture::FrameSource>, ScreenGeometry) = {
        #[cfg(windows)]
        {
            if a.synthetic {
                (Box::new(capture::SyntheticSource::new(480, 270)), ScreenGeometry::new(1920, 1080))
            } else {
                let s = capture::DxgiSource::new(4)?;
                let (w, h) = capture::FrameSource::size(&s);
                (Box::new(s), ScreenGeometry::new((w * 4) as u32, (h * 4) as u32))
            }
        }
        #[cfg(not(windows))]
        {
            let _ = a.synthetic;
            (Box::new(capture::SyntheticSource::new(480, 270)), ScreenGeometry::new(1920, 1080))
        }
    };

    let event_loop = EventLoop::<overlay::UserEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();
    let running = Arc::new(AtomicBool::new(true));

    let (utx, urx) = mpsc::channel::<String>();
    {
        let utx = utx.clone();
        std::thread::spawn(move || {
            for line in std::io::stdin().lock().lines().map_while(Result::ok) {
                if utx.send(line).is_err() {
                    break;
                }
            }
        });
    }
    let (ctx, crx) = mpsc::channel::<pipeline::AvatarCmd>();

    // The word dictionary: LanceDB (seeded from the JSON on first run) or the JSON itself.
    let loaded = match &a.lexicon_db {
        Some(db) => lexicon::open(db, a.lexicon.as_deref())?,
        None => lexicon::Loaded {
            lexicon: lexicon::read_json(a.lexicon.as_deref())?,
            anchors: Default::default(),
            anchors_model: String::new(),
            source: "JSON".into(),
        },
    };
    eprintln!(
        "[lexicon] {} — {} domains, {} screen domains",
        loaded.source,
        loaded.lexicon.nodes.len(),
        loaded.lexicon.vision.len()
    );

    // The periodic screen watcher (Raon-VisionEncoder vision tower only; anchors come from the dictionary).
    let (frame_tx, frame_rx) = mpsc::sync_channel::<watcher::ScreenFrame>(1);
    let (obs_tx, obs_rx) = mpsc::channel::<watcher::ScreenObservation>();
    let (anchor_tx, anchor_rx) = mpsc::channel::<watcher::AnchorsUpdate>();
    let screen_on = a.screen.is_some();
    if let Some(model) = &a.screen {
        let mut cfg = watcher::WatcherConfig::new(model.clone());
        cfg.backend = a.screen_device;
        cfg.format = a.screen_format;
        cfg.screen.max_patches = a.screen_patches;
        match a.screen_pass.as_str() {
            // noto-sandbox's measurements: the perforated pass pays off on the CPU only.
            "auto" => cfg.screen.perforation = raon_vision::PerforatedCfg::parse("fovea"),
            "full" => {
                cfg.screen.perforation = None;
                cfg.perforate_on_cpu_only = false;
            }
            p => {
                cfg.screen.perforation =
                    Some(raon_vision::PerforatedCfg::parse(p).ok_or_else(|| anyhow::anyhow!("unknown --screen-pass {p}"))?);
                cfg.perforate_on_cpu_only = false;
            }
        }
        anchor_tx.send(watcher::AnchorsUpdate {
            lexicon: loaded.lexicon.clone(),
            anchors: loaded.anchors.clone(),
            model: loaded.anchors_model.clone(),
        })?;
        watcher::spawn(cfg, frame_rx, anchor_rx, obs_tx);
    }
    let (lex_tx, lex_rx) = mpsc::channel();
    let learn_tx = a
        .lexicon_db
        .clone()
        .map(|db| {
            // A screen model whose anchors are missing or were computed by another checkpoint gets them recomputed
            // by the dictionary service (needs the model's tokenizer.json).
            let reanchor = a.screen.as_deref().and_then(|m| lexicon::Reanchor::find(m, a.screen_tokenizer.as_deref()));
            lexicon::spawn_service(db, Duration::from_secs(3), 3, lex_tx, screen_on.then_some(anchor_tx), reanchor)
        });

    // Lv3 backend: Raon-1-nano when a model is given (CPU / CUDA / ROCm / Vulkan / Metal through the
    // candle-rocm-vulkan fork, chosen by --llm-device), loaded on demand and dropped when idle; otherwise the
    // template stub.
    // The completion is parsed while it streams: the expression is shown as soon as its parameters close and the
    // line is spoken sentence by sentence; the answer goes back to the pipeline's dialogue memory.
    let (reply_tx, reply_rx) = mpsc::channel::<pipeline::Reply>();
    let on_done = pipeline::StreamToAvatar::new(ctx.clone(), Some(reply_tx));
    let worker = match &a.model {
        Some(model) => {
            let tok = a.tokenizer.clone().unwrap_or_else(|| {
                if model.is_dir() {
                    model.join("tokenizer.json")
                } else {
                    model.with_file_name("tokenizer.json")
                }
            });
            let (model, dev, head, preset, imatrix, cache) = (
                model.clone(),
                a.llm_device,
                a.head,
                a.quant_preset.clone(),
                a.imatrix.clone(),
                a.quant_cache.clone(),
            );
            let loader: llm::Loader = Box::new(move || {
                Ok(Box::new(llm::RaonNanoBackend::load(
                    &model,
                    &tok,
                    dev,
                    head,
                    &preset,
                    imatrix.as_deref(),
                    &cache,
                )?) as Box<dyn llm::LlmBackend>)
            });
            if a.llm_idle == 0 {
                llm::LlmWorker::spawn(loader()?, on_done)
            } else {
                llm::LlmWorker::spawn_lazy("raon-1-nano", loader, Duration::from_secs(a.llm_idle), on_done)
            }
        }
        None => llm::LlmWorker::spawn(Box::new(llm::TemplateBackend), on_done),
    };
    {
        let running = running.clone();
        let links = pipeline::Links {
            llm: worker,
            utterances: urx,
            out: ctx,
            lexicon: lex_rx,
            learn: if a.learn { learn_tx } else { None },
            screen_frames: screen_on.then_some(frame_tx),
            screen: screen_on.then_some(obs_rx),
            replies: Some(reply_rx),
        };
        let cfg = pipeline::PipelineConfig {
            screen_interval_ms: (a.screen_every * 1000.0) as u64,
            learn_min: a.learn.then_some(0.75),
            proactive_silence_ms: a.proactive_s * 1000,
            history_turns: a.history,
            nlp: a.nlp,
            language: zena_locale::current().code.clone(),
            pd: a.pd.then(|| pipeline::PdOptions { share: a.pd_share }),
            ..Default::default()
        };
        let lexicon = loaded.lexicon;
        std::thread::spawn(move || {
            let r = pipeline::run(src, links, lexicon, cfg);
            if running.load(Ordering::SeqCst) {
                if let Err(e) = r {
                    eprintln!("pipeline stopped: {e:#}");
                }
            }
        });
    }
    {
        let proxy = proxy.clone();
        std::thread::spawn(move || {
            for cmd in crx {
                if proxy.send_event(overlay::UserEvent::Cmd(cmd)).is_err() {
                    break;
                }
            }
        });
    }
    #[cfg(windows)]
    {
        let proxy = proxy.clone();
        win32::spawn_hotkeys(move |h| {
            let _ = proxy.send_event(overlay::UserEvent::Hotkey(h));
        });
    }
    // Dual-track assets: SolidCraft (Track A) and zena-subculture-mcp (Track B) over MCP in the background, built
    // in-process when they are not available.
    let assets = assets::Assets::spawn(
        zena_avatar_mcp::service::ServiceConfig {
            cache_dir: a.asset_cache.clone(),
            solidcraft: a.solidcraft.clone(),
            subculture_mcp: if a.subculture_mcp { assets::sibling_server() } else { None },
            vrm_path: a.vrm.clone(),
            chibi_kit: a.chibi_kit.clone(),
            // Without --vrm the 3D track downloads the catalog's default model (pinned, SHA-256 checked) instead of
            // falling back to the procedural face.
            catalog: a.avatar_download.then(|| a.asset_cache.clone()),
            stamp_frames: 8,
        },
        overlay::OverlayAssets(proxy.clone()),
    );
    eprintln!(
        "[assets] track {:?}; SolidCraft {}; cache {}",
        a.track,
        a.solidcraft
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "not found (built-in forge)".into()),
        a.asset_cache.display()
    );
    let mut app = overlay::Overlay::new(running, screen, utx)
        .with_assets(assets, a.track)
        .with_font(a.font.clone());
    event_loop.run_app(&mut app)?;
    Ok(())
}
