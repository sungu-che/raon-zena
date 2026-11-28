//! The periodic screen watcher: every few seconds the pipeline hands it one downscaled frame; it skips the frame when
//! the screen has not changed (checked with `LumaThumb` + `WakeRule` so small corner login modals wake it too),
//! otherwise runs the Raon-VisionEncoder vision tower via `read_guarded` (`PassGuard` with `EpochWatch` and
//! `redact_rgb_for_naflex` notification masking) and scores the dictionary's screen domains.
//!
//! Privacy: frames live in memory only — never written, never logged — and are zeroed when dropped; the patches and
//! 32 × 18 luma thumbnail are wiped immediately if `privacy_epoch` changes mid-pass or a `VisionKind::Privacy` screen
//! is detected. Fixed 60 s blind backoff is gone: the pipeline's `PrivacyStateMachine` governs lock and resume.

use raon_vision::naflex::{patchify_frame_with, redact_rgb_for_naflex};
use raon_vision::thumb::{LumaThumb, WakeRule};
use raon_vision::{
    Acceptance, AnchorBank, EpochWatch, LexiconAnchors, PassGuard, PerforationPolicy, ScreenConfig, ScreenEncoder,
    WeightFormat,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};
use zena_lexicon::{Lexicon, VisionKind};

/// One frame for the watcher (RGB8). Zeroed when dropped.
pub struct ScreenFrame {
    pub rgb: Vec<u8>,
    pub w: usize,
    pub h: usize,
    /// `privacy_epoch` read right before capture; if the epoch advances before or during encoding,
    /// the pass aborts and wipes every buffer (PS-07).
    pub captured_epoch: u64,
    /// Normalised `(x, y, w, h)` notification toast zones to black out on patch boundaries (PS-05).
    pub redact_zones: Vec<(f32, f32, f32, f32)>,
}

impl ScreenFrame {
    /// Overwrites the pixels (volatile writes, so the zeroing is not optimised away before the free).
    fn wipe(&mut self) {
        for b in self.rgb.iter_mut() {
            // SAFETY: a valid, exclusive reference to an initialised byte.
            unsafe { std::ptr::write_volatile(b, 0) };
        }
    }
}

impl Drop for ScreenFrame {
    fn drop(&mut self) {
        self.wipe();
    }
}

/// What the watcher saw.
#[derive(Clone, Debug)]
pub struct ScreenObservation {
    pub key: String,
    pub kind: VisionKind,
    /// The dictionary node whose expression Zena plays.
    pub link: String,
    pub score: f32,
    /// Cosine of the domain's best anchor.
    pub cos: f32,
    /// Centre of the domain's hottest cell, 0..1 screen coordinates.
    pub peak: Option<(f32, f32)>,
    pub cooldown_s: u32,
    pub encode_ms: f32,
    pub tokens: usize,
    pub cells: usize,
}

/// A dictionary (re)load for the watcher: the vision nodes and their anchors.
pub struct AnchorsUpdate {
    pub lexicon: Lexicon,
    pub anchors: HashMap<String, Vec<f32>>,
    /// `raon_vision::fingerprint` of the checkpoint that computed `anchors`. Anchors from another checkpoint are
    /// never used (google's SigLIP2 and Raon-VE share the 1152-dimensional shape but not the space).
    pub model: String,
}

pub struct WatcherConfig {
    pub model: PathBuf,
    pub backend: raon_infer::device::Backend,
    pub format: Option<WeightFormat>,
    pub screen: ScreenConfig,
    /// Perforated pass only when the encoder runs on the CPU (noto's finding: on a GPU it saves little).
    pub perforate_on_cpu_only: bool,
    /// Global + 3×3 local-block wake rule (PS-04: wakes on small corner login modals).
    pub wake_rule: WakeRule,
    /// When a reading counts (best cosine per kind, lead over the runner-up).
    pub accept: Acceptance,
}

impl WatcherConfig {
    pub fn new(model: PathBuf) -> Self {
        WatcherConfig {
            model,
            backend: raon_infer::device::Backend::Auto,
            format: None,
            screen: ScreenConfig::default(),
            perforate_on_cpu_only: true,
            wake_rule: WakeRule::default(),
            accept: Acceptance::default(),
        }
    }
}

/// Starts the watcher thread. It loads the vision tower first (and keeps it: it is the always-on part, ~0.4 B
/// parameters), then waits for frames and dictionary updates.
pub fn spawn(
    cfg: WatcherConfig,
    frames: Receiver<ScreenFrame>,
    updates: Receiver<AnchorsUpdate>,
    out: Sender<ScreenObservation>,
    privacy_epoch: Arc<AtomicU64>,
) {
    std::thread::Builder::new()
        .name("zena-screen".into())
        .spawn(move || {
            let dev = match raon_infer::device::select_device(cfg.backend, 0) {
                Ok(d) => d,
                Err(e) => return eprintln!("[screen] no device: {e:#}"),
            };
            let gpu = !dev.is_cpu();
            let format = cfg
                .format
                .unwrap_or(if gpu { WeightFormat::F16 } else { WeightFormat::F32 });
            let mut screen = cfg.screen;
            if gpu && cfg.perforate_on_cpu_only {
                screen.perforation = PerforationPolicy::Off;
            }
            let t = Instant::now();
            let (tower, scale, bias) = match raon_vision::load_vision(&cfg.model, &dev, format) {
                Ok(x) => x,
                Err(e) => {
                    return eprintln!(
                        "[screen] vision tower not loaded ({}): {e:#}",
                        cfg.model.display()
                    )
                }
            };
            eprintln!(
                "[screen] Raon-VisionEncoder vision tower ({format:?}) on {} in {:.1}s; {} patches, {} pass",
                raon_infer::device::describe(&dev),
                t.elapsed().as_secs_f32(),
                screen.max_patches,
                match screen.perforation {
                    PerforationPolicy::Off => "full".into(),
                    p => p.tag(),
                }
            );
            let enc = ScreenEncoder::new(tower, screen);
            let fingerprint = raon_vision::fingerprint(&cfg.model);
            // Raon-VE: CLIP mean / std, bicubic; google's SigLIP2: 0.5 / 0.5, bilinear (preprocessor_config.json).
            let preprocess = raon_vision::preprocess_for(&cfg.model);
            let mut bank = AnchorBank::empty(scale, bias);
            let mut last_thumb: Option<LumaThumb> = None;
            let mut seen_epoch = privacy_epoch.load(Ordering::Acquire);
            let mut warned = false;
            loop {
                let cur_epoch = privacy_epoch.load(Ordering::Acquire);
                if cur_epoch != seen_epoch {
                    seen_epoch = cur_epoch;
                    if let Some(t) = &mut last_thumb {
                        t.wipe();
                    }
                    last_thumb = None;
                }
                for u in updates.try_iter() {
                    if let Some(t) = &mut last_thumb {
                        t.wipe();
                    }
                    last_thumb = None;
                    if !u.anchors.is_empty() && u.model != fingerprint {
                        bank = AnchorBank::empty(scale, bias);
                        eprintln!(
                            "[screen] the anchors were computed by '{}', this model is '{fingerprint}': not used until they are recomputed",
                            u.model
                        );
                        continue;
                    }
                    bank = AnchorBank::build(&u.lexicon, &u.anchors, scale, bias);
                    eprintln!("[screen] {} screen domains anchored", bank.domains.len());
                }
                let mut frame = match frames.recv_timeout(Duration::from_millis(500)) {
                    Ok(f) => f,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(_) => return,
                };
                if bank.is_empty() {
                    if !warned {
                        eprintln!("[screen] no anchors yet: run `zena-lexicon anchors` once to let Zena read the screen");
                        warned = true;
                    }
                    continue;
                }
                if frame.captured_epoch != privacy_epoch.load(Ordering::Acquire) {
                    if let Some(t) = &mut last_thumb {
                        t.wipe();
                    }
                    last_thumb = None;
                    continue;
                }
                let masked = if frame.redact_zones.is_empty() {
                    Vec::new()
                } else {
                    redact_rgb_for_naflex(
                        &mut frame.rgb,
                        frame.w,
                        frame.h,
                        &frame.redact_zones,
                        enc.cfg.max_patches,
                        &preprocess,
                    )
                };
                let mut thumb = LumaThumb::from_rgb(&frame.rgb, frame.w, frame.h);
                if !cfg.wake_rule.check(last_thumb.as_ref(), &thumb).wakes() {
                    thumb.wipe();
                    continue;
                }
                let captured_epoch = frame.captured_epoch;
                let mut grid = patchify_frame_with(
                    &frame.rgb,
                    frame.w,
                    frame.h,
                    enc.cfg.max_patches,
                    &preprocess,
                );
                drop(frame);
                let guard = PassGuard {
                    epoch: Some(EpochWatch::new(&privacy_epoch, captured_epoch)),
                    masked: &masked,
                    ..PassGuard::default()
                };
                let reading = match enc.read_guarded(&mut grid, &bank, &guard) {
                    Ok(Some(r)) => r,
                    Ok(None) => {
                        // Privacy lock interrupted the pass mid-flight: wipe every residue and emit nothing.
                        thumb.wipe();
                        if let Some(t) = &mut last_thumb {
                            t.wipe();
                        }
                        last_thumb = None;
                        continue;
                    }
                    Err(e) => {
                        thumb.wipe();
                        eprintln!("[screen] encoder failed: {e:#}");
                        continue;
                    }
                };
                drop(grid);
                let Some(top) = cfg.accept.accept(&reading.scores) else {
                    last_thumb = Some(thumb);
                    continue;
                };
                if top.kind == VisionKind::Privacy {
                    thumb.wipe();
                    if let Some(t) = &mut last_thumb {
                        t.wipe();
                    }
                    last_thumb = None;
                } else {
                    last_thumb = Some(thumb);
                }
                let _ = out.send(ScreenObservation {
                    key: top.key.clone(),
                    kind: top.kind,
                    link: top.link.clone(),
                    score: top.score,
                    cos: top.cos,
                    peak: reading.peak,
                    cooldown_s: top.cooldown_s,
                    encode_ms: reading.encode_ms,
                    tokens: reading.report.tokens_run,
                    cells: reading.report.cells,
                });
            }
        })
        .expect("spawn screen watcher");
}

#[cfg(test)]
mod tests {
    use super::*;
    use raon_vision::thumb::Wake;

    #[test]
    fn frames_are_zeroed_when_dropped_and_change_wakes_on_global_or_local_corner() {
        let mut f = ScreenFrame {
            rgb: vec![200; 64 * 36 * 3],
            w: 64,
            h: 36,
            captured_epoch: 0,
            redact_zones: vec![],
        };
        let a = LumaThumb::from_rgb(&f.rgb, f.w, f.h);
        assert!((a.values()[0] - 200.0).abs() < 1e-3);
        let rule = WakeRule::default();
        assert_eq!(rule.check(Some(&a), &a), Wake::Still);

        let mut g = ScreenFrame {
            rgb: vec![200; 64 * 36 * 3],
            w: 64,
            h: 36,
            captured_epoch: 0,
            redact_zones: vec![],
        };
        g.rgb[..64 * 18 * 3].iter_mut().for_each(|b| *b = 0);
        let b = LumaThumb::from_rgb(&g.rgb, g.w, g.h);
        assert_eq!(rule.check(Some(&a), &b), Wake::Global);

        f.wipe();
        assert!(
            f.rgb.iter().all(|&b| b == 0),
            "what Drop does before the buffer is freed"
        );
    }
}
