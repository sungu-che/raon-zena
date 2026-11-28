//! Worker thread: capture → privacy state machine → event sensor → tracker → ingress PII guard →
//! NLP-NMS router (word dictionary) → feedback / credit / 600 ms two-track RAG / proactive 5-gate
//! coaching / browser UIA watcher → Lv2 expressions or Lv3 generation → avatar commands for UI.
//!
//! All 6 scenario state machines are wired here:
//! * **Scenario 1 (Course & badges)**: `StickyNumberer`, `SpokenSlotLock`, `NavTargetProjector`, `CourseBrief`.
//! * **Scenario 2 (Proactive coaching)**: `LiveEpisodeWindow`, `FrictionBaselines::peek_live`,
//!   `PrewarmGuard`, 5-gate `InterruptSignals`, `PendingCoachHook`, `CoachingFatigueController`,
//!   `EphemeralOffer`, `route_proactive`, `ProactiveTimer::due_gated`.
//! * **Scenario 3 (Two-track RAG & quarantine)**: `TopicTracker`, `RagCoordinator` (0 ms `PreThink`,
//!   600 ms `LocalMirrorIndex` vs `pd_search_merged`, `SceneRagCache`, `ask_epoch × privacy_epoch`),
//!   `QuarantineLedger`, `ExposureLedger::record_delivered_turn`.
//! * **Scenario 4 (Feedback & lossless sync)**: `PdBridge` (`SyncLoop`, `LocalFilter`, `LocalVerdictStore`,
//!   reversible `▲`/`▼` pins, `OutcomeVoteGate`, `extract_pragmatic_feedback`, `attribute_and_update_duels`).
//! * **Scenario 5 (Web analytic & shopping coach)**: `AsyncBrowserUiaWorker`, `BrowserSessionTracker`,
//!   `build_web_nav_overlay`, `WebCoachBrief` (`CommitGuard`), 11th MCP tool `pd_analytic` (`AnalyticBrief`).
//! * **Scenario 6 (Privacy shield)**: `PrivacyStateMachine`, 500 ms `ZeroPixelProbe`, dormant
//!   `GameOverlayUI.exe` filter, `PurgableSensor`, `IngressPiiScrubber` + checked BIP-39 seed guard,
//!   and `BlindAssist` (`SessionContext.privacy_locked`).

use crate::browser_uia::{
    build_web_nav_overlay, is_browser_exe, precompute_label_similarity, AsyncBrowserUiaWorker,
    BrowserSessionTracker, NavBadge, NavBadgeOverlay, ScanRequest, TabTransition,
};
use crate::capture::FrameSource;
use crate::lexicon::LexiconMsg;
use crate::llm::{JobHandler, LlmJob, LlmWorker};
use crate::pd_bridge::{
    default_quarantine_path, detect_seed_phrase_checked, filter_dormant_game_overlays,
    topic_hash_id, DropOutcome, OverlaySyncView, PdBridge, RagOutcome,
};
use crate::watcher::{ScreenFrame, ScreenObservation};
use nlp_nms_router::{Caught, Hit, NlpNmsRouter, RouterInput};
use premium_discount_nav::{
    evaluate_privacy, payment_route, receipt_finish_page, BehaviorParams, Event as NavEvent,
    FrictionAlertStage, FrictionBaselines, LiveEpisodeWindow, PrivacyProbeInput, PrivacyShield,
    PrivacyStateMachine, PrivacyTrigger, Signal as NavSignal, StickyNumberer, SurfaceSignature,
    ZeroPixelProbe,
};
use premium_discount_schema::envelope;
use raon_vision::naflex::bgra_to_rgb;
use raon_vision::{extract_rois, LogLumSensor, PurgableSensor, RoiTracker};
use premium_discount_window::{
    assess_surfaces, foreground_window, visible_windows_intersecting, PxRect,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use zena_core::avatar::viseme_timeline;
use zena_core::brief::{
    arbitrate_rag_with_telemetry, CourseBrief, NextStepHint, TelemetryTier, WebCoachBrief,
};
use zena_core::claims::PackedVerdictSlot;
use zena_core::coach::{
    route_proactive_with_reflex, CoachEvidence, CoachingFatigueController, EphemeralOffer,
    InterruptSignals, PendingCoach, PendingCoachHook, ProactiveRoute, ReflexCompiler,
};
use zena_core::cues::{cue, cue_for_reason, CuePlayer};
use zena_core::exposure::{ExposureLedger, ExposureRole, ExposureTargetKind};
use zena_core::geometry::{FlowDirection, ScreenGeometry};
use zena_core::ingress_guard::{IngressPiiScrubber, IngressVerdict};
use zena_core::nav_target::SpokenSlotLock;
use zena_core::prompt::{build_fast_path_prompt, Dialogue, Situation};
use zena_core::reaction::{Observation, ReactionLevel, ReactionRouter};
use zena_core::reference::{
    normalize_cc, CoalesceOutcome, FocusedRegion, ScreenAnchorSnapshot, SpeechCoalescer,
    TopicTracker,
};
use zena_core::session::{Proactive, ProactiveTimer, SessionTracker};
use zena_core::stream::{StreamEvent, ToolCallStream};
use zena_core::tools::{
    BodyMotion, EmotionState, GazeTarget, SystemCommand, ToolCall, WindowLayerMode, ZenaAssetArgs,
    ZenaAvatarArgs,
};
use zena_lexicon::{Action, Express, ExpressLevel, Lexicon, VisionKind};
use zena_verdict::{
    AttributionCandidate, CreditTargetKind, FeedbackContext, FeedbackToken, Tier, UserPin,
};

/// Actions sent from the overlay UI thread back to the pipeline worker.
#[derive(Debug, Clone)]
pub enum OverlayAction {
    TypingChanged(bool),
    DraggingChanged(bool),
    AvatarHidden(bool),
    /// User dropped a note onto a target window (`cc`).
    DropNote {
        cc: String,
        to: Option<String>,
        text: String,
    },
    /// User clicked ▲ or ▼ on a Stage 2 drawer row (local-only, reversible, 0 server packets).
    TogglePin {
        record_id: String,
        pin: UserPin,
    },
    /// User viewed Stage 1 rejection chip or opened Stage 2 drawer.
    AckRejections,
    /// User opened/refreshed Stage 2 drawer.
    RefreshDrawer,
    /// User selected badge 1..=9 (respects `CommitGuard` for final payment buttons).
    BadgePicked {
        number: u8,
        commit_guarded: bool,
    },
    /// User dismissed a proactive coaching bubble.
    DismissCoaching,
    /// Overlay began graceful shutdown (Farewell 0 ms).
    ShutdownStarted,
}

/// What the UI thread is asked to do.
#[derive(Debug, Clone)]
pub enum AvatarCmd {
    /// Lv1: look toward an event, no speech.
    Gaze(GazeTarget),
    /// 0 ms immediate pre-think pose when an interactive question arrives (CR-05).
    PreThink {
        #[cfg_attr(not(test), allow(dead_code))]
        epoch: u64,
    },
    /// Lv2/Lv3: a whole reaction — expression with optional speech and its viseme timeline (replaces what Zena was
    /// saying).
    Act {
        args: ZenaAvatarArgs,
        level: ReactionLevel,
        visemes: Vec<(u32, zena_core::avatar::Viseme)>,
        speech_ms: u32,
    },
    /// Streamed: gaze / emotion / motion / emote to play now; the line follows as [`AvatarCmd::Speak`].
    Expression {
        args: ZenaAvatarArgs,
        level: ReactionLevel,
    },
    /// Streamed: the next sentence, queued after what Zena is saying.
    Speak {
        text: String,
        level: ReactionLevel,
        visemes: Vec<(u32, zena_core::avatar::Viseme)>,
        speech_ms: u32,
    },
    System(SystemCommand, Option<WindowLayerMode>),
    /// Switch the avatar track / put on or take off an accessory.
    Asset(ZenaAssetArgs),
    /// Privacy lock transition (`PrivacyStateMachine` locked ↔ normal, PS-12).
    PrivacyLock { locked: bool },
    /// Immediately clears speech bubble & TTS (no 2.5 s linger).
    ClearSpeech,
    /// 1–9 numbered overlay badges for the active screen / browser tab.
    NavBadges(NavBadgeOverlay),
    /// Stage 0 status pill, Stage 1 rejections, Stage 2 history/verdict drawer rows.
    SyncStatus(OverlaySyncView),
}

impl AvatarCmd {
    /// The words Zena says with this command.
    pub fn speech(&self) -> Option<&str> {
        match self {
            AvatarCmd::Act { args, .. } => args.speech_text.as_deref(),
            AvatarCmd::Speak { text, .. } => Some(text),
            _ => None,
        }
    }
}

pub struct PipelineConfig {
    pub sensor_threshold: f32,
    pub min_tile_events: u32,
    /// How often a frame goes to the screen watcher (it skips unchanged frames itself).
    pub screen_interval_ms: u64,
    /// Learn spans caught by vectors alone with at least this score (`None`: no learning).
    pub learn_min: Option<f32>,
    /// Minimum spacing of two dictionary expressions of the same domain.
    pub expression_gap_ms: u64,
    /// Quiet time before Zena speaks up on her own (0: never).
    pub proactive_silence_ms: u64,
    /// Exchanges remembered for Lv3 prompts.
    pub history_turns: usize,
    /// Which natural-language router runs.
    pub nlp: NlpMode,
    /// Optional path to `premium-discount-mcp` binary (`None`: offline mirror mode).
    pub pd_mcp: Option<PathBuf>,
    /// Explicit user opt-in for outcome-confirmed server votes (`vote_outcome`).
    pub consent_share_outcomes: bool,
    /// Whether to poll host OS foreground/visible windows & UIA browser tabs (disabled in headless unit tests).
    pub enable_host_probes: bool,
    /// Custom path for `quarantine_v1.json` (default `%LOCALAPPDATA%\RaonZena\sds\quarantine_v1.json`).
    pub quarantine_path: Option<PathBuf>,
}

/// The natural-language router (`nlp_nms_router::router::Models`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NlpMode {
    /// v1: hashed embedder, no analyzer (deterministic; tests).
    V1,
    /// v1 scoring + the rule analyzer (auxiliaries, negation).
    Rules,
    /// NLP-NMS v2 with the installed models (granite, Stanza, SDS); falls back to `Rules` piece by piece.
    Auto,
}

impl std::str::FromStr for NlpMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "v1" | "off" => Ok(NlpMode::V1),
            "rules" => Ok(NlpMode::Rules),
            "auto" | "v2" => Ok(NlpMode::Auto),
            _ => Err(format!("unknown --nlp {s} (auto, rules, v1)")),
        }
    }
}

/// The router for `mode`, with the SDS ledger of its vector space under `%LOCALAPPDATA%\RaonZena\score_dynamics`.
fn build_router(lexicon: &Lexicon, mode: NlpMode) -> NlpNmsRouter {
    use nlp_nms_router::router::Models;
    let models = match mode {
        NlpMode::V1 => Models::v1(),
        NlpMode::Rules => Models::rules(),
        #[cfg(feature = "nlp")]
        NlpMode::Auto => {
            use nlp_nms_router::embed::TextEmbedder;
            let m = Models::auto("korean");
            let space = m.embedder.space().to_string();
            let dir = std::env::var_os("LOCALAPPDATA")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(std::env::temp_dir)
                .join("RaonZena")
                .join("score_dynamics")
                .join("spaces")
                .join(&space);
            match zena_sds::Ledger::open(&dir, "nms") {
                Ok(l) => m.with_sds(l),
                Err(e) => {
                    eprintln!("[sds] ledger unavailable ({e:#}); fixed thresholds");
                    m
                }
            }
        }
        #[cfg(not(feature = "nlp"))]
        NlpMode::Auto => Models::rules(),
    };
    eprintln!("[nlp] router: {}", models.label);
    NlpNmsRouter::with_models(lexicon, models)
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            sensor_threshold: 0.15,
            min_tile_events: 6,
            screen_interval_ms: 3000,
            learn_min: None,
            expression_gap_ms: 1500,
            proactive_silence_ms: 90_000,
            history_turns: 4,
            nlp: NlpMode::V1,
            pd_mcp: None,
            consent_share_outcomes: false,
            enable_host_probes: false,
            quarantine_path: None,
        }
    }
}

/// What Zena answered to a request, reported back by the LLM thread for the dialogue memory and
/// the 5-minute exposure ledger (CR-12).
#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    pub user: Option<String>,
    pub zena: String,
    pub epoch: u64,
    pub slots: Vec<PackedVerdictSlot>,
}

/// The channels the pipeline talks through besides the frame source.
pub struct Links {
    pub llm: LlmWorker,
    pub utterances: Receiver<String>,
    pub out: Sender<AvatarCmd>,
    /// Dictionary revisions (hot reload).
    pub lexicon: Receiver<Lexicon>,
    pub learn: Option<Sender<LexiconMsg>>,
    /// Frames for the screen watcher and what it saw.
    pub screen_frames: Option<SyncSender<ScreenFrame>>,
    pub screen: Option<Receiver<ScreenObservation>>,
    /// Zena's Lv3 answers (from [`StreamToAvatar`]).
    pub replies: Option<Receiver<Reply>>,
    /// UI interaction events from the overlay (typing, dragging, drop notes, ▲/▼ pins, badge picks).
    pub actions: Option<Receiver<OverlayAction>>,
    /// Shared ask epoch (bumped on every new question and on privacy lock).
    pub ask_epoch: Arc<AtomicU64>,
    /// Shared privacy epoch (observed by screen watcher & RAG cache).
    pub privacy_epoch: Arc<AtomicU64>,
    /// Shared spoken badge slot lock (updated while speech streams).
    pub spoken_lock: Arc<Mutex<SpokenSlotLock>>,
}

/// Rough speaking time for Korean TTS (≈ 7 syllables per second) until real TTS durations exist.
pub fn speech_ms(text: &str) -> u32 {
    (text.chars().filter(|c| !c.is_whitespace()).count() as u32 * 140).max(400)
}

fn unix_now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Plays dictionary expressions: lines rotate per domain, the same domain is not repeated within `gap_ms`.
#[derive(Default)]
pub struct ExpressionPlayer {
    next_line: HashMap<String, usize>,
    last_ms: HashMap<String, u64>,
}

impl ExpressionPlayer {
    pub fn play(
        &mut self,
        key: &str,
        e: &Express,
        gaze: Option<GazeTarget>,
        now: u64,
        gap_ms: u64,
    ) -> Option<AvatarCmd> {
        if self
            .last_ms
            .get(key)
            .is_some_and(|&t| now.saturating_sub(t) < gap_ms)
        {
            return None;
        }
        self.last_ms.insert(key.to_string(), now);
        let line = (e.level == ExpressLevel::Lv2 && !e.lines.is_empty()).then(|| {
            let i = self.next_line.entry(key.to_string()).or_default();
            let l = e.lines[*i % e.lines.len()].clone();
            *i += 1;
            l
        });
        let (level, ms, visemes) = match &line {
            Some(l) => {
                let ms = speech_ms(l);
                (ReactionLevel::Lv2Cached, ms, viseme_timeline(l, ms))
            }
            None => (ReactionLevel::Lv1Silent, 0, Vec::new()),
        };
        Some(AvatarCmd::Act {
            args: ZenaAvatarArgs {
                gaze_target: gaze,
                emotion_state: e.emotion,
                body_motion: e.motion,
                character_emote: e.emote,
                speech_text: line,
            },
            level,
            visemes,
            speech_ms: ms,
        })
    }
}

/// When a screen observation is worth reacting to: a scene when it changes, an event or a private screen at most
/// once per its cooldown.
#[derive(Default)]
pub struct ScreenGate {
    scene: Option<String>,
    last_ms: HashMap<String, u64>,
}

impl ScreenGate {
    pub fn admit(&mut self, o: &ScreenObservation, now: u64) -> bool {
        let recent = self
            .last_ms
            .get(&o.key)
            .is_some_and(|&t| now.saturating_sub(t) < o.cooldown_s as u64 * 1000);
        let ok = match o.kind {
            VisionKind::Scene => {
                let changed = self.scene.as_deref() != Some(o.key.as_str());
                self.scene = Some(o.key.clone());
                changed && !recent
            }
            VisionKind::Event | VisionKind::Privacy => !recent,
        };
        if ok {
            self.last_ms.insert(o.key.clone(), now);
        }
        ok
    }
}

/// Extra lines the LLM gets for a request topic.
fn topic_context(c: &Caught, lex: &Lexicon, screen: Option<&(String, f32)>) -> Vec<String> {
    let mut out = Vec::new();
    if let Action::Ask { topic, .. } = &c.action {
        if topic == "screen" || topic == "market" {
            if let Some((key, score)) = screen {
                let what = lex
                    .vision_node(key)
                    .map(|v| v.semantic.clone())
                    .unwrap_or_else(|| key.clone());
                out.push(format!("지금 화면: {what} (확신도 {:.0}%)", score * 100.0));
            } else {
                out.push("지금 화면은 아직 모른다.".into());
            }
        }
        if topic == "market" {
            out.push("투자 조언(사라/팔아라)은 하지 말고 플레이어의 기분에 공감한다.".into());
        }
    }
    out
}

/// How a caught dictionary domain moves the session summary.
fn session_signal(session: &mut SessionTracker, key: &str, now: u64) {
    match key {
        "emotion.celebrate" => session.victory(now),
        "emotion.frustrated" => session.defeat(now),
        "emotion.praise" | "emotion.thanks" => session.affinity(0.08),
        _ => {}
    }
}

/// Sends avatar commands; anything Zena says counts as activity for the proactive timer and tracks
/// when the speech channel is busy (G4).
struct Out<'a> {
    tx: &'a Sender<AvatarCmd>,
    speaking_until_ms: u64,
}

impl Out<'_> {
    fn send(&mut self, cmd: AvatarCmd, now: u64, proactive: &mut ProactiveTimer) -> anyhow::Result<()> {
        match &cmd {
            AvatarCmd::Act { speech_ms, args, .. } if args.speech_text.is_some() => {
                proactive.touch(now);
                self.speaking_until_ms = self.speaking_until_ms.max(now + *speech_ms as u64 + 2_500);
            }
            AvatarCmd::Speak { speech_ms, .. } => {
                proactive.touch(now);
                self.speaking_until_ms = self.speaking_until_ms.max(now + *speech_ms as u64 + 2_500);
            }
            AvatarCmd::ClearSpeech => {
                self.speaking_until_ms = 0;
            }
            _ => {}
        }
        self.tx.send(cmd)?;
        Ok(())
    }

    fn channel_busy(&self, now: u64) -> bool {
        now < self.speaking_until_ms
    }
}

fn plain_line_cmd(line: &str, emotion: EmotionState, motion: BodyMotion, level: ReactionLevel) -> AvatarCmd {
    let ms = speech_ms(line);
    AvatarCmd::Act {
        args: ZenaAvatarArgs {
            emotion_state: emotion,
            body_motion: motion,
            speech_text: Some(line.to_string()),
            ..Default::default()
        },
        level,
        visemes: viseme_timeline(line, ms),
        speech_ms: ms,
    }
}

pub fn run(
    mut src: Box<dyn FrameSource>,
    links: Links,
    mut lexicon: Lexicon,
    cfg: PipelineConfig,
) -> anyhow::Result<()> {
    let Links {
        llm,
        utterances,
        out,
        lexicon: reloads,
        learn,
        screen_frames,
        screen,
        replies,
        actions,
        ask_epoch,
        privacy_epoch,
        spoken_lock,
    } = links;
    let out_tx = out.clone();
    let mut out = Out {
        tx: &out_tx,
        speaking_until_ms: 0,
    };
    let (w, h) = src.size();
    let geo = ScreenGeometry::new(w as u32, h as u32);
    let mut sensor = LogLumSensor::new(w, h, cfg.sensor_threshold);
    let mut tracker = RoiTracker::default();
    let mut router = build_router(&lexicon, cfg.nlp);
    let mut sds_flushed = 0u64;
    let mut reactions = ReactionRouter::default();
    let mut cues = CuePlayer::default();
    let mut expressions = ExpressionPlayer::default();
    let mut gate = ScreenGate::default();
    let mut session = SessionTracker::default();
    let mut dialogue = Dialogue::new(cfg.history_turns);
    let mut proactive = ProactiveTimer::new(cfg.proactive_silence_ms);

    // Scenario 1–6 state machines
    let mut pd_bridge = PdBridge::new(
        cfg.pd_mcp.as_deref(),
        router.models.embedder.clone(),
        ask_epoch.clone(),
        privacy_epoch.clone(),
        cfg.consent_share_outcomes,
        cfg.quarantine_path
            .clone()
            .unwrap_or_else(default_quarantine_path),
    );
    let mut privacy_sm = PrivacyStateMachine::new();
    let mut live_episode = LiveEpisodeWindow::new("desktop.local", "screen:init", 0);
    let mut friction_baselines = FrictionBaselines::default();
    let behavior_params = BehaviorParams::default();
    let mut coach_hook = PendingCoachHook::default();
    let mut fatigue = CoachingFatigueController::default();
    let reflex_compiler = ReflexCompiler::default();
    let mut speech_coalescer = SpeechCoalescer::new();
    let mut topic_tracker = TopicTracker::default().with_hasher(topic_hash_id);
    let mut exposure_ledger = ExposureLedger::default();
    let mut sticky_numberer = StickyNumberer::default();
    let mut permanent_web_tiers: HashMap<String, Tier> = HashMap::new();
    let mut browser_tracker = BrowserSessionTracker::default();
    #[cfg(windows)]
    if cfg.enable_host_probes && crate::win32::is_screen_reader_active() {
        fatigue.set_screen_reader_mode(true);
        if let Ok(mut sl) = spoken_lock.lock() {
            sl.set_screen_reader_mode(true);
        }
    }
    let browser_worker = cfg
        .enable_host_probes
        .then(AsyncBrowserUiaWorker::spawn_platform);

    let mut scene_link: Option<String> = None;
    let mut active_cc = "desktop.local".to_string();
    let mut active_screen_key = "screen:init".to_string();
    let mut active_surface = SurfaceSignature::default();
    let mut active_web_brief: Option<WebCoachBrief> = None;
    let mut active_course_brief: Option<CourseBrief> = None;
    let mut active_defeat_trap: Option<String> = None;
    let mut active_nav_overlay: Option<NavBadgeOverlay> = None;
    let mut last_screen: Option<(String, f32)> = None;
    let mut last_frame_sent: Option<u64> = None;
    let mut last_probe_ms = 0u64;
    let mut last_sync_push_ms = 0u64;
    let mut last_uia_scan_ms = 0u64;
    let mut last_defeat_ms: Option<u64> = None;
    let mut last_victory_ms: Option<u64> = None;
    let mut defeat_shielded_until_ms = 0u64;
    let mut vision_privacy_hit = false;
    let mut redact_zones: Vec<(f32, f32, f32, f32)> = Vec::new();
    let mut privacy_locked_since: Option<u64> = None;
    let mut user_typing = false;
    let mut user_dragging = false;
    let mut avatar_hidden = false;
    let mut shutting_down = false;

    let t0 = Instant::now();
    eprintln!(
        "pipeline: {w}x{h} sensor, dictionary revision {} ({} domains), Lv3 backend = {}, screen watcher {}, pd-mcp {}, proactive {}",
        lexicon.version,
        router.bank.len(),
        llm.name,
        if screen_frames.is_some() { "on" } else { "off" },
        if pd_bridge.client.is_some() { "on" } else { "offline-mirror" },
        if cfg.proactive_silence_ms > 0 {
            format!("after {}s", cfg.proactive_silence_ms / 1000)
        } else {
            "off".into()
        }
    );

    loop {
        // Static screen gives no frame; everything else still ticks.
        let frame = src.next_frame()?;
        if let Some(f) = &frame {
            if (f.width, f.height) != (w, h) {
                anyhow::bail!("frame size changed {}x{} -> {}x{}", w, h, f.width, f.height);
            }
        }
        let now = t0.elapsed().as_millis() as u64;
        let now_i64 = now as i64;
        let wall_ms = unix_now_ms();
        let mut just_defeated = false;
        let mut progress_advanced = false;

        // 1. Drain overlay UI actions (typing, dragging, drop notes, ▲/▼ pins, badge picks, shutdown).
        if let Some(rx) = &actions {
            for act in rx.try_iter() {
                match act {
                    OverlayAction::TypingChanged(t) => user_typing = t,
                    OverlayAction::DraggingChanged(d) => user_dragging = d,
                    OverlayAction::AvatarHidden(h) => avatar_hidden = h,
                    OverlayAction::DropNote { cc, to, text } => {
                        let target_cc = if cc.trim().is_empty() {
                            active_cc.clone()
                        } else {
                            cc
                        };
                        match pd_bridge.enqueue_drop(&target_cc, to.as_deref(), &text, wall_ms) {
                            DropOutcome::BlockedPii { reaction } => {
                                out.send(
                                    plain_line_cmd(
                                        &reaction,
                                        EmotionState::Worried,
                                        BodyMotion::HideCrouch,
                                        ReactionLevel::Lv2Cached,
                                    ),
                                    now,
                                    &mut proactive,
                                )?;
                            }
                            DropOutcome::BlockedProfanity => {
                                out.send(
                                    plain_line_cmd(
                                        "욕설이 들어간 메모는 저장할 수 없어. 표현을 조금만 다듬어 줄래?",
                                        EmotionState::Worried,
                                        BodyMotion::Idle,
                                        ReactionLevel::Lv2Cached,
                                    ),
                                    now,
                                    &mut proactive,
                                )?;
                            }
                            DropOutcome::Enqueued { .. } => {
                                let _ = pd_bridge.curate_cc(&target_cc, wall_ms);
                                let view = pd_bridge.poll_overlay_view(Some(&active_cc), wall_ms);
                                out.send(AvatarCmd::SyncStatus(view), now, &mut proactive)?;
                                out.send(
                                    plain_line_cmd(
                                        "메모를 저장해 뒀어! 다음에 여기서 바로 꺼내 쓸게.",
                                        EmotionState::Joy,
                                        BodyMotion::Nod,
                                        ReactionLevel::Lv2Cached,
                                    ),
                                    now,
                                    &mut proactive,
                                )?;
                            }
                        }
                    }
                    OverlayAction::TogglePin { record_id, pin } => {
                        let q = pd_bridge.rag.quarantine.lock().unwrap().clone();
                        let _ = pd_bridge.verdicts.toggle_pin(&record_id, pin, wall_ms, &q);
                        if pin == UserPin::Demoted {
                            fatigue.on_ignored_or_dismissed(true);
                        }
                        let view = pd_bridge.poll_overlay_view(Some(&active_cc), wall_ms);
                        out.send(AvatarCmd::SyncStatus(view), now, &mut proactive)?;
                    }
                    OverlayAction::AckRejections => {
                        let _ = pd_bridge.ack_rejections();
                        let view = pd_bridge.poll_overlay_view(Some(&active_cc), wall_ms);
                        out.send(AvatarCmd::SyncStatus(view), now, &mut proactive)?;
                    }
                    OverlayAction::RefreshDrawer => {
                        if let Some(sl) = &pd_bridge.sync_loop {
                            sl.request(false);
                        }
                        let view = pd_bridge.poll_overlay_view(Some(&active_cc), wall_ms);
                        out.send(AvatarCmd::SyncStatus(view), now, &mut proactive)?;
                    }
                    OverlayAction::BadgePicked {
                        number,
                        commit_guarded,
                    } => {
                        if commit_guarded {
                            // CommitGuard (WA-10): never auto-click final payment button!
                            out.send(
                                plain_line_cmd(
                                    &format!("{number}번은 최종 결제 버튼이야! 금액이랑 배송지 먼저 확인하고 직접 눌러 줘."),
                                    EmotionState::Focused,
                                    BodyMotion::LeanForwardPoint,
                                    ReactionLevel::Lv2Cached,
                                ),
                                now,
                                &mut proactive,
                            )?;
                        } else {
                            progress_advanced = true;
                            live_episode.push_filtered(
                                NavEvent {
                                    at_ms: now_i64,
                                    signal: NavSignal::Picked,
                                    anchor: Some(format!("#{number}")),
                                },
                                false,
                            );
                            // the choice book (with the visible pool) and this run's checkpoint
                            if let Some(ov) = active_nav_overlay.as_ref() {
                                if let Some(b) = ov.badges.iter().find(|b| b.number == number) {
                                    let labels: Vec<&str> = ov.badges.iter().map(|x| x.label.as_str()).collect();
                                    let spread = |f: fn(&NavBadge) -> f32| {
                                        let v: Vec<f32> = ov.badges.iter().map(f).collect();
                                        v.iter().cloned().fold(f32::MIN, f32::max) - v.iter().cloned().fold(f32::MAX, f32::min)
                                    };
                                    let vertical = spread(|x| x.rect.center().0) < spread(|x| x.rect.center().1);
                                    pd_bridge.on_badge_picked(&active_cc, &active_screen_key, &b.anchor, &b.label, &labels, vertical, wall_ms);
                                }
                            }
                            if let Some(accepted) = fatigue.on_target_picked(&active_cc, number, now)
                            {
                                if let Some(exp) = accepted.exposure() {
                                    exposure_ledger.insert(exp);
                                }
                            }
                        }
                    }
                    OverlayAction::DismissCoaching => {
                        fatigue.on_ignored_or_dismissed(true);
                    }
                    OverlayAction::ShutdownStarted => {
                        shutting_down = true;
                        let to_wall = wall_ms - now_i64;
                        let frozen: Vec<(i64, i64)> =
                            privacy_sm.active_frozen_spans(now_i64).iter().map(|(a, b)| (a + to_wall, b + to_wall)).collect();
                        let _ = pd_bridge.on_app_left(&frozen);
                        pd_bridge.shutdown();
                        if let Some(l) = &router.models.sds {
                            let _ = l.lock().unwrap().flush();
                        }
                    }
                }
            }
        }

        // 2. SDS periodic flush every 15 s & overlay sync status refresh every 2 s.
        if now.saturating_sub(sds_flushed) >= 15_000 {
            sds_flushed = now;
            if let Some(l) = &router.models.sds {
                if let Err(e) = l.lock().unwrap().flush() {
                    eprintln!("[sds] flush failed: {e:#}");
                }
            }
        }
        if now.saturating_sub(last_sync_push_ms) >= 2_000 {
            last_sync_push_ms = now;
            let view = pd_bridge.poll_overlay_view(Some(&active_cc), wall_ms);
            let _ = out.send(AvatarCmd::SyncStatus(view), now, &mut proactive);
        }

        // 3. Dictionary hot reload.
        for lex in reloads.try_iter() {
            router.reload(&lex);
            eprintln!(
                "[lexicon] router now on revision {} ({} domains)",
                lex.version,
                router.bank.len()
            );
            lexicon = lex;
        }

        // 4. Consume completed Lv3 replies -> Dialogue memory + 5-min ExposureLedger (CR-12).
        if let Some(rx) = &replies {
            for r in rx.try_iter().filter(|r| !r.zena.trim().is_empty()) {
                proactive.touch(now);
                if let Ok(mut sl) = spoken_lock.lock() {
                    sl.end_line(now);
                }
                if r.user.is_some() {
                    dialogue.push(r.user.as_deref(), &r.zena);
                }
                if !r.slots.is_empty() {
                    let mirror_ref = pd_bridge.rag.mirror.clone();
                    let _ = exposure_ledger.record_delivered_turn(
                        &r.slots,
                        &r.zena,
                        |a, b| mirror_ref.lock().map(|m| m.similarity(a, b)).unwrap_or(0.0),
                        now,
                    );
                }
            }
        }

        // 5. Async Browser UIA snapshot handling (Scenario 5 WA-09 / WA-10).
        if let Some(worker) = &browser_worker {
            if let Some(snap) = worker.try_recv_latest() {
                match browser_tracker.observe(&snap) {
                    TabTransition::IncognitoBlocked => {
                        active_web_brief = None;
                        active_course_brief = None;
                        active_nav_overlay = None;
                        sticky_numberer.reset();
                        let _ = out.send(
                            AvatarCmd::NavBadges(NavBadgeOverlay::default()),
                            now,
                            &mut proactive,
                        );
                    }
                    TabTransition::PendingUrlResolution { .. } => {}
                    TabTransition::ActiveTab { cc, page_hash, .. } => {
                        if active_cc != cc {
                            // leaving an app / site: its checkpoints become a run in the on-device ZoneVault (privacy
                            // lock spans out, moved from the probe clock to the wall clock the run's day is counted on)
                            let to_wall = wall_ms - now_i64;
                            let frozen: Vec<(i64, i64)> =
                                privacy_sm.active_frozen_spans(now_i64).iter().map(|(a, b)| (a + to_wall, b + to_wall)).collect();
                            let _ = pd_bridge.on_app_left(&frozen);
                            active_cc = cc.clone();
                            progress_advanced = true;
                        }
                        if let Some(ph) = &page_hash {
                            if active_screen_key != *ph {
                                let done = live_episode.flush_completed(ph, now_i64);
                                if let Some(done) = &done {
                                    friction_baselines.decide(
                                        &active_cc,
                                        &active_screen_key,
                                        None,
                                        done,
                                        &behavior_params,
                                    );
                                }
                                // the 1-hop outcome of the last badge pick, and the new screen as a checkpoint
                                pd_bridge.on_screen_changed(&active_cc, &active_screen_key, done.as_ref(), ph, &behavior_params, wall_ms);
                                active_screen_key = ph.clone();
                            }
                        }
                        let locked_mask = spoken_lock
                            .lock()
                            .map(|s| s.mask(now))
                            .unwrap_or(0);
                        let embedder = router.models.embedder.clone();
                        // Phase 4 Part 3 + Phase 6 Part 3 (`CN2-55`, `CN2-78`):
                        // 빈 `&[]` 대신 3-소스 병합 라이브 `WebAnalyticTierRow` 를 투입한다!
                        let visible_labels: Vec<&str> = snap.regions.iter().map(|r| r.label.as_str()).collect();
                        let live_web_rows = pd_bridge.live_web_rows_with_choices(&cc, &visible_labels, wall_ms);
                        let label_sim =
                            precompute_label_similarity(&embedder, &live_web_rows, &snap.regions);
                        if let Some((badge_ov, web_brief)) = build_web_nav_overlay(
                            &snap,
                            &live_web_rows,
                            &mut permanent_web_tiers,
                            &mut sticky_numberer,
                            locked_mask,
                            None,
                            label_sim,
                            now_i64,
                        ) {
                            if let Ok(mut m) = pd_bridge.rag.mirror.lock() {
                                for b in &badge_ov.badges {
                                    m.register_label(&b.anchor, &b.label);
                                }
                            }
                            active_course_brief = build_course_brief_from_overlay(&badge_ov);
                            active_web_brief = Some(web_brief);
                            active_nav_overlay = Some(badge_ov.clone());
                            let _ = out.send(AvatarCmd::NavBadges(badge_ov), now, &mut proactive);
                        }
                    }
                }
            }
        }

        // 6. Scenario 6: 500 ms ZeroPixelProbe + PrivacyStateMachine (replaces fixed 60 s timer).
        let should_probe = now.saturating_sub(last_probe_ms) >= 500 || vision_privacy_hit;
        if should_probe {
            last_probe_ms = now;
            let mut probe_trigger = None;
            let mut in_pay_flow = false;
            let mut receipt_hash = None;
            let mut new_redact_zones: Vec<(f32, f32, f32, f32)> = Vec::new();

            if cfg.enable_host_probes {
                let monitor = PxRect {
                    left: 0,
                    top: 0,
                    right: (w * 4) as i32,
                    bottom: (h * 4) as i32,
                };
                let fg_window = foreground_window();
                if let Some(fg) = fg_window.as_ref() {
                    let fg_exe = fg.exe_name.as_deref().unwrap_or_default();
                    let fg_cc = fg
                        .app_uri()
                        .as_deref()
                        .and_then(|u| envelope::normalize_cc(u).ok())
                        .map(|c| c.cc)
                        .unwrap_or_else(|| normalize_cc(fg_exe));
                    let route_or_title = format!("{fg_exe}|{}", fg.title);
                    active_surface = SurfaceSignature::new(fg.hwnd, &route_or_title);
                    if is_browser_exe(fg_exe) {
                        if let Some(worker) = &browser_worker {
                            if now.saturating_sub(last_uia_scan_ms) >= 1_000 {
                                last_uia_scan_ms = now;
                                worker.request_scan(ScanRequest {
                                    hwnd: fg.hwnd,
                                    exe: fg_exe.to_string(),
                                    title: fg.title.clone(),
                                    now_ms: now_i64,
                                });
                            }
                        }
                    } else if !fg_cc.is_empty() && active_cc != fg_cc {
                        active_cc = fg_cc;
                        active_web_brief = None;
                        progress_advanced = true;
                    }
                    in_pay_flow = payment_route(&fg.title);
                    receipt_hash = receipt_finish_page(&fg.title);
                }

                let raw_windows = visible_windows_intersecting(monitor);
                let filtered_windows = filter_dormant_game_overlays(raw_windows);
                let surface_verdict = assess_surfaces(&filtered_windows, monitor);
                let notif_rects: Vec<premium_discount_nav::Rect> = surface_verdict
                    .redact_zones
                    .iter()
                    .filter_map(|&(rx, ry, rw, rh)| premium_discount_nav::Rect::new(rx, ry, rw, rh))
                    .collect();
                let seed_in_title = fg_window
                    .as_ref()
                    .is_some_and(|w| detect_seed_phrase_checked(&w.title).is_some());
                let shield = evaluate_privacy(&PrivacyProbeInput {
                    regions: &[],
                    os_capture_protected: surface_verdict.capture_protected(),
                    os_overlay_payment_detected: surface_verdict.overlay_payment(),
                    os_notification_zones: &notif_rects,
                    vision_privacy_hit,
                    seed_phrase_detected: seed_in_title,
                    auth_or_payment_context: in_pay_flow,
                });
                match shield {
                    PrivacyShield::FullLockdown(trig) => probe_trigger = Some(trig),
                    PrivacyShield::PartialRedaction { zones } => {
                        new_redact_zones = zones.into_iter().map(|z| (z.x, z.y, z.w, z.h)).collect();
                    }
                    PrivacyShield::Clear => {}
                }
            } else if vision_privacy_hit {
                let shield = evaluate_privacy(&PrivacyProbeInput {
                    vision_privacy_hit: true,
                    ..Default::default()
                });
                probe_trigger = shield.trigger();
            }

            redact_zones = new_redact_zones;
            let probe = ZeroPixelProbe {
                now_ms: now_i64,
                active_cc: active_cc.clone(),
                surface: active_surface,
                unoccluded_trigger: probe_trigger,
                in_payment_flow: in_pay_flow,
                receipt_finish_page_hash: receipt_hash,
            };
            let outcome = privacy_sm.step(&probe);
            vision_privacy_hit = false;

            if outcome.privacy_epoch_started {
                let _ = pd_bridge.rag.bump_privacy_epoch();
                sensor.purge_privacy_residue();
                tracker.purge_privacy_residue();
                live_episode.begin_privacy_lock(now_i64);
                sticky_numberer.reset();
                if let Ok(mut sl) = spoken_lock.lock() {
                    sl.clear();
                }
                privacy_locked_since = Some(now);
                out.send(AvatarCmd::ClearSpeech, now, &mut proactive)?;
                out.send(AvatarCmd::PrivacyLock { locked: true }, now, &mut proactive)?;
            }
            if outcome.trigger_sensor_reprime {
                sensor.purge_privacy_residue();
                tracker.purge_privacy_residue();
                live_episode.end_privacy_lock(now_i64);
                if let Some(since) = privacy_locked_since.take() {
                    session.add_paused_ms(now.saturating_sub(since));
                }
                out.send(AvatarCmd::PrivacyLock { locked: false }, now, &mut proactive)?;
            }
        }

        let privacy_locked = !privacy_sm.capture_allowed();

        // 7. Periodic screen sampling (strictly gated by PrivacyStateMachine::capture_allowed).
        if !privacy_locked {
            if let (Some(tx), Some(f)) = (&screen_frames, &frame) {
                if last_frame_sent.is_none_or(|t| now.saturating_sub(t) >= cfg.screen_interval_ms) {
                    last_frame_sent = Some(now);
                    let _ = tx.try_send(ScreenFrame {
                        rgb: bgra_to_rgb(&f.bgra),
                        w,
                        h,
                        captured_epoch: privacy_epoch.load(Ordering::Acquire),
                        redact_zones: redact_zones.clone(),
                    });
                }
            }
        }
        if let Some(rx) = &screen {
            for o in rx.try_iter() {
                if o.kind == VisionKind::Privacy {
                    vision_privacy_hit = true;
                    // Immediately step the PrivacyStateMachine on VisionKind::Privacy!
                    let probe = ZeroPixelProbe {
                        now_ms: now_i64,
                        active_cc: active_cc.clone(),
                        surface: active_surface,
                        unoccluded_trigger: Some(PrivacyTrigger::VisionPrivacyDomain),
                        in_payment_flow: false,
                        receipt_finish_page_hash: None,
                    };
                    let outcome = privacy_sm.step(&probe);
                    if outcome.privacy_epoch_started {
                        let _ = pd_bridge.rag.bump_privacy_epoch();
                        sensor.purge_privacy_residue();
                        tracker.purge_privacy_residue();
                        live_episode.begin_privacy_lock(now_i64);
                        sticky_numberer.reset();
                        privacy_locked_since = Some(now);
                        out.send(AvatarCmd::ClearSpeech, now, &mut proactive)?;
                        out.send(AvatarCmd::PrivacyLock { locked: true }, now, &mut proactive)?;
                    }
                    continue;
                }
                if !privacy_sm.capture_allowed() {
                    continue;
                }
                last_screen = Some((o.key.clone(), o.score));
                if o.kind == VisionKind::Scene {
                    let what = lexicon
                        .vision_node(&o.key)
                        .map(|v| v.semantic.clone())
                        .unwrap_or_else(|| o.key.clone());
                    session.scene(&o.key, &what, now);
                    scene_link = Some(o.link.clone());
                    if active_screen_key != o.key {
                        if let Some(done) = live_episode.flush_completed(&o.key, now_i64) {
                            friction_baselines.decide(
                                &active_cc,
                                &active_screen_key,
                                None,
                                &done,
                                &behavior_params,
                            );
                        }
                        active_screen_key = o.key.clone();
                        progress_advanced = true;
                    }
                }
                if !gate.admit(&o, now) {
                    continue;
                }
                if o.kind == VisionKind::Event {
                    match o.key.as_str() {
                        "vision.game_defeat" => {
                            session.defeat(now);
                            last_defeat_ms = Some(now);
                            just_defeated = true;
                            pd_bridge.on_defeat(wall_ms);
                            // Phase 7 Part 3 (AC-05): 사용자가 자책/설명을 입력 중이면 패배 방패 창을 보류·연장
                            if user_typing {
                                defeat_shielded_until_ms = defeat_shielded_until_ms.max(now + 15_000);
                            }
                            let shielded = now <= defeat_shielded_until_ms;
                            if !shielded {
                                if let Some(b) = active_nav_overlay.as_ref().and_then(|ov| ov.badges.first()) {
                                    active_defeat_trap = Some(b.label.clone());
                                }
                            }
                            let linked_ccs = pd_bridge.trajectory.linked_ccs(&active_cc, wall_ms);
                            let cands = active_credit_candidates_meta(&mut exposure_ledger, &linked_ccs, now);
                            let outbound = pd_bridge.verdicts.apply_outcome_credit(
                                &cands,
                                -1,
                                wall_ms,
                                shielded,
                                &mut pd_bridge.vote_gate,
                            );
                            for (id, sign) in outbound {
                                pd_bridge.send_outcome_vote(&id, sign);
                            }
                        }
                        "vision.game_victory" => {
                            session.victory(now);
                            last_victory_ms = Some(now);
                            active_defeat_trap = None;
                            let linked_ccs = pd_bridge.trajectory.linked_ccs(&active_cc, wall_ms);
                            let cands = active_credit_candidates_meta(&mut exposure_ledger, &linked_ccs, now);
                            let outbound = pd_bridge.verdicts.apply_outcome_credit(
                                &cands,
                                1,
                                wall_ms,
                                false,
                                &mut pd_bridge.vote_gate,
                            );
                            for (id, sign) in outbound {
                                pd_bridge.send_outcome_vote(&id, sign);
                            }
                        }
                        _ => {}
                    }
                }
                let Some(e) = lexicon.vision_express(&o.key).cloned() else {
                    continue;
                };
                let gaze = o.peak.map(|(x, y)| {
                    let p = geo.to_polar(x * w as f32, y * h as f32);
                    GazeTarget {
                        angle_deg: p.theta_deg,
                        distance_norm: p.r,
                    }
                });
                eprintln!(
                    "[screen] {} (cos {:.3}, p {:.2}, {} of {} tokens, {:.0} ms) → {}",
                    o.key, o.cos, o.score, o.tokens, o.cells, o.encode_ms, o.link
                );
                if let Some(cmd) = expressions.play(&o.link, &e, gaze, now, cfg.expression_gap_ms) {
                    out.send(cmd, now, &mut proactive)?;
                }
            }
        }

        // 8. Event sensor + ROI tracker (only when capture is allowed).
        let privacy_locked = !privacy_sm.capture_allowed();
        let tracks = if privacy_locked {
            Vec::new()
        } else {
            match &frame {
                Some(f) => {
                    let stats = sensor.process_bgra(&f.bgra);
                    let rois = extract_rois(&stats, &geo, cfg.min_tile_events);
                    tracker.update(rois, now).to_vec()
                }
                None => Vec::new(),
            }
        };
        let max_energy = tracks.iter().map(|t| t.roi.energy).fold(1.0f32, f32::max);

        // 9. Front-gate every user utterance through IngressPiiScrubber (PS-12) BEFORE router/dialogue/LLM!
        let scrubber = IngressPiiScrubber::with_seed_detector(&detect_seed_phrase_checked);
        let mut inputs: Vec<RouterInput> = Vec::new();
        let mut allow_lexicon_learn = true;
        for mut raw_text in utterances.try_iter() {
            match scrubber.inspect_and_scrub(&mut raw_text) {
                IngressVerdict::HardBlocked { kind } => {
                    out.send(
                        plain_line_cmd(
                            kind.warning_line(),
                            EmotionState::Worried,
                            BodyMotion::HideCrouch,
                            ReactionLevel::Lv2Cached,
                        ),
                        now,
                        &mut proactive,
                    )?;
                }
                IngressVerdict::Scrubbed {
                    sanitized_text,
                    allow_lexicon_learn: allow,
                } => {
                    allow_lexicon_learn = allow;
                    // Promote recent ephemeral proactive offer into Dialogue if user responded within 45 s (PF-11).
                    if let Some(promoted) = fatigue.on_user_utterance(&active_cc, now) {
                        dialogue.push_turn(promoted);
                    }
                    inputs.push(RouterInput::Utterance {
                        text: sanitized_text,
                        t_ms: now,
                    });
                }
                IngressVerdict::Clean(clean) => {
                    if let Some(promoted) = fatigue.on_user_utterance(&active_cc, now) {
                        dialogue.push_turn(promoted);
                    }
                    match speech_coalescer.push(&clean, now) {
                        CoalesceOutcome::Buffering => {}
                        CoalesceOutcome::Ready(merged) | CoalesceOutcome::CancelledAndReplaced(merged) => {
                            inputs.push(RouterInput::Utterance {
                                text: merged,
                                t_ms: now,
                            });
                        }
                    }
                }
            }
        }
        if let Some(flushed) = speech_coalescer.flush_if_due(now) {
            inputs.push(RouterInput::Utterance {
                text: flushed,
                t_ms: now,
            });
        }
        let user_spoke = !inputs.is_empty();
        if user_spoke {
            proactive.touch(now);
            session.affinity(0.01);
        }
        for t in tracks.iter().filter(|t| t.age >= 1) {
            let label = match (t.flow, t.roi.polar.r) {
                (FlowDirection::Inward, r) if r < 0.22 => "근접 위협",
                (FlowDirection::Inward, _) => "몬스터 접근",
                (FlowDirection::Outward, _) => "스킬 시전",
                _ => "움직임",
            };
            inputs.push(RouterInput::GameEvent {
                label: label.into(),
                score: t.roi.energy / max_energy,
                polar: t.roi.polar,
                t_ms: now,
            });
        }
        let routed = if inputs.is_empty() {
            Default::default()
        } else {
            router.route(inputs, now)
        };

        // 10. Scenario 4: consume `routed.feedback` via pragmatic feedback extractor!
        if !routed.feedback.is_empty() {
            let fb_ctx = FeedbackContext {
                recent_defeat_within_15s: last_defeat_ms
                    .is_some_and(|t| now.saturating_sub(t) <= 15_000),
                recent_victory_within_15s: last_victory_ms
                    .is_some_and(|t| now.saturating_sub(t) <= 15_000),
                high_friction_burst: session.tension(now) >= 0.6,
            };
            let active_exposed = exposure_ledger.active_in_window(&active_cc, now);
            for fb in &routed.feedback {
                let toks: Vec<FeedbackToken<'_>> = fb
                    .words
                    .iter()
                    .map(|w| FeedbackToken {
                        surface: &w.surface,
                        lemma: &w.lemma,
                        is_auxiliary_request: w.is_auxiliary_request,
                        is_negation: w.is_negation,
                        negated_in_clause: w.negated_in_clause,
                        is_subject_or_topic: w.is_subject_or_topic,
                    })
                    .collect();
                let target_id = active_exposed
                    .last()
                    .map(|e| e.record_id.clone())
                    .unwrap_or_else(|| active_cc.clone());
                if let Some(extracted) = pd_bridge.verdicts.apply_spoken_feedback(
                    &target_id,
                    &fb.utterance,
                    &toks,
                    &fb_ctx,
                    1.0,
                    wall_ms,
                ) {
                    if extracted.shield_defeat_outcome {
                        defeat_shielded_until_ms = now + 15_000;
                    }
                }
            }
        }

        // Learning: 4-게이트 안전 자동학습(Command Air-Gap · PII 스크럽 · Tier 2 앱 스코프 격리).
        if allow_lexicon_learn {
            if let (Some(tx), Some(min)) = (&learn, cfg.learn_min) {
                for c in routed
                    .expressions
                    .iter()
                    .chain(&routed.queries)
                    .filter(|c| c.hit == Hit::Vector && c.score >= min)
                {
                    let _ = tx.send(LexiconMsg::LearnSafe {
                        cc: Some(active_cc.clone()),
                        key: c.key.clone(),
                        span: c.span.clone(),
                        session_id: format!("sess-{}", wall_ms / 1_800_000),
                        score: c.score,
                        margin: 0.15,
                        outcome_sign: 0,
                    });
                }
            }
        }
        for c in &routed.expressions {
            session_signal(&mut session, &c.key, now);
            if c.key == "emotion.frustrated" {
                last_defeat_ms = Some(now);
                just_defeated = true;
            }
        }

        // Dictionary expressions first, then system & asset commands.
        if let Some(c) = routed.expressions.iter().find(|c| {
            !matches!(
                c.action,
                Action::System {
                    command: SystemCommand::ExitApp,
                    ..
                }
            )
        }) {
            if let Some(e) = c.action.express() {
                if let Some(cmd) = expressions.play(&c.key, e, None, now, cfg.expression_gap_ms) {
                    if let Some(line) = cmd.speech() {
                        dialogue.push(Some(&c.utterance), line);
                    }
                    out.send(cmd, now, &mut proactive)?;
                }
            }
        }
        for (c, m) in &routed.system {
            out.send(AvatarCmd::System(*c, *m), now, &mut proactive)?;
        }
        for a in &routed.assets {
            out.send(AvatarCmd::Asset(a.clone()), now, &mut proactive)?;
        }

        // 11. Pick strongest surviving signal for this tick + 600 ms two-track RAG race on Ask!
        let mut obs = Observation::default();
        let mut label = None;
        let query = routed.queries.first().cloned();
        if let Some(q) = &query {
            obs.user_query = Some(q.utterance.clone());
            if let Some(e) = q.action.express() {
                if let Some(cmd) = expressions.play(&q.key, e, None, now, cfg.expression_gap_ms) {
                    out.send(cmd, now, &mut proactive)?;
                }
            }
        }
        if let Some((l, p, s)) = routed.events.iter().max_by(|a, b| a.2.total_cmp(&b.2)) {
            let tr = tracks.iter().min_by(|a, b| {
                let da = (a.roi.polar.r - p.r).abs()
                    + (a.roi.polar.theta_deg - p.theta_deg).abs() / 360.0;
                let db = (b.roi.polar.r - p.r).abs()
                    + (b.roi.polar.theta_deg - p.theta_deg).abs() / 360.0;
                da.total_cmp(&db)
            });
            obs.polar = Some(*p);
            obs.energy = *s;
            obs.flow = tr.map(|t| t.flow);
            obs.radial_velocity = tr.map(|t| t.radial_velocity).unwrap_or(0.0);
            label = Some(l.clone());
        }

        let mut high_tension_combat = session.tension(now) >= zena_core::coach::COMBAT_TENSION;
        if obs.user_query.is_some() || obs.polar.is_some() {
            let d = reactions.route(&obs, now);
            high_tension_combat =
                InterruptSignals::combat_from(d.level, d.reason, session.tension(now));
            if obs.user_query.is_none() && d.level >= ReactionLevel::Lv2Cached {
                session.combat(obs.energy, now);
            }
            match d.level {
                ReactionLevel::Lv1Silent => {
                    if !privacy_locked {
                        if let Some(p) = obs.polar {
                            out.send(
                                AvatarCmd::Gaze(GazeTarget {
                                    angle_deg: p.theta_deg,
                                    distance_norm: p.r,
                                }),
                                now,
                                &mut proactive,
                            )?;
                        }
                    }
                }
                ReactionLevel::Lv2Cached => {
                    if !privacy_locked {
                        out.send(
                            cached_cue(&mut cues, d.reason, obs.polar, d.level),
                            now,
                            &mut proactive,
                        )?;
                    }
                }
                ReactionLevel::Lv3Coaching => {
                    let mut context = Vec::new();
                    let mut sess_ctx = session.snapshot(now);
                    let mut job_epoch = 0u64;

                    if privacy_locked {
                        // BlindAssist (PS-12): answer text query only, zero screen pixels/verdicts.
                        sess_ctx.privacy_locked = true;
                        sess_ctx.scene = None;
                        job_epoch = pd_bridge.rag.next_ask_epoch();
                    } else if let Some(q) = &query {
                        context = topic_context(q, &lexicon, last_screen.as_ref());
                        let is_analytic = matches!(
                            &q.action,
                            Action::Ask { topic, .. } if topic == "analytic"
                        );
                        if is_analytic {
                            if let Some(brief) = pd_bridge.query_analytic(&active_cc, &q.utterance) {
                                sess_ctx.analytic = Some(brief);
                            }
                            job_epoch = pd_bridge.rag.next_ask_epoch();
                        } else {
                            // Phase 6 Part 3 (`CN2-78`): 활성 배지 오버레이에서 checkpoint_to · focused 를 직접 주입!
                            let screen_snap = build_screen_anchor_snapshot(
                                &active_cc,
                                &active_screen_key,
                                last_screen
                                    .as_ref()
                                    .and_then(|(k, _)| lexicon.vision_node(k))
                                    .map(|v| v.semantic.as_str()),
                                active_nav_overlay.as_ref(),
                                now,
                            );
                            let extracted: Vec<String> = (!q.span.trim().is_empty())
                                .then(|| q.span.clone())
                                .into_iter()
                                .collect();
                            let plan = topic_tracker.resolve_and_plan(
                                &q.utterance,
                                &extracted,
                                &screen_snap,
                                now,
                            );
                            if let Some(guard_line) = plan.context_guard() {
                                context.push(guard_line);
                            }
                            let q_snap = pd_bridge.rag.quarantine.lock().unwrap().clone();
                            let local_grades: HashMap<String, zena_core::claims::Grade> = pd_bridge
                                .verdicts
                                .records
                                .keys()
                                .cloned()
                                .collect::<Vec<_>>()
                                .into_iter()
                                .map(|id| {
                                    let g = pd_bridge.verdicts.local_grade(&id, wall_ms, &q_snap);
                                    (id, g)
                                })
                                .collect();
                            match pd_bridge.rag.retrieve(
                                &plan,
                                pd_bridge.sync_loop.as_ref(),
                                Some(&out_tx),
                                Arc::new(move |id| {
                                    local_grades
                                        .get(id)
                                        .copied()
                                        .unwrap_or(zena_core::claims::Grade::Unrated)
                                }),
                                now,
                            ) {
                                RagOutcome::Ready {
                                    epoch,
                                    slots,
                                    ..
                                } => {
                                    job_epoch = epoch;
                                    // Phase 6 Part 3 (`CN2-79`): 하이브리드 RAG × 실시간 텔레메트리 중재 경고 주입
                                    if let Some(arbiter_line) =
                                        hybrid_arbiter_context_line(&slots, active_nav_overlay.as_ref())
                                    {
                                        context.push(arbiter_line);
                                    }
                                    sess_ctx.verdict_slots = slots;
                                }
                                RagOutcome::Cancelled { .. } => {
                                    continue;
                                }
                            }
                        }
                        if sess_ctx.web_coach.is_none() {
                            sess_ctx.web_coach = active_web_brief.clone();
                        }
                        if sess_ctx.course_brief.is_none() {
                            sess_ctx.course_brief = active_course_brief.clone();
                        }
                        // Phase 6 Part 3 (`CN2-80`): 같은 턴의 활성 코스 체크포인트를 "route" 슬롯으로 병행 등록
                        if let Some(primary) = active_nav_overlay
                            .as_ref()
                            .and_then(|ov| ov.badges.iter().find(|b| !b.unavailable))
                        {
                            let role = if primary.tier == Some(Tier::Discount) {
                                ExposureRole::WarnedTrap
                            } else {
                                ExposureRole::RecommendedSolution
                            };
                            exposure_ledger.record_course_checkpoint(
                                &primary.anchor,
                                &active_cc,
                                Some(&primary.anchor),
                                &primary.label,
                                role,
                                now,
                            );
                        }
                    }

                    let sit = Situation {
                        event_label: if privacy_locked { None } else { label },
                        polar: if privacy_locked { None } else { obs.polar },
                        flow: if privacy_locked { None } else { obs.flow },
                        hp_ratio: obs.hp_ratio,
                        user_query: obs.user_query.clone(),
                        context,
                        history: dialogue.turns(),
                        session: Some(sess_ctx),
                    };
                    let prompt = build_fast_path_prompt(&sit);
                    eprintln!(
                        "[lv3] {}{} ({} remembered turns, epoch {job_epoch})",
                        d.reason,
                        query
                            .as_ref()
                            .map(|q| format!(
                                " → {} ({:?} {:.2} on '{}'): {}",
                                q.key, q.hit, q.score, q.span, q.utterance
                            ))
                            .unwrap_or_default(),
                        sit.history.len()
                    );
                    if !llm.try_submit(LlmJob {
                        prompt,
                        situation: sit,
                        level: d.level,
                        epoch: job_epoch,
                    }) {
                        out.send(
                            cached_cue(&mut cues, d.reason, obs.polar, ReactionLevel::Lv2Cached),
                            now,
                            &mut proactive,
                        )?;
                    }
                }
            }
        }

        // 12. Scenario 2: Live friction peek + 5-gate InterruptSignals + PendingCoachHook + ProactiveTimer::due_gated.
        let _ = fatigue.expire(now);
        let sig = InterruptSignals {
            now_ms: now,
            privacy_or_hidden: privacy_locked || avatar_hidden || shutting_down,
            user_composing: user_typing || user_dragging,
            high_tension_combat,
            speech_channel_busy: out.channel_busy(now),
            just_defeated,
        };

        if !privacy_locked && fatigue.allows_proactive() {
            let live_reading = live_episode.peek(&friction_baselines, now_i64, 0, &behavior_params);
            match live_reading.stage {
                FrictionAlertStage::PreWarm => {
                    pd_bridge.maybe_prewarm(now, privacy_locked, out.channel_busy(now));
                }
                FrictionAlertStage::CoachReady => {
                    let reason_tag = match live_reading.dominant_signal {
                        Some(NavSignal::RageClick) => "rage_click",
                        Some(NavSignal::Retried) => "retried",
                        Some(NavSignal::Searched { .. }) => "pre_action_stall",
                        _ => "ghost_overrun",
                    };
                    let mut cand = PendingCoach::new(
                        &active_cc,
                        &active_screen_key,
                        live_reading.bottleneck_anchor.as_deref(),
                        reason_tag,
                        now,
                        ask_epoch.load(Ordering::Acquire),
                    );
                    cand.friction_z = live_reading.screen.z;
                    if let Some((ready, mode)) = coach_hook.submit(cand, &sig) {
                        deliver_proactive_coach(
                            &ready,
                            mode,
                            &active_nav_overlay,
                            &active_web_brief,
                            active_defeat_trap.as_deref(),
                            &reflex_compiler,
                            &mut fatigue,
                            &llm,
                            &mut session,
                            &dialogue,
                            &ask_epoch,
                            &mut out,
                            &mut proactive,
                            now,
                        )?;
                    }
                }
                FrictionAlertStage::Calm => {}
            }
            if let Some((released, mode)) = coach_hook.poll(
                &active_cc,
                &active_screen_key,
                progress_advanced,
                ask_epoch.load(Ordering::Acquire),
                &sig,
            ) {
                deliver_proactive_coach(
                    &released,
                    mode,
                    &active_nav_overlay,
                    &active_web_brief,
                    active_defeat_trap.as_deref(),
                    &reflex_compiler,
                    &mut fatigue,
                    &llm,
                    &mut session,
                    &dialogue,
                    &ask_epoch,
                    &mut out,
                    &mut proactive,
                    now,
                )?;
            }
        }

        // Unprompted chatter / stretch break: gated by all 5 InterruptSignals via `due_gated`!
        if fatigue.allows_proactive() {
            if let Some(kind) = proactive.due_gated(now, session.scene_ms(now), &sig) {
                if let Some((key, e)) =
                    lexicon.proactive(scene_link.as_deref(), kind == Proactive::Break)
                {
                    let (key, e) = (key.to_string(), e.clone());
                    if let Some(cmd) = expressions.play(&key, &e, None, now, 0) {
                        if let Some(line) = cmd.speech() {
                            eprintln!("[proactive] {key}");
                            fatigue.offer(EphemeralOffer::new(&active_cc, None, None, line, now));
                        }
                        out.send(cmd, now, &mut proactive)?;
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn deliver_proactive_coach(
    coach: &PendingCoach,
    mode: zena_core::coach::CoachDeliveryMode,
    nav_overlay: &Option<NavBadgeOverlay>,
    web_brief: &Option<WebCoachBrief>,
    defeat_trap: Option<&str>,
    reflex_compiler: &ReflexCompiler,
    fatigue: &mut CoachingFatigueController,
    llm: &LlmWorker,
    session: &mut SessionTracker,
    dialogue: &Dialogue,
    ask_epoch: &Arc<AtomicU64>,
    out: &mut Out<'_>,
    proactive: &mut ProactiveTimer,
    now: u64,
) -> anyhow::Result<()> {
    let primary_badge = nav_overlay
        .as_ref()
        .and_then(|ov| ov.badges.iter().find(|b| !b.unavailable && !b.commit_guarded));
    let evidence = CoachEvidence {
        bottleneck_label: coach.bottleneck_anchor.clone(),
        verified_alternative: web_brief
            .as_ref()
            .and_then(|w| w.primary.as_ref().filter(|p| p.verified))
            .map(|p| format!("#{} '{}'", p.number, p.label)),
        confirmed_trap: web_brief
            .as_ref()
            .and_then(|w| w.friction_trap.clone())
            .or_else(|| defeat_trap.map(str::to_string)),
        point_at: primary_badge.map(|b| b.number),
    };
    let cur_epoch = ask_epoch.load(Ordering::Acquire);
    match route_proactive_with_reflex(
        coach,
        mode,
        &evidence,
        true,
        cur_epoch,
        fatigue,
        reflex_compiler,
    ) {
        ProactiveRoute::Lv3(brief) => {
            let mut sess = session.snapshot(now);
            sess.friction_brief = Some(brief);
            let sit = Situation {
                history: dialogue.turns(),
                session: Some(sess),
                ..Default::default()
            };
            let prompt = build_fast_path_prompt(&sit);
            if !llm.try_submit(LlmJob {
                prompt,
                situation: sit,
                level: ReactionLevel::Lv3Coaching,
                epoch: cur_epoch,
            }) {
                let fallback = "잠깐 막혔어? 다른 경로도 한번 볼까?";
                fatigue.offer(EphemeralOffer::new(
                    &coach.cc,
                    None,
                    evidence.point_at,
                    fallback,
                    now,
                ));
                out.send(
                    plain_line_cmd(
                        fallback,
                        EmotionState::Focused,
                        BodyMotion::LeanForwardPoint,
                        ReactionLevel::Lv2Cached,
                    ),
                    now,
                    proactive,
                )?;
            }
        }
        ProactiveRoute::Lv2(cached) => {
            fatigue.offer(EphemeralOffer::new(
                &coach.cc,
                None,
                cached.point_at,
                &cached.line,
                now,
            ));
            out.send(
                plain_line_cmd(
                    &cached.line,
                    EmotionState::Focused,
                    BodyMotion::LeanForwardPoint,
                    ReactionLevel::Lv2Cached,
                ),
                now,
                proactive,
            )?;
        }
        ProactiveRoute::Cancel(_) => {}
    }
    Ok(())
}

/// Phase 6 Part 3 (`CN2-78`): 활성 배지 오버레이에서 `ScreenAnchorSnapshot` (`checkpoint_to` · `focused`) 을 채운다.
fn build_screen_anchor_snapshot(
    cc: &str,
    screen_key: &str,
    screen_label: Option<&str>,
    overlay: Option<&NavBadgeOverlay>,
    now_ms: u64,
) -> ScreenAnchorSnapshot {
    let primary = overlay.and_then(|ov| ov.badges.iter().find(|b| !b.unavailable));
    ScreenAnchorSnapshot {
        cc: cc.to_string(),
        screen_key: Some(screen_key.to_string()),
        screen_label: screen_label.map(str::to_string),
        checkpoint_to: primary.map(|b| b.anchor.clone()),
        focused: primary.map(|b| FocusedRegion {
            number: Some(b.number),
            label: b.label.clone(),
            to: Some(b.anchor.clone()),
            at_ms: now_ms,
        }),
        changed_at_ms: Some(now_ms),
    }
}

/// Phase 6 Part 3 (`CN2-78`): 활성 배지 오버레이에서 `CourseBrief` 를 구축한다.
fn build_course_brief_from_overlay(overlay: &NavBadgeOverlay) -> Option<CourseBrief> {
    let primary = overlay
        .badges
        .iter()
        .find(|b| !b.unavailable && !b.commit_guarded)?;
    Some(CourseBrief {
        passed: 1,
        length: overlay.badges.len().max(1),
        next_hint: NextStepHint::NumberedOnScreen {
            number: primary.number,
            clean_label: primary.label.clone(),
            direction_label: String::new(),
        },
        verdict_judged_premium: primary.tier == Some(Tier::Premium),
        ..Default::default()
    })
}

/// Phase 6 Part 3 (`CN2-79`): RAG 슬롯(90일 `pd` 트랙)과 실시간 배지(14일 `nav` 트랙) 티어를 4사분면 중재한다.
fn hybrid_arbiter_context_line(
    slots: &[PackedVerdictSlot],
    overlay: Option<&NavBadgeOverlay>,
) -> Option<String> {
    let ov = overlay?;
    let primary = ov.badges.iter().find(|b| !b.unavailable)?;
    let nav_tier = match primary.tier {
        Some(Tier::Premium) => TelemetryTier::Premium,
        Some(Tier::Discount) => TelemetryTier::Discount,
        Some(Tier::Neutral) => TelemetryTier::Neutral,
        _ => TelemetryTier::Unrated,
    };
    let rag_tier = slots
        .first()
        .map(|s| match s.role {
            zena_core::claims::SlotRole::Solution => TelemetryTier::Premium,
            zena_core::claims::SlotRole::Warning => TelemetryTier::Discount,
            _ => TelemetryTier::Unrated,
        })
        .unwrap_or(TelemetryTier::Unrated);
    arbitrate_rag_with_telemetry(rag_tier, nav_tier, &primary.label).1
}

/// Phase 6 Part 1 & 3 (`CN2-72`, `CN2-80`): 크로스-앱 메타 궤적과 `RagRecord` × `CourseCheckpoint` 시너지 귀속 후보 생성.
fn active_credit_candidates_meta(
    ledger: &mut ExposureLedger,
    linked_ccs: &[String],
    now_ms: u64,
) -> Vec<AttributionCandidate> {
    let cc_refs: Vec<&str> = linked_ccs.iter().map(String::as_str).collect();
    ledger
        .active_in_meta_trajectory(&cc_refs, now_ms)
        .into_iter()
        .map(|rec| AttributionCandidate {
            id: rec.record_id,
            kind: match rec.target_kind {
                ExposureTargetKind::RagRecord => CreditTargetKind::RagRecord,
                ExposureTargetKind::CourseCheckpoint => CreditTargetKind::CourseCheckpoint,
                ExposureTargetKind::NavRegion => CreditTargetKind::NavRegion,
            },
            exclusive_slot_group: rec.exclusive_slot_group.or(rec.to),
            is_warned_trap: rec.role == ExposureRole::WarnedTrap,
            exposure_weight: rec.w_spoken,
            age_ms: now_ms.saturating_sub(rec.exposed_at_ms),
            semantic_sim: 0.7,
            matches_explicit_mention: false,
            friction_z: None,
            ambient_prior: 0.20,
            prior_same_sign_outcomes: 0,
        })
        .collect()
}

fn cached_cue(
    cues: &mut CuePlayer,
    reason: &str,
    polar: Option<zena_core::Polar>,
    level: ReactionLevel,
) -> AvatarCmd {
    let kind = cue_for_reason(reason);
    let c = cue(kind);
    let line = cues.next_line(kind).to_string();
    let ms = speech_ms(&line);
    AvatarCmd::Act {
        args: ZenaAvatarArgs {
            gaze_target: polar.map(|p| GazeTarget {
                angle_deg: p.theta_deg,
                distance_norm: p.r,
            }),
            emotion_state: c.emotion,
            body_motion: c.motion,
            speech_text: Some(line.clone()),
            ..Default::default()
        },
        level,
        visemes: viseme_timeline(&line, ms),
        speech_ms: ms,
    }
}

/// One streamed event as an avatar command.
pub fn event_cmd(ev: StreamEvent, level: ReactionLevel) -> Option<AvatarCmd> {
    match ev {
        StreamEvent::Expression(args) | StreamEvent::Call(ToolCall::ControlZenaAvatar(args)) => {
            Some(AvatarCmd::Expression { args, level })
        }
        StreamEvent::Speech(text) => {
            let ms = speech_ms(&text);
            Some(AvatarCmd::Speak {
                visemes: viseme_timeline(&text, ms),
                speech_ms: ms,
                text,
                level,
            })
        }
        StreamEvent::Call(ToolCall::ManageZenaSystem(s)) => {
            Some(AvatarCmd::System(s.command, s.window_layer_mode))
        }
        StreamEvent::Call(ToolCall::ManageAvatarAsset(a)) => Some(AvatarCmd::Asset(a)),
        StreamEvent::Rejected { name, error } => {
            eprintln!("[llm] rejected tool call {name}: {error}");
            None
        }
    }
}

/// Turn a whole model completion into avatar commands (the same commands streaming produces).
#[cfg_attr(not(test), allow(dead_code))]
pub fn completion_to_cmds(completion: &str, level: ReactionLevel) -> Vec<AvatarCmd> {
    let mut s = ToolCallStream::new();
    let mut ev = s.push(completion);
    ev.extend(s.finish());
    ev.into_iter()
        .filter_map(|e| event_cmd(e, level))
        .collect()
}

/// The LLM thread's handler: parses the completion while it streams and sends avatar commands at once —
/// guarding every delta & completion by `ask_epoch` (CR-06 / PS-12) and tracking spoken badge numbers in
/// `SpokenSlotLock` (CN-12) — then reports the delivered answer & injected slots for `ExposureLedger` (CR-12).
pub struct StreamToAvatar {
    out: Sender<AvatarCmd>,
    replies: Option<Sender<Reply>>,
    ask_epoch: Arc<AtomicU64>,
    spoken_lock: Arc<Mutex<SpokenSlotLock>>,
    stream: ToolCallStream,
    streamed: bool,
}

impl StreamToAvatar {
    #[allow(dead_code)]
    pub fn new(out: Sender<AvatarCmd>, replies: Option<Sender<Reply>>) -> Self {
        Self::with_guards(
            out,
            replies,
            Arc::new(AtomicU64::new(0)),
            Arc::new(Mutex::new(SpokenSlotLock::default())),
        )
    }

    pub fn with_guards(
        out: Sender<AvatarCmd>,
        replies: Option<Sender<Reply>>,
        ask_epoch: Arc<AtomicU64>,
        spoken_lock: Arc<Mutex<SpokenSlotLock>>,
    ) -> Self {
        Self {
            out,
            replies,
            ask_epoch,
            spoken_lock,
            stream: ToolCallStream::new(),
            streamed: false,
        }
    }

    fn is_current(&self, epoch: u64) -> bool {
        epoch == 0 || epoch == self.ask_epoch.load(Ordering::Acquire)
    }

    fn emit(&self, events: Vec<StreamEvent>, level: ReactionLevel) {
        for ev in events {
            if let StreamEvent::Speech(ref t) = ev {
                if let Ok(mut sl) = self.spoken_lock.lock() {
                    sl.push_spoken(t);
                }
            }
            if let Some(cmd) = event_cmd(ev, level) {
                let _ = self.out.send(cmd);
            }
        }
    }
}

impl JobHandler for StreamToAvatar {
    fn delta(&mut self, job: &LlmJob, text: &str) {
        if !self.is_current(job.epoch) {
            return;
        }
        self.streamed = true;
        let ev = self.stream.push(text);
        self.emit(ev, job.level);
    }

    fn done(&mut self, job: LlmJob, text: String) {
        if !self.is_current(job.epoch) {
            self.stream = ToolCallStream::new();
            self.streamed = false;
            return;
        }
        let mut ev = if self.streamed {
            Vec::new()
        } else {
            self.stream.push(&text)
        };
        ev.extend(self.stream.finish());
        self.emit(ev, job.level);
        if let Some(tx) = &self.replies {
            let slots = job
                .situation
                .session
                .as_ref()
                .map(|s| s.verdict_slots.clone())
                .unwrap_or_default();
            let _ = tx.send(Reply {
                user: job.situation.user_query.clone(),
                zena: self.stream.spoken().to_string(),
                epoch: job.epoch,
                slots,
            });
        }
        self.stream = ToolCallStream::new();
        self.streamed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LlmBackend, OnDone, TemplateBackend};
    use std::sync::mpsc::channel;
    use std::time::Duration;
    use zena_core::ingress_guard::SecretKind;
    use zena_core::tools::{AssetAction, AvatarTrack, BodyMotion, EmotionState};

    #[test]
    fn plain_text_completion_is_spoken_and_tool_calls_are_mapped() {
        let c = completion_to_cmds("인벤토리 한번 같이 찾아볼까?", ReactionLevel::Lv3Coaching);
        assert!(matches!(&c[0], AvatarCmd::Expression { args, .. } if args.body_motion == BodyMotion::Nod));
        assert!(matches!(&c[1], AvatarCmd::Speak { text, .. } if text == "인벤토리 한번 같이 찾아볼까?"));
        let c = completion_to_cmds(
            "<tool_call>\n<function=manage_zena_system>\n<parameter=command>\nEXIT_APP\n</parameter>\n</function>\n</tool_call>",
            ReactionLevel::Lv3Coaching,
        );
        assert!(matches!(&c[..], [AvatarCmd::System(SystemCommand::ExitApp, None)]));
        let c = completion_to_cmds(
            "<tool_call>\n<function=manage_avatar_asset>\n<parameter=action>\nSWITCH_TRACK\n</parameter>\n<parameter=track>\nSOLIDCRAFT\n</parameter>\n</function>\n</tool_call>",
            ReactionLevel::Lv3Coaching,
        );
        assert!(matches!(&c[..], [AvatarCmd::Asset(a)] if a.track == Some(AvatarTrack::Solidcraft)));
    }

    #[test]
    fn streamed_generation_shows_the_expression_before_the_line_and_reports_the_reply() {
        let (otx, orx) = channel();
        let (rtx, rrx) = channel();
        let w = LlmWorker::spawn(Box::new(TemplateBackend), StreamToAvatar::new(otx, Some(rtx)));
        let sit = Situation {
            user_query: Some("보스 패턴 어떻게 피해?".into()),
            ..Default::default()
        };
        let job = || LlmJob {
            prompt: String::new(),
            situation: sit.clone(),
            level: ReactionLevel::Lv3Coaching,
            epoch: 0,
        };
        while !w.try_submit(job()) {
            std::thread::sleep(Duration::from_millis(5));
        }
        let reply = rrx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(reply.user.as_deref(), Some("보스 패턴 어떻게 피해?"));
        assert!(reply.zena.contains("반격"), "{reply:?}");
        let cmds: Vec<AvatarCmd> = orx.try_iter().collect();
        let first_expr = cmds
            .iter()
            .position(|c| matches!(c, AvatarCmd::Expression { .. }))
            .unwrap();
        let first_speak = cmds
            .iter()
            .position(|c| matches!(c, AvatarCmd::Speak { .. }))
            .unwrap();
        assert!(first_expr < first_speak, "{cmds:?}");
        let spoken: Vec<&str> = cmds.iter().filter_map(AvatarCmd::speech).collect();
        assert_eq!(spoken.join(" "), reply.zena);
    }

    #[test]
    fn dictionary_expressions_rotate_lines_and_respect_the_gap() {
        let lex = Lexicon::bundled();
        let e = lex
            .node("emotion.celebrate")
            .unwrap()
            .action
            .express()
            .unwrap()
            .clone();
        let mut p = ExpressionPlayer::default();
        let line = |c: Option<AvatarCmd>| match c {
            Some(AvatarCmd::Act { args, level, .. }) => (args.speech_text, level),
            other => panic!("{other:?}"),
        };
        let (a, lv) = line(p.play("emotion.celebrate", &e, None, 0, 1500));
        assert_eq!(lv, ReactionLevel::Lv2Cached);
        assert!(p.play("emotion.celebrate", &e, None, 500, 1500).is_none(), "within the gap");
        let (b, _) = line(p.play("emotion.celebrate", &e, None, 2000, 1500));
        assert_ne!(a, b, "the next line of the node");
        let silent = lex
            .node("react.combat")
            .unwrap()
            .action
            .express()
            .unwrap()
            .clone();
        assert_eq!(
            line(p.play("react.combat", &silent, None, 0, 1500)),
            (None, ReactionLevel::Lv1Silent)
        );
        match p.play(
            "emotion.praise",
            lex.node("emotion.praise").unwrap().action.express().unwrap(),
            None,
            0,
            0,
        ) {
            Some(AvatarCmd::Act { args, .. }) => {
                assert_eq!(
                    args.character_emote,
                    Some(zena_core::tools::CharacterEmote::ChibiHeart)
                )
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn scenes_react_on_change_and_events_after_their_cooldown() {
        let obs = |key: &str, kind: VisionKind, cooldown_s: u32| ScreenObservation {
            key: key.into(),
            kind,
            link: String::new(),
            score: 0.9,
            cos: 0.15,
            peak: None,
            cooldown_s,
            encode_ms: 0.0,
            tokens: 0,
            cells: 0,
        };
        let mut g = ScreenGate::default();
        assert!(g.admit(&obs("vision.stock_chart", VisionKind::Scene, 600), 0));
        assert!(
            !g.admit(&obs("vision.stock_chart", VisionKind::Scene, 600), 3000),
            "same scene"
        );
        assert!(
            g.admit(&obs("vision.coding", VisionKind::Scene, 600), 6000),
            "the scene changed"
        );
        assert!(
            !g.admit(&obs("vision.stock_chart", VisionKind::Scene, 600), 9000),
            "back to a scene seen within its cooldown"
        );
        assert!(g.admit(&obs("vision.game_victory", VisionKind::Event, 20), 10_000));
        assert!(!g.admit(&obs("vision.game_victory", VisionKind::Event, 20), 15_000));
        assert!(g.admit(&obs("vision.game_victory", VisionKind::Event, 20), 31_000));
    }

    /// A screen that never changes: the capture never has a new frame.
    struct StaticScreen;
    impl FrameSource for StaticScreen {
        fn next_frame(&mut self) -> anyhow::Result<Option<crate::capture::Frame>> {
            std::thread::sleep(Duration::from_millis(20));
            Ok(None)
        }
        fn size(&self) -> (usize, usize) {
            (160, 90)
        }
    }

    /// Records every Lv3 situation it is asked about and answers with the template.
    struct Recorder(Sender<Situation>);
    impl LlmBackend for Recorder {
        fn name(&self) -> &'static str {
            "recorder"
        }
        fn generate(&mut self, p: &str, s: &Situation, on_text: &mut dyn FnMut(&str)) -> String {
            let _ = self.0.send(s.clone());
            TemplateBackend.generate(p, s, on_text)
        }
    }

    fn links(
        llm: LlmWorker,
    ) -> (
        Links,
        Sender<String>,
        Receiver<AvatarCmd>,
        Sender<ScreenObservation>,
    ) {
        let (utx, urx) = channel();
        let (otx, orx) = channel();
        let (_ltx, lrx) = channel();
        let (stx, srx) = channel();
        let links = Links {
            llm,
            utterances: urx,
            out: otx,
            lexicon: lrx,
            learn: None,
            screen_frames: None,
            screen: Some(srx),
            replies: None,
            actions: None,
            ask_epoch: Arc::new(AtomicU64::new(0)),
            privacy_epoch: Arc::new(AtomicU64::new(0)),
            spoken_lock: Arc::new(Mutex::new(SpokenSlotLock::default())),
        };
        (links, utx, orx, stx)
    }

    #[test]
    fn utterances_and_screen_observations_become_dictionary_expressions() {
        let (links, utx, orx, stx) =
            links(LlmWorker::spawn(Box::new(TemplateBackend), OnDone(|_, _| {})));
        let src = Box::new(crate::capture::SyntheticSource::new(160, 90));
        std::thread::spawn(move || run(src, links, Lexicon::bundled(), PipelineConfig::default()));
        utx.send("와 보스 잡았다! 이겼어!".to_string()).unwrap();
        stx.send(ScreenObservation {
            key: "vision.stock_chart".into(),
            kind: VisionKind::Scene,
            link: "react.market".into(),
            score: 0.8,
            cos: 0.14,
            peak: Some((0.75, 0.25)),
            cooldown_s: 600,
            encode_ms: 0.0,
            tokens: 0,
            cells: 0,
        })
        .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let (mut cheered, mut chart) = (false, false);
        while std::time::Instant::now() < deadline && !(cheered && chart) {
            if let Ok(AvatarCmd::Act { args, .. }) = orx.recv_timeout(Duration::from_millis(200)) {
                if (args.emotion_state, args.body_motion) == (EmotionState::Excited, BodyMotion::Cheer)
                {
                    cheered = args.speech_text.is_some();
                }
                if args.body_motion == BodyMotion::Think
                    && args.gaze_target.is_some_and(|g| g.angle_deg < 90.0)
                {
                    chart = true;
                }
            }
        }
        assert!(cheered, "the user's win is played from the dictionary");
        assert!(
            chart,
            "the chart on screen is played from react.market, looking at its peak"
        );
    }

    #[test]
    fn chat_works_on_a_static_screen_and_remembers_the_previous_exchange() {
        let (stx, srx) = channel();
        let (otx, _orx) = channel::<AvatarCmd>();
        let (rtx, rrx) = channel();
        let ask_epoch = Arc::new(AtomicU64::new(0));
        let spoken_lock = Arc::new(Mutex::new(SpokenSlotLock::default()));
        let llm = LlmWorker::spawn_with_epoch(
            Box::new(Recorder(stx)),
            StreamToAvatar::with_guards(otx, Some(rtx), ask_epoch.clone(), spoken_lock.clone()),
            ask_epoch.clone(),
        );
        let (mut links, utx, orx, _screen) = links(llm);
        links.replies = Some(rrx);
        links.ask_epoch = ask_epoch;
        links.spoken_lock = spoken_lock;
        std::thread::spawn(move || {
            run(
                Box::new(StaticScreen),
                links,
                Lexicon::bundled(),
                PipelineConfig::default(),
            )
        });
        utx.send("제나야 보스 패턴 어떻게 피해?".into()).unwrap();
        let first = srx
            .recv_timeout(Duration::from_secs(5))
            .expect("handled without any frame");
        assert!(first.history.is_empty());
        assert!(first.session.is_some());
        std::thread::sleep(Duration::from_millis(300));
        utx.send("제나야 인벤토리 어디 있어?".into()).unwrap();
        let second = srx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(second.history.len(), 1, "{:?}", second.history);
        assert!(second.history[0]
            .user
            .as_deref()
            .unwrap()
            .contains("보스 패턴"));
        assert!(!second.history[0].zena.is_empty());
        let prompt = build_fast_path_prompt(&second);
        assert!(prompt.contains("[최근 대화]"));
        drop(orx);
    }

    #[test]
    fn zena_speaks_up_after_a_silence_and_switches_assets_on_request() {
        let (links, utx, orx, stx) =
            links(LlmWorker::spawn(Box::new(TemplateBackend), OnDone(|_, _| {})));
        let cfg = PipelineConfig {
            proactive_silence_ms: 400,
            ..Default::default()
        };
        std::thread::spawn(move || run(Box::new(StaticScreen), links, Lexicon::bundled(), cfg));
        stx.send(ScreenObservation {
            key: "vision.coding".into(),
            kind: VisionKind::Scene,
            link: "react.coding".into(),
            score: 0.8,
            cos: 0.14,
            peak: None,
            cooldown_s: 600,
            encode_ms: 0.0,
            tokens: 0,
            cells: 0,
        })
        .unwrap();
        let lex = Lexicon::bundled();
        let coding_lines = &lex
            .node("proactive.coding")
            .unwrap()
            .action
            .express()
            .unwrap()
            .lines;
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let mut proactive_line = None;
        while std::time::Instant::now() < deadline && proactive_line.is_none() {
            if let Ok(AvatarCmd::Act { args, .. }) = orx.recv_timeout(Duration::from_millis(100)) {
                if let Some(s) = args.speech_text.filter(|s| coding_lines.contains(s)) {
                    proactive_line = Some(s);
                }
            }
        }
        assert!(proactive_line.is_some(), "a proactive.coding line after the silence");
        utx.send("제나야 로봇 모드로 변신해 줘".into()).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let mut switched = false;
        while std::time::Instant::now() < deadline && !switched {
            if let Ok(AvatarCmd::Asset(a)) = orx.recv_timeout(Duration::from_millis(100)) {
                switched =
                    a.action == AssetAction::SwitchTrack && a.track == Some(AvatarTrack::Solidcraft);
            }
        }
        assert!(switched);
    }

    #[test]
    fn ingress_pii_hard_blocks_card_and_seed_before_llm_and_privacy_lock_activates_blind_assist() {
        let (stx, srx) = channel();
        let (otx, _orx) = channel::<AvatarCmd>();
        let ask_epoch = Arc::new(AtomicU64::new(0));
        let llm = LlmWorker::spawn_with_epoch(
            Box::new(Recorder(stx)),
            StreamToAvatar::with_guards(
                otx,
                None,
                ask_epoch.clone(),
                Arc::new(Mutex::new(SpokenSlotLock::default())),
            ),
            ask_epoch.clone(),
        );
        let (mut links, utx, orx, scr_tx) = links(llm);
        links.ask_epoch = ask_epoch;
        std::thread::spawn(move || {
            run(
                Box::new(StaticScreen),
                links,
                Lexicon::bundled(),
                PipelineConfig::default(),
            )
        });

        // 1. Valid-checksum BIP-39 seed phrase in chat -> blocked at ingress, NEVER reaches LLM!
        utx.send("제나야 legal winner thank year wave sausage worth useful legal winner thank yellow 이거 뭐야?".into())
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        let mut got_block_reaction = false;
        while std::time::Instant::now() < deadline && !got_block_reaction {
            if let Ok(AvatarCmd::Act { args, .. }) = orx.recv_timeout(Duration::from_millis(80)) {
                if args
                    .speech_text
                    .as_deref()
                    .is_some_and(|s| s == SecretKind::WalletSeedPhrase.warning_line())
                {
                    got_block_reaction = true;
                }
            }
        }
        assert!(got_block_reaction, "seed phrase must trigger block reaction");
        assert!(
            srx.recv_timeout(Duration::from_millis(120)).is_err(),
            "blocked seed phrase must NEVER reach LLM"
        );

        // 2. Trigger VisionKind::Privacy -> PrivacyStateMachine locks, sends PrivacyLock(true),
        //    and a text question asked while locked runs in BlindAssist (`privacy_locked == true`).
        scr_tx
            .send(ScreenObservation {
                key: "vision.privacy_bank".into(),
                kind: VisionKind::Privacy,
                link: String::new(),
                score: 0.95,
                cos: 0.4,
                peak: None,
                cooldown_s: 60,
                encode_ms: 1.0,
                tokens: 1,
                cells: 1,
            })
            .unwrap();
        let mut locked_event = false;
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while std::time::Instant::now() < deadline && !locked_event {
            if let Ok(AvatarCmd::PrivacyLock { locked: true }) =
                orx.recv_timeout(Duration::from_millis(80))
            {
                locked_event = true;
            }
        }
        assert!(locked_event, "VisionKind::Privacy must emit PrivacyLock(true)");

        utx.send("제나야 보스 패턴 어떻게 피해?".into()).unwrap();
        let blind = srx
            .recv_timeout(Duration::from_secs(5))
            .expect("BlindAssist answers text question while locked");
        assert!(
            blind.session.as_ref().is_some_and(|s| s.privacy_locked),
            "session must carry privacy_locked = true during BlindAssist"
        );
    }

    #[test]
    fn cn278_pipeline_wires_live_web_rows_checkpoint_to_focused_and_course_brief() {
        use crate::browser_uia::NavBadge;
        use premium_discount_nav::Rect;
        let overlay = NavBadgeOverlay {
            cc: "shop.example.com".into(),
            screen_key: Some("0xpage1".into()),
            window_rect_px: None,
            badges: vec![NavBadge {
                number: 1,
                anchor: "0xcp-easypay".into(),
                label: "간편결제".into(),
                rect: Rect { x: 0.2, y: 0.3, w: 0.2, h: 0.1 },
                tier: Some(Tier::Premium),
                commit_guarded: false,
                unavailable: false,
            }],
            occluded_by_overlay: false,
            primary_arrow_deg: None,
        };
        let snap = build_screen_anchor_snapshot(
            "shop.example.com",
            "0xpage1",
            Some("결제 화면"),
            Some(&overlay),
            5_000,
        );
        assert_eq!(snap.checkpoint_to.as_deref(), Some("0xcp-easypay"));
        assert_eq!(snap.focused.as_ref().and_then(|f| f.number), Some(1));
        let brief = build_course_brief_from_overlay(&overlay).expect("course brief built from live overlay");
        assert!(brief.verdict_judged_premium);
        assert!(brief.to_prompt_lines()[1].contains("#1 '간편결제'"));
    }

    #[test]
    fn cn280_hybrid_turn_attributes_05_05_synergy_credit_across_rag_and_course_checkpoint() {
        let mut ledger = ExposureLedger::new();
        // Same hybrid turn delivers a RAG tactics note ("tactics") AND a course checkpoint ("route").
        ledger.insert(zena_core::exposure::ExposedRecord {
            record_id: "rag-ice-jar".into(),
            cc: "eldenring.exe".into(),
            to: None,
            canonical_topic: "빙결 항아리".into(),
            role: ExposureRole::RecommendedSolution,
            w_spoken: 1.0,
            exposed_at_ms: 10_000,
            target_kind: ExposureTargetKind::RagRecord,
            exclusive_slot_group: Some("tactics".into()),
        });
        ledger.record_course_checkpoint(
            "0xcp-right-ladder",
            "eldenring.exe",
            Some("0xcp-right-ladder"),
            "우측 사다리",
            ExposureRole::RecommendedSolution,
            10_000,
        );
        let cands = active_credit_candidates_meta(&mut ledger, &["eldenring.exe".to_string()], 20_000);
        assert_eq!(cands.len(), 2);
        assert!(cands.iter().any(|c| c.kind == CreditTargetKind::RagRecord));
        assert!(cands.iter().any(|c| c.kind == CreditTargetKind::CourseCheckpoint));

        let mut verdicts = crate::pd_bridge::LocalVerdictStore::default();
        verdicts.ensure_record("rag-ice-jar", "eldenring.exe", None, "빙결 항아리", "stored");
        let cp = verdicts.ensure_record("0xcp-right-ladder", "eldenring.exe", None, "우측 사다리", "local");
        cp.track = "nav".into();
        let mut gate = crate::pd_bridge::OutcomeVoteGate::new(false);
        verdicts.apply_outcome_credit(&cands, 1, 20_000, false, &mut gate);
        // Both RagRecord and CourseCheckpoint receive complementary 0.5 / 0.5 synergy credit!
        assert_eq!(verdicts.records["rag-ice-jar"].wins, 1);
        assert_eq!(verdicts.records["0xcp-right-ladder"].wins, 1);
        assert!((verdicts.records["rag-ice-jar"].obs[0].credit - 0.5).abs() < 1e-6);
        assert!((verdicts.records["0xcp-right-ladder"].obs[0].credit - 0.5).abs() < 1e-6);
    }
}
