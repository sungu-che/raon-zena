//! Worker thread: capture → event sensor → tracker → NLP-NMS router (word dictionary) → reaction routing →
//! dictionary expressions / (Lv3) generation → avatar commands for the UI thread. Every few seconds a downscaled
//! frame also goes to the screen watcher, whose observations come back as dictionary expressions too.
//!
//! Character chat on top of that:
//! * **multi-turn memory** — the last few exchanges ([`Dialogue`]) and a session summary ([`SessionTracker`]: scene
//!   time, win/loss streaks, affinity, tension) go into every Lv3 prompt's user turn;
//! * **streamed tool calls** — [`StreamToAvatar`] turns Raon-1-nano's deltas into an expression first and the line
//!   sentence by sentence ([`AvatarCmd::Expression`], [`AvatarCmd::Speak`]);
//! * **proactive chatter** — after a silence Zena says a `proactive.*` dictionary line for the current scene, or
//!   suggests a break after a long session ([`ProactiveTimer`]);
//! * **avatar assets** — "로봇 모드로 변신해 줘" / "안경 써 봐" become [`AvatarCmd::Asset`].
//!
//! Utterances are handled on every tick, also while the screen is static and the capture has no new frame.

use crate::capture::FrameSource;
use crate::lexicon::LexiconMsg;
use crate::llm::{JobHandler, LlmJob, LlmWorker};
use crate::course::CourseTracker;
use crate::pd_bridge::{self, Analysis};
use crate::watcher::{ScreenFrame, ScreenObservation};
use nlp_nms_router::{Caught, Hit, NlpNmsRouter, RouterInput};
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::time::Instant;
use zena_core::avatar::viseme_timeline;
use zena_core::cues::{cue, cue_for_reason, CuePlayer};
use zena_core::geometry::{FlowDirection, ScreenGeometry};
use zena_core::prompt::{build_fast_path_prompt, Dialogue, Situation};
use zena_core::reaction::{Observation, ReactionLevel, ReactionRouter};
use zena_core::session::{Proactive, ProactiveTimer, SessionTracker};
use zena_core::stream::{StreamEvent, ToolCallStream};
use zena_core::tools::{GazeTarget, SystemCommand, ToolCall, WindowLayerMode, ZenaAssetArgs, ZenaAvatarArgs};
use zena_lexicon::{Action, Express, ExpressLevel, Lexicon, VisionKind};
use premium_discount_sensor::naflex::bgra_to_rgb;
use premium_discount_sensor::{extract_rois, LogLumSensor, RoiTracker};

/// What the UI thread is asked to do.
#[derive(Debug, Clone)]
pub enum AvatarCmd {
    /// Lv1: look toward an event, no speech.
    Gaze(GazeTarget),
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
    /// Language pack code (`zena_locale`): also picks the Stanza model of the router
    pub language: String,
    /// The premium-discount bridge (search, verdict lines, votes); `None` keeps Zena offline
    pub pd: Option<PdOptions>,
}

/// premium-discount bridge settings
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PdOptions {
    /// The user agreed to share feedback and outcomes as premium-discount votes
    pub share: bool,
}

/// How long a question waits for premium-discount's context before Zena answers without it (the THINK pose covers it)
pub const ASK_CONTEXT_BUDGET: std::time::Duration = std::time::Duration::from_millis(2500);

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

/// The router for `mode`, with the SDS ledger of its vector space under `%LOCALAPPDATA%\RaonZena\score_dynamics`, and the
/// models it runs with (the premium-discount bridge borrows the analyzer and the embedder).
fn build_router(lexicon: &Lexicon, mode: NlpMode, language: &str) -> (NlpNmsRouter, nlp_nms_router::router::Models) {
    use nlp_nms_router::router::Models;
    let models = match mode {
        NlpMode::V1 => Models::v1(),
        NlpMode::Rules => Models::rules(),
        #[cfg(feature = "nlp")]
        NlpMode::Auto => {
            use nlp_nms_router::embed::TextEmbedder;
            let m = Models::auto(if language == "ko" { "korean" } else { language });
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
    let _ = language;
    (NlpNmsRouter::with_models(lexicon, models.clone()), models)
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
            language: zena_locale::DEFAULT_LANGUAGE.into(),
            pd: None,
        }
    }
}

/// What Zena answered to a request, reported back by the LLM thread for the dialogue memory.
#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    pub user: Option<String>,
    pub zena: String,
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
}

/// Rough speaking time for Korean TTS (≈ 7 syllables per second) until real TTS durations exist.
pub fn speech_ms(text: &str) -> u32 {
    (text.chars().filter(|c| !c.is_whitespace()).count() as u32 * 140).max(400)
}

/// Plays dictionary expressions: lines rotate per domain, the same domain is not repeated within `gap_ms`.
#[derive(Default)]
pub struct ExpressionPlayer {
    next_line: HashMap<String, usize>,
    last_ms: HashMap<String, u64>,
}

impl ExpressionPlayer {
    pub fn play(&mut self, key: &str, e: &Express, gaze: Option<GazeTarget>, now: u64, gap_ms: u64) -> Option<AvatarCmd> {
        if self.last_ms.get(key).is_some_and(|&t| now.saturating_sub(t) < gap_ms) {
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
    let l = zena_locale::current();
    if let Action::Ask { topic, .. } = &c.action {
        if topic == "screen" || topic == "market" {
            if let Some((key, score)) = screen {
                let what = lex.vision_node(key).map(|v| v.semantic.clone()).unwrap_or_else(|| key.clone());
                out.push(zena_locale::fill(l.app("screen_now"), &[("what", &what), ("pct", &format!("{:.0}", score * 100.0))]));
            } else {
                out.push(l.app("screen_unknown").to_string());
            }
        }
        if topic == "market" {
            out.push(l.app("market_empathy").to_string());
        }
    }
    out
}

/// The game courses (premium-discount time-trial collection): `%LOCALAPPDATA%\RaonZena\courses.json`
fn course_store() -> std::path::PathBuf {
    std::env::var_os("LOCALAPPDATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("RaonZena")
        .join("courses.json")
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

/// Sends avatar commands; anything Zena says counts as activity for the proactive timer.
struct Out<'a> {
    tx: &'a Sender<AvatarCmd>,
}

impl Out<'_> {
    fn send(&self, cmd: AvatarCmd, now: u64, proactive: &mut ProactiveTimer) -> anyhow::Result<()> {
        if cmd.speech().is_some() {
            proactive.touch(now);
        }
        self.tx.send(cmd)?;
        Ok(())
    }
}

pub fn run(mut src: Box<dyn FrameSource>, links: Links, mut lexicon: Lexicon, cfg: PipelineConfig) -> anyhow::Result<()> {
    let Links {
        llm,
        utterances,
        out,
        lexicon: reloads,
        learn,
        screen_frames,
        screen,
        replies,
    } = links;
    let out = Out { tx: &out };
    let (w, h) = src.size();
    let geo = ScreenGeometry::new(w as u32, h as u32);
    let mut sensor = LogLumSensor::new(w, h, cfg.sensor_threshold);
    let mut tracker = RoiTracker::default();
    let (mut router, models) = build_router(&lexicon, cfg.nlp, &cfg.language);
    let analysis = Analysis { embedder: models.embedder.clone(), analyzer: models.analyzer.clone() };
    let bridge = cfg.pd.as_ref().and_then(|o| pd_bridge::connect(analysis.clone(), o.share));
    let mut sds_flushed = 0u64;
    let mut reactions = ReactionRouter::default();
    let mut cues = CuePlayer::default();
    let mut expressions = ExpressionPlayer::default();
    let mut gate = ScreenGate::default();
    let mut session = SessionTracker::default();
    // The game course (premium-discount time-trial): scenes are checkpoints, a victory is the finish, my best run is the ghost
    let mut course = CourseTracker::open(&course_store());
    let mut dialogue = Dialogue::new(cfg.history_turns);
    let mut proactive = ProactiveTimer::new(cfg.proactive_silence_ms);
    let mut scene_link: Option<String> = None;
    let mut quiet_until = 0u64;
    let mut last_screen: Option<(String, f32)> = None;
    let mut last_frame_sent: Option<u64> = None;
    let t0 = Instant::now();
    eprintln!(
        "pipeline: {w}x{h} sensor, dictionary revision {} ({} domains), Lv3 backend = {}, screen watcher {}, proactive {}",
        lexicon.version,
        router.bank.len(),
        llm.name,
        if screen_frames.is_some() { "on" } else { "off" },
        if cfg.proactive_silence_ms > 0 {
            format!("after {}s", cfg.proactive_silence_ms / 1000)
        } else {
            "off".into()
        }
    );
    loop {
        // The app in front (not Zena's own window): the scope of premium-discount searches for questions
        if bridge.is_some() {
            pd_bridge::note_foreground();
        }
        // A static screen gives no frame; everything but the sensor still runs.
        let frame = src.next_frame()?;
        if let Some(f) = &frame {
            if (f.width, f.height) != (w, h) {
                anyhow::bail!("frame size changed {}x{} -> {}x{}", w, h, f.width, f.height);
            }
        }
        let now = t0.elapsed().as_millis() as u64;

        // SDS observations go to this run's file every 15 s (folded into the ledger at the next start).
        if now.saturating_sub(sds_flushed) >= 15_000 {
            sds_flushed = now;
            if let Some(l) = &router.models.sds {
                if let Err(e) = l.lock().unwrap().flush() {
                    eprintln!("[sds] flush failed: {e:#}");
                }
            }
        }

        // A new dictionary revision (edited with `zena-lexicon`, or learned).
        for lex in reloads.try_iter() {
            router.reload(&lex);
            eprintln!("[lexicon] router now on revision {} ({} domains)", lex.version, router.bank.len());
            lexicon = lex;
        }

        // What Zena answered (dialogue memory). Her automatic coaching calls are not conversation: only answers to
        // the player are remembered, so warnings do not push the exchange out of the window.
        if let Some(rx) = &replies {
            for r in rx.try_iter().filter(|r| !r.zena.trim().is_empty()) {
                proactive.touch(now);
                if r.user.is_some() {
                    dialogue.push(r.user.as_deref(), &r.zena);
                }
            }
        }

        // Periodic screen sampling: one RGB copy every interval, dropped if the watcher is still busy.
        if let (Some(tx), Some(f)) = (&screen_frames, &frame) {
            if last_frame_sent.is_none_or(|t| now.saturating_sub(t) >= cfg.screen_interval_ms) {
                last_frame_sent = Some(now);
                let _ = tx.try_send(ScreenFrame {
                    rgb: bgra_to_rgb(&f.bgra),
                    w,
                    h,
                });
            }
        }
        if let Some(rx) = &screen {
            for o in rx.try_iter() {
                last_screen = (o.kind != VisionKind::Privacy).then(|| (o.key.clone(), o.score));
                match o.kind {
                    VisionKind::Scene => {
                        let what = lexicon
                            .vision_node(&o.key)
                            .map(|v| v.semantic.clone())
                            .unwrap_or_else(|| o.key.clone());
                        session.scene(&o.key, &what, now);
                        scene_link = Some(o.link.clone());
                        course.scene(pd_bridge::foreground_cc().as_deref(), &o.key, &what, now as i64);
                        session.set_course(course.brief());
                    }
                    VisionKind::Privacy => quiet_until = now + o.cooldown_s as u64 * 1000,
                    VisionKind::Event => {}
                }
                if !gate.admit(&o, now) {
                    continue;
                }
                if o.kind == VisionKind::Event {
                    match o.key.as_str() {
                        "vision.game_defeat" => {
                            session.defeat(now);
                            course.defeat();
                            session.set_course(course.brief());
                        }
                        "vision.game_victory" => {
                            session.victory(now);
                            pd_bridge::outcome_in_background(bridge.as_ref(), 1, now as i64);
                            if let Some(added) = course.victory(now as i64) {
                                eprintln!("[course] run recorded: {added:?}");
                            }
                            session.set_course(course.brief());
                        }
                        _ => {}
                    }
                }
                let Some(e) = lexicon.vision_express(&o.key).cloned() else {
                    continue;
                };
                let gaze = match (o.kind, o.peak) {
                    (VisionKind::Privacy, _) | (_, None) => None,
                    (_, Some((x, y))) => {
                        let p = geo.to_polar(x * w as f32, y * h as f32);
                        Some(GazeTarget {
                            angle_deg: p.theta_deg,
                            distance_norm: p.r,
                        })
                    }
                };
                eprintln!(
                    "[screen] {} (cos {:.3}, p {:.2}, {} of {} tokens, {:.0} ms) → {}",
                    o.key, o.cos, o.score, o.tokens, o.cells, o.encode_ms, o.link
                );
                if let Some(cmd) = expressions.play(&o.link, &e, gaze, now, cfg.expression_gap_ms) {
                    out.send(cmd, now, &mut proactive)?;
                }
            }
        }

        let tracks = match &frame {
            Some(f) => {
                let stats = sensor.process_bgra(&f.bgra);
                let rois = extract_rois(&stats, &geo, cfg.min_tile_events);
                tracker.update(rois, now).to_vec()
            }
            None => Vec::new(),
        };
        let max_energy = tracks.iter().map(|t| t.roi.energy).fold(1.0f32, f32::max);

        let mut inputs: Vec<RouterInput> = Vec::new();
        for text in utterances.try_iter() {
            // The topic keys of this turn (Relay Keys) restore the subject a follow-up question leaves out
            session.push_turn_keys(analysis.keys(&text));
            if let Some((intent, negated)) = pd_bridge::detect_feedback(&text, &analysis) {
                pd_bridge::feedback_in_background(bridge.as_ref(), intent, negated, &text, now as i64);
            }
            inputs.push(RouterInput::Utterance { text, t_ms: now });
        }
        if !inputs.is_empty() {
            proactive.touch(now);
            session.affinity(0.01);
        }
        for t in tracks.iter().filter(|t| t.age >= 1) {
            let key = match (t.flow, t.roi.polar.r) {
                (FlowDirection::Inward, r) if r < 0.22 => "event_close_threat",
                (FlowDirection::Inward, _) => "event_monster",
                (FlowDirection::Outward, _) => "event_skill",
                _ => "event_motion",
            };
            inputs.push(RouterInput::GameEvent {
                label: zena_locale::current().app(key).to_string(),
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

        // Learning: spans the dictionary caught by vectors alone.
        if let (Some(tx), Some(min)) = (&learn, cfg.learn_min) {
            for c in routed
                .expressions
                .iter()
                .chain(&routed.queries)
                .filter(|c| c.hit == Hit::Vector && c.score >= min)
            {
                let _ = tx.send(LexiconMsg::Learn {
                    key: c.key.clone(),
                    span: c.span.clone(),
                });
            }
        }
        for c in &routed.expressions {
            session_signal(&mut session, &c.key, now);
            if c.key == "emotion.celebrate" {
                pd_bridge::outcome_in_background(bridge.as_ref(), 1, now as i64);
            }
        }
        // Dictionary expressions first, then the commands (quitting plays the overlay's own farewell sequence).
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

        // Pick the strongest surviving signal for this tick.
        let mut obs = Observation::default();
        let mut label = None;
        let query = routed.queries.first().cloned();
        if let Some(q) = &query {
            obs.user_query = Some(q.utterance.clone());
            // Acknowledge at once (e.g. a thinking pose) while Raon-1-nano answers.
            if let Some(e) = q.action.express() {
                if let Some(cmd) = expressions.play(&q.key, e, None, now, cfg.expression_gap_ms) {
                    out.send(cmd, now, &mut proactive)?;
                }
            }
        }
        if let Some((l, p, s)) = routed.events.iter().max_by(|a, b| a.2.total_cmp(&b.2)) {
            let tr = tracks.iter().min_by(|a, b| {
                let da = (a.roi.polar.r - p.r).abs() + (a.roi.polar.theta_deg - p.theta_deg).abs() / 360.0;
                let db = (b.roi.polar.r - p.r).abs() + (b.roi.polar.theta_deg - p.theta_deg).abs() / 360.0;
                da.total_cmp(&db)
            });
            obs.polar = Some(*p);
            obs.energy = *s;
            obs.flow = tr.map(|t| t.flow);
            obs.radial_velocity = tr.map(|t| t.radial_velocity).unwrap_or(0.0);
            label = Some(l.clone());
        }
        if obs.user_query.is_some() || obs.polar.is_some() {
            let d = reactions.route(&obs, now);
            if obs.user_query.is_none() && d.level >= ReactionLevel::Lv2Cached {
                session.combat(obs.energy, now);
            }
            match d.level {
                ReactionLevel::Lv1Silent => {
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
                ReactionLevel::Lv2Cached => {
                    out.send(cached_cue(&mut cues, d.reason, obs.polar, d.level), now, &mut proactive)?;
                }
                ReactionLevel::Lv3Coaching => {
                    let mut context = Vec::new();
                    if let Some(q) = &query {
                        context = topic_context(q, &lexicon, last_screen.as_ref());
                    }
                    let sit = Situation {
                        event_label: label,
                        polar: obs.polar,
                        flow: obs.flow,
                        hp_ratio: obs.hp_ratio,
                        user_query: obs.user_query.clone(),
                        context,
                        history: dialogue.turns(),
                        session: Some(session.snapshot(now)),
                    };
                    let prompt = build_fast_path_prompt(&sit);
                    eprintln!(
                        "[lv3] {}{} ({} remembered turns)",
                        d.reason,
                        query
                            .as_ref()
                            .map(|q| format!(" → {} ({:?} {:.2} on '{}'): {}", q.key, q.hit, q.score, q.span, q.utterance))
                            .unwrap_or_default(),
                        sit.history.len()
                    );
                    // Generation runs on the LLM thread (loading the model first if it is not resident); if it is
                    // busy, fall back to an immediate cached cue so Zena still reacts.
                    // A question: premium-discount's records and verdicts for it, fetched on the LLM thread within a
                    // budget (the acknowledging THINK pose is already playing)
                    let prepare = match (&bridge, &query) {
                        (Some(b), Some(q)) => Some(pd_bridge::ask_step(
                            b.clone(),
                            pd_bridge::foreground_cc(),
                            session.active_keys(),
                            q.utterance.clone(),
                            now as i64,
                            ASK_CONTEXT_BUDGET,
                        )),
                        _ => None,
                    };
                    if !llm.try_submit(LlmJob {
                        prompt,
                        situation: sit,
                        level: d.level,
                        prepare,
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

        // Nothing has happened for a while: Zena speaks up on her own.
        if let Some(kind) = proactive.due(now, session.scene_ms(now), now < quiet_until) {
            if let Some((key, e)) = lexicon.proactive(scene_link.as_deref(), kind == Proactive::Break) {
                let (key, e) = (key.to_string(), e.clone());
                if let Some(cmd) = expressions.play(&key, &e, None, now, 0) {
                    if let Some(line) = cmd.speech() {
                        eprintln!("[proactive] {key}");
                        dialogue.push(None, line);
                    }
                    out.send(cmd, now, &mut proactive)?;
                }
            }
        }
    }
}

fn cached_cue(cues: &mut CuePlayer, reason: &str, polar: Option<zena_core::Polar>, level: ReactionLevel) -> AvatarCmd {
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
        StreamEvent::Expression(args) | StreamEvent::Call(ToolCall::ControlZenaAvatar(args)) => Some(AvatarCmd::Expression { args, level }),
        StreamEvent::Speech(text) => {
            let ms = speech_ms(&text);
            Some(AvatarCmd::Speak {
                visemes: viseme_timeline(&text, ms),
                speech_ms: ms,
                text,
                level,
            })
        }
        StreamEvent::Call(ToolCall::ManageZenaSystem(s)) => Some(AvatarCmd::System(s.command, s.window_layer_mode)),
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
    ev.into_iter().filter_map(|e| event_cmd(e, level)).collect()
}

/// The LLM thread's handler: parses the completion while it streams and sends avatar commands at once — the
/// expression as soon as its parameters close, the line sentence by sentence — then reports the answer for the
/// dialogue memory.
pub struct StreamToAvatar {
    out: Sender<AvatarCmd>,
    replies: Option<Sender<Reply>>,
    stream: ToolCallStream,
    streamed: bool,
}

impl StreamToAvatar {
    pub fn new(out: Sender<AvatarCmd>, replies: Option<Sender<Reply>>) -> Self {
        Self {
            out,
            replies,
            stream: ToolCallStream::new(),
            streamed: false,
        }
    }

    fn emit(&self, events: Vec<StreamEvent>, level: ReactionLevel) {
        for cmd in events.into_iter().filter_map(|e| event_cmd(e, level)) {
            let _ = self.out.send(cmd);
        }
    }
}

impl JobHandler for StreamToAvatar {
    fn delta(&mut self, job: &LlmJob, text: &str) {
        self.streamed = true;
        let ev = self.stream.push(text);
        self.emit(ev, job.level);
    }

    fn done(&mut self, job: LlmJob, text: String) {
        let mut ev = if self.streamed { Vec::new() } else { self.stream.push(&text) };
        ev.extend(self.stream.finish());
        self.emit(ev, job.level);
        if let Some(tx) = &self.replies {
            let _ = tx.send(Reply {
                user: job.situation.user_query.clone(),
                zena: self.stream.spoken().to_string(),
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
            prepare: None,
        };
        while !w.try_submit(job()) {
            std::thread::sleep(Duration::from_millis(5));
        }
        let reply = rrx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(reply.user.as_deref(), Some("보스 패턴 어떻게 피해?"));
        assert!(reply.zena.contains("반격"), "{reply:?}");
        let cmds: Vec<AvatarCmd> = orx.try_iter().collect();
        let first_expr = cmds.iter().position(|c| matches!(c, AvatarCmd::Expression { .. })).unwrap();
        let first_speak = cmds.iter().position(|c| matches!(c, AvatarCmd::Speak { .. })).unwrap();
        assert!(first_expr < first_speak, "{cmds:?}");
        let spoken: Vec<&str> = cmds.iter().filter_map(AvatarCmd::speech).collect();
        assert_eq!(spoken.join(" "), reply.zena);
    }

    #[test]
    fn dictionary_expressions_rotate_lines_and_respect_the_gap() {
        let lex = Lexicon::bundled();
        let e = lex.node("emotion.celebrate").unwrap().action.express().unwrap().clone();
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
        let silent = lex.node("react.combat").unwrap().action.express().unwrap().clone();
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
                assert_eq!(args.character_emote, Some(zena_core::tools::CharacterEmote::ChibiHeart))
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
        assert!(!g.admit(&obs("vision.stock_chart", VisionKind::Scene, 600), 3000), "same scene");
        assert!(g.admit(&obs("vision.coding", VisionKind::Scene, 600), 6000), "the scene changed");
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

    fn links(llm: LlmWorker) -> (Links, Sender<String>, Receiver<AvatarCmd>, Sender<ScreenObservation>) {
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
        };
        (links, utx, orx, stx)
    }

    #[test]
    fn utterances_and_screen_observations_become_dictionary_expressions() {
        let (links, utx, orx, stx) = links(LlmWorker::spawn(Box::new(TemplateBackend), OnDone(|_, _| {})));
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
                if (args.emotion_state, args.body_motion) == (EmotionState::Excited, BodyMotion::Cheer) {
                    cheered = args.speech_text.is_some();
                }
                if args.body_motion == BodyMotion::Think && args.gaze_target.is_some_and(|g| g.angle_deg < 90.0) {
                    chart = true; // looks up-right, where the chart's peak cell is
                }
            }
        }
        assert!(cheered, "the user's win is played from the dictionary");
        assert!(chart, "the chart on screen is played from react.market, looking at its peak");
    }

    #[test]
    fn chat_works_on_a_static_screen_and_remembers_the_previous_exchange() {
        let (stx, srx) = channel();
        let (otx, _orx) = channel::<AvatarCmd>();
        let (rtx, rrx) = channel();
        let llm = LlmWorker::spawn(Box::new(Recorder(stx)), StreamToAvatar::new(otx, Some(rtx)));
        let (mut links, utx, orx, _screen) = links(llm);
        links.replies = Some(rrx);
        std::thread::spawn(move || run(Box::new(StaticScreen), links, Lexicon::bundled(), PipelineConfig::default()));
        utx.send("제나야 보스 패턴 어떻게 피해?".into()).unwrap();
        let first = srx.recv_timeout(Duration::from_secs(5)).expect("handled without any frame");
        assert!(first.history.is_empty());
        assert!(first.session.is_some());
        std::thread::sleep(Duration::from_millis(300));
        utx.send("제나야 인벤토리 어디 있어?".into()).unwrap();
        let second = srx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(second.history.len(), 1, "{:?}", second.history);
        assert!(second.history[0].user.as_deref().unwrap().contains("보스 패턴"));
        assert!(!second.history[0].zena.is_empty());
        let prompt = build_fast_path_prompt(&second);
        assert!(prompt.contains("[최근 대화]"));
        drop(orx);
    }

    #[test]
    fn zena_speaks_up_after_a_silence_and_switches_assets_on_request() {
        let (links, utx, orx, stx) = links(LlmWorker::spawn(Box::new(TemplateBackend), OnDone(|_, _| {})));
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
        let coding_lines = &lex.node("proactive.coding").unwrap().action.express().unwrap().lines;
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
                switched = a.action == AssetAction::SwitchTrack && a.track == Some(AvatarTrack::Solidcraft);
            }
        }
        assert!(switched);
    }
}
