//! The 350×450 transparent micro-viewport: winit window + one wgpu device, the three-pass scene
//! ([`crate::renderers::scene`]: body → emote & props → speech bubble), dynamic FPS, Win32 layering, the in-overlay
//! chat box and the graceful shutdown sequence.
//!
//! Streamed answers arrive as [`AvatarCmd::Expression`] (the face and motion first) and [`AvatarCmd::Speak`]
//! sentences that queue after what Zena is saying and append to the bubble.

use crate::assets::{AssetLink, Assets};
use crate::pipeline::AvatarCmd;
use crate::renderers::scene::{EmoteShow, Loaded, Missing, Scene, Stage};
use crate::renderers::{depth_texture, FrameParams};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, Ime, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId, WindowLevel};
use zena_core::avatar::{live2d_gaze_params, live2d_mouth, look_at_event, AvatarPlacement, LookAt, Viseme};
use zena_core::geometry::{Polar, ScreenGeometry};
use zena_core::reaction::ReactionLevel;
use zena_core::tools::{
    AssetAction, AvatarPart, AvatarTrack, BodyMotion, CharacterEmote, EmotionState, SystemCommand, WindowLayerMode, ZenaAssetArgs,
    ZenaAvatarArgs,
};
use zena_core::window::{apply_command, frame_interval, Hotkey, ShutdownPhase, WindowEffect, WindowState, VIEWPORT_H, VIEWPORT_W};

#[derive(Debug)]
pub enum UserEvent {
    Cmd(AvatarCmd),
    #[cfg_attr(not(windows), allow(dead_code))]
    Hotkey(Hotkey),
    /// An asset decoded by the asset service.
    Asset(Box<Loaded>),
    /// An asset that could not be produced.
    AssetFailed(Missing),
}

impl std::fmt::Debug for Loaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Loaded::Stamp { emote, frames, .. } => write!(f, "Stamp({emote:?}, {frames} frames)"),
            Loaded::Prop { emote, angle, .. } => write!(f, "Prop({emote:?}, {angle}°)"),
            Loaded::Part { part, .. } => write!(f, "Part({part:?})"),
            Loaded::Body2d { w, h, .. } => write!(f, "Body2d({w}x{h})"),
            Loaded::BodyMesh { track, mesh, vrm } => {
                write!(f, "BodyMesh({track:?}, {} vertices, vrm {})", mesh.positions.len(), vrm.is_some())
            }
        }
    }
}

struct Gfx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth: wgpu::TextureView,
    scene: Scene,
}

/// Smoothed animation state (targets set by commands, current values eased per frame).
struct Anim {
    gaze: Option<Polar>,
    eye: [f32; 2],
    head: [f32; 2],
    emotion: EmotionState,
    motion: BodyMotion,
    motion_start: Instant,
    level: ReactionLevel,
    speech_start: Instant,
    speech_ms: u32,
    visemes: Vec<(u32, Viseme)>,
    expression_until: Instant,
    emote: Option<(CharacterEmote, Instant)>,
    look: LookAt,
}

/// The line on show (streamed sentences append while the answer lasts).
#[derive(Default)]
struct Bubble {
    text: String,
    /// The next sentence starts a new bubble (a new answer began).
    fresh: bool,
}

/// The in-overlay chat box.
#[derive(Default)]
struct Chat {
    open: bool,
    text: String,
    preedit: String,
}

pub struct Overlay {
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
    state: WindowState,
    shutdown: ShutdownPhase,
    anim: Anim,
    bubble: Bubble,
    chat: Chat,
    track: AvatarTrack,
    parts: Vec<AvatarPart>,
    assets: Option<Assets>,
    requested: HashSet<Missing>,
    pending: Vec<Loaded>,
    font: Option<std::path::PathBuf>,
    t0: Instant,
    screen: ScreenGeometry,
    utterances: Sender<String>,
    pub capture_running: Arc<AtomicBool>,
    /// 누르기 vs 끌기 (premium-discount-window 의 5 px 문턱)
    press: crate::pet::Press,
    /// 제자리 복귀 (Auto-Dock)
    dock: crate::pet::Dock,
    /// 창 안의 커서 (물리 픽셀)
    cursor: Option<PhysicalPosition<f64>>,
    /// 사용자가 끌어 옮기는 중 (그 사이의 Moved 는 점 아래 창을 다시 본다)
    dragged: bool,
}

const BUBBLE_HOLD_MS: u64 = 2500;
const EMOTE_FADE_S: f32 = 0.45;

impl Overlay {
    pub fn new(capture_running: Arc<AtomicBool>, screen: ScreenGeometry, utterances: Sender<String>) -> Self {
        let now = Instant::now();
        Self {
            window: None,
            gfx: None,
            state: WindowState::default(),
            shutdown: ShutdownPhase::Running,
            anim: Anim {
                gaze: None,
                eye: [0.0; 2],
                head: [0.0; 2],
                emotion: EmotionState::Neutral,
                motion: BodyMotion::Idle,
                motion_start: now,
                level: ReactionLevel::Lv1Silent,
                speech_start: now,
                speech_ms: 0,
                visemes: vec![],
                expression_until: now,
                emote: None,
                look: LookAt::default(),
            },
            bubble: Bubble::default(),
            chat: Chat::default(),
            track: AvatarTrack::Procedural,
            parts: Vec::new(),
            assets: None,
            requested: HashSet::new(),
            pending: Vec::new(),
            font: None,
            t0: now,
            screen,
            utterances,
            capture_running,
            press: crate::pet::Press::default(),
            dock: crate::pet::Dock::default(),
            cursor: None,
            dragged: false,
        }
    }

    /// 말하거나 다루는 중 (제자리로 돌아가지 않고, 불투명도도 100 %)
    fn busy(&self) -> bool {
        self.speaking() || self.chat.open || self.press.holding() || self.shutdown != ShutdownPhase::Running
    }

    /// 제자리: 창이 있는 모니터의 작업 영역 오른쪽 아래 (premium-discount-window dock_position)
    fn home(&self) -> Option<premium_discount_window::Position> {
        let w = self.window.as_ref()?;
        #[cfg(windows)]
        let area = self.hwnd().and_then(crate::win32::work_area);
        #[cfg(not(windows))]
        let area: Option<premium_discount_window::WorkArea> = None;
        let area = area.or_else(|| {
            let m = w.current_monitor()?;
            let (p, s) = (m.position(), m.size());
            Some(premium_discount_window::WorkArea { x: p.x, y: p.y, width: s.width, height: s.height })
        })?;
        let size = premium_discount_window::Size::new(VIEWPORT_W as u32, VIEWPORT_H as u32);
        Some(premium_discount_window::dock_position(size, area, w.scale_factor()))
    }

    /// 지금 창 위치 (물리 픽셀)
    fn position(&self) -> Option<premium_discount_window::Position> {
        let p = self.window.as_ref()?.outer_position().ok()?;
        Some(premium_discount_window::Position::new(p.x, p.y))
    }

    /// 끌어 놓은 자리 아래의 창 (캐릭터 발밑 바로 아래 점)을 질문 범위로 기억한다
    fn note_window_below(&self) {
        let (Some(w), Some(p)) = (&self.window, self.position()) else { return };
        let size = w.outer_size();
        let uri = crate::pd_bridge::note_window_at(p.x + size.width as i32 / 2, p.y + size.height as i32 + 4);
        if let Some(uri) = uri {
            eprintln!("[pet] dropped on {uri}");
        }
    }

    /// The asset service (track bodies, emote stamps and props, accessories) and the body track to start with.
    pub fn with_assets(mut self, assets: Assets, track: AvatarTrack) -> Self {
        // Prefetch every stamp and the props that do not depend on a direction, so the first emote of each kind
        // does not pop in late; the arrow, the parts and the bodies are asked for when first shown.
        let mut first: Vec<Missing> = CharacterEmote::ALL.iter().map(|e| Missing::Stamp(*e)).collect();
        first.extend(
            CharacterEmote::ALL
                .iter()
                .filter(|e| e.form() == zena_core::tools::EmoteForm::Prop && **e != CharacterEmote::ThreatArrow3d)
                .map(|e| Missing::Prop(*e, 0)),
        );
        if track != AvatarTrack::Procedural {
            first.push(Missing::Body(track));
        }
        for m in first {
            assets.request(&m);
            self.requested.insert(m);
        }
        self.assets = Some(assets);
        self.track = track;
        self
    }

    pub fn with_font(mut self, font: Option<std::path::PathBuf>) -> Self {
        self.font = font;
        self
    }

    fn now_ms(&self) -> u64 {
        self.t0.elapsed().as_millis() as u64
    }

    fn speaking(&self) -> bool {
        self.anim.speech_start.elapsed() < Duration::from_millis(self.anim.speech_ms as u64)
    }

    /// Gaze, emotion, motion and emote (no speech).
    fn express(&mut self, args: &ZenaAvatarArgs, level: ReactionLevel) {
        if let Some(g) = args.gaze_target {
            self.anim.gaze = Some(Polar {
                r: g.distance_norm,
                theta_deg: g.angle_deg,
            });
        }
        if args.body_motion != self.anim.motion || args.body_motion != BodyMotion::Idle {
            self.anim.motion_start = Instant::now();
        }
        self.anim.emotion = args.emotion_state;
        self.anim.motion = args.body_motion;
        self.anim.level = self.anim.level.max(level);
        let now = Instant::now();
        let showing = self.anim.emote.is_some() && now < self.anim.expression_until;
        match args.emote() {
            Some(e) if showing && self.anim.emote.map(|x| x.0) == Some(e) => {}
            Some(e) => self.anim.emote = Some((e, now)),
            None => {}
        }
        let speech_end = self.anim.speech_start + Duration::from_millis(self.anim.speech_ms as u64);
        self.anim.expression_until = speech_end.max(now) + Duration::from_millis(1800);
    }

    /// A whole reaction: replaces what Zena was saying.
    fn act(&mut self, args: &ZenaAvatarArgs, level: ReactionLevel, visemes: Vec<(u32, Viseme)>, speech_ms: u32) {
        self.anim.level = level;
        self.anim.visemes = visemes;
        self.anim.speech_ms = speech_ms;
        self.anim.speech_start = Instant::now();
        self.express(args, level);
        if let Some(t) = &args.speech_text {
            // TTS backend hook (Raon-OpenTTS); until then the line is logged and shown in the bubble.
            eprintln!("[zena:{level:?}] {t}");
            self.bubble.text = t.clone();
            self.bubble.fresh = false;
        }
    }

    /// A streamed sentence: queued after the current speech.
    fn speak(&mut self, text: &str, level: ReactionLevel, visemes: Vec<(u32, Viseme)>, ms: u32) {
        eprintln!("[zena:{level:?}] {text}");
        if self.speaking() {
            let off = self.anim.speech_ms;
            // The previous timeline ends with a closing rest; the next sentence takes its place.
            if self.anim.visemes.last().is_some_and(|(t, v)| *v == Viseme::Rest && *t >= off) {
                self.anim.visemes.pop();
            }
            self.anim.visemes.extend(visemes.into_iter().map(|(t, v)| (t + off, v)));
            self.anim.speech_ms += ms;
        } else {
            self.anim.speech_start = Instant::now();
            self.anim.visemes = visemes;
            self.anim.speech_ms = ms;
        }
        if self.bubble.fresh || self.bubble.text.is_empty() {
            self.bubble.text = text.to_string();
            self.bubble.fresh = false;
        } else {
            self.bubble.text.push(' ');
            self.bubble.text.push_str(text);
        }
        self.anim.level = self.anim.level.max(level);
        let end = self.anim.speech_start + Duration::from_millis(self.anim.speech_ms as u64);
        self.anim.expression_until = self.anim.expression_until.max(end + Duration::from_millis(1800));
    }

    fn asset(&mut self, a: &ZenaAssetArgs) {
        match a.action {
            AssetAction::SwitchTrack => {
                if let Some(t) = a.track {
                    eprintln!("[avatar] track → {t:?}");
                    self.track = t;
                }
            }
            AssetAction::EquipPart => {
                if let Some(p) = a.part.filter(|p| !self.parts.contains(p)) {
                    eprintln!("[avatar] equip {p:?}");
                    self.parts.push(p);
                    self.parts.sort();
                }
            }
            AssetAction::RemovePart => {
                if let Some(p) = a.part {
                    self.parts.retain(|x| *x != p);
                }
            }
            AssetAction::ResetParts => self.parts.clear(),
        }
    }

    fn system(&mut self, cmd: SystemCommand, mode: Option<WindowLayerMode>) {
        for fx in apply_command(&mut self.state, cmd, mode) {
            match fx {
                WindowEffect::BeginShutdown => {
                    if self.shutdown == ShutdownPhase::Running {
                        self.shutdown = ShutdownPhase::Farewell { started_ms: self.now_ms() };
                        let line = "오늘도 수고했어! 또 봐~";
                        let bye = ZenaAvatarArgs {
                            emotion_state: EmotionState::Joy,
                            body_motion: BodyMotion::WaveGoodbye,
                            character_emote: Some(CharacterEmote::ChibiHeart),
                            speech_text: Some(line.into()),
                            ..Default::default()
                        };
                        let ms = 1800;
                        self.act(&bye, ReactionLevel::Lv3Coaching, zena_core::avatar::viseme_timeline(line, ms), ms);
                    }
                }
                WindowEffect::ApplyStyle => self.platform_style(),
                WindowEffect::SetTopmost(on) => {
                    #[cfg(windows)]
                    if let Some(h) = self.hwnd() {
                        crate::win32::set_topmost(h, on);
                    }
                    #[cfg(not(windows))]
                    if let Some(w) = &self.window {
                        w.set_window_level(if on { WindowLevel::AlwaysOnTop } else { WindowLevel::Normal });
                    }
                }
                WindowEffect::Show(v) => {
                    // SW_HIDE: no rendering load; capture + coaching keep running.
                    #[cfg(windows)]
                    if let Some(h) = self.hwnd() {
                        crate::win32::show(h, v);
                    }
                    #[cfg(not(windows))]
                    if let Some(w) = &self.window {
                        w.set_visible(v);
                    }
                }
                WindowEffect::None => {}
            }
        }
    }

    /// Open / close the chat box: the window takes focus and IME input while it is open.
    fn set_chat(&mut self, open: bool) {
        if self.chat.open == open {
            return;
        }
        self.chat.open = open;
        self.chat.preedit.clear();
        self.state.typing = open;
        if !open {
            self.chat.text.clear();
        }
        self.platform_style();
        if let Some(w) = &self.window {
            w.set_ime_allowed(open);
            if open {
                w.focus_window();
            }
        }
        #[cfg(windows)]
        if open {
            if let Some(h) = self.hwnd() {
                crate::win32::focus(h);
            }
        }
    }

    fn submit_chat(&mut self) {
        let text = std::mem::take(&mut self.chat.text);
        if !text.trim().is_empty() {
            eprintln!("[you] {text}");
            let _ = self.utterances.send(text);
        }
        self.set_chat(false);
    }

    #[cfg(windows)]
    fn hwnd(&self) -> Option<windows::Win32::Foundation::HWND> {
        use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
        let w = self.window.as_ref()?;
        match w.window_handle().ok()?.as_raw() {
            RawWindowHandle::Win32(h) => Some(crate::win32::hwnd_from_raw(h.hwnd.get())),
            _ => None,
        }
    }

    fn platform_style(&self) {
        #[cfg(windows)]
        if let Some(h) = self.hwnd() {
            crate::win32::apply_style(h, &self.state);
        }
        #[cfg(not(windows))]
        if let Some(w) = &self.window {
            // Best effort elsewhere: click-through via winit where supported.
            let _ = w.set_cursor_hittest(self.state.mode == WindowLayerMode::InteractivePet || self.state.typing);
        }
    }

    fn frame_params(&mut self, width: u32, height: u32) -> FrameParams {
        let t = self.t0.elapsed().as_secs_f32();
        // Gaze: from the avatar's head on the desktop toward the event.
        let (mut tx, mut ty, mut hx, mut hy) = (0.0, 0.0, 0.0, 0.0);
        if let (Some(p), Some(w)) = (self.anim.gaze, &self.window) {
            let pos = w.outer_position().unwrap_or(PhysicalPosition::new(0, 0));
            let place = AvatarPlacement {
                head_x: pos.x as f32 + VIEWPORT_W as f32 / 2.0,
                head_y: pos.y as f32 + VIEWPORT_H as f32 * 0.3,
                viewer_depth_px: 900.0,
            };
            let look = look_at_event(&self.screen, &place, p);
            self.anim.look = look;
            let prm = live2d_gaze_params(look);
            (hx, hy, tx, ty) = (prm[0].1 / 30.0, prm[1].1 / 30.0, prm[3].1, prm[4].1);
        }
        let k = 0.25;
        self.anim.eye[0] += (tx - self.anim.eye[0]) * k;
        self.anim.eye[1] += (ty - self.anim.eye[1]) * k;
        self.anim.head[0] += (hx - self.anim.head[0]) * k * 0.5;
        self.anim.head[1] += (hy - self.anim.head[1]) * k * 0.5;
        if Instant::now() > self.anim.expression_until {
            self.anim.emotion = EmotionState::Neutral;
            self.anim.motion = BodyMotion::Idle;
        }
        let (brow, eye_open, base_form) = match self.anim.emotion {
            EmotionState::Joy | EmotionState::Excited => (0.0, 0.9, 0.8),
            EmotionState::Surprised | EmotionState::SurprisedUrgent => (-0.3, 1.35, 0.0),
            EmotionState::Angry => (1.0, 0.9, -0.3),
            EmotionState::Sorrow | EmotionState::Worried => (-1.0, 0.85, -0.4),
            EmotionState::Focused => (0.4, 0.95, 0.0),
            EmotionState::Relieved => (-0.2, 0.8, 0.5),
            EmotionState::Neutral => (0.0, 1.0, 0.2),
        };
        let (mut mouth_open, mut mouth_form, mut viseme) = (0.0, base_form, Viseme::Rest);
        if self.speaking() {
            let el = self.anim.speech_start.elapsed().as_millis() as u32;
            viseme = self
                .anim
                .visemes
                .iter()
                .rev()
                .find(|(s, _)| *s <= el)
                .map(|x| x.1)
                .unwrap_or(Viseme::Rest);
            // Stand-in for the TTS RMS envelope.
            let rms = 0.25 + 0.2 * (t * 17.0).sin().abs();
            let m = live2d_mouth(rms, viseme);
            mouth_open = m[0].1;
            mouth_form = 0.5 * base_form + 0.5 * m[1].1;
        }
        let hover = {
            #[cfg(windows)]
            {
                self.hwnd().map(crate::win32::cursor_over).unwrap_or(false)
            }
            #[cfg(not(windows))]
            {
                false
            }
        };
        // Hover fade (30 %) is applied in the shaders so the layered window keeps alpha 255.
        FrameParams {
            gaze: self.anim.eye,
            head: self.anim.head,
            mouth_open,
            mouth_form,
            brow,
            eye_open,
            // 제자리에서 쉬는 동안 82 % (premium-discount-window PILL_OPACITY), 고스트 모드의 커서 위 30 % 는 state.alpha 가
            alpha: self.state.alpha(hover) as f32 / 255.0
                * crate::pet::resting_opacity(self.home().is_some_and(|h| self.position() == Some(h)), self.busy()),
            time: t,
            width,
            height,
            emotion: self.anim.emotion,
            motion: self.anim.motion,
            viseme,
            look: self.anim.look,
            motion_t: self.anim.motion_start.elapsed().as_secs_f32(),
            lift: 0.0,
        }
    }

    fn emote_show(&self) -> Option<EmoteShow> {
        let (e, since) = self.anim.emote?;
        let now = Instant::now();
        let left = self.anim.expression_until.saturating_duration_since(now).as_secs_f32();
        let fade = (left / EMOTE_FADE_S).clamp(0.0, 1.0);
        if fade <= 0.0 {
            return None;
        }
        // The arrow points from Zena's head toward what she looks at.
        let l = self.anim.look;
        let angle_deg = if l.yaw_deg.abs() + l.pitch_deg.abs() > 1.0 {
            l.pitch_deg.atan2(l.yaw_deg).to_degrees()
        } else {
            self.anim.gaze.map(|g| g.theta_deg).unwrap_or(0.0)
        };
        Some(EmoteShow {
            emote: e,
            age: since.elapsed().as_secs_f32(),
            fade,
            angle_deg,
        })
    }

    fn bubble_alpha(&self) -> f32 {
        if self.bubble.text.is_empty() {
            return 0.0;
        }
        let end = self.anim.speech_start + Duration::from_millis(self.anim.speech_ms as u64 + BUBBLE_HOLD_MS);
        let left = end.saturating_duration_since(Instant::now()).as_secs_f32();
        (left / 0.5).clamp(0.0, 1.0)
    }

    fn render(&mut self) {
        let Some((w, h)) = self.gfx.as_ref().map(|g| (g.config.width, g.config.height)) else {
            return;
        };
        for asset in std::mem::take(&mut self.pending) {
            if let Some(g) = &mut self.gfx {
                g.scene.load(&g.device, &g.queue, asset);
            }
        }
        let params = self.frame_params(w, h);
        let emote = self.emote_show();
        let bubble_alpha = self.bubble_alpha();
        if bubble_alpha <= 0.0 && !self.speaking() {
            self.bubble.fresh = true;
        }
        let caret = (self.t0.elapsed().as_millis() / 500) % 2 == 0;
        let Some(g) = &mut self.gfx else { return };
        let stage = Stage {
            params,
            track: self.track,
            parts: &self.parts,
            emote,
            bubble: (bubble_alpha > 0.0).then_some((self.bubble.text.as_str(), bubble_alpha)),
            input: self
                .chat
                .open
                .then_some((self.chat.text.as_str(), self.chat.preedit.as_str(), caret)),
        };
        if let Some(a) = &self.assets {
            for m in g.scene.missing(&stage) {
                if self.requested.insert(m.clone()) {
                    a.request(&m);
                }
            }
        }
        let frame = match g.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) => f,
            wgpu::CurrentSurfaceTexture::Suboptimal(f) => {
                g.surface.configure(&g.device, &g.config);
                f
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            _ => {
                g.surface.configure(&g.device, &g.config);
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("zena") });
        g.scene.render(&g.device, &g.queue, &mut enc, &view, &g.depth, &stage);
        g.queue.submit([enc.finish()]);
        g.queue.present(frame);
    }

    fn init_gfx(window: Arc<Window>, font: Option<&std::path::Path>) -> anyhow::Result<Gfx> {
        #[allow(unused_mut)]
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        #[cfg(windows)]
        {
            // Per-pixel transparency on Windows needs a DirectComposition-backed swap chain;
            // capture, sensor, vision and avatar all share this single DX12 device.
            desc.backends = wgpu::Backends::DX12;
            desc.backend_options.dx12.presentation_system = wgpu::Dx12SwapchainKind::DxgiFromVisual;
        }
        let instance = wgpu::Instance::new(desc);
        let surface = instance.create_surface(window.clone())?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("raon-zena"),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            ..Default::default()
        }))?;
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| anyhow::anyhow!("surface unsupported"))?;
        let caps = surface.get_capabilities(&adapter);
        config.alpha_mode = [wgpu::CompositeAlphaMode::PreMultiplied, wgpu::CompositeAlphaMode::PostMultiplied]
            .into_iter()
            .find(|m| caps.alpha_modes.contains(m))
            .unwrap_or(caps.alpha_modes[0]);
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);
        eprintln!(
            "[overlay] {} ({:?}), {}x{} {:?}",
            adapter.get_info().name,
            adapter.get_info().backend,
            config.width,
            config.height,
            config.format
        );
        let depth = depth_texture(&device, config.width, config.height);
        let scene = Scene::new(&device, config.format, font);
        Ok(Gfx {
            surface,
            device,
            queue,
            config,
            depth,
            scene,
        })
    }

    fn on_key(&mut self, ev: &winit::event::KeyEvent) {
        if !self.chat.open || ev.state != ElementState::Pressed {
            return;
        }
        match &ev.logical_key {
            Key::Named(NamedKey::Enter) if self.chat.preedit.is_empty() => self.submit_chat(),
            Key::Named(NamedKey::Escape) => self.set_chat(false),
            Key::Named(NamedKey::Backspace) if self.chat.preedit.is_empty() => {
                self.chat.text.pop();
            }
            _ => {
                if let Some(t) = &ev.text {
                    if self.chat.preedit.is_empty() {
                        self.chat.text.extend(t.chars().filter(|c| !c.is_control()));
                    }
                }
            }
        }
    }
}

impl ApplicationHandler<UserEvent> for Overlay {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Raon Zena")
            .with_inner_size(LogicalSize::new(VIEWPORT_W, VIEWPORT_H))
            .with_transparent(true)
            .with_decorations(false)
            .with_resizable(false)
            .with_window_level(WindowLevel::AlwaysOnTop);
        #[cfg(windows)]
        let attrs = {
            use winit::platform::windows::WindowAttributesExtWindows;
            attrs.with_skip_taskbar(true)
        };
        let window = Arc::new(el.create_window(attrs).expect("create window"));
        match Self::init_gfx(window.clone(), self.font.as_deref()) {
            Ok(g) => self.gfx = Some(g),
            Err(e) => eprintln!("GPU init failed: {e:#}"),
        }
        self.window = Some(window);
        self.platform_style();
    }

    fn user_event(&mut self, _el: &ActiveEventLoop, ev: UserEvent) {
        match ev {
            UserEvent::Cmd(AvatarCmd::Gaze(g)) => {
                self.anim.gaze = Some(Polar {
                    r: g.distance_norm,
                    theta_deg: g.angle_deg,
                })
            }
            UserEvent::Cmd(AvatarCmd::Act {
                args,
                level,
                visemes,
                speech_ms,
            }) => self.act(&args, level, visemes, speech_ms),
            UserEvent::Cmd(AvatarCmd::Expression { args, level }) => {
                // A new streamed answer: its first sentence opens a new bubble.
                if !self.speaking() {
                    self.bubble.fresh = true;
                }
                self.express(&args, level)
            }
            UserEvent::Cmd(AvatarCmd::Speak {
                text,
                level,
                visemes,
                speech_ms,
            }) => self.speak(&text, level, visemes, speech_ms),
            UserEvent::Cmd(AvatarCmd::System(c, m)) => self.system(c, m),
            UserEvent::Cmd(AvatarCmd::Asset(a)) => self.asset(&a),
            UserEvent::Hotkey(Hotkey::Chat) => {
                let open = !self.chat.open;
                self.set_chat(open);
            }
            UserEvent::Hotkey(h) => {
                let (c, m) = h.command(&self.state);
                self.system(c, m);
            }
            UserEvent::Asset(a) => self.pending.push(*a),
            UserEvent::AssetFailed(m) => {
                eprintln!("[assets] {m:?} unavailable");
                if let Some(g) = &mut self.gfx {
                    g.scene.failed.insert(m);
                }
            }
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, ev: WindowEvent) {
        match ev {
            WindowEvent::CloseRequested => self.system(SystemCommand::ExitApp, None),
            WindowEvent::Resized(s) => {
                if let Some(g) = &mut self.gfx {
                    g.config.width = s.width.max(1);
                    g.config.height = s.height.max(1);
                    g.surface.configure(&g.device, &g.config);
                    g.depth = depth_texture(&g.device, g.config.width, g.config.height);
                }
            }
            // 누르기 vs 끌기: 누른 채 5 논리 px 를 넘으면 OS 끌기, 그 전에 떼면 누르기(채팅 열기)
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if self.state.mode == WindowLayerMode::InteractivePet {
                    let scale = self.window.as_ref().map_or(1.0, |w| w.scale_factor());
                    let at = self.cursor.map_or((0.0, 0.0), |c| (c.x / scale, c.y / scale));
                    self.press.down(at);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = Some(position);
                let scale = self.window.as_ref().map_or(1.0, |w| w.scale_factor());
                if self.press.moved((position.x / scale, position.y / scale)) == Some(premium_discount_window::PointerAction::Drag) {
                    self.dragged = true;
                    if let Some(w) = &self.window {
                        let _ = w.drag_window();
                    }
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                if self.press.up() == Some(premium_discount_window::PointerAction::Click) {
                    let open = !self.chat.open;
                    self.set_chat(open);
                }
            }
            WindowEvent::CursorLeft { .. } => {
                // OS 끌기가 시작되면 뗌은 OS 가 가져간다: 눌림을 풀어 둔다
                let _ = self.press.up();
                self.cursor = None;
            }
            WindowEvent::Moved(_) if self.dragged && !self.dock.moving() => self.note_window_below(),
            WindowEvent::KeyboardInput { event, .. } => self.on_key(&event),
            WindowEvent::Ime(Ime::Preedit(text, _)) if self.chat.open => self.chat.preedit = text,
            WindowEvent::Ime(Ime::Commit(text)) if self.chat.open => {
                self.chat.preedit.clear();
                self.chat.text.push_str(&text);
            }
            WindowEvent::Focused(false) if self.chat.open => self.set_chat(false),
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
        if self.chat.open {
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
        if self.shutdown == ShutdownPhase::Exit {
            el.exit();
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        // Shutdown sequence: farewell → stop capture → release GPU/mmaps → exit.
        let next = self.shutdown.step(self.now_ms());
        if next != self.shutdown {
            match next {
                ShutdownPhase::StopCapture => self.capture_running.store(false, Ordering::SeqCst),
                ShutdownPhase::ReleaseResources => {
                    self.gfx = None;
                }
                ShutdownPhase::Exit => {
                    el.exit();
                    return;
                }
                _ => {}
            }
            self.shutdown = next;
        }
        let animating = self.emote_show().is_some() || self.bubble_alpha() > 0.0;
        let level = if self.speaking() || self.chat.open {
            ReactionLevel::Lv3Coaching
        } else if animating {
            ReactionLevel::Lv2Cached
        } else {
            self.anim.level.min(ReactionLevel::Lv2Cached)
        };
        match frame_interval(level, self.speaking(), self.state.visible) {
            Some(dt) => {
                el.set_control_flow(ControlFlow::WaitUntil(Instant::now() + dt));
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            None => el.set_control_flow(ControlFlow::Wait),
        }
        if self.shutdown != ShutdownPhase::Running {
            el.set_control_flow(ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(16)));
        }
        // 제자리 복귀 (Auto-Dock): 조용해진 지 2.5초가 지나면 오른쪽 아래로 0.45초 동안
        if let (Some(home), Some(at)) = (self.home(), self.position()) {
            let busy = self.busy();
            if busy {
                self.dragged = self.dragged && self.press.holding();
            }
            if let Some(p) = self.dock.tick(Instant::now(), at, home, busy) {
                self.dragged = false;
                if let Some(w) = &self.window {
                    w.set_outer_position(PhysicalPosition::new(p.x, p.y));
                }
                el.set_control_flow(ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(16)));
            } else if let Some(t) = self.dock.wake_at().filter(|_| at != home) {
                if !matches!(el.control_flow(), ControlFlow::WaitUntil(d) if d <= t) {
                    el.set_control_flow(ControlFlow::WaitUntil(t));
                }
            }
        }
        if !animating && !self.speaking() {
            self.anim.level = ReactionLevel::Lv1Silent;
        }
    }
}

/// Forwards what the asset service produced to the UI thread.
pub struct OverlayAssets(pub winit::event_loop::EventLoopProxy<UserEvent>);

impl AssetLink for OverlayAssets {
    fn ready(&self, l: Loaded) {
        let _ = self.0.send_event(UserEvent::Asset(Box::new(l)));
    }
    fn failed(&self, m: Missing) {
        let _ = self.0.send_event(UserEvent::AssetFailed(m));
    }
}
