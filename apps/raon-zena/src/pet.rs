//! 오버레이 캐릭터 다루기 — premium-discount-window 의 규칙을 그대로 쓴다.
//!
//! - **누르기 vs 끌기** ([`Press`]): 누른 점에서 [`DRAG_THRESHOLD`](5 논리 px) 이상 움직이면 그때 OS 끌기를 시작하고, 그 전에 손을
//!   떼면 누르기다(채팅을 연다). 예전에는 누르는 즉시 끌기라 누르기가 없었다.
//! - **제자리 복귀 (Auto-Dock)** ([`Dock`]): 말 · 끌기 · 채팅이 끝나고 [`DOCK_IDLE`] 동안 조용하면 작업 영역 오른쪽 아래
//!   ([`premium_discount_window::dock_position`])로 [`DOCK_DURATION`](0.45초, cubic ease-out) 동안 돌아간다.
//! - **쉬는 동안 불투명도** ([`resting_opacity`]): 제자리에서 쉬면 [`PILL_OPACITY`](82 %), 말하거나 다루는 동안은 100 %.

use std::time::{Duration, Instant};

use premium_discount_window::{classify_pointer, dock_step, PointerAction, Position, DOCK_DURATION, DRAG_THRESHOLD, PILL_OPACITY};

/// 조용해진 뒤 제자리로 돌아가기까지
pub const DOCK_IDLE: Duration = Duration::from_millis(2_500);

/// 왼쪽 버튼 하나의 누르기 · 끌기 판정 (좌표는 창 안의 논리 픽셀)
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Press {
    origin: Option<(f64, f64)>,
    dragging: bool,
}

impl Press {
    pub fn down(&mut self, at: (f64, f64)) {
        self.origin = Some(at);
        self.dragging = false;
    }

    /// 누른 채 움직임. 문턱을 처음 넘는 순간 `Some(Drag)` — 그때 OS 끌기를 시작한다(그 뒤의 뗌은 OS 가 가져간다)
    pub fn moved(&mut self, at: (f64, f64)) -> Option<PointerAction> {
        let o = self.origin?;
        if classify_pointer(at.0 - o.0, at.1 - o.1, DRAG_THRESHOLD) == PointerAction::Drag {
            self.origin = None;
            self.dragging = true;
            return Some(PointerAction::Drag);
        }
        None
    }

    /// 뗌: 끌기 없이 떼면 `Some(Click)`
    pub fn up(&mut self) -> Option<PointerAction> {
        let pressed = self.origin.take().is_some();
        let dragged = std::mem::take(&mut self.dragging);
        (pressed && !dragged).then_some(PointerAction::Click)
    }

    /// 누르고 있는 중 (아직 누르기인지 끌기인지 모름)
    pub fn holding(&self) -> bool {
        self.origin.is_some()
    }
}

/// 제자리 복귀
#[derive(Debug, Default, Clone)]
pub struct Dock {
    idle_since: Option<Instant>,
    anim: Option<(Position, Position, Instant)>,
}

impl Dock {
    /// 한 프레임. `busy`(말 · 끌기 · 채팅)면 멈추고 기다림을 새로 시작한다. 조용해진 지 [`DOCK_IDLE`] 이 지났는데 `current` 가
    /// `home` 이 아니면 돌아가기 시작한다. 이번 프레임에 창을 옮길 위치를 돌려준다
    pub fn tick(&mut self, now: Instant, current: Position, home: Position, busy: bool) -> Option<Position> {
        if busy {
            self.idle_since = None;
            self.anim = None;
            return None;
        }
        let since = *self.idle_since.get_or_insert(now);
        if self.anim.is_none() {
            if current == home || now.saturating_duration_since(since) < DOCK_IDLE {
                return None;
            }
            self.anim = Some((current, home, now));
        }
        let (from, to, start) = self.anim?;
        let (pos, done) = dock_step(from, to, now.saturating_duration_since(start), DOCK_DURATION);
        if done {
            self.anim = None;
        }
        Some(pos)
    }

    /// 돌아가는 중
    pub fn moving(&self) -> bool {
        self.anim.is_some()
    }

    /// 다음에 깨어날 때 (돌아가기를 시작할 시각). 기다리는 중이 아니면 None
    pub fn wake_at(&self) -> Option<Instant> {
        if self.anim.is_some() {
            return None;
        }
        self.idle_since.map(|t| t + DOCK_IDLE)
    }
}

/// 제자리에서 쉬는 동안의 불투명도 (알약 단계와 같은 82 %), 다루거나 말하는 동안 1
pub fn resting_opacity(docked: bool, busy: bool) -> f32 {
    if docked && !busy {
        PILL_OPACITY as f32
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_press_is_a_click_and_five_pixels_start_a_drag() {
        let mut p = Press::default();
        p.down((100.0, 100.0));
        assert_eq!(p.moved((103.0, 103.0)), None, "4.2 px");
        assert_eq!(p.up(), Some(PointerAction::Click));
        p.down((100.0, 100.0));
        assert_eq!(p.moved((103.0, 104.0)), Some(PointerAction::Drag), "exactly 5 px");
        assert_eq!(p.moved((150.0, 150.0)), None, "the drag starts once");
        assert_eq!(p.up(), None, "a drag is not a click");
        assert_eq!(p.up(), None, "nothing pressed");
        assert!(!p.holding());
    }

    #[test]
    fn the_pet_waits_then_glides_home_and_stops_when_busy() {
        let t0 = Instant::now();
        let (away, home) = (Position::new(100, 100), Position::new(1500, 800));
        let mut d = Dock::default();
        assert_eq!(d.tick(t0, away, home, false), None, "just went quiet");
        assert_eq!(d.wake_at(), Some(t0 + DOCK_IDLE));
        let start = t0 + DOCK_IDLE;
        let first = d.tick(start, away, home, false).unwrap();
        assert_eq!(first, away, "the glide starts where the window is");
        assert!(d.moving());
        let mid = d.tick(start + DOCK_DURATION / 2, away, home, false).unwrap();
        assert!(mid.x > 100 + (1500 - 100) / 2, "ease-out: past half way at half time ({mid:?})");
        assert_eq!(d.tick(start + DOCK_DURATION, mid, home, false), Some(home));
        assert!(!d.moving());
        assert_eq!(d.tick(start + DOCK_DURATION * 2, home, home, false), None, "home already");
        // speaking again stops the wait
        let mut d = Dock::default();
        d.tick(t0, away, home, false);
        assert_eq!(d.tick(t0 + DOCK_IDLE, away, home, true), None);
        assert_eq!(d.tick(t0 + DOCK_IDLE * 2, away, home, false), None, "the wait starts over");
    }

    #[test]
    fn resting_at_home_is_82_percent() {
        assert_eq!(resting_opacity(true, false), 0.82);
        assert_eq!(resting_opacity(true, true), 1.0);
        assert_eq!(resting_opacity(false, false), 1.0);
    }
}
