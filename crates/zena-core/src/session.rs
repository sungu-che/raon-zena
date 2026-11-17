//! Play-session state for the character chat: what the screen has shown and for how long, win / loss streaks,
//! how close Zena and the player are, how tense the moment is — summarised into [`SessionContext`] for Lv3
//! prompts — and when Zena should speak up on her own ([`ProactiveTimer`]).

use std::collections::VecDeque;

use crate::prompt::{CourseBrief, SessionContext};

/// How many recent turns keep their topic keys (Relay Keys) active: the same window as the dialogue history
pub const ACTIVE_KEY_TURNS: usize = 4;

/// Collects session signals; everything decays or resets on its own, nothing is stored.
#[derive(Debug, Clone)]
pub struct SessionTracker {
    scene: Option<(String, String, u64)>,
    defeat_streak: u32,
    victory_streak: u32,
    affinity: f32,
    tension: f32,
    tension_ms: u64,
    /// Tension halves every this many ms.
    pub tension_half_life_ms: u64,
    /// Topic keys (premium-discount-relay Relay Keys) of the last few turns, newest last. A turn with no key
    /// still takes a slot, so a topic fades after [`ACTIVE_KEY_TURNS`] turns without it
    turn_keys: VecDeque<Vec<String>>,
    /// The current course and checkpoint (premium-discount time-trial), set by the app
    course: Option<CourseBrief>,
}

impl Default for SessionTracker {
    fn default() -> Self {
        Self {
            scene: None,
            defeat_streak: 0,
            victory_streak: 0,
            affinity: 0.3,
            tension: 0.0,
            tension_ms: 0,
            tension_half_life_ms: 20_000,
            turn_keys: VecDeque::new(),
            course: None,
        }
    }
}

impl SessionTracker {
    /// The screen shows scene `key` (described as `semantic`); the clock restarts only when the scene changes.
    pub fn scene(&mut self, key: &str, semantic: &str, now_ms: u64) {
        if self.scene.as_ref().is_none_or(|(k, _, _)| k != key) {
            self.scene = Some((key.to_string(), semantic.to_string(), now_ms));
        }
    }

    pub fn scene_key(&self) -> Option<&str> {
        self.scene.as_ref().map(|(k, _, _)| k.as_str())
    }

    pub fn scene_ms(&self, now_ms: u64) -> u64 {
        self.scene.as_ref().map(|(_, _, t)| now_ms.saturating_sub(*t)).unwrap_or(0)
    }

    pub fn defeat(&mut self, now_ms: u64) {
        self.defeat_streak += 1;
        self.victory_streak = 0;
        self.bump_tension(0.25, now_ms);
    }

    pub fn victory(&mut self, now_ms: u64) {
        self.victory_streak += 1;
        self.defeat_streak = 0;
        self.decay(now_ms);
        self.tension *= 0.5;
    }

    /// Praise, thanks (+) or a scolding (−); a conversation turn adds a little.
    pub fn affinity(&mut self, delta: f32) {
        self.affinity = (self.affinity + delta).clamp(0.0, 1.0);
    }

    /// A fight or a threat close to the player, weighted by its energy 0..1.
    pub fn combat(&mut self, energy: f32, now_ms: u64) {
        self.bump_tension(0.08 * energy.clamp(0.0, 1.0), now_ms);
    }

    fn decay(&mut self, now_ms: u64) {
        let dt = now_ms.saturating_sub(self.tension_ms) as f32;
        self.tension *= 0.5f32.powf(dt / self.tension_half_life_ms.max(1) as f32);
        self.tension_ms = now_ms;
    }

    fn bump_tension(&mut self, by: f32, now_ms: u64) {
        self.decay(now_ms);
        self.tension = (self.tension + by).min(1.0);
    }

    /// One conversation turn's topic keys (may be empty). Keeps the last [`ACTIVE_KEY_TURNS`] turns
    pub fn push_turn_keys(&mut self, keys: Vec<String>) {
        self.turn_keys.push_back(keys);
        while self.turn_keys.len() > ACTIVE_KEY_TURNS {
            self.turn_keys.pop_front();
        }
    }

    /// The active topic keys, newest turn first, each once. They restore the subject a follow-up question leaves
    /// out ("그럼 2페이즈는?" after a question about Malenia)
    pub fn active_keys(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for turn in self.turn_keys.iter().rev() {
            for k in turn {
                if !out.contains(k) {
                    out.push(k.clone());
                }
            }
        }
        out
    }

    pub fn set_course(&mut self, course: Option<CourseBrief>) {
        self.course = course;
    }

    pub fn course(&self) -> Option<&CourseBrief> {
        self.course.as_ref()
    }

    pub fn snapshot(&mut self, now_ms: u64) -> SessionContext {
        self.decay(now_ms);
        SessionContext {
            verdicts: Vec::new(),
            scene: self.scene.as_ref().map(|(_, s, _)| s.clone()),
            scene_secs: self.scene_ms(now_ms) / 1000,
            defeat_streak: self.defeat_streak,
            victory_streak: self.victory_streak,
            affinity: self.affinity,
            tension: self.tension,
            course: self.course.clone(),
        }
    }
}

/// What kind of unprompted line is due.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Proactive {
    /// The player and Zena have both been quiet for a while: a greeting / comment on the scene.
    Chatter,
    /// The same scene for a long time: suggest a break / stretching.
    Break,
}

/// Decides when Zena speaks up on her own. Activity (the player talking, Zena speaking, a reaction) resets the
/// silence; lines are spaced by `cooldown_ms` and a break is suggested at most once per `break_every_ms`.
#[derive(Debug, Clone)]
pub struct ProactiveTimer {
    pub silence_ms: u64,
    pub cooldown_ms: u64,
    pub break_after_ms: u64,
    pub break_every_ms: u64,
    last_activity: u64,
    last_line: Option<u64>,
    last_break: Option<u64>,
}

impl Default for ProactiveTimer {
    fn default() -> Self {
        Self::new(90_000)
    }
}

impl ProactiveTimer {
    /// `silence_ms` of quiet before Zena talks (0 disables proactive lines).
    pub fn new(silence_ms: u64) -> Self {
        Self {
            silence_ms,
            cooldown_ms: silence_ms.max(1) * 2,
            break_after_ms: 50 * 60_000,
            break_every_ms: 50 * 60_000,
            last_activity: 0,
            last_line: None,
            last_break: None,
        }
    }

    pub fn touch(&mut self, now_ms: u64) {
        self.last_activity = self.last_activity.max(now_ms);
    }

    /// `scene_ms`: how long the current scene has been on screen. `quiet`: Zena must not talk (private screen).
    pub fn due(&mut self, now_ms: u64, scene_ms: u64, quiet: bool) -> Option<Proactive> {
        if self.silence_ms == 0 || quiet || now_ms.saturating_sub(self.last_activity) < self.silence_ms {
            return None;
        }
        if self.last_line.is_some_and(|t| now_ms.saturating_sub(t) < self.cooldown_ms) {
            return None;
        }
        let kind = if scene_ms >= self.break_after_ms && self.last_break.is_none_or(|t| now_ms.saturating_sub(t) >= self.break_every_ms) {
            self.last_break = Some(now_ms);
            Proactive::Break
        } else {
            Proactive::Chatter
        };
        self.last_line = Some(now_ms);
        self.last_activity = now_ms;
        Some(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_keys_fade_after_four_turns_and_the_course_reaches_the_snapshot() {
        let mut s = SessionTracker::default();
        s.push_turn_keys(vec!["말레니아".into()]);
        s.push_turn_keys(vec![]);
        s.push_turn_keys(vec!["2페이즈".into(), "말레니아".into()]);
        assert_eq!(s.active_keys(), ["2페이즈", "말레니아"], "newest turn first, once each");
        s.push_turn_keys(vec![]);
        s.push_turn_keys(vec![]);
        s.push_turn_keys(vec![]);
        assert_eq!(s.active_keys(), ["2페이즈", "말레니아"], "still within four turns");
        s.push_turn_keys(vec![]);
        assert!(s.active_keys().is_empty(), "four turns without the topic");

        assert!(s.snapshot(0).course.is_none());
        s.set_course(Some(CourseBrief { checkpoint_label: "보스방".into(), ..Default::default() }));
        assert_eq!(s.snapshot(0).course.unwrap().checkpoint_label, "보스방");
        s.set_course(None);
        assert!(s.course().is_none());
    }

    #[test]
    fn streaks_scene_clock_and_tension() {
        let mut s = SessionTracker::default();
        s.scene("vision.game_combat", "게임 전투", 1_000);
        s.scene("vision.game_combat", "게임 전투", 50_000);
        assert_eq!(s.snapshot(61_000).scene_secs, 60, "same scene keeps its clock");
        s.defeat(61_000);
        s.defeat(62_000);
        s.defeat(63_000);
        let c = s.snapshot(63_000);
        assert_eq!((c.defeat_streak, c.victory_streak), (3, 0));
        assert!(c.tension > 0.6, "{}", c.tension);
        assert!(s.snapshot(63_000 + 60_000).tension < 0.1, "tension decays");
        s.victory(200_000);
        assert_eq!(s.snapshot(200_000).defeat_streak, 0);
        s.affinity(2.0);
        assert_eq!(s.snapshot(200_000).affinity, 1.0);
        s.scene("vision.coding", "코드 편집기", 300_000);
        assert_eq!(s.snapshot(300_000).scene_secs, 0, "a new scene restarts the clock");
    }

    #[test]
    fn proactive_lines_wait_for_silence_and_space_out() {
        let mut p = ProactiveTimer::new(60_000);
        assert_eq!(p.due(30_000, 0, false), None);
        assert_eq!(p.due(61_000, 0, false), Some(Proactive::Chatter));
        assert_eq!(p.due(125_000, 0, false), None, "cooldown after a line");
        p.touch(150_000);
        assert_eq!(p.due(200_000, 0, false), None, "the player spoke");
        assert_eq!(p.due(300_000, 0, true), None, "a private screen keeps Zena quiet");
        assert_eq!(p.due(300_000, 60 * 60_000, false), Some(Proactive::Break));
        assert_eq!(p.due(500_000, 70 * 60_000, false), Some(Proactive::Chatter), "one break per period");
        assert_eq!(ProactiveTimer::new(0).due(10_000_000, 0, false), None, "disabled");
    }
}
