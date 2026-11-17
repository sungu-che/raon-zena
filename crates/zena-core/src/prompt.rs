//! ChatML prompt builder for Raon-1-nano's non-thinking fast path.
//!
//! The released chat template opens the assistant turn with `<think>\n` unless
//! `enable_thinking=false`, in which case it emits an empty `<think></think>\n\n` block.
//! Pre-filling that empty block skips reasoning tokens entirely (Lv3 fast path).
//!
//! Multi-turn memory and the session summary go into the **user** turn ([`Situation::history`],
//! [`Situation::session`]), never into the system prompt: the backend prefills the system + tools prefix once and
//! clones its state per request, so that prefix must stay byte-identical for every request.

use crate::geometry::{FlowDirection, Polar};
use crate::tools::tool_definitions;
use std::collections::VecDeque;

/// One exchange remembered for the next prompts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DialogueTurn {
    /// What the player said (`None`: Zena spoke on her own).
    pub user: Option<String>,
    pub zena: String,
}

/// Sliding window of the last few exchanges.
#[derive(Debug, Clone)]
pub struct Dialogue {
    turns: VecDeque<DialogueTurn>,
    pub capacity: usize,
    /// Each remembered utterance is cut to this many characters.
    pub max_chars: usize,
}

impl Default for Dialogue {
    fn default() -> Self {
        Self::new(4)
    }
}

fn clip(s: &str, n: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

impl Dialogue {
    pub fn new(capacity: usize) -> Self {
        Self {
            turns: VecDeque::new(),
            capacity,
            max_chars: 60,
        }
    }

    pub fn push(&mut self, user: Option<&str>, zena: &str) {
        if zena.trim().is_empty() && user.is_none_or(|u| u.trim().is_empty()) {
            return;
        }
        self.turns.push_back(DialogueTurn {
            user: user.map(|u| clip(u, self.max_chars)).filter(|u| !u.is_empty()),
            zena: clip(zena, self.max_chars),
        });
        while self.turns.len() > self.capacity {
            self.turns.pop_front();
        }
    }

    pub fn turns(&self) -> Vec<DialogueTurn> {
        self.turns.iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.turns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    pub fn clear(&mut self) {
        self.turns.clear();
    }
}

/// Where the player is on their current course (premium-discount's time-trial: checkpoints, ghost, friction),
/// language-neutral: the language pack (`zena_locale`) turns it into the `[코스]` line.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CourseBrief {
    /// Course chain hash up to the current checkpoint (premium-discount-nav `Step::course`)
    pub course_id: String,
    /// The current checkpoint's display name (already scrubbed of personal data)
    pub checkpoint_label: String,
    /// Split time against the ghost (my best run or the public one): negative = ahead, positive = behind
    pub ghost_delta_ms: Option<i64>,
    /// The checkpoint `pd_nav_number` puts first next, with the local and the global tier
    pub next_recommended: Option<NextCheckpoint>,
    /// This section's friction is high (premium-discount-nav `FrictionBaselines::decide` gave discount)
    pub high_friction: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NextCheckpoint {
    pub label: String,
    pub local: Option<zena_verdict::Tier>,
    pub global: Option<zena_verdict::Tier>,
}

impl CourseBrief {
    /// The `[코스]` line's content (no block name), in the current language pack
    pub fn render(&self) -> String {
        self.render_in(&zena_locale::current())
    }

    pub fn render_in(&self, l: &zena_locale::Locale) -> String {
        let fill = zena_locale::fill;
        let mut parts = Vec::new();
        if !self.checkpoint_label.is_empty() {
            parts.push(fill(l.prompt("course_checkpoint"), &[("label", &self.checkpoint_label)]));
        }
        if let Some(d) = self.ghost_delta_ms {
            let secs = format!("{:.1}", d.unsigned_abs() as f64 / 1000.0);
            parts.push(match d {
                d if d.abs() < 50 => l.prompt("course_even").to_string(),
                d if d < 0 => fill(l.prompt("course_ahead"), &[("secs", &secs)]),
                _ => fill(l.prompt("course_behind"), &[("secs", &secs)]),
            });
        }
        if let Some(n) = &self.next_recommended {
            let word = |t: zena_verdict::Tier| match t {
                zena_verdict::Tier::Premium => l.speak("premium").to_string(),
                zena_verdict::Tier::Discount => l.speak("discount").to_string(),
                zena_verdict::Tier::Neutral => l.speak("neutral").to_string(),
                zena_verdict::Tier::Unrated => String::new(),
            };
            let stance = match zena_verdict::stance(n.local, n.global) {
                zena_verdict::Stance::Agree(t) | zena_verdict::Stance::LocalOnly(t) | zena_verdict::Stance::GlobalOnly(t) => word(t),
                zena_verdict::Stance::Differ { local, global } => format!("{}/{}", word(local), word(global)),
                zena_verdict::Stance::FactsOnly => String::new(),
            };
            parts.push(if stance.is_empty() {
                fill(l.prompt("course_next"), &[("label", &n.label)])
            } else {
                fill(l.prompt("course_next_stance"), &[("label", &n.label), ("stance", &stance)])
            });
        }
        if self.high_friction {
            parts.push(l.prompt("course_friction").to_string());
        }
        parts.join(l.prompt("join"))
    }
}

/// What Zena knows about the play session, rendered as one compact block so her tone follows it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionContext {
    /// What the screen shows (the scene's description) and for how long.
    pub scene: Option<String>,
    pub scene_secs: u64,
    /// Defeats / victories in a row (screen events and what the player said).
    pub defeat_streak: u32,
    pub victory_streak: u32,
    /// 0 (stranger) .. 1 (close): praise, thanks and conversation raise it.
    pub affinity: f32,
    /// 0 (calm) .. 1 (frantic): combat and defeats raise it, it decays over time.
    pub tension: f32,
    /// Recommendations for what is on screen: one line per record (at most 3), each saying the local and the
    /// global verdict side by side (`zena_verdict::Stance::line`). Empty: nothing to recommend.
    pub verdicts: Vec<String>,
    /// The current course and checkpoint (premium-discount time-trial), when Zena knows one
    pub course: Option<CourseBrief>,
}

impl SessionContext {
    pub fn render(&self) -> String {
        self.render_in(&zena_locale::current())
    }

    /// [`Self::render`] in a given language pack
    pub fn render_in(&self, l: &zena_locale::Locale) -> String {
        let fill = zena_locale::fill;
        let mut parts = Vec::new();
        if let Some(s) = &self.scene {
            let m = self.scene_secs / 60;
            parts.push(if m > 0 {
                fill(l.prompt("scene_minutes"), &[("scene", s), ("minutes", &m.to_string())])
            } else {
                fill(l.prompt("scene"), &[("scene", s)])
            });
        }
        if self.defeat_streak > 0 {
            parts.push(fill(l.prompt("defeat_streak"), &[("n", &self.defeat_streak.to_string())]));
        }
        if self.victory_streak > 0 {
            parts.push(fill(l.prompt("victory_streak"), &[("n", &self.victory_streak.to_string())]));
        }
        parts.push(
            match self.affinity {
                a if a >= 0.66 => l.prompt("affinity_high"),
                a if a >= 0.33 => l.prompt("affinity_mid"),
                _ => l.prompt("affinity_low"),
            }
            .to_string(),
        );
        if self.tension >= 0.5 {
            parts.push(l.prompt("tension_high").to_string());
        }
        let mut s = format!("{} {}\n", l.prompt("block_session"), parts.join(l.prompt("join")));
        if let Some(t) = self.tone_in(l) {
            s.push_str(&format!("{} {t}\n", l.prompt("block_tone")));
        }
        for v in self.verdicts.iter().take(3) {
            s.push_str(&format!("{} {v}\n", l.prompt("block_recommend")));
        }
        if let Some(c) = &self.course {
            let line = c.render_in(l);
            if !line.is_empty() {
                s.push_str(&format!("{} {line}\n", l.prompt("block_course")));
            }
        }
        s
    }

    /// The tone hint the summary implies (current language pack).
    pub fn tone(&self) -> Option<String> {
        self.tone_in(&zena_locale::current())
    }

    pub fn tone_in(&self, l: &zena_locale::Locale) -> Option<String> {
        let key = if self.defeat_streak >= 3 {
            "tone_defeats"
        } else if self.tension >= 0.7 {
            "tone_tense"
        } else if self.victory_streak >= 2 {
            "tone_streak"
        } else if self.affinity >= 0.66 {
            "tone_close"
        } else {
            return None;
        };
        Some(l.prompt(key).to_string())
    }
}

/// Zena's system persona in the current language pack. It goes into the cached system prefix, so a backend that
/// cached the prefix must rebuild it after the language changes.
pub fn system_persona() -> String {
    zena_locale::current().persona.clone()
}

/// Compact, model-readable description of the situation for one Lv3 call.
#[derive(Debug, Clone, Default)]
pub struct Situation {
    pub event_label: Option<String>,
    pub polar: Option<Polar>,
    pub flow: Option<FlowDirection>,
    pub hp_ratio: Option<f32>,
    pub user_query: Option<String>,
    /// Retrieved memory / game-guide snippets (already NMS-filtered).
    pub context: Vec<String>,
    /// The last few exchanges, oldest first.
    pub history: Vec<DialogueTurn>,
    pub session: Option<SessionContext>,
}

/// Direction name of an angle in the current language pack (eight 45° sectors, 0 = right, counter-clockwise)
pub fn direction(theta: f32) -> String {
    zena_locale::current().direction(theta).to_string()
}

impl Situation {
    pub fn render(&self) -> String {
        self.render_in(&zena_locale::current())
    }

    /// [`Self::render`] in a given language pack
    pub fn render_in(&self, l: &zena_locale::Locale) -> String {
        let mut s = String::new();
        if !self.history.is_empty() {
            s.push_str(&format!("{}\n", l.prompt("block_history")));
            for t in &self.history {
                if let Some(u) = &t.user {
                    s.push_str(&format!("{}: {u}\n", l.prompt("speaker_player")));
                }
                s.push_str(&format!("{}: {}\n", l.prompt("speaker_zena"), t.zena));
            }
        }
        if let Some(c) = &self.session {
            s.push_str(&c.render_in(l));
        }
        if let Some(q) = &self.user_query {
            s.push_str(&format!("{} {q}\n", l.prompt("block_question")));
        }
        if let Some(e) = &self.event_label {
            s.push_str(&format!("{} {e}", l.prompt("block_event")));
            if let Some(p) = self.polar {
                let dist = if p.r < 0.22 {
                    l.prompt("dist_near")
                } else if p.r < 0.5 {
                    l.prompt("dist_mid")
                } else {
                    l.prompt("dist_far")
                };
                s.push_str(&format!(
                    " / {} {} (angle_deg={:.0}, distance_norm={:.2})",
                    l.direction(p.theta_deg),
                    dist,
                    p.theta_deg,
                    p.r
                ));
            }
            if let Some(f) = self.flow {
                s.push_str(match f {
                    FlowDirection::Inward => l.prompt("flow_inward"),
                    FlowDirection::Outward => l.prompt("flow_outward"),
                    _ => "",
                });
            }
            s.push('\n');
        }
        if let Some(hp) = self.hp_ratio {
            s.push_str(&format!("{} {:.0}%\n", l.prompt("block_hp"), hp * 100.0));
        }
        for c in &self.context {
            s.push_str(&format!("{} {c}\n", l.prompt("block_reference")));
        }
        s
    }
}

/// Full prompt string in Raon-1-nano's chat format, ending with the pre-filled empty think block.
pub fn build_fast_path_prompt(sit: &Situation) -> String {
    let mut p = String::from("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for t in tool_definitions().as_array().expect("array") {
        p.push('\n');
        p.push_str(&t.to_string());
    }
    p.push_str("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>\n\n");
    p.push_str(&system_persona());
    p.push_str("<|im_end|>\n<|im_start|>user\n");
    p.push_str(sit.render().trim_end());
    p.push_str("<|im_end|>\n<|im_start|>assistant\n<think></think>\n\n");
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prompt_has_fast_path_prefill_and_tools() {
        let s = Situation {
            event_label: Some("보스 등장".into()),
            polar: Some(Polar { r: 0.1, theta_deg: 10.0 }),
            flow: Some(FlowDirection::Inward),
            hp_ratio: Some(0.12),
            ..Default::default()
        };
        let p = build_fast_path_prompt(&s);
        assert!(p.ends_with("<|im_start|>assistant\n<think></think>\n\n"));
        assert!(p.contains("control_zena_avatar"));
        assert!(p.contains("오른쪽 바로 코앞"));
        assert!(p.contains("[HP] 12%"));
        assert_eq!(direction(350.0), "오른쪽");
        assert_eq!(direction(135.0), "왼쪽 위");
    }

    #[test]
    fn history_and_session_go_into_the_user_turn_so_the_cached_prefix_never_changes() {
        let user_turn = "<|im_start|>user\n";
        let bare = build_fast_path_prompt(&Situation::default());
        let mut d = Dialogue::new(2);
        d.push(Some("보스 패턴 알려줘"), "내려찍기 전에 옆으로 굴러!");
        d.push(Some("고마워"), "천만에!");
        d.push(None, "슬슬 스트레칭 한 번 할까?");
        assert_eq!(d.len(), 2, "the window keeps the last two");
        let s = Situation {
            user_query: Some("또 졌어".into()),
            history: d.turns(),
            session: Some(SessionContext {
                scene: Some("게임 전투".into()),
                scene_secs: 754,
                defeat_streak: 3,
                affinity: 0.7,
                ..Default::default()
            }),
            ..Default::default()
        };
        let p = build_fast_path_prompt(&s);
        let cut = |p: &str| p[..p.find(user_turn).unwrap()].to_string();
        assert_eq!(cut(&p), cut(&bare), "system + tools prefix is byte-identical");
        let user = &p[p.find(user_turn).unwrap()..];
        assert!(user.contains("[최근 대화]\n플레이어: 고마워\n제나: 천만에!\n제나: 슬슬 스트레칭 한 번 할까?\n"));
        assert!(!user.contains("보스 패턴"), "dropped from the window");
        assert!(user.contains("[세션] 화면: 게임 전투 (12분째) / 연속 패배 3회 / 친밀도 높음"));
        assert!(user.contains("[말투] 계속 지고 있다"));
        assert!(user.contains("[플레이어 질문] 또 졌어"));
    }

    #[test]
    fn dialogue_clips_long_lines() {
        let mut d = Dialogue::new(4);
        d.push(Some(&"가".repeat(100)), "응");
        assert_eq!(d.turns()[0].user.as_ref().unwrap().chars().count(), 61);
        d.push(None, "  ");
        assert_eq!(d.len(), 1, "nothing said, nothing remembered");
    }

    #[test]
    fn the_course_line_follows_the_recommendations() {
        let course = CourseBrief {
            course_id: "0xcourse".into(),
            checkpoint_label: "쿠폰 입력".into(),
            ghost_delta_ms: Some(-800),
            next_recommended: Some(NextCheckpoint { label: "간편결제".into(), local: Some(zena_verdict::Tier::Premium), global: Some(zena_verdict::Tier::Premium) }),
            high_friction: true,
        };
        let ctx = SessionContext { verdicts: vec!["공략: 추천".into()], course: Some(course.clone()), ..Default::default() };
        let s = ctx.render();
        assert!(
            s.ends_with("[추천] 공략: 추천\n[코스] 체크포인트 '쿠폰 입력' / 고스트보다 0.8초 빠름 / 다음 추천 '간편결제' (추천) / 이 구간에서 막혀 있다 (마찰 큼)\n"),
            "{s}"
        );
        let behind = CourseBrief { ghost_delta_ms: Some(1250), next_recommended: None, high_friction: false, ..course.clone() };
        assert_eq!(behind.render(), "체크포인트 '쿠폰 입력' / 고스트보다 1.2초 느림");
        let differ = CourseBrief {
            ghost_delta_ms: Some(20),
            next_recommended: Some(NextCheckpoint { label: "A".into(), local: Some(zena_verdict::Tier::Premium), global: Some(zena_verdict::Tier::Discount) }),
            ..course.clone()
        };
        assert!(differ.render().contains("고스트와 같은 속도 / 다음 추천 'A' (추천/비추천)"));
        let unrated = CourseBrief { next_recommended: Some(NextCheckpoint { label: "B".into(), local: None, global: Some(zena_verdict::Tier::Unrated) }), ..CourseBrief::default() };
        assert_eq!(unrated.render(), "다음 추천 'B'");
        assert!(!SessionContext::default().render().contains("[코스]"));
    }

    #[test]
    fn another_language_pack_changes_every_line_but_not_the_layout() {
        let en = zena_locale::get("en").unwrap();
        let s = Situation {
            event_label: Some("boss".into()),
            polar: Some(Polar { r: 0.1, theta_deg: 10.0 }),
            hp_ratio: Some(0.12),
            user_query: Some("how do I dodge?".into()),
            session: Some(SessionContext { scene: Some("boss fight".into()), scene_secs: 754, defeat_streak: 3, ..Default::default() }),
            ..Default::default()
        };
        let user = s.render_in(&en);
        assert!(user.starts_with("[session] Screen: boss fight (12 min) / 3 defeats in a row / new acquaintance\n[tone] They keep losing."), "{user}");
        assert!(user.contains("[player question] how do I dodge?\n[screen event] boss / right right in front"), "{user}");
        assert!(user.contains("[HP] 12%"));
        assert!(!user.chars().any(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c)), "no Korean left: {user}");
    }

    #[test]
    fn verdict_lines_follow_the_session_line_at_most_three() {
        let ctx = SessionContext {
            verdicts: (1..=4).map(|i| format!("공략 {i}: 내 기록과 서버 평가 모두 추천")).collect(),
            ..Default::default()
        };
        let s = ctx.render();
        assert!(s.contains("[추천] 공략 1: 내 기록과 서버 평가 모두 추천\n"));
        assert!(s.contains("[추천] 공략 3:"));
        assert!(!s.contains("공략 4"), "at most three");
        assert!(!SessionContext::default().render().contains("[추천]"));
    }
}
