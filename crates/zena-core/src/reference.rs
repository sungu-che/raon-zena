//! What a short question refers to (`contracts/contextual_rag_v1.json` CR-01..CR-04, scenario 3 part 1):
//! "여기서 주의할 거 있어?" means the checkpoint on screen, "그거 방패로 막아?" the weapon just talked about,
//! "지력 제한은?" continues the last topic — and after Alt-Tab the previous app's topics are out of play.
//!
//! [`TopicTracker`] keeps a decaying topic frame per window (`cc`) and turns a question into a
//! [`ContextualQueryPlan`]: the [`ReferenceKind`], the rewritten query (to be embedded once, as `Role::Query`)
//! and the (cc, to) scopes of a three-level backoff — exact checkpoint (+0.12), entity / screen (+0.06),
//! app-wide (+0). Two antecedents in a near tie are not guessed: the plan carries `ambiguous_pair` and a
//! "don't assert, ask back" guard line for `Situation::context` ([`ContextualQueryPlan::context_guard`]).
//!
//! Entity extraction (Stanza head-argument spans, noun chunks) stays upstream in nlp-nms-router /
//! premium-discount-nms; this module only sees the entity strings and the screen state.

use crate::nav_target::spoken_slot_mask;
use crate::text::{collapse_ws, josa};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Salience decay per user turn in the same window.
pub const TOPIC_TURN_DECAY: f32 = 0.65;
/// Salience half-life in ms.
pub const TOPIC_HALF_LIFE_MS: f32 = 120_000.0;
/// Below this salience an entity is not an antecedent.
pub const SALIENCE_FLOOR: f32 = 0.35;
/// Two antecedents closer than this are ambiguous.
pub const AMBIGUITY_MARGIN: f32 = 0.12;
/// Role weight of an entity the player named.
pub const USER_MENTION_WEIGHT: f32 = 1.0;
/// Role weight of an entity only Zena named (a proactive offer).
pub const ZENA_MENTION_WEIGHT: f32 = 0.85;
/// "이거" right after pointing at a badge or a window / screen change means the screen.
pub const DEICTIC_RECENT_MS: u64 = 3_000;
/// "이거" within this long of the last exchange (and no screen change) means the dialogue topic.
pub const DIALOGUE_CONTINUITY_MS: u64 = 15_000;
/// A bare predicate ("수수료 있어?") continues a topic mentioned within this long.
pub const ELLIPSIS_WINDOW_MS: u64 = 60_000;
/// Topics kept per window.
pub const MAX_TOPICS_PER_CC: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKind {
    /// Names its subject itself.
    SelfContained,
    /// Points at the screen: "여기", "이 버튼", "#1번", "here".
    DeicticScreen,
    /// Points back into the dialogue: "그거", "그 무기", "아까 말한 거", "it".
    AnaphoricDialogue,
    /// Leaves the subject out: "지력 제한은?", "수수료 있어?".
    EllipticalFollowUp,
}

/// How well a record's (cc, to) matches the question's scopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScopeMatchLevel {
    /// L1: the checkpoint / badge in focus (+0.12).
    ExactCheckpoint,
    /// L2: the restored topic entity or the current screen / page (+0.06).
    Entity,
    /// L3: the whole app / site (`to` = none or the zero address, +0).
    AppWide,
}

impl ScopeMatchLevel {
    pub fn boost(self) -> f32 {
        match self {
            ScopeMatchLevel::ExactCheckpoint => 0.12,
            ScopeMatchLevel::Entity => 0.06,
            ScopeMatchLevel::AppWide => 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ScopeTargets {
    /// Normalised window / site id ("eldenring.exe", "steampowered.com").
    pub cc: String,
    /// L1: the current checkpoint or the selected badge's hash.
    pub l1_checkpoint_to: Option<String>,
    /// L2: restored topic entity hashes, then the current screen / page hash.
    pub l2_entity_tos: Vec<String>,
}

fn is_zero_address(to: &str) -> bool {
    to.strip_prefix("0x").is_some_and(|h| !h.is_empty() && h.bytes().all(|c| c == b'0'))
}

impl ScopeTargets {
    /// The level of a record in window `cc` with address `to`; `None` when it is out of scope (another window,
    /// or another checkpoint / entity of this one).
    pub fn match_level(&self, cc: &str, to: Option<&str>) -> Option<ScopeMatchLevel> {
        if normalize_cc(cc) != self.cc {
            return None;
        }
        match to {
            None => Some(ScopeMatchLevel::AppWide),
            Some(t) if is_zero_address(t) => Some(ScopeMatchLevel::AppWide),
            Some(t) if self.l1_checkpoint_to.as_deref() == Some(t) => Some(ScopeMatchLevel::ExactCheckpoint),
            Some(t) if self.l2_entity_tos.iter().any(|x| x == t) => Some(ScopeMatchLevel::Entity),
            Some(_) => None,
        }
    }

    /// `score` plus the level's boost, or `None` when out of scope.
    pub fn boosted(&self, cc: &str, to: Option<&str>, score: f32) -> Option<(ScopeMatchLevel, f32)> {
        self.match_level(cc, to).map(|l| (l, score + l.boost()))
    }

    /// The backoff order: L1 (if any), each L2 address, then the app-wide scope.
    pub fn backoff(&self) -> Vec<(ScopeMatchLevel, Option<String>)> {
        let mut v = Vec::new();
        if let Some(t) = &self.l1_checkpoint_to {
            v.push((ScopeMatchLevel::ExactCheckpoint, Some(t.clone())));
        }
        v.extend(self.l2_entity_tos.iter().map(|t| (ScopeMatchLevel::Entity, Some(t.clone()))));
        v.push((ScopeMatchLevel::AppWide, None));
        v
    }
}

/// The resolved question: the input of the RAG coordinator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextualQueryPlan {
    pub raw_query: String,
    /// The query with its reference restored (embed this once, as `Role::Query`).
    pub rewritten_query: String,
    pub kind: ReferenceKind,
    pub scopes: ScopeTargets,
    /// The single antecedent / screen label restored into the query.
    pub resolved_entity: Option<String>,
    /// Two antecedents within [`AMBIGUITY_MARGIN`]: search both, ask back instead of asserting.
    pub ambiguous_pair: Option<(String, String)>,
}

impl ContextualQueryPlan {
    /// The one-line guard for `Situation::context` when the antecedent is ambiguous.
    pub fn context_guard(&self) -> Option<String> {
        self.ambiguous_pair
            .as_ref()
            .map(|(a, b)| format!("선행사 모호함('{a}' vs '{b}') — 단정하지 말고 어느 쪽인지 되묻거나 둘을 나란히 안내할 것"))
    }
}

/// The badge / region the player pointed at or hovered.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FocusedRegion {
    pub number: Option<u8>,
    /// Scrubbed label.
    pub label: String,
    /// Region / element hash.
    pub to: Option<String>,
    pub at_ms: u64,
}

/// The screen state the question is asked against.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ScreenAnchorSnapshot {
    /// The foreground window / site (normalised here: lowercase, no scheme, no `www.`).
    pub cc: String,
    /// Current screen / page hash (an L2 scope).
    pub screen_key: Option<String>,
    /// Scrubbed label of the screen or the current checkpoint ("보스방 입구").
    pub screen_label: Option<String>,
    /// Current checkpoint hash (L1).
    pub checkpoint_to: Option<String>,
    pub focused: Option<FocusedRegion>,
    /// When the window or the screen last changed.
    pub changed_at_ms: Option<u64>,
}

/// One remembered topic.
#[derive(Debug, Clone, PartialEq)]
pub struct TopicEntity {
    pub surface: String,
    /// Its L2 address (`hashId(canonical key)`), when the tracker has a hasher.
    pub to: Option<String>,
    pub role_weight: f32,
    /// The window's turn counter when it was last mentioned.
    pub turn: u64,
    pub at_ms: u64,
    /// Mention order (ties go to the earlier mention).
    pub seq: u64,
}

#[derive(Debug, Clone, Default)]
struct TopicFrame {
    entities: Vec<TopicEntity>,
    turn: u64,
    last_turn_ms: Option<u64>,
}

/// Window-isolated, decaying topic memory: `S = w_role · 0.65^Δturn · 2^(−Δt / 120 s)`.
#[derive(Debug, Clone)]
pub struct TopicTracker {
    by_cc: BTreeMap<String, TopicFrame>,
    seq: u64,
    hasher: Option<fn(&str) -> Option<String>>,
    reverse: BTreeMap<String, String>,
    pub max_topics_per_cc: usize,
}

impl Default for TopicTracker {
    fn default() -> Self {
        Self {
            by_cc: BTreeMap::new(),
            seq: 0,
            hasher: None,
            reverse: BTreeMap::new(),
            max_topics_per_cc: MAX_TOPICS_PER_CC,
        }
    }
}

/// Lowercase, no scheme (`app://`, `https://`), no path, no leading `www.`.
pub fn normalize_cc(cc: &str) -> String {
    let mut s = cc.trim().to_lowercase();
    if let Some(i) = s.find("://") {
        s = s[i + 3..].to_string();
        if let Some(j) = s.find('/') {
            s.truncate(j);
        }
    }
    let s = s.trim_end_matches('/');
    s.strip_prefix("www.").unwrap_or(s).to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkerKind {
    Spatial,
    Anaphoric,
    Proximal,
}

#[derive(Debug, Clone, Copy)]
struct Marker {
    kind: MarkerKind,
    /// Byte range of the marker word (punctuation excluded).
    start: usize,
    end: usize,
    /// The word may be replaced by the antecedent ("그거" yes, "아까" no).
    replaceable: bool,
}

const SPATIAL_PREFIXES: [&str; 8] = ["여기", "이곳", "이쪽", "저기", "저거", "저것", "저건", "저게"];
const SPATIAL_NOUNS: [&str; 12] = [
    "구간",
    "화면",
    "버튼",
    "칸",
    "창",
    "페이지",
    "메뉴",
    "보스",
    "방",
    "문",
    "곳",
    "상품",
];
const NOW_NOUNS: [&str; 3] = ["보스", "화면", "구간"];
const ANAPHORIC_PREFIXES: [&str; 5] = ["그거", "그것", "그건", "그게", "그걸"];
const PROXIMAL_PREFIXES: [&str; 5] = ["이거", "이것", "이건", "이게", "이걸"];

fn tokens(s: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in s.char_indices() {
        match (c.is_whitespace(), start) {
            (true, Some(st)) => {
                out.push((st, i));
                start = None;
            }
            (false, None) => start = Some(i),
            _ => {}
        }
    }
    if let Some(st) = start {
        out.push((st, s.len()));
    }
    out
}

fn core_end(s: &str, (a, b): (usize, usize)) -> usize {
    a + s[a..b].trim_end_matches(|c: char| !c.is_alphanumeric() && c != '#').len()
}

fn scan_markers(q: &str) -> Vec<Marker> {
    let toks = tokens(q);
    let mut out = Vec::new();
    for (i, &(a, b)) in toks.iter().enumerate() {
        let end = core_end(q, (a, b));
        let core = &q[a..end];
        let lower = core.to_ascii_lowercase();
        let next = toks.get(i + 1).map(|&(c, d)| &q[c..core_end(q, (c, d))]);
        let next_lower = next.map(str::to_ascii_lowercase);
        let mark = |kind, replaceable| Marker {
            kind,
            start: a,
            end,
            replaceable,
        };
        if SPATIAL_PREFIXES.iter().any(|p| core.starts_with(p))
            || (core == "이" && next.is_some_and(|n| SPATIAL_NOUNS.iter().any(|p| n.starts_with(p))))
            || (core == "지금" && next.is_some_and(|n| NOW_NOUNS.iter().any(|p| n.starts_with(p))))
            || spoken_slot_mask(core) != 0
            || lower == "here"
            || (lower == "this" && next_lower.as_deref().is_some_and(|n| ["button", "screen", "page"].contains(&n)))
        {
            out.push(mark(MarkerKind::Spatial, false));
        } else if ANAPHORIC_PREFIXES.iter().any(|p| core.starts_with(p))
            || (core == "그" && next.is_some())
            || lower == "it"
            || lower == "that"
        {
            out.push(mark(MarkerKind::Anaphoric, true));
        } else if core.starts_with("아까") || (core == "방금" && next.is_some_and(|n| n.starts_with("말한") || n.starts_with('그')))
        {
            out.push(mark(MarkerKind::Anaphoric, false));
        } else if PROXIMAL_PREFIXES.iter().any(|p| core.starts_with(p)) || lower == "this" {
            out.push(mark(MarkerKind::Proximal, true));
        }
    }
    out
}

/// `word` (a "그거 / 이거" form with its particle) with the antecedent put in its place, particle re-attached.
fn substitute(word: &str, antecedent: &str) -> String {
    let prefixes = ANAPHORIC_PREFIXES.iter().chain(PROXIMAL_PREFIXES.iter());
    let Some(prefix) = prefixes.into_iter().find(|p| word.starts_with(**p)) else {
        return antecedent.to_string();
    };
    let rest = &word[prefix.len()..];
    // 그건 = 그것+은, 그게 = 그것+이, 그걸 = 그것+을 (and the 이- forms).
    let implied = match prefix.chars().last() {
        Some('건') => Some(("은", "는")),
        Some('게') => Some(("이", "가")),
        Some('걸') => Some(("을", "를")),
        _ => None,
    };
    if let Some((c, v)) = implied {
        return format!("{}{rest}", josa(antecedent, c, v));
    }
    for (consonant, vowel) in [
        ("이랑", "랑"),
        ("으로", "로"),
        ("은", "는"),
        ("을", "를"),
        ("이", "가"),
        ("과", "와"),
    ] {
        if let Some(r) = rest.strip_prefix(vowel).or_else(|| rest.strip_prefix(consonant)) {
            return format!("{}{r}", josa(antecedent, consonant, vowel));
        }
    }
    format!("{antecedent}{rest}")
}

fn replace_range(q: &str, m: &Marker, with: &str) -> String {
    collapse_ws(&format!("{}{}{}", &q[..m.start], with, &q[m.end..]))
}

fn salience(e: &TopicEntity, frame_turn: u64, now_ms: u64) -> f32 {
    let dturn = frame_turn.saturating_sub(e.turn).min(64) as i32;
    let dt = now_ms.saturating_sub(e.at_ms) as f32;
    e.role_weight * TOPIC_TURN_DECAY.powi(dturn) * 2f32.powf(-dt / TOPIC_HALF_LIFE_MS)
}

impl TopicTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// With an L2 address function for topic keys (premium-discount-identity's `hash_id`, or a wrapper that
    /// canonicalises the key first).
    pub fn with_hasher(mut self, hasher: fn(&str) -> Option<String>) -> Self {
        self.hasher = Some(hasher);
        self
    }

    /// The surface phrase of a topic address (label hydration step 1).
    pub fn surface_for_to(&self, to: &str) -> Option<&str> {
        self.reverse.get(to).map(String::as_str)
    }

    /// The salience of `surface` for a question asked in `active_cc`: 0.0 for topics of other windows.
    pub fn salience(&self, active_cc: &str, surface: &str, now_ms: u64) -> f32 {
        self.by_cc
            .get(&normalize_cc(active_cc))
            .and_then(|f| {
                f.entities
                    .iter()
                    .find(|e| e.surface == surface)
                    .map(|e| salience(e, f.turn, now_ms))
            })
            .unwrap_or(0.0)
    }

    /// The topics of `cc` by salience (highest first, earlier mention first on ties).
    pub fn ranked(&self, cc: &str, now_ms: u64) -> Vec<(f32, TopicEntity)> {
        let Some(f) = self.by_cc.get(&normalize_cc(cc)) else {
            return Vec::new();
        };
        let mut v: Vec<(f32, TopicEntity)> = f.entities.iter().map(|e| (salience(e, f.turn, now_ms), e.clone())).collect();
        v.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.seq.cmp(&b.1.seq)));
        v
    }

    /// Forgets every topic of `cc`.
    pub fn forget_cc(&mut self, cc: &str) {
        self.by_cc.remove(&normalize_cc(cc));
    }

    fn remember(&mut self, cc: &str, surface: &str, weight: f32, now_ms: u64) {
        let surface = collapse_ws(surface);
        if surface.is_empty() {
            return;
        }
        let to = self.hasher.and_then(|h| h(&surface));
        if let Some(t) = &to {
            self.reverse.insert(t.clone(), surface.clone());
        }
        self.seq += 1;
        let seq = self.seq;
        let cap = self.max_topics_per_cc.max(1);
        let frame = self.by_cc.entry(cc.to_string()).or_default();
        let turn = frame.turn;
        if let Some(e) = frame.entities.iter_mut().find(|e| e.surface == surface) {
            e.turn = turn;
            e.at_ms = now_ms;
            e.role_weight = e.role_weight.max(weight);
            return;
        }
        frame.entities.push(TopicEntity {
            surface,
            to,
            role_weight: weight,
            turn,
            at_ms: now_ms,
            seq,
        });
        if frame.entities.len() > cap {
            let (t, now) = (frame.turn, now_ms);
            if let Some(i) = frame
                .entities
                .iter()
                .enumerate()
                .min_by(|a, b| salience(a.1, t, now).total_cmp(&salience(b.1, t, now)))
                .map(|(i, _)| i)
            {
                frame.entities.remove(i);
            }
        }
    }

    /// Zena named `surface` herself (a proactive offer that was promoted, or her answer's subject).
    pub fn note_zena_mention(&mut self, cc: &str, surface: &str, now_ms: u64) {
        self.remember(&normalize_cc(cc), surface, ZENA_MENTION_WEIGHT, now_ms);
    }

    /// Resolves `raw_query` against the screen and the window's topic frame, then records this turn (the
    /// player's named entities, a refreshed antecedent).
    pub fn resolve_and_plan(
        &mut self,
        raw_query: &str,
        extracted_entities: &[String],
        screen: &ScreenAnchorSnapshot,
        now_ms: u64,
    ) -> ContextualQueryPlan {
        let cc = normalize_cc(&screen.cc);
        let raw = collapse_ws(raw_query);
        let markers = scan_markers(&raw);
        let first = |k: MarkerKind| markers.iter().find(|m| m.kind == k).copied();
        let entities: Vec<String> = extracted_entities
            .iter()
            .map(|e| collapse_ws(e))
            .filter(|e| !e.is_empty())
            .collect();
        let frame_last = self.by_cc.get(&cc).and_then(|f| f.last_turn_ms);
        let focus = screen
            .focused
            .as_ref()
            .filter(|f| now_ms.saturating_sub(f.at_ms) <= DEICTIC_RECENT_MS);
        let screen_changed = screen.changed_at_ms.is_some_and(|t| now_ms.saturating_sub(t) <= DEICTIC_RECENT_MS);
        let dialogue_recent = frame_last.is_some_and(|t| now_ms.saturating_sub(t) <= DIALOGUE_CONTINUITY_MS);
        let ranked = self.ranked(&cc, now_ms);
        let recent_topic = ranked
            .iter()
            .any(|(s, e)| *s >= SALIENCE_FLOOR && now_ms.saturating_sub(e.at_ms) <= ELLIPSIS_WINDOW_MS);

        let (kind, marker) = if let Some(m) = first(MarkerKind::Spatial) {
            (ReferenceKind::DeicticScreen, Some(m))
        } else if let Some(m) = first(MarkerKind::Anaphoric) {
            (ReferenceKind::AnaphoricDialogue, Some(m))
        } else if let Some(m) = first(MarkerKind::Proximal) {
            let screen_first = focus.is_some() || screen_changed || !dialogue_recent;
            (
                if screen_first {
                    ReferenceKind::DeicticScreen
                } else {
                    ReferenceKind::AnaphoricDialogue
                },
                Some(m),
            )
        } else if !entities.is_empty() {
            (ReferenceKind::SelfContained, None)
        } else if recent_topic && raw.split_whitespace().count() <= 4 {
            (ReferenceKind::EllipticalFollowUp, None)
        } else {
            (ReferenceKind::SelfContained, None)
        };

        let mut scopes = ScopeTargets {
            cc: cc.clone(),
            ..Default::default()
        };
        let mut rewritten = raw.clone();
        let mut resolved_entity = None;
        let mut ambiguous_pair = None;
        match kind {
            ReferenceKind::DeicticScreen => {
                // The screen, never the previous dialogue topic.
                let label = focus
                    .map(|f| f.label.clone())
                    .or_else(|| screen.screen_label.clone())
                    .map(|l| collapse_ws(&l))
                    .filter(|l| !l.is_empty());
                scopes.l1_checkpoint_to = focus.and_then(|f| f.to.clone()).or_else(|| screen.checkpoint_to.clone());
                if let Some(l) = label {
                    rewritten = format!("{l} {raw}");
                    resolved_entity = Some(l);
                }
            }
            ReferenceKind::AnaphoricDialogue | ReferenceKind::EllipticalFollowUp => {
                // A left-out subject only continues a topic of the last minute; a pronoun may reach further back.
                let window = if kind == ReferenceKind::EllipticalFollowUp {
                    ELLIPSIS_WINDOW_MS
                } else {
                    u64::MAX
                };
                let live: Vec<&(f32, TopicEntity)> = ranked
                    .iter()
                    .filter(|(s, e)| *s >= SALIENCE_FLOOR && now_ms.saturating_sub(e.at_ms) <= window)
                    .collect();
                match live.as_slice() {
                    [a, b, ..] if a.0 - b.0 < AMBIGUITY_MARGIN => {
                        let both = format!("{}·{}", a.1.surface, b.1.surface);
                        rewritten = match marker.filter(|m| m.replaceable) {
                            Some(m) => replace_range(&raw, &m, &substitute(&raw[m.start..m.end], &both)),
                            None => format!("{} {} {raw}", a.1.surface, b.1.surface),
                        };
                        scopes.l2_entity_tos.extend(a.1.to.clone());
                        scopes.l2_entity_tos.extend(b.1.to.clone());
                        ambiguous_pair = Some((a.1.surface.clone(), b.1.surface.clone()));
                    }
                    [a, ..] => {
                        let e = &a.1;
                        rewritten = match marker.filter(|m| m.replaceable) {
                            Some(m) => replace_range(&raw, &m, &substitute(&raw[m.start..m.end], &e.surface)),
                            None => format!("{} {raw}", e.surface),
                        };
                        scopes.l2_entity_tos.extend(e.to.clone());
                        resolved_entity = Some(e.surface.clone());
                    }
                    [] => {}
                }
            }
            ReferenceKind::SelfContained => {}
        }
        for e in &entities {
            if let Some(t) = self.hasher.and_then(|h| h(e)) {
                if !scopes.l2_entity_tos.contains(&t) {
                    scopes.l2_entity_tos.push(t);
                }
            }
        }
        if let Some(k) = screen.screen_key.clone() {
            if !scopes.l2_entity_tos.contains(&k) {
                scopes.l2_entity_tos.push(k);
            }
        }

        // Record the turn: the player's own entities, and the antecedent they referred to again.
        for e in &entities {
            self.remember(&cc, e, USER_MENTION_WEIGHT, now_ms);
        }
        if kind != ReferenceKind::DeicticScreen {
            if let Some(e) = &resolved_entity {
                self.remember(&cc, e, USER_MENTION_WEIGHT, now_ms);
            }
        }
        let frame = self.by_cc.entry(cc).or_default();
        frame.turn += 1;
        frame.last_turn_ms = Some(now_ms);

        ContextualQueryPlan {
            raw_query: raw_query.to_string(),
            rewritten_query: rewritten,
            kind,
            scopes,
            resolved_entity,
            ambiguous_pair,
        }
    }

    /// Phase 6 Part 1 (`CN2-70`): 크로스-앱 메타 코스의 선택적 토픽 브리지.
    ///
    /// * `shared_relay_keys` 가 비어 있으면 아무것도 전이하지 않는다 (**CR-02 PLINKO 단절 0.0 완전 보존**).
    /// * 공통 캐노니컬 `RelayKey` 가 1개 이상 존재할 때만 `from_cc` 의 활성 상위 토픽을 `to_cc` 로 감쇠 전이(`0.80`)한다.
    pub fn bridge_cross_cc(
        &mut self,
        from_cc: &str,
        to_cc: &str,
        shared_relay_keys: &[String],
        now_ms: u64,
    ) -> usize {
        if shared_relay_keys.is_empty() {
            return 0;
        }
        let src = normalize_cc(from_cc);
        let dst = normalize_cc(to_cc);
        if src.is_empty() || dst.is_empty() || src == dst {
            return 0;
        }
        let active: Vec<(f32, TopicEntity)> = self
            .ranked(&src, now_ms)
            .into_iter()
            .filter(|(s, _)| *s >= SALIENCE_FLOOR)
            .take(3)
            .collect();
        let mut bridged = 0usize;
        for (sal, ent) in active {
            let weight = (sal * 0.80).clamp(SALIENCE_FLOOR, USER_MENTION_WEIGHT);
            self.remember(&dst, &ent.surface, weight, now_ms);
            bridged += 1;
        }
        if bridged > 0 {
            let dst_frame = self.by_cc.entry(dst).or_default();
            dst_frame.last_turn_ms = Some(now_ms);
        }
        bridged
    }
}

/// Phase 7 Part 3 (`AC-05`): 구음장애 · 말더듬으로 잘게 끊어 들어오는 음성 조각의 병합(2.5초 창, CR-06 "아니" · "취소" ·
/// "말고" 가 오면 버퍼를 교체)은 premium-discount-nav 의 것을 그대로 쓴다 (같은 규칙의 상태 기계, 캐릭터 대사 없음).
/// `in_flight` 로 병합 중인지도 읽는다.
pub use premium_discount_nav::{CoalesceOutcome, SpeechCoalescer, SPEECH_COALESCE_WINDOW_MS};

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_hash(s: &str) -> Option<String> {
        Some(format!("0x{:040x}", s.chars().map(|c| c as u64).sum::<u64>()))
    }

    fn game(now_ms: u64) -> ScreenAnchorSnapshot {
        ScreenAnchorSnapshot {
            cc: "app://EldenRing.exe".into(),
            screen_key: Some("0xscreen-boss-room".into()),
            screen_label: Some("보스방 입구".into()),
            checkpoint_to: Some("0xcheckpoint-gate".into()),
            focused: None,
            changed_at_ms: Some(now_ms.saturating_sub(60_000)),
        }
    }

    #[test]
    fn cr01_here_binds_the_checkpoint_and_that_restores_the_weapon() {
        let mut t = TopicTracker::new().with_hasher(fake_hash);
        let p0 = t.resolve_and_plan("출혈 쌍곡도 방패로 막아져?", &["출혈 쌍곡도".into()], &game(10_000), 10_000);
        assert_eq!(p0.kind, ReferenceKind::SelfContained);
        let mut here = t.clone();
        let p = here.resolve_and_plan("여기서 주의할 거 있어?", &[], &game(15_000), 15_000);
        assert_eq!(p.kind, ReferenceKind::DeicticScreen);
        assert_eq!(p.scopes.l1_checkpoint_to.as_deref(), Some("0xcheckpoint-gate"));
        assert!(
            !p.rewritten_query.contains("출혈 쌍곡도"),
            "no dialogue topic in a screen question: {}",
            p.rewritten_query
        );
        assert!(p.rewritten_query.contains("보스방 입구"));
        assert!(p.scopes.l2_entity_tos.contains(&"0xscreen-boss-room".to_string()));

        let mut that = t.clone();
        let p = that.resolve_and_plan("그거 방패로 막아?", &[], &game(15_000), 15_000);
        assert_eq!(p.kind, ReferenceKind::AnaphoricDialogue);
        assert_eq!(p.rewritten_query, "출혈 쌍곡도 방패로 막아?");
        assert_eq!(p.resolved_entity.as_deref(), Some("출혈 쌍곡도"));
        assert_eq!(p.scopes.l1_checkpoint_to, None);
        assert_eq!(
            p.scopes.l2_entity_tos.first(),
            fake_hash("출혈 쌍곡도").as_ref(),
            "the entity address leads L2"
        );
        assert_eq!(t.surface_for_to(&fake_hash("출혈 쌍곡도").unwrap()), Some("출혈 쌍곡도"));
    }

    #[test]
    fn cr02_alt_tab_isolates_the_previous_apps_topics() {
        let mut t = TopicTracker::new().with_hasher(fake_hash);
        t.resolve_and_plan("말레니아 물새 난무 어떻게 피해?", &["말레니아".into()], &game(0), 0);
        assert!(t.salience("eldenring.exe", "말레니아", 1_000) > 0.6);
        let browser = ScreenAnchorSnapshot {
            cc: "https://www.SteamPowered.com/app/1245620".into(),
            screen_key: Some("0xpage-store".into()),
            screen_label: Some("상점 페이지".into()),
            changed_at_ms: Some(4_000),
            ..Default::default()
        };
        assert_eq!(t.salience("steampowered.com", "말레니아", 5_000), 0.0, "weight 0 in another window");
        let mut now = t.clone();
        let p = now.resolve_and_plan("이거 할인율 어때?", &[], &browser, 5_000);
        assert_eq!(p.scopes.cc, "steampowered.com");
        assert_eq!(p.kind, ReferenceKind::DeicticScreen, "just switched: the screen");
        assert!(!p.rewritten_query.contains("말레니아"), "{}", p.rewritten_query);
        assert!(!p.scopes.l2_entity_tos.contains(&fake_hash("말레니아").unwrap()));
        assert_eq!(p.scopes.l2_entity_tos, vec!["0xpage-store".to_string()]);
        // Even an explicit back-reference finds nothing of the game in the browser.
        let p = t.resolve_and_plan("그거 할인율 어때?", &[], &browser, 30_000);
        assert_eq!((p.resolved_entity, p.ambiguous_pair), (None, None));
        assert!(!p.rewritten_query.contains("말레니아"));
        // Back in the game the topic is there again, decayed by time only.
        let back = t.salience("eldenring.exe", "말레니아", 30_000);
        assert!(back > 0.5 && back < 0.65, "{back}");
    }

    #[test]
    fn cr03_two_antecedents_in_a_tie_are_asked_back_not_guessed() {
        let mut t = TopicTracker::new().with_hasher(fake_hash);
        t.resolve_and_plan(
            "암월의 대검이랑 명도 월은 중에 뭐가 좋아?",
            &["암월의 대검".into(), "명도 월은".into()],
            &game(0),
            0,
        );
        let p = t.resolve_and_plan("그거 지력 제한 몇이야?", &[], &game(5_000), 5_000);
        assert_eq!(p.ambiguous_pair, Some(("암월의 대검".to_string(), "명도 월은".to_string())));
        assert_eq!(p.resolved_entity, None, "no single substitution");
        assert!(
            p.rewritten_query.contains("암월의 대검") && p.rewritten_query.contains("명도 월은"),
            "both are searched"
        );
        let guard = p.context_guard().unwrap();
        assert!(guard.contains("'암월의 대검' vs '명도 월은'") && guard.contains("되묻"), "{guard}");
        assert_eq!(p.scopes.l2_entity_tos.len(), 3, "both entity addresses and the screen");
    }

    #[test]
    fn a_clear_leader_resolves_and_zena_mentions_count_slightly_less() {
        let mut t = TopicTracker::new();
        t.resolve_and_plan("암월의 대검 어디서 얻어?", &["암월의 대검".into()], &game(0), 0);
        t.note_zena_mention("eldenring.exe", "명도 월은", 1_000);
        // Zena's mention is newer (Δturn 0, w 0.85) than the player's (Δturn 1, w 1.0 · 0.65).
        let p = t.resolve_and_plan("그건 지력 제한 몇이야?", &[], &game(2_000), 2_000);
        assert_eq!(p.resolved_entity.as_deref(), Some("명도 월은"));
        assert_eq!(
            p.rewritten_query, "명도 월은은 지력 제한 몇이야?",
            "the particle follows the antecedent"
        );
        assert_eq!(p.ambiguous_pair, None);
    }

    #[test]
    fn elliptical_follow_ups_and_the_proximal_rule() {
        let mut t = TopicTracker::new();
        t.resolve_and_plan(
            "간편결제 수수료 있어?",
            &["간편결제".into()],
            &ScreenAnchorSnapshot {
                cc: "shop.example.com".into(),
                ..Default::default()
            },
            0,
        );
        let shop = ScreenAnchorSnapshot {
            cc: "shop.example.com".into(),
            screen_label: Some("결제 페이지".into()),
            ..Default::default()
        };
        let mut e = t.clone();
        let p = e.resolve_and_plan("환불은 돼?", &[], &shop, 10_000);
        assert_eq!(p.kind, ReferenceKind::EllipticalFollowUp);
        assert_eq!(p.rewritten_query, "간편결제 환불은 돼?");
        // Still salient after 70 s, but a left-out subject only continues the last minute.
        let p = t.clone().resolve_and_plan("환불은 돼?", &[], &shop, 70_000);
        assert_eq!(p.kind, ReferenceKind::SelfContained);
        assert_eq!(p.rewritten_query, "환불은 돼?");
        // "이거" in an ongoing conversation (no screen change, no pointing) is the dialogue topic …
        let mut d = t.clone();
        let p = d.resolve_and_plan("이거 안전해?", &[], &shop, 8_000);
        assert_eq!(p.kind, ReferenceKind::AnaphoricDialogue);
        assert_eq!(p.rewritten_query, "간편결제 안전해?");
        // … right after pointing at badge #2 it is the screen.
        let pointed = ScreenAnchorSnapshot {
            focused: Some(FocusedRegion {
                number: Some(2),
                label: "일반결제".into(),
                to: Some("0xregion-2".into()),
                at_ms: 7_000,
            }),
            ..shop.clone()
        };
        let p = t.resolve_and_plan("이거 안전해?", &[], &pointed, 8_000);
        assert_eq!(p.kind, ReferenceKind::DeicticScreen);
        assert_eq!(p.scopes.l1_checkpoint_to.as_deref(), Some("0xregion-2"));
        assert_eq!(p.rewritten_query, "일반결제 이거 안전해?");
        // Long after the conversation, a bare question stays self-contained.
        let p = t.resolve_and_plan("오늘 날씨 어때?", &[], &shop, 500_000);
        assert_eq!(p.kind, ReferenceKind::SelfContained);
        assert_eq!(p.rewritten_query, "오늘 날씨 어때?");
    }

    #[test]
    fn markers_badges_and_english() {
        let kinds = |q: &str| scan_markers(q).iter().map(|m| m.kind).collect::<Vec<_>>();
        assert_eq!(kinds("#1번 눌러도 돼?"), vec![MarkerKind::Spatial]);
        assert_eq!(kinds("이 버튼 뭐야?"), vec![MarkerKind::Spatial]);
        assert_eq!(kinds("지금 보스 패턴 알려줘"), vec![MarkerKind::Spatial]);
        assert_eq!(kinds("그 무기 무거워?"), vec![MarkerKind::Anaphoric]);
        assert_eq!(kinds("아까 말한 거 다시"), vec![MarkerKind::Anaphoric]);
        assert_eq!(kinds("is it safe here?"), vec![MarkerKind::Anaphoric, MarkerKind::Spatial]);
        assert!(kinds("그래 고마워").is_empty() && kinds("그리고 이번엔").is_empty());
        assert_eq!(substitute("그걸", "물병"), "물병을");
        assert_eq!(substitute("그거랑", "대검"), "대검이랑");
        assert_eq!(substitute("그거", "대검"), "대검");
    }

    #[test]
    fn scopes_back_off_from_checkpoint_to_entity_to_app() {
        let s = ScopeTargets {
            cc: "eldenring.exe".into(),
            l1_checkpoint_to: Some("0xcp".into()),
            l2_entity_tos: vec!["0xmalenia".into(), "0xscreen".into()],
        };
        assert_eq!(s.match_level("EldenRing.exe", Some("0xcp")), Some(ScopeMatchLevel::ExactCheckpoint));
        assert_eq!(s.match_level("eldenring.exe", Some("0xmalenia")), Some(ScopeMatchLevel::Entity));
        assert_eq!(s.match_level("eldenring.exe", None), Some(ScopeMatchLevel::AppWide));
        assert_eq!(
            s.match_level("eldenring.exe", Some(&format!("0x{}", "0".repeat(40)))),
            Some(ScopeMatchLevel::AppWide)
        );
        assert_eq!(s.match_level("eldenring.exe", Some("0xother")), None);
        assert_eq!(s.match_level("steampowered.com", None), None, "another window is out of scope");
        let (lvl, sc) = s.boosted("eldenring.exe", Some("0xmalenia"), 0.5).unwrap();
        assert_eq!(lvl, ScopeMatchLevel::Entity);
        assert!((sc - 0.56).abs() < 1e-6);
        let order: Vec<ScopeMatchLevel> = s.backoff().into_iter().map(|(l, _)| l).collect();
        assert_eq!(
            order,
            vec![
                ScopeMatchLevel::ExactCheckpoint,
                ScopeMatchLevel::Entity,
                ScopeMatchLevel::Entity,
                ScopeMatchLevel::AppWide
            ]
        );
        assert_eq!(normalize_cc(" APP://Code.exe "), "code.exe");
        assert_eq!(normalize_cc("www.shop.example.com"), "shop.example.com");
    }

    #[test]
    fn cn270_cross_cc_topic_bridge_transfers_only_on_shared_relay_key_preserving_cr02_cliff() {
        let mut tracker = TopicTracker::new().with_hasher(fake_hash);
        tracker.remember("fextralife.com", "말레니아 빙결 항아리", USER_MENTION_WEIGHT, 1_000);

        // 1) Unrelated app switch (no shared RelayKey): CR-02 PLINKO cliff stays strictly 0.0!
        assert_eq!(tracker.bridge_cross_cc("fextralife.com", "notepad.exe", &[], 2_000), 0);
        assert_eq!(tracker.salience("notepad.exe", "말레니아 빙결 항아리", 2_000), 0.0);

        // 2) Linked meta-course Alt-Tab (shared canonical RelayKey): topic bridges cleanly!
        let bridged = tracker.bridge_cross_cc(
            "fextralife.com",
            "eldenring.exe",
            &["0xrelay-malenia-0001".to_string()],
            2_500,
        );
        assert_eq!(bridged, 1);
        assert!(tracker.salience("eldenring.exe", "말레니아 빙결 항아리", 2_500) >= SALIENCE_FLOOR);

        let game_snap = ScreenAnchorSnapshot {
            cc: "eldenring.exe".into(),
            ..Default::default()
        };
        let plan = tracker.resolve_and_plan("그거 어떻게 써?", &[], &game_snap, 3_000);
        assert_eq!(plan.kind, ReferenceKind::AnaphoricDialogue);
        assert_eq!(plan.resolved_entity.as_deref(), Some("말레니아 빙결 항아리"));
    }

    #[test]
    fn ac05_speech_coalescer_merges_stuttered_fragments_without_breaking_cr06_cancel() {
        let mut coal = SpeechCoalescer::new();
        assert_eq!(coal.push("그...", 1_000), CoalesceOutcome::Buffering);
        assert_eq!(coal.push("어... 2번", 1_900), CoalesceOutcome::Buffering);
        assert_eq!(
            coal.push("눌러줘", 2_800),
            CoalesceOutcome::Ready("그 2번 눌러줘".to_string())
        );

        // CR-06 explicit correction/cancel clears buffered fragments immediately.
        assert_eq!(coal.push("어... 3번", 5_000), CoalesceOutcome::Buffering);
        assert_eq!(
            coal.push("아니 취소 1번으로", 5_700),
            CoalesceOutcome::CancelledAndReplaced("아니 취소 1번으로".to_string())
        );
        assert_eq!(coal.flush_if_due(10_000), None);
    }
}
