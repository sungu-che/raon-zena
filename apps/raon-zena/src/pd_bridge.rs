//! Bridge between Raon Zena and `premium-discount` (`PdSyncClient`, `SyncLoop`, `LocalFilter`,
//! `QuarantineLedger`, `zena-verdict` and the two-track 600 ms RAG coordinator).
//!
//! Covers:
//! * **0 ms `PreThink` & 600 ms two-track RAG race** (`RagCoordinator`, CR-05) — Track Fast searches
//!   the in-memory `LocalMirrorIndex` (excluding `rejected` / `failed`, keeping `missing` as local-only)
//!   while Track Deep runs `pd_search_merged` under `SyncLoop::hold_for_interactive()` + `try_lock_for`.
//!   Late deep results populate `SceneRagCache` (120 s TTL) only when `privacy_epoch` has not changed.
//! * **Rapid re-ask & privacy cache isolation** (`ask_epoch × privacy_epoch`, CR-06).
//! * **Cross-scope supersession & third-party quarantine** (`curate_cross_scope`, `QuarantineLedger`
//!   persisted atomically to `%LOCALAPPDATA%\RaonZena\sds\quarantine_v1.json`).
//! * **Drop intake**: 0 ms `IngressPiiScrubber` (with checked BIP-39 seed detection) + `LocalFilter`
//!   profanity gate before touching the child app, optimistic mirror insert, and coalesced `SyncLoop`.
//! * **Pragmatic feedback, credit attribution & Stage 2 drawer state** (`LocalVerdictStore`,
//!   reversible `▲`/`▼` manual pins with zero outbound packets, 4-gate `OutcomeVoteGate`).
//! * **Non-blocking `PrewarmGuard`** (15 s debounce, `avail_ram_mb >= 650`) and 11th MCP tool
//!   `pd_analytic` report parsing into `AnalyticBrief`.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nlp_nms_router::embed::{HashedNgramEmbedder, SharedEmbedder, TextEmbedder};
use premium_discount_filter::{LocalFilter, Verdict as FilterVerdict};
use premium_discount_identity::{hash_id, is_bip39_word, is_valid_phrase};
use premium_discount_mcp::client::{
    try_lock_for, PdSyncClient, RejectionNotice, SyncLoop, SyncSnapshot, SYNC_INTERVAL,
};
use premium_discount_nav::{
    app_hash as nav_app_hash, element_hash, BehaviorParams, Checkpoint, Collection as NavCollection, Episode, Gesture, ReduceOptions,
    Role as NavRole, Run as NavRun, StepKind, UniversalChoiceBook, UniversalChoiceKind, CROSS_BOUNDARY_BOUNCE_WINDOW_MS,
};
use premium_discount_relay::{
    curate_cross_scope, supersede_sync, Entry as RelayEntry, QuarantineEntry, QuarantineLedger,
    QuarantineSyncState, Record as RelayRecord, RelayConfig, Sentence as RelaySentence, Standing,
    SupersedeFacts, Token as RelayToken,
};
use premium_discount_schema::envelope;
use premium_discount_window::{sensitive_title, VisibleWindowEntry, WS_EX_NOACTIVATE, WS_EX_TRANSPARENT};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use zena_core::brief::{prompt_label, AnalyticBrief};
use zena_core::claims::{
    extract_claim_for_query, CandidateKind, Grade, PackedVerdictSlot, RecordStatus,
    VerdictCandidate, VerdictSlotPacker,
};
use zena_core::hydrate::{LabelHydrator, RemoteHit};
use zena_core::ingress_guard::{IngressPiiScrubber, IngressVerdict};
use zena_core::reference::{normalize_cc, ContextualQueryPlan};
use zena_sds::ScopeStat;
use zena_verdict::feedback::RETRACT_WINDOW_MS;
use zena_verdict::{
    apply_feedback_to_ledger, attribute_and_update_duels, duel_for_item, extract_pragmatic_feedback,
    local_params, outperforms, record_demotion, toggle_pin, AttributionCandidate, Baselines, Case,
    CreditTargetKind, CrossTrialMemory, ExtractedFeedback, FeedbackContext, FeedbackToken,
    GlobalVerdict, LocalVerdictEntry, Obs, Tier, UserPin, VerdictPairView,
};

use crate::browser_uia::{build_live_web_rows, WebAnalyticTierRow};
use crate::pipeline::AvatarCmd;

/// Soft latency budget for interactive RAG before answering from `LocalMirrorIndex` (CR-05).
pub const SOFT_DEADLINE_MS: u64 = 600;
/// Maximum mirrored local notes kept per window / domain `cc`.
pub const MIRROR_PER_CC_CAP: usize = 256;
/// Cosine floor below which mirror hits are ignored (CR-04).
pub const COSINE_FLOOR: f32 = 0.42;
/// Small preference for my own note at equal scope & similarity.
pub const MINE_BOOST: f32 = 0.02;
/// TTL of `SceneRagCache` entries (CR-05 / CR-06).
pub const SCENE_CACHE_TTL_MS: u64 = 120_000;
/// Minimum free RAM (MiB) required to allow speculative pre-warming (PF-09).
pub const MIN_PREWARM_AVAIL_RAM_MB: u64 = 650;
/// Minimum spacing between `prewarm` requests.
pub const PREWARM_DEBOUNCE_MS: u64 = 15_000;
/// Cap on outcome votes held while signed out.
pub const MAX_HELD_OUTCOME_VOTES: usize = 32;

const SEED_CONTEXT_WINDOW_BYTES: usize = 64;
const SEED_CONTEXT_WORDS: [&str; 13] = [
    "seed",
    "mnemonic",
    "recovery",
    "phrase",
    "wallet",
    "backup",
    "bip39",
    "bip-39",
    "시드",
    "복구",
    "니모닉",
    "지갑",
    "백업",
];
const GAME_OVERLAY_PREFIXES: [&str; 2] = ["gameoverlayui", "eosoverlayrenderer"];

// -------------------------------------------------------------------------------------------------
// Real-device edge-case guards (work identically against pinned premium-discount rev d68508d)
// -------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct WordTok<'a> {
    start: usize,
    end: usize,
    word: &'a str,
    bip39: bool,
}

fn is_numbered_sep(sep: &str) -> bool {
    let trimmed = sep.trim();
    if trimmed.is_empty() {
        return false;
    }
    let digits_end = trimmed.find(|c: char| !c.is_ascii_digit()).unwrap_or(trimmed.len());
    if !(1..=2).contains(&digits_end) {
        return false;
    }
    let rest = &trimmed[digits_end..];
    rest.is_empty() || matches!(rest, "." | ")" | ":" | "-" | "번")
}

fn run_has_valid_checksum(words: &[&str]) -> bool {
    for &len in &[12usize, 15, 18, 21, 24] {
        if words.len() >= len && words.windows(len).any(|w| is_valid_phrase(&w.join(" "))) {
            return true;
        }
    }
    false
}

fn run_has_seed_context(text: &str, start: usize, end: usize) -> bool {
    let mut lo = start.saturating_sub(SEED_CONTEXT_WINDOW_BYTES);
    while lo > 0 && !text.is_char_boundary(lo) {
        lo -= 1;
    }
    let mut hi = (end + SEED_CONTEXT_WINDOW_BYTES).min(text.len());
    while hi < text.len() && !text.is_char_boundary(hi) {
        hi += 1;
    }
    let around = format!("{} {}", &text[lo..start], &text[end..hi]).to_lowercase();
    SEED_CONTEXT_WORDS.iter().any(|k| around.contains(k))
}

/// Checked BIP-39 seed phrase detector: 12+ consecutive BIP-39 words match only when at least one of
/// the following holds:
/// 1. any 12/15/18/21/24-word window has a valid BIP-39 SHA-256 checksum;
/// 2. the run is a numbered backup grid (`1. ... 2. ...`, >= 6 numbered gaps); or
/// 3. a seed / wallet / recovery keyword appears within 64 bytes.
///
/// Ordinary English gaming chatter ("game boss sword shield ...") is therefore never false-blocked.
pub fn detect_seed_phrase_checked(text: &str) -> Option<(usize, usize)> {
    let mut tokens: Vec<WordTok<'_>> = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
            let raw = &text[start..i];
            let lower = raw.to_ascii_lowercase();
            tokens.push(WordTok {
                start,
                end: i,
                word: raw,
                bip39: is_bip39_word(&lower),
            });
        } else {
            i += 1;
        }
    }

    let mut idx = 0;
    while idx < tokens.len() {
        if !tokens[idx].bip39 {
            idx += 1;
            continue;
        }
        let mut end_idx = idx;
        let mut numbered_gaps = 0usize;
        while end_idx + 1 < tokens.len() && tokens[end_idx + 1].bip39 {
            let sep = &text[tokens[end_idx].end..tokens[end_idx + 1].start];
            let sep_ok = sep.bytes().all(|b| {
                b.is_ascii_whitespace()
                    || b.is_ascii_digit()
                    || matches!(b, b',' | b'.' | b'-' | b'/' | b':' | b')' | b'(' | b';' | b'#')
            });
            if !sep_ok {
                break;
            }
            if is_numbered_sep(sep) {
                numbered_gaps += 1;
            }
            end_idx += 1;
        }
        let run_len = end_idx - idx + 1;
        if run_len >= 12 {
            let span_start = tokens[idx].start;
            let span_end = tokens[end_idx].end;
            let lowers: Vec<String> = tokens[idx..=end_idx]
                .iter()
                .map(|t| t.word.to_ascii_lowercase())
                .collect();
            let refs: Vec<&str> = lowers.iter().map(String::as_str).collect();
            let numbered = numbered_gaps >= 6
                || (idx == 0 && is_numbered_sep(&text[..span_start]) && numbered_gaps >= 5);
            if run_has_valid_checksum(&refs)
                || numbered
                || run_has_seed_context(text, span_start, span_end)
            {
                return Some((span_start, span_end));
            }
        }
        idx = end_idx + 1;
    }
    None
}

/// Filters out dormant Steam / Epic game overlay host windows (transparent, non-activating without
/// a checkout/payment title, or tiny <= 16 px host windows) so `GameOverlayUI.exe` sitting behind a
/// Steam game does not permanently lock capture. Interactive checkout overlays remain untouched.
pub fn filter_dormant_game_overlays(windows: Vec<VisibleWindowEntry>) -> Vec<VisibleWindowEntry> {
    windows
        .into_iter()
        .filter(|w| !is_dormant_game_overlay(w))
        .collect()
}

pub fn is_dormant_game_overlay(w: &VisibleWindowEntry) -> bool {
    let exe = w
        .info
        .exe_name
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let is_overlay = GAME_OVERLAY_PREFIXES.iter().any(|p| exe.starts_with(p));
    if !is_overlay {
        return false;
    }
    let click_through = w.ex_style & WS_EX_TRANSPARENT != 0;
    let no_activate = w.ex_style & WS_EX_NOACTIVATE != 0;
    let tiny = w.rect.width() <= 16 || w.rect.height() <= 16;
    tiny || click_through || (no_activate && !sensitive_title(&w.info.title))
}

/// Available physical RAM in MiB (Windows `GlobalMemoryStatusEx`; `None` elsewhere).
pub fn avail_ram_mb() -> Option<u64> {
    #[cfg(windows)]
    {
        use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut status = MEMORYSTATUSEX {
            dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        unsafe { GlobalMemoryStatusEx(&mut status).ok()? };
        Some(status.ullAvailPhys / (1024 * 1024))
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// Default local persistence path for `quarantine_v1.json`.
pub fn default_quarantine_path() -> PathBuf {
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("RaonZena")
        .join("sds")
        .join("quarantine_v1.json")
}

// -------------------------------------------------------------------------------------------------
// LocalMirrorIndex & SceneRagCache (Track Fast + 120 s privacy-isolated cache)
// -------------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MirroredLocalNote {
    pub request_id: String,
    pub server_id: Option<String>,
    pub cc: String,
    #[allow(dead_code)]
    pub to: Option<String>,
    pub label: String,
    pub text: String,
    /// `"queued" | "pending" | "stored" | "missing" | "rejected" | "failed"`.
    pub state: String,
    pub created_at_ms: i64,
    pub embedding: Vec<f32>,
    /// Phase 6 Part 1 (`CN2-71`): 크로스-앱 릴레이 연합으로 선택적 브리지된 대상 `cc` 목록.
    pub bridged_targets: Vec<String>,
}

/// In-memory mirror of my own outbox rows, kept in sync with `SyncSnapshot` for 0–600 ms fast RAG.
pub struct LocalMirrorIndex {
    embedder: SharedEmbedder,
    by_cc: BTreeMap<String, Vec<MirroredLocalNote>>,
    reverse_to: HashMap<String, String>,
    note_by_id: HashMap<String, String>,
    pub cap_per_cc: usize,
}

impl LocalMirrorIndex {
    pub fn new(embedder: SharedEmbedder) -> Self {
        Self {
            embedder,
            by_cc: BTreeMap::new(),
            reverse_to: HashMap::new(),
            note_by_id: HashMap::new(),
            cap_per_cc: MIRROR_PER_CC_CAP,
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn hashed() -> Self {
        Self::new(SharedEmbedder(Arc::new(HashedNgramEmbedder::default())))
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        self.embedder.embed(text)
    }

    pub fn shared_embedder(&self) -> SharedEmbedder {
        self.embedder.clone()
    }

    /// Phase 6 Part 1 (`CN2-71`): 공통 캐노니컬 RelayKey 가 검증된 경우에만 외부 `cc` 노트를 `target_cc` 검색에 브리지한다.
    pub fn bridge_note_to_cc(&mut self, request_id: &str, target_cc: &str) -> bool {
        let dst = normalize_cc(target_cc);
        if dst.is_empty() {
            return false;
        }
        for notes in self.by_cc.values_mut() {
            if let Some(n) = notes.iter_mut().find(|n| {
                n.request_id == request_id || n.server_id.as_deref() == Some(request_id)
            }) {
                if n.cc != dst && !n.bridged_targets.contains(&dst) {
                    n.bridged_targets.push(dst);
                }
                return true;
            }
        }
        false
    }

    pub fn similarity(&self, a: &str, b: &str) -> f32 {
        cosine(&self.embedder.embed(a), &self.embedder.embed(b))
    }

    /// Registers a `to` hash -> human label mapping (from NavMap, checkpoints, or TopicTracker).
    pub fn register_label(&mut self, to: &str, label: &str) {
        let clean = prompt_label(label);
        if !to.is_empty() && !clean.is_empty() {
            self.reverse_to.insert(to.to_string(), clean);
        }
    }

    pub fn label_for_to(&self, to: &str) -> Option<String> {
        self.reverse_to.get(to).cloned()
    }

    pub fn my_note_for_id(&self, id: &str) -> Option<String> {
        self.note_by_id.get(id).cloned()
    }

    /// Incrementally applies a `SyncSnapshot`: drops `rejected`/`failed`, marks `missing`, and
    /// embeds only newly seen or changed notes.
    pub fn apply_sync_snapshot(&mut self, snapshot: &SyncSnapshot) {
        for (req_id, row) in &snapshot.rows {
            let state = row.get("state").and_then(Value::as_str).unwrap_or("queued");
            let cc_raw = row.get("cc").and_then(Value::as_str).unwrap_or("");
            let cc = normalize_cc(cc_raw);
            if cc.is_empty() {
                continue;
            }
            if matches!(state, "rejected" | "failed") {
                if let Some(list) = self.by_cc.get_mut(&cc) {
                    list.retain(|n| n.request_id != *req_id);
                }
                self.note_by_id.remove(req_id);
                if let Some(sid) = row.get("server_id").and_then(Value::as_str) {
                    self.note_by_id.remove(sid);
                }
                continue;
            }
            let text = row
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            if text.is_empty() {
                continue;
            }
            let to = row.get("to").and_then(Value::as_str).map(str::to_string);
            let server_id = row
                .get("server_id")
                .or_else(|| row.get("id"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty() && *s != req_id)
                .map(str::to_string);
            let created_at_ms = row
                .get("created_at_ms")
                .or_else(|| row.get("updated_at_ms"))
                .and_then(Value::as_i64)
                .unwrap_or(0);
            self.upsert_note(req_id, server_id, &cc, to, &text, state, created_at_ms);
        }
    }

    /// Immediately mirrors a locally dropped note before `SyncLoop` completes.
    pub fn enqueue_optimistic(
        &mut self,
        request_id: &str,
        cc: &str,
        to: Option<&str>,
        text: &str,
        now_ms: i64,
    ) {
        let cc = normalize_cc(cc);
        if cc.is_empty() || text.trim().is_empty() {
            return;
        }
        self.upsert_note(
            request_id,
            None,
            &cc,
            to.map(str::to_string),
            text.trim(),
            "queued",
            now_ms,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn upsert_note(
        &mut self,
        request_id: &str,
        server_id: Option<String>,
        cc: &str,
        to: Option<String>,
        text: &str,
        state: &str,
        created_at_ms: i64,
    ) {
        let label = prompt_label(text);
        self.note_by_id.insert(request_id.to_string(), label.clone());
        if let Some(sid) = &server_id {
            self.note_by_id.insert(sid.clone(), label.clone());
        }
        let list = self.by_cc.entry(cc.to_string()).or_default();
        if let Some(existing) = list.iter_mut().find(|n| n.request_id == request_id) {
            let changed = existing.text != text;
            existing.server_id = server_id.or(existing.server_id.clone());
            existing.to = to.or(existing.to.clone());
            existing.state = state.to_string();
            existing.label = label;
            if changed {
                existing.text = text.to_string();
                existing.embedding = self.embedder.embed(text);
            }
            if created_at_ms > 0 {
                existing.created_at_ms = created_at_ms;
            }
        } else {
            let embedding = self.embedder.embed(text);
            list.push(MirroredLocalNote {
                request_id: request_id.to_string(),
                server_id,
                cc: cc.to_string(),
                to,
                label,
                text: text.to_string(),
                state: state.to_string(),
                created_at_ms,
                embedding,
                bridged_targets: Vec::new(),
            });
            while list.len() > self.cap_per_cc {
                if let Some((idx, _)) = list.iter().enumerate().min_by_key(|(_, n)| n.created_at_ms) {
                    let removed = list.remove(idx);
                    self.note_by_id.remove(&removed.request_id);
                } else {
                    break;
                }
            }
        }
    }

    pub fn notes_in_cc(&self, cc: &str) -> &[MirroredLocalNote] {
        self.by_cc
            .get(&normalize_cc(cc))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Fast in-memory search (Track Fast): applies scope boost (+0.12 L1, +0.06 L2) and cosine floor,
    /// guarantees 0 `rejected` / `failed` notes, marks `missing` as local-only, and surfaces notes
    /// explicitly bridged into `plan.scopes.cc` via cross-app RelayKey federation (`CN2-71`).
    pub fn search_fast(
        &self,
        plan: &ContextualQueryPlan,
        query_vec: &[f32],
        local_grade: &dyn Fn(&str) -> Grade,
    ) -> Vec<VerdictCandidate> {
        let mut out = Vec::new();
        let target_cc = &plan.scopes.cc;
        let matching_notes = self
            .by_cc
            .values()
            .flat_map(|v| v.iter())
            .filter(|n| &n.cc == target_cc || n.bridged_targets.iter().any(|b| b == target_cc));
        for note in matching_notes {
            if matches!(note.state.as_str(), "rejected" | "failed") {
                continue;
            }
            let cos = cosine(query_vec, &note.embedding);
            if cos < COSINE_FLOOR {
                continue;
            }
            let effective_cc = if &note.cc == target_cc {
                note.cc.as_str()
            } else {
                target_cc.as_str()
            };
            let Some((_, boosted)) = plan.scopes.boosted(effective_cc, note.to.as_deref(), cos) else {
                continue;
            };
            let id = note
                .server_id
                .clone()
                .unwrap_or_else(|| note.request_id.clone());
            let local = local_grade(&id);
            let status = if note.state == "missing" {
                RecordStatus::Missing
            } else {
                RecordStatus::Active
            };
            let claim = extract_claim_for_query(&note.text, &plan.rewritten_query);
            out.push(VerdictCandidate {
                record_id: id,
                cc: note.cc.clone(),
                to: note.to.clone(),
                canonical_topic: canonical_topic_of(&note.label, claim.as_deref()),
                label: note.label.clone(),
                claim,
                local,
                global: Grade::Unrated,
                kind: if local == Grade::Discount {
                    CandidateKind::Trap
                } else {
                    CandidateKind::Solution
                },
                status,
                score: boosted + MINE_BOOST,
                value_diff: None,
            });
        }
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        out
    }
}

#[derive(Debug, Clone)]
struct CachedRagEntry {
    privacy_epoch: u64,
    stored_at_ms: u64,
    slots: Vec<PackedVerdictSlot>,
}

/// Phase 6 Part 3: 정밀 키 `(cc, screen_key, checkpoint_to, prereq_digest, rewritten_query)` 기반 120s RAG 캐시 + 선행 프리페치.
#[derive(Debug, Default)]
pub struct SceneRagCache {
    entries: HashMap<(String, String, String, String, String), CachedRagEntry>,
    pub ttl_ms: u64,
}

impl SceneRagCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            ttl_ms: SCENE_CACHE_TTL_MS,
        }
    }

    fn key_scoped(
        cc: &str,
        screen_key: &str,
        checkpoint_to: &str,
        prereq_digest: &str,
        rewritten_query: &str,
    ) -> (String, String, String, String, String) {
        (
            normalize_cc(cc),
            screen_key.trim().to_lowercase(),
            checkpoint_to.trim().to_lowercase(),
            prereq_digest.trim().to_lowercase(),
            rewritten_query.trim().to_lowercase(),
        )
    }

    /// Inserts slots ONLY if `captured_privacy_epoch == current_privacy_epoch`.
    pub fn insert(
        &mut self,
        cc: &str,
        rewritten_query: &str,
        captured_privacy_epoch: u64,
        current_privacy_epoch: u64,
        slots: Vec<PackedVerdictSlot>,
        now_ms: u64,
    ) -> bool {
        self.insert_scoped(
            cc,
            "",
            "",
            "",
            rewritten_query,
            captured_privacy_epoch,
            current_privacy_epoch,
            slots,
            now_ms,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert_scoped(
        &mut self,
        cc: &str,
        screen_key: &str,
        checkpoint_to: &str,
        prereq_digest: &str,
        rewritten_query: &str,
        captured_privacy_epoch: u64,
        current_privacy_epoch: u64,
        slots: Vec<PackedVerdictSlot>,
        now_ms: u64,
    ) -> bool {
        if captured_privacy_epoch != current_privacy_epoch {
            return false;
        }
        self.entries.insert(
            Self::key_scoped(cc, screen_key, checkpoint_to, prereq_digest, rewritten_query),
            CachedRagEntry {
                privacy_epoch: captured_privacy_epoch,
                stored_at_ms: now_ms,
                slots,
            },
        );
        true
    }

    /// Looks up cached slots for the current `privacy_epoch` within TTL.
    pub fn get(
        &mut self,
        cc: &str,
        rewritten_query: &str,
        current_privacy_epoch: u64,
        now_ms: u64,
    ) -> Option<Vec<PackedVerdictSlot>> {
        self.get_scoped(cc, "", "", "", rewritten_query, current_privacy_epoch, now_ms)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn get_scoped(
        &mut self,
        cc: &str,
        screen_key: &str,
        checkpoint_to: &str,
        prereq_digest: &str,
        rewritten_query: &str,
        current_privacy_epoch: u64,
        now_ms: u64,
    ) -> Option<Vec<PackedVerdictSlot>> {
        let ttl = self.ttl_ms;
        self.entries.retain(|_, v| {
            v.privacy_epoch == current_privacy_epoch
                && now_ms.saturating_sub(v.stored_at_ms) <= ttl
        });
        self.entries
            .get(&Self::key_scoped(cc, screen_key, checkpoint_to, prereq_digest, rewritten_query))
            .map(|e| e.slots.clone())
    }

    /// Phase 6 Part 3: 다음 체크포인트 진입 직전 선행 프리페치 슬롯을 적재한다.
    #[allow(dead_code)]
    pub fn prefetch_lookahead(
        &mut self,
        cc: &str,
        next_checkpoint_to: &str,
        query: &str,
        privacy_epoch: u64,
        slots: Vec<PackedVerdictSlot>,
        now_ms: u64,
    ) -> bool {
        self.insert_scoped(cc, "", next_checkpoint_to, "", query, privacy_epoch, privacy_epoch, slots, now_ms)
    }

    pub fn purge_all(&mut self) {
        self.entries.clear();
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Phase 6 Part 1 (`CN2-72`): 크로스-앱 메타 궤적 간선(예: 공략 위키 → 인게임 전환) 원장.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrajectoryEdge {
    pub from_cc: String,
    pub to_cc: String,
    pub shared_relay_keys: Vec<String>,
    pub observed_at_ms: i64,
}

#[derive(Debug, Default, Clone)]
pub struct TrajectoryLedger {
    edges: VecDeque<TrajectoryEdge>,
}

impl TrajectoryLedger {
    pub const WINDOW_MS: i64 = 300_000;

    pub fn record_edge(&mut self, from_cc: &str, to_cc: &str, shared_relay_keys: Vec<String>, now_ms: i64) {
        if shared_relay_keys.is_empty() {
            return;
        }
        let f = normalize_cc(from_cc);
        let t = normalize_cc(to_cc);
        if f.is_empty() || t.is_empty() || f == t {
            return;
        }
        self.edges.push_back(TrajectoryEdge {
            from_cc: f,
            to_cc: t,
            shared_relay_keys,
            observed_at_ms: now_ms,
        });
        while self.edges.len() > 32 {
            self.edges.pop_front();
        }
    }

    /// 현재 `active_cc` 와 5분 이내에 공통 RelayKey 로 연결된 모든 앱 `cc` 목록(`active_cc` 포함).
    pub fn linked_ccs(&mut self, active_cc: &str, now_ms: i64) -> Vec<String> {
        let cur = normalize_cc(active_cc);
        self.edges.retain(|e| now_ms.saturating_sub(e.observed_at_ms) <= Self::WINDOW_MS);
        let mut out = vec![cur.clone()];
        for e in &self.edges {
            if e.to_cc == cur && !out.contains(&e.from_cc) {
                out.push(e.from_cc.clone());
            } else if e.from_cc == cur && !out.contains(&e.to_cc) {
                out.push(e.to_cc.clone());
            }
        }
        out
    }
}

/// Phase 5 Part 2 (`CN2-64`): 온디바이스 64-슬롯 구역별 보관함(`ZoneVault`, Pinned-SLRU + `batch_id` 멱등 동기화).
pub const ZONE_VAULT_MAX_SLOTS: usize = 64;
/// 한 앱 · 사이트에서 모으는 체크포인트 상한 (오래된 것부터 버린다. 런은 10단계로 솎인다)
pub const LIVE_RUN_MAX_CHECKPOINTS: usize = 256;

/// `ZoneVault` 파일: 격리 목록 파일 옆 (`<이름>.zone_vault.json`)
pub fn zone_vault_path(quarantine_path: &Path) -> PathBuf {
    quarantine_path.with_extension("zone_vault.json")
}

/// 1-hop 결과를 기다리는 배지 선택
#[derive(Debug, Clone)]
struct PendingPick {
    key: String,
    pick_screen: String,
    /// 고른 화면을 떠날 때 그 화면 에피소드의 마찰
    same_screen_friction: Option<f64>,
    /// 도착 화면과 도착 시각
    arrived: Option<(String, i64)>,
}

/// 지금 앱 · 사이트에서 모으는 체크포인트
#[derive(Debug, Clone, Default)]
struct LiveRunBuffer {
    cc: String,
    checkpoints: Vec<Checkpoint>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneVault {
    pub collection: NavCollection,
    #[serde(default)]
    pending_batch: Option<String>,
    #[serde(default)]
    batch_seq: u64,
}

impl ZoneVault {
    pub fn new() -> Self {
        Self::default()
    }

    /// 기기 안 파일에서 읽는다 (없거나 깨졌으면 빈 보관함). 코스 · 단계 해시와 구간 시간만 들어 있다 (글 · 라벨 없음)
    pub fn load(path: &Path) -> Self {
        std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    /// 기기 안 파일에 쓴다 (임시 파일에 쓰고 바꿔 넣는다)
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(self)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// 64-슬롯 Pinned-SLRU 상한으로 완주 기록을 보관한다.
    pub fn record_run(&mut self, run: &NavRun) {
        self.collection.add_with_cap(run, ZONE_VAULT_MAX_SLOTS);
    }

    /// 특정 코스를 수동 고정(또는 해제)해 슬롯 축출에서 보호한다.
    pub fn pin_course(&mut self, app: &str, course: &str, pinned: bool) -> bool {
        self.collection.pin_course(app, course, pinned)
    }

    /// 서버 `POST /nav/trials` 배치 재전송 시 동일한 `batch_id` 를 유지해 멱등성을 보장한다.
    pub fn next_or_pending_batch_id(&mut self, now_ms: i64) -> String {
        if let Some(existing) = &self.pending_batch {
            return existing.clone();
        }
        self.batch_seq += 1;
        let bid = format!("batch_{now_ms}_{:08x}", self.batch_seq);
        self.pending_batch = Some(bid.clone());
        bid
    }

    pub fn ack_batch(&mut self, batch_id: &str) {
        if self.pending_batch.as_deref() == Some(batch_id) {
            self.pending_batch = None;
        }
    }
}

// -------------------------------------------------------------------------------------------------
// RagCoordinator: 0 ms PreThink, 600 ms two-track race, ask_epoch × privacy_epoch cancellation
// -------------------------------------------------------------------------------------------------

/// Abstraction over `pd_search_merged` so production (`PdSyncClient`) and tests run the exact same
/// 600 ms race, epoch cancellation, and background cache population code.
pub trait DeepSearcher: Send + Sync + 'static {
    fn search_merged(&self, plan: &ContextualQueryPlan, budget: Duration) -> anyhow::Result<Value>;
}

/// Real `DeepSearcher` backed by the shared `PdSyncClient`, respecting `try_lock_for` so an in-flight
/// background sync never stalls interactive search past its soft deadline.
pub struct McpDeepSearcher {
    client: Arc<Mutex<PdSyncClient>>,
}

impl McpDeepSearcher {
    pub fn new(client: Arc<Mutex<PdSyncClient>>) -> Self {
        Self { client }
    }
}

impl DeepSearcher for McpDeepSearcher {
    fn search_merged(&self, plan: &ContextualQueryPlan, budget: Duration) -> anyhow::Result<Value> {
        let mut guard = try_lock_for(&self.client, budget)
            .ok_or_else(|| anyhow::anyhow!("sync client busy past interactive budget"))?;
        let mut args = json!({
            "query": plan.rewritten_query,
            "cc": plan.scopes.cc,
            "top_k": 16,
        });
        if let Some(l1) = &plan.scopes.l1_checkpoint_to {
            args["l1_checkpoint_to"] = json!(l1);
        }
        if !plan.scopes.l2_entity_tos.is_empty() {
            args["l2_entity_tos"] = json!(plan.scopes.l2_entity_tos);
        }
        guard.search_merged(args)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RagHealth {
    FreshMerged,
    SoftDeadlineFastOnly,
    CacheHit,
    OfflineFastOnly,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RagOutcome {
    Ready {
        epoch: u64,
        health: RagHealth,
        slots: Vec<PackedVerdictSlot>,
        prethink_emitted: bool,
    },
    Cancelled {
        epoch: u64,
    },
}

pub struct RagCoordinator {
    pub ask_epoch: Arc<AtomicU64>,
    pub privacy_epoch: Arc<AtomicU64>,
    pub soft_deadline: Duration,
    pub mirror: Arc<Mutex<LocalMirrorIndex>>,
    pub cache: Arc<Mutex<SceneRagCache>>,
    pub quarantine: Arc<Mutex<QuarantineLedger>>,
    pub deep: Option<Arc<dyn DeepSearcher>>,
}

impl RagCoordinator {
    pub fn new(
        ask_epoch: Arc<AtomicU64>,
        privacy_epoch: Arc<AtomicU64>,
        mirror: Arc<Mutex<LocalMirrorIndex>>,
        quarantine: Arc<Mutex<QuarantineLedger>>,
        deep: Option<Arc<dyn DeepSearcher>>,
    ) -> Self {
        Self {
            ask_epoch,
            privacy_epoch,
            soft_deadline: Duration::from_millis(SOFT_DEADLINE_MS),
            mirror,
            cache: Arc::new(Mutex::new(SceneRagCache::new())),
            quarantine,
            deep,
        }
    }

    pub fn next_ask_epoch(&self) -> u64 {
        self.ask_epoch.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub fn bump_privacy_epoch(&self) -> u64 {
        let p = self.privacy_epoch.fetch_add(1, Ordering::AcqRel) + 1;
        self.ask_epoch.fetch_add(1, Ordering::AcqRel);
        if let Ok(mut c) = self.cache.lock() {
            c.purge_all();
        }
        p
    }

    /// Runs the 0 ms PreThink + 600 ms two-track RAG race.
    pub fn retrieve(
        &self,
        plan: &ContextualQueryPlan,
        sync_loop: Option<&SyncLoop>,
        prethink_tx: Option<&Sender<AvatarCmd>>,
        local_grade: Arc<dyn Fn(&str) -> Grade + Send + Sync>,
        now_ms: u64,
    ) -> RagOutcome {
        let epoch = self.next_ask_epoch();
        let p_epoch = self.privacy_epoch.load(Ordering::Acquire);
        let mut prethink_emitted = false;
        if let Some(tx) = prethink_tx {
            if tx.send(AvatarCmd::PreThink { epoch }).is_ok() {
                prethink_emitted = true;
            }
        }
        // Yield SyncLoop for the interactive search window.
        let _hold = sync_loop.map(|s| s.hold_for_interactive());

        if let Ok(mut cache) = self.cache.lock() {
            if let Some(slots) = cache.get(&plan.scopes.cc, &plan.rewritten_query, p_epoch, now_ms) {
                if self.ask_epoch.load(Ordering::Acquire) != epoch
                    || self.privacy_epoch.load(Ordering::Acquire) != p_epoch
                {
                    return RagOutcome::Cancelled { epoch };
                }
                return RagOutcome::Ready {
                    epoch,
                    health: RagHealth::CacheHit,
                    slots,
                    prethink_emitted,
                };
            }
        }

        // Track Fast: instant in-memory mirror search.
        let fast_candidates = {
            let mirror = self.mirror.lock().unwrap();
            let qvec = mirror.embed(&plan.rewritten_query);
            mirror.search_fast(plan, &qvec, local_grade.as_ref())
        };
        let fast_slots = self.pack_with_quarantine(&plan.scopes.cc, fast_candidates.clone(), None);

        let Some(deep) = self.deep.clone() else {
            if self.ask_epoch.load(Ordering::Acquire) != epoch
                || self.privacy_epoch.load(Ordering::Acquire) != p_epoch
            {
                return RagOutcome::Cancelled { epoch };
            }
            return RagOutcome::Ready {
                epoch,
                health: RagHealth::OfflineFastOnly,
                slots: fast_slots,
                prethink_emitted,
            };
        };

        // Track Deep: runs on a worker thread so we can return at soft_deadline (600 ms) and still
        // store late results into SceneRagCache (iff privacy_epoch is unchanged!).
        let (tx, rx) = mpsc::channel::<Vec<PackedVerdictSlot>>();
        let plan_bg = plan.clone();
        let mirror_bg = self.mirror.clone();
        let cache_bg = self.cache.clone();
        let quarantine_bg = self.quarantine.clone();
        let privacy_bg = self.privacy_epoch.clone();
        let grade_bg = local_grade.clone();
        let budget = self.soft_deadline;
        std::thread::Builder::new()
            .name("zena-rag-deep".into())
            .spawn(move || {
                let Ok(val) = deep.search_merged(&plan_bg, budget) else {
                    return;
                };
                let slots = build_merged_slots(
                    &val,
                    &plan_bg,
                    &mirror_bg,
                    &quarantine_bg,
                    grade_bg.as_ref(),
                );
                let cur_p = privacy_bg.load(Ordering::Acquire);
                if cur_p == p_epoch && !slots.is_empty() {
                    if let Ok(mut c) = cache_bg.lock() {
                        c.insert(
                            &plan_bg.scopes.cc,
                            &plan_bg.rewritten_query,
                            p_epoch,
                            cur_p,
                            slots.clone(),
                            now_ms,
                        );
                    }
                }
                let _ = tx.send(slots);
            })
            .ok();

        let deadline = Instant::now() + self.soft_deadline;
        loop {
            if self.ask_epoch.load(Ordering::Acquire) != epoch
                || self.privacy_epoch.load(Ordering::Acquire) != p_epoch
            {
                return RagOutcome::Cancelled { epoch };
            }
            let rem = deadline.saturating_duration_since(Instant::now());
            if rem.is_zero() {
                return RagOutcome::Ready {
                    epoch,
                    health: RagHealth::SoftDeadlineFastOnly,
                    slots: fast_slots,
                    prethink_emitted,
                };
            }
            match rx.recv_timeout(rem.min(Duration::from_millis(10))) {
                Ok(deep_slots) => {
                    if self.ask_epoch.load(Ordering::Acquire) != epoch
                        || self.privacy_epoch.load(Ordering::Acquire) != p_epoch
                    {
                        return RagOutcome::Cancelled { epoch };
                    }
                    let slots = if deep_slots.is_empty() {
                        fast_slots
                    } else {
                        deep_slots
                    };
                    return RagOutcome::Ready {
                        epoch,
                        health: RagHealth::FreshMerged,
                        slots,
                        prethink_emitted,
                    };
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    if self.ask_epoch.load(Ordering::Acquire) != epoch
                        || self.privacy_epoch.load(Ordering::Acquire) != p_epoch
                    {
                        return RagOutcome::Cancelled { epoch };
                    }
                    return RagOutcome::Ready {
                        epoch,
                        health: RagHealth::OfflineFastOnly,
                        slots: fast_slots,
                        prethink_emitted,
                    };
                }
            }
        }
    }

    fn pack_with_quarantine(
        &self,
        cc: &str,
        mut candidates: Vec<VerdictCandidate>,
        barometer: Option<&zena_core::claims::Barometer>,
    ) -> Vec<PackedVerdictSlot> {
        if let Ok(q) = self.quarantine.lock() {
            for c in &mut candidates {
                let standing = |_id: &str| Standing::default();
                if let Some(entry) = q.active_superseder(cc, &c.record_id, standing) {
                    c.kind = CandidateKind::SupersededOld;
                    c.local = Grade::Discount;
                    c.value_diff = entry.value_diff.clone();
                }
            }
        }
        VerdictSlotPacker::default().pack(&candidates, barometer)
    }
}

fn build_merged_slots(
    merged_json: &Value,
    plan: &ContextualQueryPlan,
    mirror_arc: &Arc<Mutex<LocalMirrorIndex>>,
    quarantine_arc: &Arc<Mutex<QuarantineLedger>>,
    local_grade: &dyn Fn(&str) -> Grade,
) -> Vec<PackedVerdictSlot> {
    let matches = merged_json
        .get("matches")
        .or_else(|| merged_json.get("results"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let mut candidates: Vec<VerdictCandidate> = Vec::new();
    let mut remote_hits: Vec<RemoteHit> = Vec::new();

    for m in &matches {
        let state = m.get("state").and_then(Value::as_str).unwrap_or("stored");
        if matches!(state, "rejected" | "failed") {
            continue;
        }
        let id = m
            .get("id")
            .or_else(|| m.get("request_id"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if id.is_empty() {
            continue;
        }
        let cc = m
            .get("cc")
            .and_then(Value::as_str)
            .map(normalize_cc)
            .unwrap_or_else(|| plan.scopes.cc.clone());
        let to = m.get("to").and_then(Value::as_str).map(str::to_string);
        let score = m
            .get("score")
            .or_else(|| m.get("similarity"))
            .and_then(Value::as_f64)
            .unwrap_or(0.5) as f32;
        let global = parse_grade(
            m.get("verdict")
                .and_then(|v| v.get("tier"))
                .or_else(|| m.get("tier"))
                .and_then(Value::as_str),
        );
        let text = m
            .get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty());

        if let Some(txt) = text {
            let local = local_grade(&id);
            let label = prompt_label(txt);
            let claim = extract_claim_for_query(txt, &plan.rewritten_query);
            let status = if state == "missing" {
                RecordStatus::Missing
            } else {
                RecordStatus::Active
            };
            let global = if status == RecordStatus::Missing {
                Grade::Unrated
            } else {
                global
            };
            candidates.push(VerdictCandidate {
                record_id: id,
                cc,
                to,
                canonical_topic: canonical_topic_of(&label, claim.as_deref()),
                label,
                claim,
                local,
                global,
                kind: if local == Grade::Discount || global == Grade::Discount {
                    CandidateKind::Trap
                } else {
                    CandidateKind::Solution
                },
                status,
                score,
                value_diff: None,
            });
        } else {
            remote_hits.push(RemoteHit {
                id,
                cc,
                to,
                root: m.get("root").and_then(Value::as_str).map(str::to_string),
                ref_: m.get("ref").and_then(Value::as_str).map(str::to_string),
                score,
                global,
            });
        }
    }

    let barometer = {
        let mirror = mirror_arc.lock().unwrap();
        let label_for_to = |t: &str| mirror.label_for_to(t);
        let my_note_for_id = |id: &str| mirror.my_note_for_id(id);
        let report = LabelHydrator {
            label_for_to: &label_for_to,
            my_note_for_id: &my_note_for_id,
        }
        .hydrate(&remote_hits);
        for h in &report.hydrated {
            candidates.push(h.to_candidate(&canonical_topic_of(&h.label, None)));
        }
        report.barometer
    };

    if let Ok(q) = quarantine_arc.lock() {
        for c in &mut candidates {
            let standing = |_id: &str| Standing::default();
            if let Some(entry) = q.active_superseder(&plan.scopes.cc, &c.record_id, standing) {
                c.kind = CandidateKind::SupersededOld;
                c.local = Grade::Discount;
                c.value_diff = entry.value_diff.clone();
            }
        }
    }

    VerdictSlotPacker::default().pack(&candidates, barometer.as_ref())
}

fn parse_grade(tier: Option<&str>) -> Grade {
    match tier.map(|s| s.to_ascii_lowercase()).as_deref() {
        Some("premium") => Grade::Premium,
        Some("neutral") => Grade::Neutral,
        Some("discount") => Grade::Discount,
        _ => Grade::Unrated,
    }
}

fn tier_to_grade(t: Tier) -> Grade {
    match t {
        Tier::Premium => Grade::Premium,
        Tier::Neutral => Grade::Neutral,
        Tier::Discount => Grade::Discount,
        Tier::Unrated => Grade::Unrated,
    }
}

fn canonical_topic_of(label: &str, claim: Option<&str>) -> String {
    let src = if !label.trim().is_empty() {
        label
    } else {
        claim.unwrap_or("")
    };
    src.split_whitespace()
        .find(|w| w.chars().count() >= 2)
        .unwrap_or(src)
        .trim_matches(|c: char| !c.is_alphanumeric() && !('\u{AC00}'..='\u{D7A3}').contains(&c))
        .to_string()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot = raon_cpu_kernels::dot_f32_simd(a, b);
    let na = raon_cpu_kernels::dot_f32_simd(a, a);
    let nb = raon_cpu_kernels::dot_f32_simd(b, b);
    let denom = (na * nb).sqrt();
    if denom > 1e-8 {
        dot / denom
    } else {
        0.0
    }
}

// -------------------------------------------------------------------------------------------------
// PrewarmGuard & OutcomeVoteGate
// -------------------------------------------------------------------------------------------------

/// Debounced, RAM-guarded pre-warm trigger for `pd_model_manage { action: "prewarm" }`.
#[derive(Debug, Default)]
pub struct PrewarmGuard {
    last_attempt_ms: Option<u64>,
    pub suppressed: Arc<AtomicBool>,
}

impl PrewarmGuard {
    pub fn should_fire(
        &mut self,
        now_ms: u64,
        privacy_locked: bool,
        busy: bool,
        ram_mb: Option<u64>,
    ) -> bool {
        if self.suppressed.load(Ordering::Acquire) || privacy_locked || busy {
            return false;
        }
        if ram_mb.is_some_and(|mb| mb < MIN_PREWARM_AVAIL_RAM_MB) {
            return false;
        }
        if self
            .last_attempt_ms
            .is_some_and(|t| now_ms.saturating_sub(t) < PREWARM_DEBOUNCE_MS)
        {
            return false;
        }
        self.last_attempt_ms = Some(now_ms);
        true
    }
}

/// 4-gate outcome vote transmitter (`pd_vote { kind: "vote_outcome" }`), holding votes in a bounded
/// queue while signed out.
#[derive(Debug, Default)]
pub struct OutcomeVoteGate {
    pub consent_share_outcomes: bool,
    pub signed_in: bool,
    pub held_votes: VecDeque<(String, i32)>,
}

impl OutcomeVoteGate {
    pub fn new(consent_share_outcomes: bool) -> Self {
        Self {
            consent_share_outcomes,
            signed_in: false,
            held_votes: VecDeque::new(),
        }
    }

    /// Evaluates the 4 gates (`consent && remote && eval_id && decisive && credit >= 0.8 && eff >= 0.8`).
    /// Returns `Some((eval_id, sign))` when ready to transmit immediately, or queues into `held_votes`
    /// while signed out.
    pub fn consider(
        &mut self,
        eval_id: Option<&str>,
        is_remote: bool,
        decisive: bool,
        credit: f64,
        effective_weight: f64,
        sign: i32,
    ) -> Option<(String, i32)> {
        if !self.consent_share_outcomes
            || !is_remote
            || !decisive
            || sign == 0
            || credit < 0.8
            || effective_weight < 0.8
        {
            return None;
        }
        let Some(id) = eval_id.filter(|s| !s.is_empty()) else {
            return None;
        };
        if self.signed_in {
            Some((id.to_string(), sign.signum()))
        } else {
            if let Some(pos) = self.held_votes.iter().position(|(k, _)| k == id) {
                self.held_votes.remove(pos);
            }
            if self.held_votes.len() >= MAX_HELD_OUTCOME_VOTES {
                self.held_votes.pop_front();
            }
            self.held_votes.push_back((id.to_string(), sign.signum()));
            None
        }
    }

    pub fn drain_ready_on_sign_in(&mut self) -> Vec<(String, i32)> {
        if !self.signed_in || !self.consent_share_outcomes {
            return Vec::new();
        }
        self.held_votes.drain(..).collect()
    }
}

// -------------------------------------------------------------------------------------------------
// LocalVerdictStore & Stage 2 Drawer View
// -------------------------------------------------------------------------------------------------

/// One row shown in the overlay's Stage 2 history/verdict drawer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawerItem {
    pub record_id: String,
    pub cc: String,
    pub label: String,
    pub sync_state: String,
    pub pair: VerdictPairView,
    pub pinned: Option<UserPin>,
}

/// Snapshot of sync status + Stage 1 rejections + Stage 2 drawer rows sent from pipeline → overlay.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OverlaySyncView {
    pub status_dot: Option<String>,
    pub auth_required: bool,
    pub offline: bool,
    pub queued_count: usize,
    pub pending_count: usize,
    pub votes_remaining: usize,
    pub unacked_rejections: Vec<RejectionNotice>,
    pub drawer_items: Vec<DrawerItem>,
}

#[derive(Debug, Clone)]
pub struct LocalRecordState {
    pub record_id: String,
    pub server_id: Option<String>,
    pub cc: String,
    #[allow(dead_code)]
    pub to: Option<String>,
    pub label: String,
    pub sync_state: String,
    pub obs: Vec<Obs>,
    pub entry: LocalVerdictEntry,
    pub pin: Option<UserPin>,
    pub pin_set_at_ms: i64,
    pub wins: u32,
    pub losses: u32,
    pub global: Option<(Tier, Option<i32>, Option<(u32, u32)>, String)>,
    /// Phase 4 Part 2 & Phase 5 Part 1: `"eval"` (공략 노트, 90일 반감기) 또는 `"nav"` (체크포인트·지름길, 14일 반감기).
    pub track: String,
}

/// Local PD-Verdict (`pd-verdict/2`) store, cross-trial duels, and reversible `▲`/`▼` manual pins.
#[derive(Debug, Default)]
pub struct LocalVerdictStore {
    pub records: BTreeMap<String, LocalRecordState>,
    pub scope_stat: ScopeStat,
    pub cross_trial: CrossTrialMemory,
}

impl LocalVerdictStore {
    pub fn ensure_record(
        &mut self,
        record_id: &str,
        cc: &str,
        to: Option<&str>,
        label: &str,
        sync_state: &str,
    ) -> &mut LocalRecordState {
        self.records
            .entry(record_id.to_string())
            .or_insert_with(|| LocalRecordState {
                record_id: record_id.to_string(),
                server_id: None,
                cc: normalize_cc(cc),
                to: to.map(str::to_string),
                label: prompt_label(label),
                sync_state: sync_state.to_string(),
                obs: Vec::new(),
                entry: LocalVerdictEntry::default(),
                pin: None,
                pin_set_at_ms: 0,
                wins: 0,
                losses: 0,
                global: None,
                track: "eval".to_string(),
            })
    }

    /// Phase 4 Part 2 (`CN2-51`): 개인 지름길 승리(`ShortcutWinV2`)를 로컬 `nav` 트랙 + 대결 원장에 기록한다.
    /// 동일 갈림길에서 3연승(`outperforms` 활성) 달성 시 즉시 로컬 추천(`Premium`)으로 승격해
    /// Stage 2 서랍과 가역적 `[▲]/[▼]` 핀 조작에 노출한다.
    #[allow(clippy::too_many_arguments)]
    pub fn ingest_shortcut_win(
        &mut self,
        cc: &str,
        fork_group: &str,
        taken_cp: &str,
        skipped_cp: &str,
        label: &str,
        saved_ms: u32,
        now_ms: i64,
        quarantine: &QuarantineLedger,
    ) -> (bool, VerdictPairView) {
        let params = local_params();
        let rec = self.ensure_record(taken_cp, cc, Some(fork_group), label, "local");
        rec.track = "nav".to_string();
        rec.wins += 1;
        rec.obs.push(Obs {
            kind: "outcome".to_string(),
            sign: 1,
            weight: 1.0,
            credit: 1.0,
            at_ms: now_ms,
        });
        // `ScopeStat.confusion` 에 (taken_cp vs skipped_cp) 갈림길 대결 승리를 직접 누적한다.
        let (key, taken_is_a) = if taken_cp <= skipped_cp {
            (format!("{taken_cp}|{skipped_cp}"), true)
        } else {
            (format!("{skipped_cp}|{taken_cp}"), false)
        };
        let conf = self.scope_stat.confusion.entry(key).or_default();
        conf.ties += 1;
        if taken_is_a {
            conf.a_wins += 1;
        } else {
            conf.b_wins += 1;
        }
        conf.margin.push(f64::from(saved_ms), 32);
        self.scope_stat.updated_ms = self.scope_stat.updated_ms.max(now_ms);
        let promoted = outperforms(&self.scope_stat, taken_cp, skipped_cp, &params);
        let view = self.pair_view(taken_cp, now_ms, quarantine);
        (promoted, view)
    }

    pub fn ingest_sync_snapshot(&mut self, snapshot: &SyncSnapshot) {
        for (req_id, row) in &snapshot.rows {
            let state = row.get("state").and_then(Value::as_str).unwrap_or("queued");
            let cc = row.get("cc").and_then(Value::as_str).unwrap_or("");
            if cc.is_empty() {
                continue;
            }
            let text = row.get("text").and_then(Value::as_str).unwrap_or("");
            let server_id = row
                .get("server_id")
                .or_else(|| row.get("id"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string);
            let id = server_id.clone().unwrap_or_else(|| req_id.clone());
            let to = row.get("to").and_then(Value::as_str);
            let rec = self.ensure_record(&id, cc, to, text, state);
            rec.server_id = server_id;
            rec.sync_state = state.to_string();
            if !text.is_empty() {
                rec.label = prompt_label(text);
            }
        }
    }

    /// Reversible `▲`/`▼` manual pin (FC-09): overrides statistics immediately, releases on second
    /// click, records `demoted` in `ScopeStat` on `▼`, and produces ZERO outbound server packets.
    pub fn toggle_pin(
        &mut self,
        record_id: &str,
        clicked: UserPin,
        now_ms: i64,
        quarantine: &QuarantineLedger,
    ) -> Option<VerdictPairView> {
        let params = local_params();
        let rec = self.records.get_mut(record_id)?;
        let was_demoted = rec.pin == Some(UserPin::Demoted);
        let res = toggle_pin(&mut rec.obs, rec.pin, clicked, now_ms, &params);
        rec.pin = res.pin;
        if res.pin.is_some() {
            rec.pin_set_at_ms = now_ms;
        }
        let now_demoted = res.pin == Some(UserPin::Demoted);
        if now_demoted != was_demoted {
            record_demotion(&mut self.scope_stat, record_id, now_demoted, now_ms);
        }
        Some(self.pair_view(record_id, now_ms, quarantine))
    }

    /// Applies spoken pragmatic feedback (FC-01..04) to the target record's local observation ledger.
    pub fn apply_spoken_feedback(
        &mut self,
        record_id: &str,
        utterance: &str,
        tokens: &[FeedbackToken<'_>],
        ctx: &FeedbackContext,
        credit: f64,
        now_ms: i64,
    ) -> Option<ExtractedFeedback> {
        let extracted = extract_pragmatic_feedback(utterance, tokens, "ko", *ctx)?;
        let params = local_params();
        if let Some(rec) = self.records.get_mut(record_id) {
            apply_feedback_to_ledger(
                &mut rec.obs,
                &extracted,
                credit,
                now_ms,
                RETRACT_WINDOW_MS,
                &params,
            );
        }
        Some(extracted)
    }

    /// Distributes outcome / feedback credit across active candidates and updates cross-trial duels.
    pub fn apply_outcome_credit(
        &mut self,
        candidates: &[AttributionCandidate],
        sign: i32,
        now_ms: i64,
        shielded: bool,
        vote_gate: &mut OutcomeVoteGate,
    ) -> Vec<(String, i32)> {
        let params = local_params();
        let obs_kind = "outcome";
        let report = attribute_and_update_duels(
            obs_kind,
            sign,
            1.0,
            shielded,
            candidates,
            &mut self.scope_stat,
            self.cross_trial.failures(),
            now_ms,
            &params,
        );
        self.cross_trial.remember(&report, candidates, now_ms);
        let mut outbound = Vec::new();
        let decisive = report.shares.len() == 1;
        for share in &report.shares {
            if !matches!(
                share.target_kind,
                CreditTargetKind::RagRecord | CreditTargetKind::CourseCheckpoint
            ) {
                continue;
            }
            if let Some(rec) = self.records.get_mut(&share.target_id) {
                rec.obs.push(share.obs.clone());
                if sign > 0 {
                    rec.wins += 1;
                } else if sign < 0 {
                    rec.losses += 1;
                }
                // 서버 vote_outcome 은 원격 평가 서버 ID 가 있는 RagRecord 에만 전송(로컬 코스/체크포인트 격리)
                if share.target_kind == CreditTargetKind::RagRecord {
                    let eff_w = share.obs.weight * share.credit;
                    if let Some(vote) = vote_gate.consider(
                        rec.server_id.as_deref(),
                        rec.server_id.is_some(),
                        decisive,
                        share.credit,
                        eff_w,
                        sign,
                    ) {
                        outbound.push(vote);
                    }
                }
            }
        }
        outbound
    }

    pub fn local_grade(&mut self, record_id: &str, now_ms: i64, quarantine: &QuarantineLedger) -> Grade {
        let view = self.pair_view(record_id, now_ms, quarantine);
        tier_to_grade(view.local_tier)
    }

    /// Computes non-blended `VerdictPairView` (`pd-verdict/2` local vs server recipe global).
    pub fn pair_view(
        &mut self,
        record_id: &str,
        now_ms: i64,
        quarantine: &QuarantineLedger,
    ) -> VerdictPairView {
        let params = local_params();
        let Some(rec) = self.records.get_mut(record_id) else {
            return VerdictPairView::resolve(None, None, None, None, None);
        };
        let standing = |_id: &str| Standing::default();
        let active_sup = quarantine.active_superseder(&rec.cc, record_id, standing);
        let superseded = active_sup.is_some();
        let pin_superseded_after = rec.pin == Some(UserPin::Promoted)
            && active_sup.is_some_and(|e| e.new_at_ms > rec.pin_set_at_ms);
        let duel = duel_for_item(&self.scope_stat, record_id);
        let track = if rec.track.is_empty() {
            "eval".to_string()
        } else {
            rec.track.clone()
        };
        let case = Case {
            now_ms,
            track,
            prev: rec.entry.prev,
            obs: rec.obs.clone(),
            baseline: Baselines::default(),
            duel,
            superseded,
        };
        let judged = rec.entry.judge(&case, &params).map(|(v, _)| v);
        let recall_in = rec.entry.recall_input(&rec.obs);
        let user_pin = rec.pin.map(|p| (p, pin_superseded_after));
        let wl = (rec.wins + rec.losses > 0).then_some((rec.wins, rec.losses));
        let global: Option<GlobalVerdict<'_>> = rec
            .global
            .as_ref()
            .map(|(t, g, ud, recipe)| (*t, *g, *ud, recipe.as_str()));
        VerdictPairView::resolve(judged.as_ref(), recall_in, user_pin, wl, global)
    }

    pub fn drawer_items(
        &mut self,
        cc: Option<&str>,
        now_ms: i64,
        quarantine: &QuarantineLedger,
    ) -> Vec<DrawerItem> {
        let norm_cc = cc.map(normalize_cc);
        let ids: Vec<String> = self
            .records
            .values()
            .filter(|r| !matches!(r.sync_state.as_str(), "rejected" | "failed"))
            .filter(|r| norm_cc.as_deref().is_none_or(|c| r.cc == c))
            .map(|r| r.record_id.clone())
            .collect();
        let mut items = Vec::new();
        for id in ids.into_iter().take(12) {
            let pair = self.pair_view(&id, now_ms, quarantine);
            if let Some(rec) = self.records.get(&id) {
                items.push(DrawerItem {
                    record_id: rec.record_id.clone(),
                    cc: rec.cc.clone(),
                    label: rec.label.clone(),
                    sync_state: rec.sync_state.clone(),
                    pair,
                    pinned: rec.pin,
                });
            }
        }
        items
    }
}

// -------------------------------------------------------------------------------------------------
// Drop result & PdBridge facade
// -------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DropOutcome {
    BlockedPii { reaction: String },
    BlockedProfanity,
    Enqueued { request_id: String, cc: String },
}

#[derive(Serialize, Deserialize)]
struct PersistedQuarantineEntry {
    cc: String,
    old_id: String,
    new_id: String,
    old_to: Option<String>,
    new_to: Option<String>,
    old_at_ms: i64,
    new_at_ms: i64,
    value_diff: Option<(String, String)>,
    recorded_at_ms: i64,
    sync_state: String,
}

pub fn load_quarantine(path: &Path) -> QuarantineLedger {
    let Ok(raw) = std::fs::read_to_string(path) else {
        return QuarantineLedger::new();
    };
    let Ok(rows) = serde_json::from_str::<Vec<PersistedQuarantineEntry>>(&raw) else {
        return QuarantineLedger::new();
    };
    let entries = rows
        .into_iter()
        .filter_map(|r| {
            Some(QuarantineEntry {
                cc: r.cc,
                old_id: r.old_id,
                new_id: r.new_id,
                old_to: r.old_to,
                new_to: r.new_to,
                old_at_ms: r.old_at_ms,
                new_at_ms: r.new_at_ms,
                value_diff: r.value_diff,
                recorded_at_ms: r.recorded_at_ms,
                sync_state: QuarantineSyncState::from_name(&r.sync_state)?,
            })
        })
        .collect();
    QuarantineLedger::from_entries(entries)
}

pub fn save_quarantine(path: &Path, ledger: &QuarantineLedger) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let rows: Vec<PersistedQuarantineEntry> = ledger
        .entries()
        .iter()
        .map(|e| PersistedQuarantineEntry {
            cc: e.cc.clone(),
            old_id: e.old_id.clone(),
            new_id: e.new_id.clone(),
            old_to: e.old_to.clone(),
            new_to: e.new_to.clone(),
            old_at_ms: e.old_at_ms,
            new_at_ms: e.new_at_ms,
            value_diff: e.value_diff.clone(),
            recorded_at_ms: e.recorded_at_ms,
            sync_state: e.sync_state.as_str().to_string(),
        })
        .collect();
    raon_infer::device::write_file_atomic(path, &serde_json::to_vec_pretty(&rows)?)?;
    Ok(())
}

/// Searches for a sibling or PATH `premium-discount-mcp` executable.
pub fn find_pd_mcp() -> Option<PathBuf> {
    let exe_name = if cfg!(windows) {
        "premium-discount-mcp.exe"
    } else {
        "premium-discount-mcp"
    };
    if let Ok(cur) = std::env::current_exe() {
        if let Some(dir) = cur.parent() {
            let sibling = dir.join(exe_name);
            if sibling.is_file() {
                return Some(sibling);
            }
        }
    }
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let p = dir.join(exe_name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// Complete app-level bridge to `premium-discount`: sync loop, 600 ms RAG, drop intake, quarantine,
/// local verdict store, pre-warm guard, and `pd_analytic`.
pub struct PdBridge {
    pub client: Option<Arc<Mutex<PdSyncClient>>>,
    pub sync_loop: Option<SyncLoop>,
    pub filter: LocalFilter,
    pub rag: Arc<RagCoordinator>,
    pub verdicts: LocalVerdictStore,
    pub prewarm: PrewarmGuard,
    pub vote_gate: OutcomeVoteGate,
    pub quarantine_path: PathBuf,
    /// Phase 4 Part 3: 웹·인게임·네이티브 통합 선택지 편향 제거 장부. 배지 선택([`PdBridge::on_badge_picked`])과 그 뒤 1-hop
    /// 역류([`PdBridge::on_screen_changed`])를 넣고, 판정은 실시간 배지 등급([`PdBridge::live_web_rows_with_choices`])에 들어간다.
    pub choice_book: UniversalChoiceBook,
    /// Phase 5 Part 2: 64-슬롯 구역별 Pinned-SLRU 로컬 코스 보관함. 앱 · 사이트를 떠날 때 모은 체크포인트를 런으로 닫아 넣는다
    /// ([`PdBridge::on_app_left`]).
    pub zone_vault: ZoneVault,
    /// 1-hop 결과를 기다리는 배지 선택
    pending_pick: Option<PendingPick>,
    /// 지금 앱 · 사이트에서 모으는 체크포인트
    live_run: LiveRunBuffer,
    /// Phase 6 Part 1: 크로스-앱 메타 코스 궤적 원장.
    pub trajectory: TrajectoryLedger,
    /// Phase 4 Part 3 & Phase 6 Part 3: 동기화된 공개 타임트라이얼 체크포인트 / 선행 승격 갈림길 행.
    pub public_trial_rows: BTreeMap<String, Vec<WebAnalyticTierRow>>,
    last_snapshot: SyncSnapshot,
    local_seq: u64,
}

impl PdBridge {
    pub fn new(
        mcp_path: Option<&Path>,
        embedder: SharedEmbedder,
        ask_epoch: Arc<AtomicU64>,
        privacy_epoch: Arc<AtomicU64>,
        consent_share_outcomes: bool,
        quarantine_path: PathBuf,
    ) -> Self {
        let mirror = Arc::new(Mutex::new(LocalMirrorIndex::new(embedder)));
        let quarantine = Arc::new(Mutex::new(load_quarantine(&quarantine_path)));

        let mut client_opt = None;
        let mut sync_loop_opt = None;
        let mut deep_opt: Option<Arc<dyn DeepSearcher>> = None;

        if let Some(bin) = mcp_path {
            match PdSyncClient::spawn(bin, &[]) {
                Ok(c) => {
                    eprintln!("[pd-mcp] connected to {}", bin.display());
                    let shared = Arc::new(Mutex::new(c));
                    let mirror_cb = mirror.clone();
                    let loop_handle = SyncLoop::start(
                        shared.clone(),
                        SYNC_INTERVAL,
                        Some(Box::new(move |snap| {
                            if let Ok(mut m) = mirror_cb.lock() {
                                m.apply_sync_snapshot(snap);
                            }
                        })),
                    );
                    loop_handle.request(false);
                    deep_opt = Some(Arc::new(McpDeepSearcher::new(shared.clone())));
                    client_opt = Some(shared);
                    sync_loop_opt = Some(loop_handle);
                }
                Err(e) => {
                    eprintln!("[pd-mcp] unavailable ({e:#}); running local-mirror offline mode");
                }
            }
        }

        let rag = Arc::new(RagCoordinator::new(
            ask_epoch,
            privacy_epoch,
            mirror,
            quarantine,
            deep_opt,
        ));
        let zone_vault = ZoneVault::load(&zone_vault_path(&quarantine_path));

        Self {
            client: client_opt,
            sync_loop: sync_loop_opt,
            filter: LocalFilter::default(),
            rag,
            verdicts: LocalVerdictStore::default(),
            prewarm: PrewarmGuard::default(),
            vote_gate: OutcomeVoteGate::new(consent_share_outcomes),
            quarantine_path,
            choice_book: UniversalChoiceBook::default(),
            zone_vault,
            pending_pick: None,
            live_run: LiveRunBuffer::default(),
            trajectory: TrajectoryLedger::default(),
            public_trial_rows: BTreeMap::new(),
            last_snapshot: SyncSnapshot::default(),
            local_seq: 0,
        }
    }

    /// 선택 장부의 열쇠: 같은 라벨이라도 앱 · 사이트마다 따로 센다
    fn choice_key(cc: &str, label: &str) -> String {
        format!("{}|{}", normalize_cc(cc), label.trim().to_lowercase())
    }

    /// 사용자가 배지를 골랐다 (`OverlayAction::BadgePicked`). 보이는 배지 풀과 함께 선택 장부에 넣고(목록이면 위치 감쇠 사전분포),
    /// 도착 화면을 떠날 때 1-hop 결과를 넣으려고 기다린다. 고른 요소는 이번 런의 체크포인트다
    #[allow(clippy::too_many_arguments)]
    pub fn on_badge_picked(&mut self, cc: &str, screen: &str, anchor: &str, label: &str, visible_labels: &[&str], vertical_list: bool, now_ms: i64) {
        if label.trim().is_empty() {
            return;
        }
        let key = Self::choice_key(cc, label);
        let keys: Vec<String> = visible_labels.iter().map(|l| Self::choice_key(cc, l)).collect();
        let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        self.choice_book.observe_pick(&key, &refs, vertical_list);
        self.pending_pick = Some(PendingPick { key, pick_screen: screen.to_string(), same_screen_friction: None, arrived: None });
        if let Some(h) = element_hash(screen, anchor) {
            self.push_checkpoint(cc, Checkpoint::new(now_ms, StepKind::Element, h, Some(Gesture::Press)));
        }
    }

    /// 화면이 바뀌었다 (`done` = 떠난 화면의 끝난 에피소드). 고른 화면을 떠났으면 그 화면의 마찰을 적고 도착을 기다린다. 도착 화면을
    /// 떠났으면 머문 시간 · 되돌아옴(8초 안에 고른 화면으로) · 도착 화면 마찰로 1-hop 결과를 넣는다. 새 화면은 체크포인트다
    pub fn on_screen_changed(&mut self, cc: &str, done_screen: &str, done: Option<&Episode>, next_screen: &str, p: &BehaviorParams, now_ms: i64) {
        let friction = done.map_or(0.0, |e| e.friction(p));
        let settled = match self.pending_pick.as_mut() {
            Some(pp) if pp.arrived.is_none() && pp.pick_screen == done_screen => {
                pp.same_screen_friction = Some(friction);
                pp.arrived = Some((next_screen.to_string(), now_ms));
                None
            }
            Some(pp) if pp.arrived.as_ref().is_some_and(|(arrival, _)| arrival == done_screen) => {
                let at = pp.arrived.as_ref().map_or(now_ms, |(_, at)| *at);
                let dwell = now_ms - at;
                let bounce = next_screen == pp.pick_screen && dwell <= CROSS_BOUNDARY_BOUNCE_WINDOW_MS;
                Some((pp.key.clone(), pp.same_screen_friction.unwrap_or(0.0), dwell, bounce))
            }
            _ => None,
        };
        if let Some((key, same, dwell, bounce)) = settled {
            self.choice_book.observe_post_pick_1hop(&key, same, Some(dwell), bounce, friction);
            self.pending_pick = None;
        }
        if !next_screen.trim().is_empty() {
            self.push_checkpoint(cc, Checkpoint::new(now_ms, StepKind::Screen, next_screen.to_string(), None));
        }
    }

    /// 도착 화면에서 졌다 (게임 패배): 고른 배지의 1-hop 결과를 역류로 넣는다
    pub fn on_defeat(&mut self, now_ms: i64) {
        if let Some(pp) = self.pending_pick.take() {
            match pp.arrived {
                Some((_, at)) => self.choice_book.observe_post_pick_1hop(&pp.key, pp.same_screen_friction.unwrap_or(0.0), Some(now_ms - at), true, 0.0),
                None => self.pending_pick = Some(pp),
            }
        }
    }

    fn push_checkpoint(&mut self, cc: &str, cp: Checkpoint) {
        let cc = normalize_cc(cc);
        if self.live_run.cc != cc {
            self.live_run = LiveRunBuffer { cc, checkpoints: Vec::new() };
        }
        if self.live_run.checkpoints.len() >= LIVE_RUN_MAX_CHECKPOINTS {
            self.live_run.checkpoints.remove(0);
        }
        self.live_run.checkpoints.push(cp);
    }

    /// 앱 · 사이트를 떠났다 (또는 세션 끝): 모은 체크포인트를 런으로 닫아(개인정보 잠금 · 기다림 동결 구간을 뺀다) 64-슬롯
    /// `ZoneVault` 에 넣는다. 서버 공개 코스가 30일 뒤 사라져도 이 기기에서 그 코스를 다시 안내할 수 있다. 닫은 런을 돌려준다
    pub fn on_app_left(&mut self, frozen_spans: &[(i64, i64)]) -> Option<NavRun> {
        let buf = std::mem::take(&mut self.live_run);
        self.pending_pick = None;
        let app = nav_app_hash(&buf.cc)?;
        let opts = ReduceOptions { frozen_spans, ..ReduceOptions::default() };
        let run = NavRun::from_checkpoints_with(&app, &buf.checkpoints, None, &opts).ok()?;
        self.zone_vault.record_run(&run);
        if let Err(e) = self.zone_vault.save(&zone_vault_path(&self.quarantine_path)) {
            eprintln!("[pd] zone vault not saved: {e:#}");
        }
        Some(run)
    }

    /// 선택 장부의 판정을 보이는 라벨에 대해 웹 행으로 (아직 표본이 모자란 라벨은 빼고, 함정은 `is_friction_trap`)
    pub fn choice_rows_for(&self, cc: &str, labels: &[&str]) -> Vec<WebAnalyticTierRow> {
        labels
            .iter()
            .enumerate()
            .filter_map(|(i, label)| {
                let kind = self.choice_book.classify(&Self::choice_key(cc, label));
                (kind != UniversalChoiceKind::UnratedCold).then(|| WebAnalyticTierRow {
                    label: label.to_string(),
                    role: NavRole::Button,
                    ordinal_norm: (i as f32 / labels.len().max(1) as f32).clamp(0.0, 1.0),
                    tier: kind.tier(),
                    is_friction_trap: kind == UniversalChoiceKind::FrictionTrap,
                })
            })
            .collect()
    }

    /// [`PdBridge::live_web_rows_for_cc`] 에 선택 장부를 더한 것: 판정 · 공개 체크포인트가 먼저이고, 선택 장부는 빈 라벨을 채우며,
    /// 함정 표시는 어느 소스에서 왔든 지킨다
    pub fn live_web_rows_with_choices(&mut self, cc: &str, labels: &[&str], now_ms: i64) -> Vec<WebAnalyticTierRow> {
        let rows = self.live_web_rows_for_cc(cc, now_ms);
        build_live_web_rows(&rows, &self.choice_rows_for(cc, labels), &[])
    }

    /// Phase 4 Part 3 (`CN2-55`) & Phase 6 Part 3 (`CN2-78`):
    /// 로컬 `LocalVerdictStore` + 동기화된 `public_trial_rows` 를 병합해 `build_web_nav_overlay` 에 즉시 투입할
    /// `WebAnalyticTierRow` 목록을 반환한다. 선택 장부까지 더한 것은 [`PdBridge::live_web_rows_with_choices`]
    pub fn live_web_rows_for_cc(&mut self, cc: &str, now_ms: i64) -> Vec<WebAnalyticTierRow> {
        let norm_cc = normalize_cc(cc);
        let q = self.rag.quarantine.lock().unwrap().clone();
        let rec_ids: Vec<String> = self
            .verdicts
            .records
            .values()
            .filter(|r| r.cc == norm_cc && !r.label.trim().is_empty())
            .map(|r| r.record_id.clone())
            .collect();
        let mut local_rows = Vec::new();
        for (idx, id) in rec_ids.iter().enumerate() {
            let pair = self.verdicts.pair_view(id, now_ms, &q);
            if pair.local_tier == Tier::Unrated {
                continue;
            }
            if let Some(rec) = self.verdicts.records.get(id) {
                local_rows.push(WebAnalyticTierRow {
                    label: rec.label.clone(),
                    role: NavRole::Button,
                    ordinal_norm: (idx as f32 / rec_ids.len().max(1) as f32).clamp(0.0, 1.0),
                    tier: pair.local_tier,
                    is_friction_trap: pair.local_tier == Tier::Discount,
                });
            }
        }
        let pub_rows = self
            .public_trial_rows
            .get(&norm_cc)
            .cloned()
            .unwrap_or_default();
        build_live_web_rows(&local_rows, &pub_rows, &[])
    }

    /// 0 ms PII + profanity filter before waking child app, optimistic mirror insert, and enqueue.
    pub fn enqueue_drop(
        &mut self,
        raw_cc: &str,
        to: Option<&str>,
        raw_text: &str,
        now_ms: i64,
    ) -> DropOutcome {
        let scrubber = IngressPiiScrubber::with_seed_detector(&detect_seed_phrase_checked);
        let mut buf = raw_text.to_string();
        let text = match scrubber.inspect_and_scrub(&mut buf) {
            IngressVerdict::HardBlocked { kind } => {
                return DropOutcome::BlockedPii {
                    reaction: kind.warning_line().to_string(),
                };
            }
            IngressVerdict::Scrubbed { sanitized_text, .. } => sanitized_text,
            IngressVerdict::Clean(clean) => clean,
        };
        if self.filter.judge(&text) == FilterVerdict::Profane {
            return DropOutcome::BlockedProfanity;
        }
        let cc = envelope::normalize_cc(raw_cc)
            .map(|c| c.cc)
            .unwrap_or_else(|_| normalize_cc(raw_cc));
        if cc.is_empty() {
            return DropOutcome::BlockedProfanity;
        }
        self.local_seq += 1;
        let mut request_id = format!("local-{}-{now_ms}", self.local_seq);
        if let Some(client) = &self.client {
            if let Some(mut guard) = try_lock_for(client, Duration::from_millis(250)) {
                let mut payload = json!({
                    "cc": cc,
                    "text": text,
                });
                if let Some(t) = to {
                    payload["to"] = json!(t);
                }
                if let Ok(resp) = guard.enqueue(payload) {
                    if let Some(id) = resp
                        .get("local_id")
                        .or_else(|| resp.get("id"))
                        .and_then(Value::as_str)
                    {
                        request_id = id.to_string();
                    }
                }
            }
        }
        if let Ok(mut m) = self.rag.mirror.lock() {
            m.enqueue_optimistic(&request_id, &cc, to, &text, now_ms);
        }
        self.verdicts
            .ensure_record(&request_id, &cc, to, &text, "queued");
        if let Some(sl) = &self.sync_loop {
            sl.request(true);
        }
        DropOutcome::Enqueued { request_id, cc }
    }

    /// Runs cross-scope supersession curation across mirrored notes of `cc`, recording any
    /// supersessions into `QuarantineLedger` and reporting eligible same-scope pairs to `pd_supersede`.
    pub fn curate_cc(&mut self, cc: &str, now_ms: i64) -> usize {
        let (notes, embedder) = {
            let m = self.rag.mirror.lock().unwrap();
            (m.notes_in_cc(cc).to_vec(), m.shared_embedder())
        };
        if notes.len() < 2 {
            return 0;
        }
        let records: Vec<RelayRecord> = notes
            .iter()
            .map(|n| RelayRecord {
                at_ms: n.created_at_ms,
                sentences: vec![simple_relay_sentence(&n.text)],
            })
            .collect();
        let entries: Vec<RelayEntry<'_>> = notes
            .iter()
            .zip(&records)
            .map(|(n, r)| RelayEntry {
                id: n.server_id.as_deref().unwrap_or(&n.request_id),
                cc: &n.cc,
                to: n.to.as_deref(),
                record: r,
            })
            .collect();
        let report = curate_cross_scope(&entries, &embedder, &RelayConfig::default());
        let mut added = 0usize;
        if let Ok(mut q) = self.rag.quarantine.lock() {
            for s in &report.supersessions {
                let facts = SupersedeFacts {
                    old_mine: true,
                    new_mine: true,
                    signed_in: self.vote_gate.signed_in,
                    sharing_consent: self.vote_gate.consent_share_outcomes,
                    old_stored: notes.iter().any(|n| n.server_id.as_deref() == Some(&s.old_id) && n.state == "stored"),
                    new_stored: notes.iter().any(|n| n.server_id.as_deref() == Some(&s.new_id) && n.state == "stored"),
                };
                let sync = supersede_sync(s.old_to.as_deref(), s.new_to.as_deref(), &facts);
                let entry = QuarantineEntry::from_supersession(s, sync.into(), now_ms);
                if q.record(entry) {
                    added += 1;
                }
            }
            if added > 0 {
                let _ = save_quarantine(&self.quarantine_path, &q);
            }
            // Drain ready same-scope server reports.
            let ready = q.drain_ready_server_reports(
                |id| {
                    notes
                        .iter()
                        .find(|n| n.server_id.as_deref() == Some(id) && n.state == "stored")
                        .and_then(|n| n.server_id.clone())
                },
                |_| Standing::default(),
            );
            if let Some(client) = &self.client {
                for (old, new) in ready {
                    if let Some(mut guard) = try_lock_for(client, Duration::from_millis(150)) {
                        let _ = guard.call("pd_supersede", json!({ "old": old, "new": new }));
                    }
                }
            }
        }
        added
    }

    /// Triggers non-blocking speculative pre-warming on `FrictionAlertStage::PreWarm` (PF-08 / PF-09).
    pub fn maybe_prewarm(&mut self, now_ms: u64, privacy_locked: bool, busy: bool) {
        let ram = avail_ram_mb();
        if !self.prewarm.should_fire(now_ms, privacy_locked, busy, ram) {
            return;
        }
        let Some(client) = self.client.clone() else {
            return;
        };
        let suppressed = self.prewarm.suppressed.clone();
        std::thread::Builder::new()
            .name("zena-pd-prewarm".into())
            .spawn(move || {
                if let Some(mut guard) = try_lock_for(&client, Duration::from_millis(100)) {
                    let mut args = json!({ "action": "prewarm" });
                    if let Some(mb) = ram {
                        args["avail_ram_mb"] = json!(mb);
                    }
                    if let Ok(resp) = guard.call("pd_model_manage", args) {
                        if resp.get("outcome").and_then(Value::as_str) == Some("suppressed") {
                            suppressed.store(true, Ordering::Release);
                        }
                    }
                }
            })
            .ok();
    }

    /// Sends an eligible outcome vote via `pd_vote` and notifies `SyncLoop::notify_vote_queued`
    /// if the vote was queued offline (FC-12).
    pub fn send_outcome_vote(&self, eval_id: &str, sign: i32) {
        let Some(client) = &self.client else {
            return;
        };
        if let Some(mut guard) = try_lock_for(client, Duration::from_millis(150)) {
            if let Ok(resp) = guard.call(
                "pd_vote",
                json!({
                    "id": eval_id,
                    "value": sign.signum(),
                    "kind": "vote_outcome",
                }),
            ) {
                if resp.get("status").and_then(Value::as_str) == Some("queued") {
                    if let Some(sl) = &self.sync_loop {
                        sl.notify_vote_queued();
                    }
                }
            }
        }
    }

    /// Calls the 11th MCP tool `pd_analytic { action: "report" }` and maps the JSON response into
    /// `AnalyticBrief` (with `AnalyticBrief::sample_sufficient` guard, WA-11).
    pub fn query_analytic(&self, host: &str, query: &str) -> Option<AnalyticBrief> {
        let client = self.client.as_ref()?;
        let mut guard = try_lock_for(client, Duration::from_millis(450))?;
        let resp = guard
            .call(
                "pd_analytic",
                json!({
                    "action": "report",
                    "host": host,
                    "query": query,
                    "lang": "ko",
                }),
            )
            .ok()?;
        parse_analytic_brief(host, &resp)
    }

    /// Acknowledges unseen rejections when the user opens Stage 1 chip or Stage 2 drawer (E-16).
    pub fn ack_rejections(&mut self) -> Vec<RejectionNotice> {
        if let Some(sl) = &self.sync_loop {
            let drained = sl.ack_rejections();
            self.last_snapshot.unacked_rejections.clear();
            drained
        } else {
            self.last_snapshot.drain_unacked_rejections()
        }
    }

    /// Polls latest `SyncSnapshot`, refreshes `LocalVerdictStore`, and builds `OverlaySyncView`.
    pub fn poll_overlay_view(&mut self, cc: Option<&str>, now_ms: i64) -> OverlaySyncView {
        if let Some(sl) = &self.sync_loop {
            let snap = sl.snapshot();
            if snap.runs != self.last_snapshot.runs
                || snap.unacked_rejections.len() != self.last_snapshot.unacked_rejections.len()
            {
                self.verdicts.ingest_sync_snapshot(&snap);
                self.vote_gate.signed_in = !snap.auth_required && !snap.offline;
                if self.vote_gate.signed_in {
                    for (id, sign) in self.vote_gate.drain_ready_on_sign_in() {
                        self.send_outcome_vote(&id, sign);
                    }
                }
                self.last_snapshot = snap;
            }
        }
        let q = self.rag.quarantine.lock().unwrap();
        let drawer_items = self.verdicts.drawer_items(cc, now_ms, &q);
        OverlaySyncView {
            status_dot: self.last_snapshot.status_dot().map(str::to_string),
            auth_required: self.last_snapshot.auth_required,
            offline: self.last_snapshot.offline,
            queued_count: self.last_snapshot.count("queued"),
            pending_count: self.last_snapshot.count("pending"),
            votes_remaining: self.last_snapshot.votes_remaining,
            unacked_rejections: self.last_snapshot.unacked_rejections.iter().cloned().collect(),
            drawer_items,
        }
    }

    /// Evicts idle child app gracefully on shutdown.
    pub fn shutdown(&mut self) {
        if let Ok(q) = self.rag.quarantine.lock() {
            let _ = save_quarantine(&self.quarantine_path, &q);
        }
        if let Some(client) = &self.client {
            if let Some(mut guard) = try_lock_for(client, Duration::from_millis(150)) {
                let _ = guard.call("pd_model_manage", json!({ "action": "evict" }));
            }
        }
    }
}

/// Parses `pd_analytic { action: "report" }` JSON into `AnalyticBrief`.
pub fn parse_analytic_brief(host: &str, v: &Value) -> Option<AnalyticBrief> {
    let sessions = v
        .get("sessions")
        .or_else(|| v.get("funnel").and_then(|f| f.get("sessions")))
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let prev_sessions = v
        .get("prev_sessions")
        .or_else(|| v.get("compare").and_then(|c| c.get("sessions")))
        .and_then(Value::as_u64)
        .map(|n| n as u32);
    let insufficient_flag = v
        .get("insufficient_sample")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let confidence_sufficient =
        !insufficient_flag && AnalyticBrief::sample_sufficient(sessions, prev_sessions);
    let window_summary = v
        .get("window_summary")
        .or_else(|| v.get("window"))
        .and_then(Value::as_str)
        .unwrap_or("최근 기간")
        .to_string();
    let conversion_delta = confidence_sufficient
        .then(|| {
            v.get("conversion_delta")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .flatten();
    let top_friction_trap = v
        .get("top_friction_trap")
        .or_else(|| {
            v.get("bottlenecks")
                .and_then(Value::as_array)
                .and_then(|a| a.first())
                .and_then(|b| b.get("label").or_else(|| b.get("name")))
        })
        .and_then(Value::as_str)
        .map(str::to_string);
    let top_golden_path = v
        .get("top_golden_path")
        .or_else(|| {
            v.get("golden_paths")
                .and_then(Value::as_array)
                .and_then(|a| a.first())
                .and_then(|g| g.get("label").or_else(|| g.get("name")))
        })
        .and_then(Value::as_str)
        .map(str::to_string);
    Some(AnalyticBrief {
        host: normalize_cc(host),
        window_summary,
        confidence_sufficient,
        sessions,
        conversion_delta,
        top_friction_trap,
        top_golden_path,
    })
}

/// Deterministic `hashId` helper for `TopicTracker::with_hasher`.
pub fn topic_hash_id(key: &str) -> Option<String> {
    let trimmed = key.trim();
    if trimmed.is_empty() {
        None
    } else {
        hash_id(trimmed)
    }
}

fn simple_relay_sentence(text: &str) -> RelaySentence {
    let tokens = text
        .split_whitespace()
        .enumerate()
        .map(|(i, w)| {
            let upos = if w.chars().any(|c| c.is_ascii_digit()) {
                "NUM"
            } else {
                "NOUN"
            };
            RelayToken {
                text: w.to_string(),
                lemma: w.to_lowercase(),
                upos: upos.into(),
                deprel: if i == 0 { "root" } else { "obl" }.into(),
                head: usize::from(i > 0),
            }
        })
        .collect();
    RelaySentence {
        text: text.to_string(),
        tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use premium_discount_mcp::client::SyncRunner;
    use premium_discount_window::{PxRect, WindowInfo};
    use zena_core::reference::{ReferenceKind, ScopeTargets};

    struct SlowDeepSearcher {
        delay: Duration,
        payload: Value,
    }

    impl DeepSearcher for SlowDeepSearcher {
        fn search_merged(&self, _plan: &ContextualQueryPlan, _budget: Duration) -> anyhow::Result<Value> {
            std::thread::sleep(self.delay);
            Ok(self.payload.clone())
        }
    }

    struct StubSyncRunner {
        runs: Arc<AtomicU64>,
    }

    impl SyncRunner for StubSyncRunner {
        fn run(&mut self, since_ms: i64, _flush: bool) -> anyhow::Result<Value> {
            self.runs.fetch_add(1, Ordering::SeqCst);
            Ok(json!({
                "rows": [],
                "next_since_ms": since_ms + 1,
            }))
        }
    }

    fn plan(cc: &str, query: &str, l1: Option<&str>) -> ContextualQueryPlan {
        ContextualQueryPlan {
            raw_query: query.into(),
            rewritten_query: query.into(),
            kind: ReferenceKind::SelfContained,
            scopes: ScopeTargets {
                cc: normalize_cc(cc),
                l1_checkpoint_to: l1.map(str::to_string),
                l2_entity_tos: vec![],
            },
            resolved_entity: None,
            ambiguous_pair: None,
        }
    }

    /// Contract CR-05:
    /// 0 ms PreThink, SyncLoop interactive hold, 600 ms LocalMirrorIndex response containing ONLY
    /// `stored` (0 `rejected`), and late deep search result populated into `SceneRagCache`.
    #[test]
    fn cr05_prethink_600ms_mirror_fallback_and_cache_fill() {
        let ask_epoch = Arc::new(AtomicU64::new(0));
        let privacy_epoch = Arc::new(AtomicU64::new(1));
        let mirror = Arc::new(Mutex::new(LocalMirrorIndex::hashed()));
        {
            let mut snap = SyncSnapshot::default();
            snap.rows.insert(
                "local-stored".into(),
                json!({
                    "request_id": "local-stored",
                    "server_id": "srv-stored-1",
                    "cc": "eldenring.exe",
                    "text": "말레니아 2페이즈 물새난격 첫 타는 뒤로 달리고 두 번째에 앞으로 구르기",
                    "state": "stored",
                    "created_at_ms": 1000,
                }),
            );
            snap.rows.insert(
                "local-rejected".into(),
                json!({
                    "request_id": "local-rejected",
                    "server_id": "srv-rejected-1",
                    "cc": "eldenring.exe",
                    "text": "말레니아 2페이즈 물새난격 거부된 글 절대 나오면 안됨",
                    "state": "rejected",
                    "created_at_ms": 2000,
                }),
            );
            mirror.lock().unwrap().apply_sync_snapshot(&snap);
        }

        // Simulate cold-start child app taking longer than soft_deadline (600 ms vs 80 ms test budget).
        let deep = Arc::new(SlowDeepSearcher {
            delay: Duration::from_millis(220),
            payload: json!({
                "matches": [{
                    "id": "srv-deep-99",
                    "cc": "eldenring.exe",
                    "text": "말레니아 2페이즈 물새난격 냉기 항아리로 즉시 캔슬 가능",
                    "state": "stored",
                    "score": 0.91,
                    "verdict": { "tier": "premium", "grade": 2 },
                }]
            }),
        });
        let mut coord = RagCoordinator::new(
            ask_epoch,
            privacy_epoch.clone(),
            mirror,
            Arc::new(Mutex::new(QuarantineLedger::new())),
            Some(deep),
        );
        coord.soft_deadline = Duration::from_millis(80);

        let sync_runs = Arc::new(AtomicU64::new(0));
        let sync_loop = SyncLoop::start(
            StubSyncRunner {
                runs: sync_runs.clone(),
            },
            Duration::from_millis(15),
            None,
        );

        let (ptx, prx) = mpsc::channel();
        let q = plan("eldenring.exe", "말레니아 2페이즈 물새난격 어떻게 피해?", None);
        let t0 = Instant::now();
        let out = coord.retrieve(
            &q,
            Some(&sync_loop),
            Some(&ptx),
            Arc::new(|id| if id == "srv-stored-1" { Grade::Premium } else { Grade::Unrated }),
            5_000,
        );
        let elapsed = t0.elapsed();

        // 0 ms PreThink emitted immediately
        assert!(matches!(prx.try_recv(), Ok(AvatarCmd::PreThink { epoch: 1 })));
        // Returned right around soft_deadline, NOT waiting for full slow deep call
        assert!(elapsed < Duration::from_millis(200), "must not block on slow deep call: {elapsed:?}");

        match out {
            RagOutcome::Ready {
                epoch,
                health,
                slots,
                prethink_emitted,
            } => {
                assert_eq!(epoch, 1);
                assert!(prethink_emitted);
                assert_eq!(health, RagHealth::SoftDeadlineFastOnly);
                assert_eq!(slots.len(), 1, "only stored note should be returned: {slots:?}");
                assert_eq!(slots[0].record_id, "srv-stored-1");
                assert!(slots.iter().all(|s| !s.record_id.contains("rejected")));
            }
            other => panic!("expected SoftDeadlineFastOnly, got {other:?}"),
        }

        // Wait for background deep thread to finish and populate SceneRagCache
        std::thread::sleep(Duration::from_millis(250));
        let cached = coord
            .cache
            .lock()
            .unwrap()
            .get("eldenring.exe", &q.rewritten_query, 1, 5_500)
            .expect("late deep result must populate SceneRagCache");
        assert_eq!(cached[0].record_id, "srv-deep-99");
    }

    /// Contract CR-06:
    /// Rapid follow-up question B (epoch 2) cancels in-flight question A (epoch 1), and a privacy
    /// epoch bump prevents late deep results from being written to `SceneRagCache`.
    #[test]
    fn cr06_rapid_reask_cancels_old_epoch_and_privacy_epoch_isolates_cache() {
        let ask_epoch = Arc::new(AtomicU64::new(0));
        let privacy_epoch = Arc::new(AtomicU64::new(1));
        let mirror = Arc::new(Mutex::new(LocalMirrorIndex::hashed()));
        let deep = Arc::new(SlowDeepSearcher {
            delay: Duration::from_millis(180),
            payload: json!({
                "matches": [{
                    "id": "srv-late-a",
                    "cc": "eldenring.exe",
                    "text": "질문 A의 늦은 원격 결과",
                    "state": "stored",
                    "score": 0.90,
                    "verdict": { "tier": "premium", "grade": 2 },
                }]
            }),
        });
        let mut coord = RagCoordinator::new(
            ask_epoch.clone(),
            privacy_epoch.clone(),
            mirror,
            Arc::new(Mutex::new(QuarantineLedger::new())),
            Some(deep),
        );
        coord.soft_deadline = Duration::from_millis(300);

        let coord = Arc::new(coord);
        let q_a = plan("eldenring.exe", "질문 A 보스 약점", None);
        let c1 = coord.clone();
        let q1 = q_a.clone();
        let handle = std::thread::spawn(move || {
            c1.retrieve(&q1, None, None, Arc::new(|_| Grade::Unrated), 1_000)
        });

        // 40 ms later: user fires question B (bumping ask_epoch from 1 -> 2)
        std::thread::sleep(Duration::from_millis(40));
        let epoch_b = coord.next_ask_epoch();
        assert_eq!(epoch_b, 2);

        let outcome_a = handle.join().unwrap();
        assert_eq!(
            outcome_a,
            RagOutcome::Cancelled { epoch: 1 },
            "epoch 1 search must be cancelled when epoch 2 arrives"
        );

        // Now start a search during privacy_epoch 1, bump privacy_epoch while it is in flight,
        // and verify the late deep result is NEVER inserted into SceneRagCache.
        let q_priv = plan("eldenring.exe", "민감 전환 직전 질문", None);
        let c2 = coord.clone();
        let qp = q_priv.clone();
        let handle2 = std::thread::spawn(move || {
            c2.retrieve(&qp, None, None, Arc::new(|_| Grade::Unrated), 2_000)
        });
        std::thread::sleep(Duration::from_millis(40));
        let new_pe = coord.bump_privacy_epoch();
        assert_eq!(new_pe, 2);
        let outcome_priv = handle2.join().unwrap();
        assert!(matches!(outcome_priv, RagOutcome::Cancelled { .. }));

        // Wait past deep delay so the background thread finishes
        std::thread::sleep(Duration::from_millis(220));
        let cache = coord.cache.lock().unwrap();
        assert_eq!(
            cache.len(),
            0,
            "late result from old privacy_epoch must never pollute SceneRagCache"
        );
    }

    #[test]
    fn seed_phrase_checker_avoids_gaming_false_positives_and_catches_real_seeds() {
        let gaming = "game boss sword shield armor arrow magic hero level bonus reward win";
        assert_eq!(detect_seed_phrase_checked(gaming), None);

        // Official valid-checksum BIP-39 test vector
        let valid_seed = "legal winner thank year wave sausage worth useful legal winner thank yellow";
        assert!(detect_seed_phrase_checked(valid_seed).is_some());

        // Numbered wallet backup grid
        let numbered = "1. game 2. boss 3. sword 4. shield 5. armor 6. arrow 7. magic 8. hero 9. level 10. bonus 11. reward 12. win";
        assert!(detect_seed_phrase_checked(numbered).is_some());

        // Wallet/seed context keyword nearby
        let with_ctx = "내 지갑 복구 시드: game boss sword shield armor arrow magic hero level bonus reward win";
        assert!(detect_seed_phrase_checked(with_ctx).is_some());
    }

    #[test]
    fn dormant_steam_game_overlay_is_ignored_while_checkout_overlay_is_kept() {
        let dormant = VisibleWindowEntry {
            info: WindowInfo {
                hwnd: 1,
                pid: 100,
                exe_path: None,
                exe_name: Some("GameOverlayUI.exe".into()),
                title: "Steam Overlay".into(),
            },
            rect: PxRect { left: 0, top: 0, right: 1920, bottom: 1080 },
            display_affinity: 0,
            ex_style: WS_EX_NOACTIVATE,
            class_name: "vguiPopupWindow".into(),
        };
        let checkout = VisibleWindowEntry {
            info: WindowInfo {
                hwnd: 2,
                pid: 100,
                exe_path: None,
                exe_name: Some("GameOverlayUI.exe".into()),
                title: "Steam Store Checkout".into(),
            },
            rect: PxRect { left: 200, top: 150, right: 1400, bottom: 900 },
            display_affinity: 0,
            ex_style: 0,
            class_name: "vguiPopupWindow".into(),
        };
        let kept = filter_dormant_game_overlays(vec![dormant, checkout]);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].info.hwnd, 2);
    }

    #[test]
    fn drop_intake_blocks_profanity_and_pii_at_0ms_and_pins_are_reversible_local_only() {
        let tmp = std::env::temp_dir().join(format!("zena_bridge_test_{}.json", std::process::id()));
        let mut bridge = PdBridge::new(
            None,
            SharedEmbedder(Arc::new(HashedNgramEmbedder::default())),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            false,
            tmp.clone(),
        );
        assert_eq!(
            bridge.enqueue_drop("eldenring.exe", None, "이 시1발 보스 너무 어려워", 1_000),
            DropOutcome::BlockedProfanity
        );
        assert!(matches!(
            bridge.enqueue_drop("eldenring.exe", None, "legal winner thank year wave sausage worth useful legal winner thank yellow", 1_000),
            DropOutcome::BlockedPii { .. }
        ));
        let ok = bridge.enqueue_drop("eldenring.exe", None, "2페이즈는 냉기 항아리가 좋아", 1_000);
        let req_id = match ok {
            DropOutcome::Enqueued { request_id, .. } => request_id,
            other => panic!("expected Enqueued, got {other:?}"),
        };
        let q = bridge.rag.quarantine.lock().unwrap().clone();
        let pinned = bridge.verdicts.toggle_pin(&req_id, UserPin::Demoted, 2_000, &q).unwrap();
        assert_eq!(pinned.local_tier, Tier::Discount);
        let unpinned = bridge.verdicts.toggle_pin(&req_id, UserPin::Demoted, 3_000, &q).unwrap();
        assert_eq!(unpinned.local_tier, Tier::Unrated);
        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn cn251_local_shortcut_3wins_promotes_in_verdict_store_and_drawer_pin_is_reversible() {
        let mut store = LocalVerdictStore::default();
        let q = QuarantineLedger::new();
        let mut promoted = false;
        let mut last_view = None;
        for i in 0..3 {
            let (p, v) = store.ingest_shortcut_win(
                "eldenring.exe",
                "fork:boss-gate",
                "0xcp-ladder",
                "0xcp-front-gate",
                "우측 사다리 지름길",
                4_000,
                10_000 + i * 60_000,
                &q,
            );
            promoted = p;
            last_view = Some(v);
        }
        assert!(promoted, "3 consecutive shortcut wins activate local outperforms");
        let view = last_view.unwrap();
        assert_eq!(view.local_tier, Tier::Premium);
        // Stage 2 drawer exposes the local shortcut and supports reversible ▼ demotion pin.
        let drawer = store.drawer_items(Some("eldenring.exe"), 200_000, &q);
        assert!(drawer.iter().any(|d| d.record_id == "0xcp-ladder"));
        let demoted = store.toggle_pin("0xcp-ladder", UserPin::Demoted, 201_000, &q).unwrap();
        assert_eq!(demoted.local_tier, Tier::Discount);
        let restored = store.toggle_pin("0xcp-ladder", UserPin::Demoted, 202_000, &q).unwrap();
        assert_eq!(restored.local_tier, Tier::Premium);
    }

    #[test]
    fn cn264_zone_vault_64slots_pinned_slru_and_batch_id_idempotency() {
        use premium_discount_nav::{Checkpoint, StepKind};
        let mut vault = ZoneVault::new();
        let app = "0x39206c971297059b89e4e6fc35844ee87feb0407";
        let make_run = |idx: u32, day: u32| -> NavRun {
            let s0 = format!("0x{:040x}", 1u64);
            let s1 = format!("0x{:040x}", 100u64 + u64::from(idx));
            let base_ms = i64::from(day) * 86_400_000;
            NavRun::from_checkpoints(
                app,
                &[
                    Checkpoint::new(base_ms, StepKind::Screen, &s0, None),
                    Checkpoint::new(base_ms + 2_000, StepKind::Region, &s1, None),
                ],
                None,
            )
            .unwrap()
        };
        let pinned_run = make_run(0, 20_774);
        vault.record_run(&pinned_run);
        assert!(vault.pin_course(app, &pinned_run.course, true));
        // Insert 70 distinct courses (exceeding 64-slot ZoneVault cap): pinned course survives!
        for i in 1..=70 {
            vault.record_run(&make_run(i, 20_774 + (i / 10)));
        }
        let kept = vault.collection.courses(app);
        assert_eq!(kept.len(), ZONE_VAULT_MAX_SLOTS);
        assert!(kept.iter().any(|c| c.course == pinned_run.course), "pinned course protected");

        // batch_id remains identical across retries until acknowledged.
        let b1 = vault.next_or_pending_batch_id(1_000);
        let b2 = vault.next_or_pending_batch_id(2_000);
        assert_eq!(b1, b2);
        vault.ack_batch(&b1);
        let b3 = vault.next_or_pending_batch_id(3_000);
        assert_ne!(b1, b3);
    }

    #[test]
    fn live_loop_feeds_the_choice_book_and_closes_runs_into_the_zone_vault() {
        let tmp = std::env::temp_dir().join(format!("zena_live_loop_test_{}.json", std::process::id()));
        let mut b = PdBridge::new(None, SharedEmbedder(Arc::new(HashedNgramEmbedder::default())), Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)), false, tmp);
        let p = BehaviorParams::default();
        let (cc, list, detail) = ("shop.example", "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
        let (cart, paid) = ("0xcccccccccccccccccccccccccccccccccccccccc", "0xdddddddddddddddddddddddddddddddddddddddd");
        let labels = ["오늘의 특가", "장바구니", "주문 내역"];
        let fought = Episode {
            events: vec![
                premium_discount_nav::Event { at_ms: 0, signal: premium_discount_nav::Signal::Retried, anchor: Some("#1".into()) },
                premium_discount_nav::Event { at_ms: 500, signal: premium_discount_nav::Signal::RageClick, anchor: Some("#1".into()) },
            ],
        };
        let mut t = 1_000_000;
        // the slot-1 promotion: picked again and again, the user fights the detail page and bounces back within 8 s
        for _ in 0..6 {
            b.on_badge_picked(cc, list, "a.promo", "오늘의 특가", &labels, true, t);
            b.on_screen_changed(cc, list, None, detail, &p, t + 500);
            b.on_screen_changed(cc, detail, Some(&fought), list, &p, t + 3_500);
            t += 10_000;
        }
        // the calm choice: the cart, where the work goes on
        for _ in 0..3 {
            b.on_badge_picked(cc, list, "a.cart", "장바구니", &labels, true, t);
            b.on_screen_changed(cc, list, None, cart, &p, t + 500);
            b.on_screen_changed(cc, cart, None, paid, &p, t + 30_000);
            t += 40_000;
        }
        let rows = b.choice_rows_for(cc, &labels);
        let promo = rows.iter().find(|r| r.label == "오늘의 특가").expect("the promotion is rated");
        assert!(promo.is_friction_trap && promo.tier == Tier::Discount, "{rows:?}");
        assert!(rows.iter().all(|r| r.label != "주문 내역"), "an unpicked label stays unrated");
        // the choice book reaches the live overlay rows (no verdicts or public rows here)
        let live = b.live_web_rows_with_choices(cc, &labels, t);
        assert!(live.iter().any(|r| r.label == "오늘의 특가" && r.is_friction_trap));
        // leaving the site closes the run into the 64-slot vault; nothing is gathered after it
        let run = b.on_app_left(&[]).expect("a run");
        let app = nav_app_hash(cc).unwrap();
        assert!(b.zone_vault.collection.courses(&app).iter().any(|c| c.course == run.course));
        assert!(b.on_app_left(&[]).is_none(), "nothing gathered since");
        // the vault outlives the process (a public course expired on the server is still here next time)
        let path = zone_vault_path(&b.quarantine_path);
        let reopened = ZoneVault::load(&path);
        assert_eq!(reopened, b.zone_vault);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cn271_cross_cc_mirror_bridge_surfaces_wiki_note_in_game_and_stays_isolated_elsewhere() {
        let mut mirror = LocalMirrorIndex::hashed();
        mirror.enqueue_optimistic(
            "wiki-note-1",
            "fextralife.com",
            None,
            "말레니아 2페이즈 물새난격 빙결 항아리 캔슬",
            1_000,
        );
        // Bridge wiki note into eldenring.exe only (verified via shared RelayKey)
        assert!(mirror.bridge_note_to_cc("wiki-note-1", "eldenring.exe"));

        let game_plan = plan("eldenring.exe", "말레니아 물새난격 빙결 항아리", None);
        let qv = mirror.embed(&game_plan.rewritten_query);
        let hits_game = mirror.search_fast(&game_plan, &qv, &|_| Grade::Unrated);
        assert!(hits_game.iter().any(|c| c.record_id == "wiki-note-1"));

        // Unrelated app (notepad.exe) sees 0 leaked notes!
        let other_plan = plan("notepad.exe", "말레니아 물새난격 빙결 항아리", None);
        let hits_other = mirror.search_fast(&other_plan, &qv, &|_| Grade::Unrated);
        assert!(hits_other.is_empty());
    }

    #[test]
    fn cn272_cross_cc_meta_trajectory_attributes_victory_credit_to_bridged_wiki_note() {
        let mut traj = TrajectoryLedger::default();
        traj.record_edge(
            "fextralife.com",
            "eldenring.exe",
            vec!["0xrelay-malenia-0001".to_string()],
            10_000,
        );
        let linked = traj.linked_ccs("eldenring.exe", 60_000);
        assert!(linked.contains(&"eldenring.exe".to_string()));
        assert!(linked.contains(&"fextralife.com".to_string()));

        // Exposure ledger returns wiki note when querying active_in_meta_trajectory in-game,
        // and LocalVerdictStore credits the wiki note on in-game victory!
        let mut ledger = zena_core::exposure::ExposureLedger::new();
        ledger.insert(zena_core::exposure::ExposedRecord {
            record_id: "wiki-note-1".into(),
            cc: "fextralife.com".into(),
            to: None,
            canonical_topic: "빙결 항아리".into(),
            role: zena_core::exposure::ExposureRole::RecommendedSolution,
            w_spoken: 1.0,
            exposed_at_ms: 20_000,
            target_kind: zena_core::exposure::ExposureTargetKind::RagRecord,
            exclusive_slot_group: Some("tactics".into()),
        });
        let cc_refs: Vec<&str> = linked.iter().map(String::as_str).collect();
        let active = ledger.active_in_meta_trajectory(&cc_refs, 60_000);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].record_id, "wiki-note-1");

        let mut verdicts = LocalVerdictStore::default();
        verdicts.ensure_record("wiki-note-1", "fextralife.com", None, "빙결 항아리", "stored");
        let mut gate = OutcomeVoteGate::new(false);
        let cands = vec![AttributionCandidate {
            id: "wiki-note-1".into(),
            kind: CreditTargetKind::RagRecord,
            exclusive_slot_group: Some("tactics".into()),
            is_warned_trap: false,
            exposure_weight: 1.0,
            age_ms: 40_000,
            semantic_sim: 0.9,
            matches_explicit_mention: false,
            friction_z: None,
            ambient_prior: 0.20,
            prior_same_sign_outcomes: 0,
        }];
        verdicts.apply_outcome_credit(&cands, 1, 60_000, false, &mut gate);
        assert_eq!(verdicts.records["wiki-note-1"].wins, 1);
    }
}
