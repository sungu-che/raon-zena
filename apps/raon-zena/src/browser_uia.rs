//! Extension-free asynchronous UIA browser watcher, tab `cc` isolator, incognito guard,
//! `LABEL_SPACE` multilingual matcher (`WA-09`), and runtime-demoted CommitGuard badge builder (`WA-10`).
//!
//! * On Windows, UIA calls run exclusively on a dedicated COM STA thread (`uia-browser-worker`,
//!   `COINIT_APARTMENTTHREADED`), never blocking the 60 fps overlay or capture pipeline.
//! * Active browser tabs are isolated by normalized host (`amazon.com`), never pooled under
//!   `chrome.exe`. Query parameters and non-route fragments are stripped.
//! * Incognito / InPrivate / 시크릿 / 프라이빗 tabs set `incognito_guard = true`: 0 NavMap writes,
//!   0 SDS ledger writes, 0 sync/course telemetry.
//! * `web_analytic` GoldenPath / FrictionTrap rows are projected onto live translated UIA controls
//!   via `LABEL_SPACE` similarity + role + reading order (`0.65 * cos + 0.20 * role_eq + 0.15 * ord`),
//!   double-centered and assigned 1:1 with `zena_sds::matrix::exclusive_assign_by_score`.
//! * Unavailable (`disabled` / sold-out) #1 targets are demoted for the current frame only while the
//!   runner-up inherits `#1`; final payment buttons receive `CommitGuard` (`SlotDispatch::FocusOnly`).

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};

use nlp_nms_router::embed::{SharedEmbedder, TextEmbedder};
use premium_discount_nav::{
    page_hash, url_template, FrameContext, Numbered, Rect, Region, Role, Source, StickyNumberer,
};
use premium_discount_schema::envelope;
use zena_verdict::Tier;
use zena_core::brief::{
    is_commit_label, is_unavailable_label, prompt_label, SlotDispatch, WebChoice, WebCoachBrief,
};
use zena_core::reference::normalize_cc;
use zena_sds::matrix::{double_center, exclusive_assign_by_score, ABSENT};

/// Logical vector space identifier for multilingual UI label matching (`WA-09`).
#[allow(dead_code)]
pub const LABEL_SPACE_ID: &str = "granite-label-384d";
/// Minimum raw score (`0.65*cos + 0.20*role + 0.15*ord`) before entering 1:1 assignment.
pub const LABEL_MATCH_MIN: f32 = 0.72;
/// Elements with less than 35% visible area behind sticky headers/footers are clipped.
pub const MIN_VISIBLE_RATIO: f32 = 0.35;

const BROWSER_EXES: [&str; 8] = [
    "chrome.exe",
    "msedge.exe",
    "brave.exe",
    "whale.exe",
    "firefox.exe",
    "arc.exe",
    "opera.exe",
    "vivaldi.exe",
];

const INCOGNITO_MARKERS: [&str; 8] = [
    "incognito",
    "inprivate",
    "private browsing",
    "시크릿",
    "프라이빗",
    "사생활 보호",
    "シークレット",
    "プライベート",
];

// -------------------------------------------------------------------------------------------------
// Overlay badge model (shared with pipeline, overlay, and scene renderer)
// -------------------------------------------------------------------------------------------------

/// One numbered badge (1–9) displayed by the overlay.
#[derive(Debug, Clone, PartialEq)]
pub struct NavBadge {
    pub number: u8,
    pub anchor: String,
    pub label: String,
    pub rect: Rect,
    pub tier: Option<Tier>,
    pub commit_guarded: bool,
    pub unavailable: bool,
}

/// Complete badge overlay snapshot for the current window.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NavBadgeOverlay {
    pub cc: String,
    pub screen_key: Option<String>,
    pub window_rect_px: Option<[f32; 4]>,
    pub badges: Vec<NavBadge>,
    /// True if any primary/active badge is occluded by the overlay window (triggers step-aside).
    pub occluded_by_overlay: bool,
    /// Optional 3D coaching arrow angle (deg, 0° right, 90° up) pointing toward badge #1.
    pub primary_arrow_deg: Option<f32>,
}

impl NavBadgeOverlay {
    pub fn dispatch_for(&self, number: u8) -> Option<(&NavBadge, SlotDispatch)> {
        let b = self.badges.iter().find(|x| x.number == number)?;
        let d = if b.commit_guarded || is_commit_label(&b.label) {
            SlotDispatch::FocusOnly
        } else {
            SlotDispatch::Invoke
        };
        Some((b, d))
    }
}

// -------------------------------------------------------------------------------------------------
// Browser detection, incognito guard & URL normalization
// -------------------------------------------------------------------------------------------------

pub fn is_browser_exe(exe: &str) -> bool {
    let lower = exe.trim().to_ascii_lowercase();
    BROWSER_EXES.iter().any(|b| lower == *b)
}

pub fn is_incognito_title(title: &str) -> bool {
    let lower = title.to_lowercase();
    INCOGNITO_MARKERS.iter().any(|m| lower.contains(m))
}

/// Normalized browser URL pieces with all query parameters and non-route fragments stripped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedBrowserUrl {
    pub normalized_cc: String,
    pub url_template: String,
    pub page_hash: String,
}

pub fn normalize_browser_url(raw_url: &str) -> Option<NormalizedBrowserUrl> {
    let trimmed = raw_url.trim();
    if trimmed.is_empty() {
        return None;
    }
    let with_scheme = if trimmed.contains("://") {
        trimmed.to_string()
    } else if trimmed.contains('.') && !trimmed.contains(' ') {
        format!("https://{trimmed}")
    } else {
        return None;
    };
    // Strip query string before everything else so tokens never leak into cc or templates.
    let no_query = match with_scheme.split_once('?') {
        Some((before, after)) => match after.split_once('#') {
            Some((_, frag)) => format!("{before}#{frag}"),
            None => before.to_string(),
        },
        None => with_scheme,
    };
    let tmpl = url_template(&no_query)?;
    let ph = page_hash(&no_query)?;
    let cc = envelope::normalize_cc(&no_query)
        .map(|c| c.cc)
        .unwrap_or_else(|_| {
            let host = tmpl.split(|c| c == '/' || c == '#').next().unwrap_or("");
            normalize_cc(host)
        });
    if cc.is_empty() {
        return None;
    }
    Some(NormalizedBrowserUrl {
        normalized_cc: cc,
        url_template: tmpl,
        page_hash: ph,
    })
}

// -------------------------------------------------------------------------------------------------
// UIA snapshot, sticky occluder clipping & LABEL_SPACE 1:1 matching
// -------------------------------------------------------------------------------------------------

/// One interactive control collected from the browser's UIA tree.
#[derive(Debug, Clone, PartialEq)]
pub struct UiaInteractiveRegion {
    pub anchor: String,
    pub label: String,
    pub role: Role,
    pub rect: Rect,
    pub enabled: bool,
    pub sensitive: bool,
    /// True if this region is a sticky/fixed top header or bottom floating bar.
    pub is_sticky_chrome: bool,
}

impl UiaInteractiveRegion {
    pub fn to_nav_region(&self) -> Region {
        Region {
            anchor: self.anchor.clone(),
            role: self.role,
            name: self.label.clone(),
            rect: self.rect,
            source: Source::Tree,
            score: if self.enabled { 1.0 } else { 0.4 },
            changed: false,
            sensitive: self.sensitive,
        }
    }
}

/// One scanned snapshot of the active browser tab.
#[derive(Debug, Clone, PartialEq)]
pub struct BrowserTabSnapshot {
    pub hwnd: isize,
    pub exe: String,
    pub title: String,
    pub url: Option<String>,
    pub normalized_cc: Option<String>,
    pub url_template: Option<String>,
    pub page_hash: Option<String>,
    pub incognito_guard: bool,
    /// Desktop pixel rect `[x0, y0, x1, y1]` of the document pane / browser window.
    pub doc_screen_rect_px: Option<[f32; 4]>,
    pub regions: Vec<UiaInteractiveRegion>,
    pub captured_at_ms: i64,
}

impl BrowserTabSnapshot {
    /// Builds a snapshot from raw UIA readings, automatically stripping query tokens and detecting
    /// incognito / private browsing tabs.
    pub fn from_raw(
        hwnd: isize,
        exe: &str,
        title: &str,
        raw_url: Option<&str>,
        doc_screen_rect_px: Option<[f32; 4]>,
        regions: Vec<UiaInteractiveRegion>,
        captured_at_ms: i64,
    ) -> Self {
        let incognito = is_incognito_title(title);
        if incognito {
            return Self {
                hwnd,
                exe: exe.to_string(),
                title: String::new(), // never keep incognito title
                url: None,
                normalized_cc: None,
                url_template: None,
                page_hash: None,
                incognito_guard: true,
                doc_screen_rect_px,
                regions: Vec::new(), // never retain incognito controls
                captured_at_ms,
            };
        }
        let norm = raw_url.and_then(normalize_browser_url);
        Self {
            hwnd,
            exe: exe.to_string(),
            title: title.to_string(),
            url: raw_url.map(|u| u.split('?').next().unwrap_or(u).to_string()),
            normalized_cc: norm.as_ref().map(|n| n.normalized_cc.clone()),
            url_template: norm.as_ref().map(|n| n.url_template.clone()),
            page_hash: norm.as_ref().map(|n| n.page_hash.clone()),
            incognito_guard: false,
            doc_screen_rect_px,
            regions,
            captured_at_ms,
        }
    }
}

/// Drops scrolled controls whose visible unoccluded area behind a top sticky header
/// (`y < 0.18 && w > 0.70`) or bottom floating bar (`y + h > 0.85 && w > 0.70`) is `< 35%`.
pub fn clip_sticky_occluders(regions: &[UiaInteractiveRegion]) -> Vec<UiaInteractiveRegion> {
    let occluders: Vec<Rect> = regions
        .iter()
        .filter(|r| {
            r.is_sticky_chrome
                && r.rect.w > 0.70
                && (r.rect.y < 0.18 || r.rect.y + r.rect.h > 0.85)
        })
        .map(|r| r.rect)
        .collect();
    if occluders.is_empty() {
        return regions.to_vec();
    }
    regions
        .iter()
        .filter(|r| {
            if r.is_sticky_chrome {
                return true;
            }
            let area = (r.rect.w * r.rect.h).max(1e-6);
            let mut occluded = 0.0f32;
            for occ in &occluders {
                let ix0 = r.rect.x.max(occ.x);
                let iy0 = r.rect.y.max(occ.y);
                let ix1 = (r.rect.x + r.rect.w).min(occ.x + occ.w);
                let iy1 = (r.rect.y + r.rect.h).min(occ.y + occ.h);
                if ix1 > ix0 && iy1 > iy0 {
                    occluded += (ix1 - ix0) * (iy1 - iy0);
                }
            }
            let visible_ratio = (1.0 - occluded / area).clamp(0.0, 1.0);
            visible_ratio >= MIN_VISIBLE_RATIO
        })
        .cloned()
        .collect()
}

/// One site-analytics element row (e.g. from `web_analytic` GoldenPath / FrictionTrap).
#[derive(Debug, Clone, PartialEq)]
pub struct WebAnalyticTierRow {
    pub label: String,
    pub role: Role,
    /// Normalized reading order in `0.0..=1.0`.
    pub ordinal_norm: f32,
    pub tier: Tier,
    pub is_friction_trap: bool,
}

/// Output of projecting `WebAnalyticTierRow`s onto live UIA regions via `LABEL_SPACE`.
#[derive(Debug, Clone, Default)]
pub struct TierProjection {
    /// Permanent anchor -> tier map (for `StickyNumberer`).
    pub tiers: HashMap<String, Tier>,
    /// Matched `(row_idx, region_idx, raw_score)` pairs (1:1).
    pub matches: Vec<(usize, usize, f32)>,
    /// First matched FrictionTrap's scrubbed label, if any.
    pub friction_trap_label: Option<String>,
}

/// Projects `web_rows` onto `uia_regions` using `LABEL_SPACE` similarity:
/// `score = 0.65 * cos(row.label, region.label) + 0.20 * role_eq + 0.15 * (1 - |ord_i - ord_j|)`.
/// Applies `double_center` + `exclusive_assign_by_score` so translated labels ("Buy Now" ↔ "지금 구매")
/// match 1:1 without stealing unrelated buttons.
pub fn project_web_tiers_onto_uia(
    web_rows: &[WebAnalyticTierRow],
    uia_regions: &[UiaInteractiveRegion],
    label_similarity: impl Fn(&str, &str) -> f32,
) -> TierProjection {
    let mut out = TierProjection::default();
    if web_rows.is_empty() || uia_regions.is_empty() {
        return out;
    }
    let n_reg = uia_regions.len();
    let mut raw = vec![vec![ABSENT; n_reg]; web_rows.len()];
    for (i, row) in web_rows.iter().enumerate() {
        for (j, reg) in uia_regions.iter().enumerate() {
            if reg.label.trim().is_empty() || reg.sensitive {
                continue;
            }
            let cos = label_similarity(&row.label, &reg.label).clamp(0.0, 1.0);
            let role_eq = if row.role == reg.role { 1.0 } else { 0.0 };
            let reg_ord = if n_reg <= 1 {
                0.0
            } else {
                j as f32 / (n_reg - 1) as f32
            };
            let ord_sim = (1.0 - (row.ordinal_norm - reg_ord).abs()).clamp(0.0, 1.0);
            let score = 0.65 * cos + 0.20 * role_eq + 0.15 * ord_sim;
            if score >= LABEL_MATCH_MIN {
                raw[i][j] = score;
            }
        }
    }

    // Double-center when we have >= 2 present rows/cols; otherwise assign directly on raw scores.
    let centered = double_center(&raw);
    let assignment = exclusive_assign_by_score(&centered, -0.5, 0.0);

    for (row_idx, slot) in assignment.into_iter().enumerate() {
        let Some((reg_idx, _, _)) = slot else { continue };
        let raw_score = raw[row_idx][reg_idx];
        if raw_score < LABEL_MATCH_MIN {
            continue;
        }
        let row = &web_rows[row_idx];
        let reg = &uia_regions[reg_idx];
        out.tiers.insert(reg.anchor.clone(), row.tier);
        out.matches.push((row_idx, reg_idx, raw_score));
        if row.is_friction_trap && out.friction_trap_label.is_none() {
            out.friction_trap_label = Some(prompt_label(&reg.label));
        }
    }
    out
}

/// Cosine similarity helper over a `SharedEmbedder` (granite 384d `LABEL_SPACE` when installed).
#[allow(dead_code)]
pub fn embedder_similarity(embedder: &SharedEmbedder, a: &str, b: &str) -> f32 {
    let va = embedder.embed(a);
    let vb = embedder.embed(b);
    cosine_unit_or_raw(&va, &vb)
}

#[inline]
fn cosine_unit_or_raw(va: &[f32], vb: &[f32]) -> f32 {
    if va.is_empty() || va.len() != vb.len() {
        return 0.0;
    }
    let dot = raon_cpu_kernels::dot_f32_simd(va, vb);
    let na = raon_cpu_kernels::dot_f32_simd(va, va);
    let nb = raon_cpu_kernels::dot_f32_simd(vb, vb);
    let denom = (na * nb).sqrt();
    if denom > 1e-8 {
        (dot / denom).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Pre-embeds all unique labels across `web_rows` and `uia_regions` in a single `embed_batch` call
/// (`M + N` embeddings instead of `2 * M * N` repeated neural passes inside `project_web_tiers_onto_uia`).
pub fn precompute_label_similarity(
    embedder: &SharedEmbedder,
    web_rows: &[WebAnalyticTierRow],
    uia_regions: &[UiaInteractiveRegion],
) -> impl Fn(&str, &str) -> f32 {
    let mut vecs: HashMap<String, Vec<f32>> = HashMap::new();
    if !web_rows.is_empty() && !uia_regions.is_empty() {
        let mut unique: Vec<&str> = Vec::with_capacity(web_rows.len() + uia_regions.len());
        let mut seen = std::collections::HashSet::with_capacity(web_rows.len() + uia_regions.len());
        for row in web_rows {
            if !row.label.trim().is_empty() && seen.insert(row.label.as_str()) {
                unique.push(row.label.as_str());
            }
        }
        for reg in uia_regions {
            if !reg.label.trim().is_empty() && !reg.sensitive && seen.insert(reg.label.as_str()) {
                unique.push(reg.label.as_str());
            }
        }
        if !unique.is_empty() {
            vecs.reserve(unique.len());
            let embedded = embedder.embed_batch(&unique);
            for (label, mut v) in unique.into_iter().zip(embedded) {
                let n = raon_cpu_kernels::dot_f32_simd(&v, &v).sqrt();
                if n > 1e-8 {
                    for x in &mut v {
                        *x /= n;
                    }
                }
                vecs.insert(label.to_string(), v);
            }
        }
    }
    move |a: &str, b: &str| -> f32 {
        match (vecs.get(a), vecs.get(b)) {
            (Some(va), Some(vb)) if !va.is_empty() && va.len() == vb.len() => {
                raon_cpu_kernels::dot_f32_simd(va, vb).clamp(0.0, 1.0)
            }
            _ => 0.0,
        }
    }
}

/// Builds live 1–9 overlay badges + `WebCoachBrief` for a browser tab snapshot.
///
/// * Returns `None` on incognito tabs (0 recording, 0 NavMap).
/// * Runtime-only demotion: if a verified `#1` control is `disabled` or sold-out, the next available
///   verified candidate inherits `#1` for this frame while `permanent_tiers` stays unchanged!
/// * Final payment buttons (`is_commit_label`) are marked `commit_guarded = true` (`FocusOnly`).
#[allow(clippy::too_many_arguments)]
pub fn build_web_nav_overlay(
    snapshot: &BrowserTabSnapshot,
    web_rows: &[WebAnalyticTierRow],
    permanent_tiers: &mut HashMap<String, Tier>,
    numberer: &mut StickyNumberer,
    locked_numbers: u16,
    priority_anchor: Option<&str>,
    label_similarity: impl Fn(&str, &str) -> f32,
    now_ms: i64,
) -> Option<(NavBadgeOverlay, WebCoachBrief)> {
    if snapshot.incognito_guard {
        numberer.reset();
        return None;
    }
    let cc = snapshot.normalized_cc.as_deref()?;
    let visible = clip_sticky_occluders(&snapshot.regions);
    let projection = project_web_tiers_onto_uia(web_rows, &visible, label_similarity);
    for (anchor, tier) in &projection.tiers {
        permanent_tiers.insert(anchor.clone(), *tier);
    }

    // Build runtime-adjusted tiers for this frame ONLY: unavailable items are demoted below available
    // verified alternatives without mutating `permanent_tiers`.
    let mut frame_tiers = permanent_tiers.clone();
    let mut unavailable_by_anchor: HashMap<String, bool> = HashMap::new();
    for reg in &visible {
        let unavail = !reg.enabled || is_unavailable_label(&reg.label);
        unavailable_by_anchor.insert(reg.anchor.clone(), unavail);
        if unavail && frame_tiers.get(&reg.anchor) == Some(&Tier::Premium) {
            frame_tiers.insert(reg.anchor.clone(), Tier::Neutral);
        }
    }

    let regions: Vec<Region> = visible.iter().map(UiaInteractiveRegion::to_nav_region).collect();
    let cx = FrameContext {
        screen_key: snapshot.page_hash.clone(),
        now_ms,
        scrolled: false,
        locked_numbers,
        priority_anchor: priority_anchor.map(str::to_string),
    };
    let (numbered, sensitive) =
        numberer.update(regions, &frame_tiers, premium_discount_nav::MERGE_IOU, &cx);
    if sensitive {
        return None;
    }

    let mut badges: Vec<NavBadge> = Vec::new();
    let mut choices: Vec<WebChoice> = Vec::new();
    for Numbered {
        number,
        tier,
        region,
        ..
    } in numbered
    {
        let Some(num) = number else { continue };
        let unavailable = unavailable_by_anchor
            .get(&region.anchor)
            .copied()
            .unwrap_or(false);
        let perm_verified = permanent_tiers.get(&region.anchor) == Some(&Tier::Premium);
        let commit_guarded = is_commit_label(&region.name);
        badges.push(NavBadge {
            number: num,
            anchor: region.anchor.clone(),
            label: prompt_label(&region.name),
            rect: region.rect,
            tier,
            commit_guarded,
            unavailable,
        });
        choices.push(WebChoice {
            number: num,
            label: region.name.clone(),
            verified: perm_verified,
            disabled: unavailable,
        });
    }

    let brief = WebCoachBrief::from_choices(
        cc,
        snapshot.url_template.as_deref(),
        &choices,
        projection.friction_trap_label.as_deref(),
    );
    let overlay = NavBadgeOverlay {
        cc: cc.to_string(),
        screen_key: snapshot.page_hash.clone(),
        window_rect_px: snapshot.doc_screen_rect_px,
        badges,
        occluded_by_overlay: false,
        primary_arrow_deg: None,
    };
    Some((overlay, brief))
}

// -------------------------------------------------------------------------------------------------
// BrowserSessionTracker: tab cc isolation, delayed URL refinement & incognito zero-write ledger
// -------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TabTransition {
    IncognitoBlocked,
    /// Tab switched before Omnibox resolved; buffered until URL arrives.
    PendingUrlResolution { hwnd: isize },
    ActiveTab {
        cc: String,
        page_hash: Option<String>,
        refined_from_pending: bool,
    },
}

/// Tracks active browser tabs by normalized host `cc`, ensuring incognito tabs record 0 entries
/// and delayed URL resolutions refine pending tab observations cleanly.
#[derive(Debug, Default)]
pub struct BrowserSessionTracker {
    pub active_cc: Option<String>,
    pub active_page_hash: Option<String>,
    pending_hwnd: Option<isize>,
    pub recorded_tabs: Vec<(String, String)>,
}

impl BrowserSessionTracker {
    pub fn observe(&mut self, snap: &BrowserTabSnapshot) -> TabTransition {
        if snap.incognito_guard {
            self.active_cc = None;
            self.active_page_hash = None;
            self.pending_hwnd = None;
            return TabTransition::IncognitoBlocked;
        }
        let Some(cc) = &snap.normalized_cc else {
            self.pending_hwnd = Some(snap.hwnd);
            return TabTransition::PendingUrlResolution { hwnd: snap.hwnd };
        };
        let refined = self.pending_hwnd.take() == Some(snap.hwnd);
        let changed = self.active_cc.as_deref() != Some(cc.as_str())
            || self.active_page_hash != snap.page_hash;
        self.active_cc = Some(cc.clone());
        self.active_page_hash = snap.page_hash.clone();
        if changed {
            self.recorded_tabs.push((
                cc.clone(),
                snap.page_hash.clone().unwrap_or_default(),
            ));
        }
        TabTransition::ActiveTab {
            cc: cc.clone(),
            page_hash: snap.page_hash.clone(),
            refined_from_pending: refined,
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Non-blocking AsyncBrowserUiaWorker + Windows STA COM IUIAutomation implementation
// -------------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ScanRequest {
    pub hwnd: isize,
    pub exe: String,
    pub title: String,
    pub now_ms: i64,
}

/// Pluggable UIA scanner backend (real Windows STA COM in production; mock in tests).
pub trait BrowserScanner: Send + 'static {
    fn scan(&mut self, req: &ScanRequest) -> Option<BrowserTabSnapshot>;
}

struct PendingReq {
    latest: Option<ScanRequest>,
    stop: bool,
}

/// Dedicated background worker (`uia-browser-worker`) that coalesces scan requests so UIA COM
/// calls never block the overlay or capture loop.
pub struct AsyncBrowserUiaWorker {
    shared: Arc<(Mutex<PendingReq>, Condvar)>,
    rx: Receiver<BrowserTabSnapshot>,
}

impl AsyncBrowserUiaWorker {
    pub fn spawn(mut scanner: impl BrowserScanner) -> Self {
        let shared = Arc::new((
            Mutex::new(PendingReq {
                latest: None,
                stop: false,
            }),
            Condvar::new(),
        ));
        let (tx, rx): (Sender<BrowserTabSnapshot>, Receiver<BrowserTabSnapshot>) = mpsc::channel();
        let worker_shared = shared.clone();
        std::thread::Builder::new()
            .name("uia-browser-worker".into())
            .spawn(move || {
                #[cfg(windows)]
                let _com = ComStaGuard::init();
                loop {
                    let req = {
                        let (lock, cv) = &*worker_shared;
                        let Ok(mut guard) = lock.lock() else { return };
                        while guard.latest.is_none() && !guard.stop {
                            guard = match cv.wait(guard) {
                                Ok(g) => g,
                                Err(_) => return,
                            };
                        }
                        if guard.stop {
                            return;
                        }
                        guard.latest.take()
                    };
                    if let Some(r) = req {
                        if let Some(snap) = scanner.scan(&r) {
                            if tx.send(snap).is_err() {
                                return;
                            }
                        }
                    }
                }
            })
            .expect("spawn uia-browser-worker");
        Self { shared, rx }
    }

    /// Spawns the platform-default scanner (Windows STA COM `IUIAutomation`, or no-op elsewhere).
    pub fn spawn_platform() -> Self {
        #[cfg(windows)]
        {
            Self::spawn(Win32UiaScanner::default())
        }
        #[cfg(not(windows))]
        {
            Self::spawn(NoopBrowserScanner)
        }
    }

    /// Non-blocking coalesced scan request: overwrites any unstarted request.
    pub fn request_scan(&self, req: ScanRequest) {
        let (lock, cv) = &*self.shared;
        if let Ok(mut guard) = lock.lock() {
            guard.latest = Some(req);
            cv.notify_one();
        }
    }

    /// Drains the most recent completed snapshot without blocking.
    pub fn try_recv_latest(&self) -> Option<BrowserTabSnapshot> {
        let mut latest = None;
        while let Ok(s) = self.rx.try_recv() {
            latest = Some(s);
        }
        latest
    }
}

impl Drop for AsyncBrowserUiaWorker {
    fn drop(&mut self) {
        let (lock, cv) = &*self.shared;
        if let Ok(mut guard) = lock.lock() {
            guard.stop = true;
            cv.notify_all();
        }
    }
}

#[cfg(not(windows))]
struct NoopBrowserScanner;

#[cfg(not(windows))]
impl BrowserScanner for NoopBrowserScanner {
    fn scan(&mut self, _req: &ScanRequest) -> Option<BrowserTabSnapshot> {
        None
    }
}

#[cfg(windows)]
struct ComStaGuard {
    initialized: bool,
}

#[cfg(windows)]
impl ComStaGuard {
    fn init() -> Self {
        use windows::Win32::System::Com::{CoInitializeEx, COINIT_APARTMENTTHREADED};
        let hr = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) };
        Self {
            initialized: hr.is_ok(),
        }
    }
}

#[cfg(windows)]
impl Drop for ComStaGuard {
    fn drop(&mut self) {
        if self.initialized {
            unsafe { windows::Win32::System::Com::CoUninitialize() };
        }
    }
}

#[cfg(windows)]
#[derive(Default)]
struct Win32UiaScanner;

#[cfg(windows)]
thread_local! {
    static STA_UIA: std::cell::RefCell<Option<windows::Win32::UI::Accessibility::IUIAutomation>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(windows)]
impl Win32UiaScanner {
    fn automation() -> Option<windows::Win32::UI::Accessibility::IUIAutomation> {
        use windows::Win32::System::Com::{CoCreateInstance, CLSCTX_INPROC_SERVER};
        use windows::Win32::UI::Accessibility::CUIAutomation;
        STA_UIA.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.is_none() {
                *slot =
                    unsafe { CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER).ok() };
            }
            slot.clone()
        })
    }
}

#[cfg(windows)]
impl BrowserScanner for Win32UiaScanner {
    fn scan(&mut self, req: &ScanRequest) -> Option<BrowserTabSnapshot> {
        use std::ffi::c_void;
        use windows::core::Interface;
        use windows::Win32::Foundation::HWND;
        use windows::Win32::UI::Accessibility::{
            IUIAutomationValuePattern, TreeScope_Descendants, UIA_ButtonControlTypeId,
            UIA_CheckBoxControlTypeId, UIA_ComboBoxControlTypeId, UIA_DocumentControlTypeId,
            UIA_EditControlTypeId, UIA_HyperlinkControlTypeId, UIA_MenuItemControlTypeId,
            UIA_RadioButtonControlTypeId, UIA_TabItemControlTypeId, UIA_ValuePatternId,
        };

        if !is_browser_exe(&req.exe) {
            return None;
        }
        if is_incognito_title(&req.title) {
            return Some(BrowserTabSnapshot::from_raw(
                req.hwnd,
                &req.exe,
                &req.title,
                None,
                None,
                Vec::new(),
                req.now_ms,
            ));
        }

        let uia = Self::automation()?;
        let hwnd = HWND(req.hwnd as *mut c_void);
        let root = unsafe { uia.ElementFromHandle(hwnd).ok()? };
        let root_rect = unsafe { root.CurrentBoundingRectangle().ok()? };
        let mut doc_rect = [
            root_rect.left as f32,
            root_rect.top as f32,
            root_rect.right as f32,
            root_rect.bottom as f32,
        ];

        let mut found_url: Option<String> = None;
        let mut raw_controls: Vec<(i32, Role, [f32; 4], String, String, bool, bool)> = Vec::new();
        let control_roles = [
            (UIA_ButtonControlTypeId.0, Role::Button),
            (UIA_HyperlinkControlTypeId.0, Role::Link),
            (UIA_EditControlTypeId.0, Role::Input),
            (UIA_TabItemControlTypeId.0, Role::Menu),
            (UIA_MenuItemControlTypeId.0, Role::Menu),
            (UIA_ComboBoxControlTypeId.0, Role::Input),
            (UIA_CheckBoxControlTypeId.0, Role::Button),
            (UIA_RadioButtonControlTypeId.0, Role::Button),
        ];

        // Single pass over descendants (capped at 256 nodes) extracts Document rect+URL, Omnibox URL,
        // and up to 96 interactive controls without blocking or requiring VARIANT/Ole features.
        if let Ok(true_cond) = unsafe { uia.CreateTrueCondition() } {
            if let Ok(all) = unsafe { root.FindAll(TreeScope_Descendants, &true_cond) } {
                let len = unsafe { all.Length().unwrap_or(0) }.min(256);
                for i in 0..len {
                    let Ok(el) = (unsafe { all.GetElement(i) }) else { continue };
                    let Ok(ct) = (unsafe { el.CurrentControlType() }) else { continue };

                    if ct == UIA_DocumentControlTypeId {
                        if let Ok(r) = unsafe { el.CurrentBoundingRectangle() } {
                            if r.right > r.left && r.bottom > r.top {
                                doc_rect = [r.left as f32, r.top as f32, r.right as f32, r.bottom as f32];
                            }
                        }
                        if found_url.is_none() {
                            if let Ok(pat) = unsafe { el.GetCurrentPattern(UIA_ValuePatternId) } {
                                if let Ok(vp) = pat.cast::<IUIAutomationValuePattern>() {
                                    if let Ok(val) = unsafe { vp.CurrentValue() } {
                                        let s = val.to_string();
                                        if s.starts_with("http://") || s.starts_with("https://") {
                                            found_url = Some(s);
                                        }
                                    }
                                }
                            }
                        }
                        continue;
                    }

                    if ct == UIA_EditControlTypeId && found_url.is_none() {
                        if let Ok(pat) = unsafe { el.GetCurrentPattern(UIA_ValuePatternId) } {
                            if let Ok(vp) = pat.cast::<IUIAutomationValuePattern>() {
                                if let Ok(val) = unsafe { vp.CurrentValue() } {
                                    let s = val.to_string();
                                    if s.contains('.') && !s.contains(' ') {
                                        found_url = Some(s);
                                    }
                                }
                            }
                        }
                    }

                    if raw_controls.len() >= 96 {
                        continue;
                    }
                    let Some((_, role)) = control_roles.iter().find(|(id, _)| *id == ct.0) else {
                        continue;
                    };
                    let Ok(br) = (unsafe { el.CurrentBoundingRectangle() }) else { continue };
                    if br.right <= br.left || br.bottom <= br.top {
                        continue;
                    }
                    let name = unsafe { el.CurrentName().map(|b| b.to_string()).unwrap_or_default() };
                    let auto_id =
                        unsafe { el.CurrentAutomationId().map(|b| b.to_string()).unwrap_or_default() };
                    let enabled = unsafe { el.CurrentIsEnabled().map(|b| b.as_bool()).unwrap_or(true) };
                    let sensitive =
                        unsafe { el.CurrentIsPassword().map(|b| b.as_bool()).unwrap_or(false) };
                    raw_controls.push((
                        ct.0,
                        *role,
                        [br.left as f32, br.top as f32, br.right as f32, br.bottom as f32],
                        name,
                        auto_id,
                        enabled,
                        sensitive,
                    ));
                }
            }
        }

        let doc_w = (doc_rect[2] - doc_rect[0]).max(1.0);
        let doc_h = (doc_rect[3] - doc_rect[1]).max(1.0);
        let mut regions = Vec::with_capacity(raw_controls.len());
        for (ct_id, role, br, name, auto_id, enabled, sensitive) in raw_controls {
            let rx = (br[0] - doc_rect[0]) / doc_w;
            let ry = (br[1] - doc_rect[1]) / doc_h;
            let rw = (br[2] - br[0]) / doc_w;
            let rh = (br[3] - br[1]) / doc_h;
            let Some(rect) = Rect::new(rx, ry, rw, rh) else { continue };
            let anchor = if !auto_id.trim().is_empty() {
                format!("uia:{ct_id}:{auto_id}")
            } else {
                format!("uia:{ct_id}:{}:{:.2},{:.2}", prompt_label(&name), rect.x, rect.y)
            };
            regions.push(UiaInteractiveRegion {
                anchor,
                label: name,
                role,
                rect,
                enabled,
                sensitive,
                is_sticky_chrome: false,
            });
        }

        Some(BrowserTabSnapshot::from_raw(
            req.hwnd,
            &req.exe,
            &req.title,
            found_url.as_deref(),
            Some(doc_rect),
            regions,
            req.now_ms,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct MockBrowserScanner;

    impl BrowserScanner for MockBrowserScanner {
        fn scan(&mut self, req: &ScanRequest) -> Option<BrowserTabSnapshot> {
            if !is_browser_exe(&req.exe) {
                return None;
            }
            if is_incognito_title(&req.title) {
                return Some(BrowserTabSnapshot::from_raw(
                    req.hwnd,
                    &req.exe,
                    &req.title,
                    Some("https://www.amazon.com/dp/B099999?session=secret_token_123"),
                    Some([0.0, 0.0, 1280.0, 800.0]),
                    vec![UiaInteractiveRegion {
                        anchor: "uia:buy-incog".into(),
                        label: "지금 구매".into(),
                        role: Role::Button,
                        rect: Rect::new(0.65, 0.40, 0.25, 0.07).unwrap(),
                        enabled: true,
                        sensitive: false,
                        is_sticky_chrome: false,
                    }],
                    req.now_ms,
                ));
            }
            // Normal Chrome tab on amazon.com (Korean-translated page) with query token to strip
            Some(BrowserTabSnapshot::from_raw(
                req.hwnd,
                &req.exe,
                &req.title,
                Some("https://www.amazon.com/dp/B099999?tag=tracker&token=abc987"),
                Some([0.0, 0.0, 1280.0, 800.0]),
                vec![
                    UiaInteractiveRegion {
                        anchor: "uia:cart-btn".into(),
                        label: "장바구니 담기".into(),
                        role: Role::Button,
                        rect: Rect::new(0.65, 0.30, 0.25, 0.07).unwrap(),
                        enabled: true,
                        sensitive: false,
                        is_sticky_chrome: false,
                    },
                    UiaInteractiveRegion {
                        anchor: "uia:buy-now-btn".into(),
                        label: "지금 구매".into(),
                        role: Role::Button,
                        rect: Rect::new(0.65, 0.40, 0.25, 0.07).unwrap(),
                        enabled: true,
                        sensitive: false,
                        is_sticky_chrome: false,
                    },
                    UiaInteractiveRegion {
                        anchor: "uia:wishlist-link".into(),
                        label: "위시리스트에 추가".into(),
                        role: Role::Link,
                        rect: Rect::new(0.65, 0.52, 0.20, 0.05).unwrap(),
                        enabled: true,
                        sensitive: false,
                        is_sticky_chrome: false,
                    },
                ],
                req.now_ms,
            ))
        }
    }

    /// Simulates granite 384d `LABEL_SPACE` multilingual cosine similarity for deterministic tests.
    fn mock_label_space_cos(a: &str, b: &str) -> f32 {
        match (a, b) {
            ("Buy Now", "지금 구매") | ("지금 구매", "Buy Now") => 0.92,
            ("Add to Cart", "장바구니 담기") | ("장바구니 담기", "Add to Cart") => 0.90,
            ("Buy Now", "장바구니 담기") | ("Add to Cart", "지금 구매") => 0.35,
            (x, y) if x.eq_ignore_ascii_case(y) => 1.0,
            _ => 0.10,
        }
    }

    /// Contract WA-09:
    /// Non-blocking async UIA browser watcher isolates Chrome tab `cc` (`amazon.com`, never
    /// `chrome.exe`), strips query tokens, matches Korean-translated `"지금 구매"` 1:1 to English
    /// `"Buy Now"` GoldenPath via `LABEL_SPACE`, and records 0 entries for Incognito tabs.
    #[test]
    fn wa09_async_uia_browser_tab_cc_isolation_incognito_block_and_label_space_matching() {
        let worker = AsyncBrowserUiaWorker::spawn(MockBrowserScanner);
        let mut tracker = BrowserSessionTracker::default();
        let mut permanent_tiers: HashMap<String, Tier> = HashMap::new();
        let mut numberer = StickyNumberer::default();

        // 1. Chrome tab switch to amazon.com (Korean translated UI)
        worker.request_scan(ScanRequest {
            hwnd: 101,
            exe: "chrome.exe".into(),
            title: "Amazon.com: 기계식 키보드 - Google Chrome".into(),
            now_ms: 1_000,
        });
        let mut snap = None;
        for _ in 0..50 {
            if let Some(s) = worker.try_recv_latest() {
                snap = Some(s);
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let snap = snap.expect("async UIA worker returned amazon.com snapshot");
        assert!(!snap.incognito_guard);
        assert_eq!(snap.normalized_cc.as_deref(), Some("amazon.com"), "tab cc must be amazon.com, not chrome.exe");
        assert!(
            !snap.url.as_deref().unwrap_or("").contains("token="),
            "query tokens must be stripped"
        );
        assert!(matches!(
            tracker.observe(&snap),
            TabTransition::ActiveTab { ref cc, .. } if cc == "amazon.com"
        ));
        assert_eq!(tracker.recorded_tabs.len(), 1);

        // 2. Match English GoldenPath "Buy Now" against Korean UIA "지금 구매" via LABEL_SPACE
        let web_rows = vec![
            WebAnalyticTierRow {
                label: "Buy Now".into(),
                role: Role::Button,
                ordinal_norm: 0.5,
                tier: Tier::Premium,
                is_friction_trap: false,
            },
            WebAnalyticTierRow {
                label: "Add to Cart".into(),
                role: Role::Button,
                ordinal_norm: 0.0,
                tier: Tier::Neutral,
                is_friction_trap: false,
            },
        ];
        let proj = project_web_tiers_onto_uia(&web_rows, &snap.regions, mock_label_space_cos);
        assert_eq!(
            proj.tiers.get("uia:buy-now-btn"),
            Some(&Tier::Premium),
            "LABEL_SPACE must map 'Buy Now' GoldenPath 1:1 onto Korean '지금 구매'"
        );
        assert_eq!(proj.matches.len(), 2);

        let (overlay, brief) = build_web_nav_overlay(
            &snap,
            &web_rows,
            &mut permanent_tiers,
            &mut numberer,
            0,
            None,
            mock_label_space_cos,
            1_000,
        )
        .expect("overlay built for normal tab");
        assert_eq!(overlay.cc, "amazon.com");
        assert_eq!(overlay.badges[0].anchor, "uia:buy-now-btn");
        assert_eq!(overlay.badges[0].number, 1);
        assert_eq!(brief.primary.as_ref().map(|p| p.number), Some(1));

        // 3. Switch to Incognito tab -> 0 recorded tabs, 0 NavMap/badges
        worker.request_scan(ScanRequest {
            hwnd: 202,
            exe: "chrome.exe".into(),
            title: "Amazon.com - Google Chrome (시크릿)".into(),
            now_ms: 2_000,
        });
        let mut incog = None;
        for _ in 0..50 {
            if let Some(s) = worker.try_recv_latest() {
                incog = Some(s);
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let incog = incog.expect("async UIA worker returned incognito guard snapshot");
        assert!(incog.incognito_guard);
        assert!(incog.regions.is_empty());
        assert_eq!(tracker.observe(&incog), TabTransition::IncognitoBlocked);
        assert_eq!(tracker.recorded_tabs.len(), 1, "incognito tab must record 0 entries");
        assert!(
            build_web_nav_overlay(
                &incog,
                &web_rows,
                &mut permanent_tiers,
                &mut numberer,
                0,
                None,
                mock_label_space_cos,
                2_000,
            )
            .is_none(),
            "incognito tab produces no NavMap overlay"
        );
    }

    #[test]
    fn sticky_header_clips_occluded_elements_and_runtime_demotion_preserves_permanent_tiers() {
        let snap = BrowserTabSnapshot::from_raw(
            303,
            "msedge.exe",
            "Checkout - Shop",
            Some("https://shop.example.com/checkout"),
            Some([0.0, 0.0, 1280.0, 800.0]),
            vec![
                UiaInteractiveRegion {
                    anchor: "uia:sticky-nav".into(),
                    label: "상단 메뉴".into(),
                    role: Role::Menu,
                    rect: Rect::new(0.0, 0.0, 1.0, 0.12).unwrap(),
                    enabled: true,
                    sensitive: false,
                    is_sticky_chrome: true,
                },
                UiaInteractiveRegion {
                    anchor: "uia:hidden-under-header".into(),
                    label: "가려진 버튼".into(),
                    role: Role::Button,
                    rect: Rect::new(0.2, 0.02, 0.2, 0.08).unwrap(),
                    enabled: true,
                    sensitive: false,
                    is_sticky_chrome: false,
                },
                UiaInteractiveRegion {
                    anchor: "uia:easy-pay".into(),
                    label: "간편결제 (일시품절)".into(),
                    role: Role::Button,
                    rect: Rect::new(0.2, 0.35, 0.3, 0.08).unwrap(),
                    enabled: false,
                    sensitive: false,
                    is_sticky_chrome: false,
                },
                UiaInteractiveRegion {
                    anchor: "uia:card-pay".into(),
                    label: "일반 카드 결제".into(),
                    role: Role::Button,
                    rect: Rect::new(0.2, 0.48, 0.3, 0.08).unwrap(),
                    enabled: true,
                    sensitive: false,
                    is_sticky_chrome: false,
                },
                UiaInteractiveRegion {
                    anchor: "uia:commit-pay".into(),
                    label: "결제하기".into(),
                    role: Role::Button,
                    rect: Rect::new(0.2, 0.65, 0.3, 0.08).unwrap(),
                    enabled: true,
                    sensitive: false,
                    is_sticky_chrome: false,
                },
            ],
            5_000,
        );
        let clipped = clip_sticky_occluders(&snap.regions);
        assert!(clipped.iter().all(|r| r.anchor != "uia:hidden-under-header"));

        let mut perm = HashMap::from([
            ("uia:easy-pay".to_string(), Tier::Premium),
            ("uia:card-pay".to_string(), Tier::Premium),
        ]);
        let mut numberer = StickyNumberer::default();
        let (overlay, brief) = build_web_nav_overlay(
            &snap,
            &[],
            &mut perm,
            &mut numberer,
            0,
            None,
            |_, _| 0.0,
            5_000,
        )
        .unwrap();
        // Permanent tier of easy-pay must remain Premium (not mutated by temporary stockout)
        assert_eq!(perm.get("uia:easy-pay"), Some(&Tier::Premium));
        // Runner-up "일반 카드 결제" inherits #1
        assert_eq!(overlay.badges[0].anchor, "uia:card-pay");
        assert_eq!(overlay.badges[0].number, 1);
        // Final payment button "결제하기" is CommitGuard -> FocusOnly
        let commit_badge = overlay.badges.iter().find(|b| b.anchor == "uia:commit-pay").unwrap();
        assert!(commit_badge.commit_guarded);
        assert_eq!(
            overlay.dispatch_for(commit_badge.number).map(|(_, d)| d),
            Some(SlotDispatch::FocusOnly)
        );
        assert!(brief.commit_safety_guard());
        assert_eq!(brief.runtime_blocked_primary.as_deref(), Some("간편결제"));
    }
}
