//! premium-discount 다리 (마스터 기획서 3.3 (2)): 파이프라인이 premium-discount 의 프로세스 · 저장소를 직접 다루지 않게 한다.
//!
//! - [`PdBridge::resolve_context_for_ask`]: 질문 하나의 맥락. 최근 몇 턴의 Relay Key(생략된 주어)를 질의에 되살려
//!   `pd_search_merged`(내 기록 + 서버, embeddinggemma 768)로 찾고, 원문이 있는 기록끼리 신구 대체(premium-discount-relay
//!   `compare` → `Supersedes`)를 판정해 옛 기록을 discount(`superseded`)로 내린다. 남은 기록마다 내 판정(기기 장부)과 서버
//!   판정을 섞지 않고 `zena_verdict::stance(local, global).line(label)` 로 `[추천]` 줄을 만든다(최대 3줄).
//! - [`PdBridge::record_feedback`] · [`PdBridge::record_outcome`]: 브리핑한 뒤 5분 안의 피드백("덕분에 깼다", "별로였어")과
//!   결과(승리)를 브리핑한 기록에 나눠 기기 장부([`LocalLedger`], PD-Verdict v1 `chat` 트랙)에 쌓고, 사용자가 공유에 동의했으면
//!   `pd_vote`(피드백은 `vote`, 결과는 `vote_outcome`)로 보낸다. 보내지 못한 표는 premium-discount-app 의 보관함이 들고 있다가
//!   [`premium_discount_mcp::SyncLoop`] 가 다시 보낸다.
//!
//! - premium-discount-mcp 가 없거나 뜨지 않거나 검색이 실패하면 패닉 없이 기기 장부만으로 답한다: 예전에 브리핑한 기록(라벨 · 원문)을
//!   장부에 남겨 두고, 질의와 가까운 기록을 골라 내 판정(`stance(local, None)`)으로 `[추천]` 줄을 만든다. 피드백 · 결과도 장부에는
//!   쌓이고 표만 보내지 않는다. 신구 대체는 공유에 동의했을 때 `pd_supersede` 로 서버에 알린다(서버는 작성자의 보고만 받는다).
//!
//! 기획서와 다른 점: `feedback::extract` 는 없다(`feedback_obs` 를 쓴다).
//! 피드백 한 마디를 기록 k 개에 나누는 1×k 행렬에 기획서의 이중 센터링 + 1:1 배정을 쓰면, 센터링은 `v − 평균` 이 되고
//! (관측 하나인 열은 전체 평균을 쓴다) 1:1 배정의 차이는 같은 기록을 두고 다투는 다른 행(피드백)과의 차이라서, 행이 하나면
//! 1위 기록과 2위 기록이 아무리 가까워도 1위를 고른다. 그래서 기록 사이의 차이를 직접 본다: 코사인 1위가 2위보다
//! 차이(margin)만큼 앞설 때 1위에 몰아주고, 아니면 같은 무리끼리 나눈다.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use premium_discount_nms::analyze::Analyzer;
use premium_discount_nms::embed::{cosine, SharedEmbedder, TextEmbedder};
use premium_discount_nms::normalize::normalize;
use premium_discount_relay::{self as relay, Record, Relation, RelayConfig, Sentence, Token};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use zena_verdict::{baseline_of, evidence, feedback_obs, verdict, Baselines, Case, FeedbackIntent, Obs, Params, Prev, Tier};

/// 브리핑한 기록이 피드백 · 결과의 크레딧 후보로 남는 시간
pub const CREDIT_WINDOW_MS: i64 = 5 * 60_000;
/// 한 질문에 내는 `[추천]` 줄 수 (`SessionContext::render` 의 상한과 같다)
pub const MAX_VERDICT_LINES: usize = 3;
/// 피드백을 한 기록에 몰아주려면 코사인 1위가 2위보다 이만큼 앞서야 한다
pub const CREDIT_MARGIN: f32 = 0.03;
const LABEL_CHARS: usize = 32;
const SEARCH_TOP_K: u32 = 5;
/// 오프라인 질문: 장부에 남긴 기록 중 질의와 코사인이 이 이상인 것만
pub const LOCAL_MIN_COS: f32 = 0.3;
/// 장부에 남기는 기록 수 (오래된 것부터 버린다)
pub const LOCAL_RECORDS_MAX: usize = 500;
/// 판정 트랙 (PD-Verdict v1 의 대화 트랙)
pub const TRACK: &str = "chat";

/// premium-discount 의 MCP 도구. 실제는 [`premium_discount_mcp::PdSyncClient`], 시험은 가짜
pub trait PdTools: Send {
    fn search_merged(&mut self, args: Value) -> Result<Value>;
    fn vote(&mut self, id: &str, value: i32, kind: &str) -> Result<Value>;
    /// 신구 대체 보고 (`pd_supersede`). 서버는 두 기록 중 하나의 작성자일 때만 받는다
    fn supersede(&mut self, old: &str, new: &str) -> Result<Value>;
}

impl PdTools for premium_discount_mcp::PdSyncClient {
    fn search_merged(&mut self, args: Value) -> Result<Value> {
        premium_discount_mcp::PdSyncClient::search_merged(self, args)
    }
    fn vote(&mut self, id: &str, value: i32, kind: &str) -> Result<Value> {
        self.call("pd_vote", json!({"id": id, "value": value, "kind": kind}))
    }
    fn supersede(&mut self, old: &str, new: &str) -> Result<Value> {
        self.call("pd_supersede", json!({"old": old, "new": new}))
    }
}

/// 문장 분석: 라우터가 띄운 분석기(Stanza 또는 규칙)와 임베더(granite 또는 해시)를 빌려 Relay 문장 · 키를 만든다.
/// nms 의 낱말에는 머리 번호가 없어 `head` 는 0 이다(수치 뒤의 명사 단위는 붙지 않고, `%` · 기호 단위는 붙는다)
#[derive(Clone)]
pub struct Analysis {
    pub embedder: SharedEmbedder,
    pub analyzer: Arc<dyn Analyzer>,
}

impl Analysis {
    pub fn sentence(&self, text: &str) -> Sentence {
        let mut n = normalize(text);
        self.analyzer.analyze(&mut n);
        Sentence {
            text: text.trim().to_string(),
            tokens: n
                .words
                .iter()
                .map(|w| Token {
                    text: w.surface.clone(),
                    lemma: w.alt.clone().unwrap_or_else(|| w.lemma.clone()),
                    upos: w.upos.clone(),
                    deprel: w.deprel.clone(),
                    head: 0,
                })
                .collect(),
        }
    }

    /// 기록 하나 (문장 끝 · 줄바꿈으로 나눈다)
    pub fn record(&self, text: &str, at_ms: i64) -> Record {
        let sentences = text
            .split_inclusive(['.', '!', '?', '\n', '。'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| self.sentence(s))
            .collect();
        Record { sentences, at_ms }
    }

    /// 글의 Relay Key (정규화한 키 글)
    pub fn keys(&self, text: &str) -> Vec<String> {
        let s = self.sentence(text);
        let mut out: Vec<String> = Vec::new();
        for k in relay::relay_keys(&s) {
            if !out.contains(&k.key) {
                out.push(k.key);
            }
        }
        out
    }
}

/// UUIDv7 의 밀리초 (premium-discount 의 id · request_id). 아니면 None
pub fn uuid7_ms(id: &str) -> Option<i64> {
    let hex: String = id.chars().filter(|c| *c != '-').collect();
    if hex.len() != 32 || hex.as_bytes()[12] != b'7' {
        return None;
    }
    i64::from_str_radix(&hex[..12], 16).ok()
}

/// 기기의 판정 장부: 기록 id 마다 관측(피드백 · 결과 · 내 표)과 대체 여부. 판정은 PD-Verdict v1 `chat` 트랙이고,
/// 기준선은 내 다른 기록들의 값이다(또래와 견준다). 기록이 모자라면 판정하지 않는다(Unrated)
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LocalLedger {
    pub obs: BTreeMap<String, Vec<Obs>>,
    pub superseded: BTreeSet<String>,
    /// 브리핑한 기록의 라벨 · 원문 (오프라인 질문의 후보)
    #[serde(default)]
    pub records: BTreeMap<String, CachedRecord>,
    #[serde(skip)]
    pub path: Option<PathBuf>,
}

impl LocalLedger {
    pub fn open(path: &Path) -> LocalLedger {
        let mut l: LocalLedger = std::fs::read_to_string(path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        l.path = Some(path.to_path_buf());
        l
    }

    pub fn save(&self) {
        if let Some(p) = &self.path {
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            if let Ok(s) = serde_json::to_string(self) {
                let _ = std::fs::write(p, s);
            }
        }
    }

    pub fn add(&mut self, id: &str, obs: Obs) {
        self.obs.entry(id.to_string()).or_default().push(obs);
    }

    pub fn mark_superseded(&mut self, id: &str) {
        self.superseded.insert(id.to_string());
    }

    /// 기기의 판정 (내 다른 기록들의 값을 기준선으로)
    pub fn tier(&self, id: &str, now_ms: i64) -> Tier {
        let params = Params::v1();
        let Some(track) = params.track(TRACK) else { return Tier::Unrated };
        let superseded = self.superseded.contains(id);
        let obs = self.obs.get(id).cloned().unwrap_or_default();
        if obs.is_empty() && !superseded {
            return Tier::Unrated;
        }
        let peers: Vec<f64> = self
            .obs
            .iter()
            .filter(|(k, o)| k.as_str() != id && !o.is_empty())
            .map(|(_, o)| evidence(o, now_ms, track, &params).1)
            .collect();
        let baseline = Baselines { global: (!peers.is_empty()).then(|| baseline_of(&peers)), ..Default::default() };
        let case = Case { now_ms, track: TRACK.into(), prev: Prev { tier: Tier::Unrated, since_ms: 0 }, obs, baseline, duel: None, superseded };
        verdict(&case, &params).map_or(Tier::Unrated, |v| v.tier)
    }
}

/// 장부에 남긴 기록 하나
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CachedRecord {
    pub label: String,
    pub text: String,
    /// 기록을 쓴 시각 (UUIDv7, 모르면 브리핑한 시각)
    pub at_ms: i64,
    /// 마지막으로 브리핑한 시각 (오래된 것부터 버림)
    pub seen_ms: i64,
}

impl LocalLedger {
    /// 브리핑한 기록을 남긴다 ([`LOCAL_RECORDS_MAX`] 개까지)
    pub fn remember(&mut self, id: &str, label: &str, text: &str, at_ms: i64, now_ms: i64) {
        if text.trim().is_empty() && label.trim().is_empty() {
            return;
        }
        self.records.insert(id.to_string(), CachedRecord { label: label.into(), text: text.into(), at_ms, seen_ms: now_ms });
        while self.records.len() > LOCAL_RECORDS_MAX {
            let oldest = self.records.iter().min_by_key(|(_, r)| r.seen_ms).map(|(k, _)| k.clone());
            match oldest {
                Some(k) => {
                    self.records.remove(&k);
                }
                None => break,
            }
        }
    }
}

/// 브리핑한 기록 (크레딧 후보)
#[derive(Debug, Clone, PartialEq)]
pub struct Briefed {
    pub id: String,
    pub label: String,
    pub text: String,
    pub at_ms: i64,
}

/// 질문 하나의 맥락
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AskContext {
    /// `[추천]` 줄 (최대 3)
    pub verdicts: Vec<String>,
    /// `[참고]` 줄 (대체된 옛 기록 알림 등)
    pub context: Vec<String>,
    /// 실제로 쓴 검색 질의 (생략된 주어를 되살린 것)
    pub query: String,
    /// 브리핑한 기록 id
    pub records: Vec<String>,
    /// 신구 대체로 내린 옛 기록 id
    pub superseded: Vec<String>,
    /// premium-discount 없이 기기 장부만으로 만든 맥락
    pub local_only: bool,
}

/// 피드백 · 결과 하나를 나눈 결과
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Credited {
    /// (기록 id, 몫, 부호)
    pub credits: Vec<(String, f64, i32)>,
    /// `pd_vote` 로 보낸 수 (공유에 동의했을 때만)
    pub votes_sent: usize,
}

/// 생략된 주어 되살리기: 질의에 없는 활성 키를 앞에 붙인다 ("그럼 2페이즈는?" + [말레니아] → "말레니아 그럼 2페이즈는?")
pub fn restore_query(query: &str, active_keys: &[String]) -> String {
    let lower = query.to_lowercase();
    let missing: Vec<&str> = active_keys.iter().map(String::as_str).filter(|k| !k.is_empty() && !lower.contains(&k.to_lowercase())).take(3).collect();
    if missing.is_empty() {
        query.trim().to_string()
    } else {
        format!("{} {}", missing.join(" "), query.trim())
    }
}

/// 피드백 한 마디의 크레딧: 후보 하나면 전부, 코사인 1위가 2위보다 [`CREDIT_MARGIN`] 이상 앞서면 1위에 전부, 아니면 1위와
/// margin 안의 무리가 똑같이 나눈다
pub fn assign_credit(scores: &[f32]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    if scores.len() == 1 {
        return vec![1.0];
    }
    let top = scores.iter().copied().fold(f32::MIN, f32::max);
    let group: Vec<bool> = scores.iter().map(|s| top - s < CREDIT_MARGIN).collect();
    let n = group.iter().filter(|g| **g).count() as f64;
    group.iter().map(|g| if *g { 1.0 / n } else { 0.0 }).collect()
}

fn label_of(text: Option<&str>, id: &str) -> String {
    match text.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => {
            let one_line = t.split_whitespace().collect::<Vec<_>>().join(" ");
            if one_line.chars().count() > LABEL_CHARS {
                format!("{}…", one_line.chars().take(LABEL_CHARS).collect::<String>())
            } else {
                one_line
            }
        }
        None => format!("#{}", id.chars().take(8).collect::<String>()),
    }
}

fn tier_of(name: Option<&str>) -> Option<Tier> {
    match name? {
        "premium" => Some(Tier::Premium),
        "discount" => Some(Tier::Discount),
        "neutral" => Some(Tier::Neutral),
        "unrated" => Some(Tier::Unrated),
        _ => None,
    }
}

/// 검색 결과 하나 (`pd_search_merged` 의 matches)
#[derive(Debug, Clone, PartialEq)]
struct Hit {
    id: String,
    text: Option<String>,
    global: Option<Tier>,
    my_vote: Option<i64>,
    at_ms: Option<i64>,
}

fn hits_of(v: &Value) -> Vec<Hit> {
    v.get("matches")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| {
            let id = m.get("id").and_then(Value::as_str).or_else(|| m.get("request_id").and_then(Value::as_str))?.to_string();
            let request_id = m.get("request_id").and_then(Value::as_str);
            Some(Hit {
                at_ms: uuid7_ms(&id).or_else(|| request_id.and_then(uuid7_ms)),
                id,
                text: m.get("text").and_then(Value::as_str).map(str::to_string),
                global: tier_of(m.get("verdict").and_then(|v| v.get("tier")).and_then(Value::as_str)),
                my_vote: m.get("my_vote").and_then(Value::as_i64),
            })
        })
        .collect()
}

pub struct PdBridge<T: PdTools> {
    /// premium-discount-mcp 클라이언트. None 이면 기기 장부만으로 답한다
    pub tools: Option<Arc<std::sync::Mutex<T>>>,
    pub analysis: Analysis,
    pub relay: RelayConfig,
    pub ledger: LocalLedger,
    /// 사용자가 피드백 · 결과를 premium-discount 표로 공유하는 데 동의했는가
    pub share: bool,
    briefed: VecDeque<Briefed>,
    /// 표를 보낸 뒤 동기화를 앞당길 곳 (실제 클라이언트일 때)
    pub on_vote: Option<Box<dyn Fn() + Send>>,
}

impl<T: PdTools> PdBridge<T> {
    pub fn new(tools: Arc<std::sync::Mutex<T>>, analysis: Analysis, ledger: LocalLedger, share: bool) -> PdBridge<T> {
        PdBridge { tools: Some(tools), analysis, relay: RelayConfig::default(), ledger, share, briefed: VecDeque::new(), on_vote: None }
    }

    /// premium-discount 없이: 기기 장부만으로 답하는 다리
    pub fn local(analysis: Analysis, ledger: LocalLedger) -> PdBridge<T> {
        PdBridge { tools: None, analysis, relay: RelayConfig::default(), ledger, share: false, briefed: VecDeque::new(), on_vote: None }
    }

    /// 장부에 남긴 기록 중 질의와 가까운 것 ([`LOCAL_MIN_COS`] 이상, 가까운 순)
    fn search_local(&self, query: &str) -> Vec<Hit> {
        let q = self.analysis.embedder.embed(query);
        let mut scored: Vec<(f32, Hit)> = self
            .ledger
            .records
            .iter()
            .map(|(id, r)| {
                let t = if r.text.is_empty() { r.label.as_str() } else { r.text.as_str() };
                let hit = Hit { id: id.clone(), text: Some(r.text.clone()).filter(|t| !t.is_empty()), global: None, my_vote: None, at_ms: Some(r.at_ms) };
                (cosine(&q, &self.analysis.embedder.embed(t)), hit)
            })
            .filter(|(c, _)| *c >= LOCAL_MIN_COS)
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.id.cmp(&b.1.id)));
        scored.into_iter().take(SEARCH_TOP_K as usize).map(|(_, h)| h).collect()
    }

    /// 서버 검색 (생략된 주어의 대상으로 먼저, 없으면 대상 없이). 클라이언트가 없거나 실패하면 Err
    fn search_remote(&self, full: &str, cc: Option<&str>, active_keys: &[String]) -> Result<Vec<Hit>> {
        let mut hits = match active_keys.first() {
            Some(k) => self.search(full, cc, Some(k))?,
            None => Vec::new(),
        };
        if hits.is_empty() {
            hits = self.search(full, cc, None)?;
        }
        Ok(hits)
    }

    fn search(&self, query: &str, cc: Option<&str>, to_key: Option<&str>) -> Result<Vec<Hit>> {
        let tools = self.tools.as_ref().ok_or_else(|| anyhow::anyhow!("premium-discount is not connected"))?;
        let mut args = json!({"query": query, "top_k": SEARCH_TOP_K});
        if let Some(cc) = cc {
            args["cc"] = json!(cc);
        }
        if let Some(k) = to_key {
            args["to_key"] = json!(k);
        }
        let v = tools.lock().map_err(|_| anyhow::anyhow!("premium-discount client lock poisoned"))?.search_merged(args)?;
        Ok(hits_of(&v))
    }

    /// 질문 하나의 맥락 (모듈 설명). premium-discount 가 없거나 검색이 실패하면 기기 장부의 기록으로 답한다
    pub fn resolve_context_for_ask(&mut self, cc: Option<&str>, active_keys: &[String], query: &str, now_ms: i64) -> Result<AskContext> {
        let full = restore_query(query, active_keys);
        // 가장 최근 주제의 대상(to_key)으로 먼저 좁히고, 없으면 대상 없이. 서버가 없거나 실패하면 장부에서
        let (hits, local_only) = match self.search_remote(&full, cc, active_keys) {
            Ok(h) => (h, false),
            Err(e) => {
                if self.tools.is_some() {
                    eprintln!("[pd] search failed ({e:#}); answering from the local ledger");
                }
                (self.search_local(&full), true)
            }
        };
        // 신구 대체: 원문과 시각이 있는 기록끼리, 늦은 기록이 이른 기록을 대체하는가
        let mut superseded: Vec<String> = Vec::new();
        let mut pairs: Vec<(String, String)> = Vec::new();
        let with_text: Vec<(&Hit, Record)> = hits
            .iter()
            .filter_map(|h| Some((h, self.analysis.record(h.text.as_deref()?, h.at_ms?))))
            .filter(|(_, r)| !r.sentences.is_empty())
            .collect();
        if with_text.len() >= 2 {
            let keys: Vec<relay::RelayKey> = with_text.iter().flat_map(|(_, r)| r.sentences.iter().flat_map(relay::relay_keys)).collect();
            let clusters = relay::cluster_keys(&keys, &self.analysis.embedder, &self.relay);
            for (old_hit, old) in &with_text {
                for (new_hit, new) in &with_text {
                    if new.at_ms <= old.at_ms || superseded.contains(&old_hit.id) {
                        continue;
                    }
                    if let (Relation::Supersedes { .. }, _) = relay::compare(old, new, &clusters, &self.analysis.embedder, &self.relay) {
                        superseded.push(old_hit.id.clone());
                        pairs.push((old_hit.id.clone(), new_hit.id.clone()));
                    }
                }
            }
        }
        for id in &superseded {
            self.ledger.mark_superseded(id);
        }
        if !local_only {
            self.report_supersessions(&pairs);
        }
        let mut out = AskContext { query: full, superseded: superseded.clone(), local_only, ..Default::default() };
        let locale = zena_locale::current();
        for h in &hits {
            let label = label_of(h.text.as_deref(), &h.id);
            if superseded.contains(&h.id) {
                out.context.push(zena_locale::fill(locale.app("superseded"), &[("label", &label)]));
                continue;
            }
            if out.verdicts.len() >= MAX_VERDICT_LINES {
                continue;
            }
            let ledger = self.ledger.tier(&h.id, now_ms);
            let local = if ledger != Tier::Unrated {
                Some(ledger)
            } else {
                match h.my_vote {
                    Some(1) => Some(Tier::Premium),
                    Some(-1) => Some(Tier::Discount),
                    _ => None,
                }
            };
            out.verdicts.push(zena_verdict::Stance::from_tiers(local, h.global).line(&label));
            out.records.push(h.id.clone());
            let text = h.text.clone().unwrap_or_default();
            self.ledger.remember(&h.id, &label, &text, h.at_ms.unwrap_or(now_ms), now_ms);
            self.briefed.push_back(Briefed { id: h.id.clone(), label, text, at_ms: now_ms });
        }
        if !superseded.is_empty() || !out.records.is_empty() {
            self.ledger.save();
        }
        Ok(out)
    }

    /// 신구 대체를 서버에 알린다 (공유에 동의했을 때만). 서버가 받지 않는 보고(작성자가 아님 등)는 기록만 남긴다
    fn report_supersessions(&mut self, pairs: &[(String, String)]) {
        if !self.share || pairs.is_empty() {
            return;
        }
        let Some(tools) = &self.tools else { return };
        if let Ok(mut t) = tools.lock() {
            for (old, new) in pairs {
                if let Err(e) = t.supersede(old, new) {
                    eprintln!("[pd] supersede {old} -> {new}: {e:#}");
                }
            }
        }
    }

    fn recent(&mut self, now_ms: i64) -> Vec<Briefed> {
        while self.briefed.front().is_some_and(|b| now_ms - b.at_ms > CREDIT_WINDOW_MS) {
            self.briefed.pop_front();
        }
        // 같은 기록을 여러 번 브리핑했으면 가장 최근 것 하나
        let mut seen = BTreeSet::new();
        let mut out: Vec<Briefed> = self.briefed.iter().rev().filter(|b| seen.insert(b.id.clone())).cloned().collect();
        out.reverse();
        out
    }

    fn send_votes(&mut self, votes: &[(String, i32, &str)]) -> usize {
        if !self.share || votes.is_empty() {
            return 0;
        }
        let mut sent = 0;
        let Some(tools) = &self.tools else { return 0 };
        if let Ok(mut t) = tools.lock() {
            for (id, value, kind) in votes {
                match t.vote(id, *value, kind) {
                    Ok(_) => sent += 1,
                    Err(e) => eprintln!("[pd] vote {id}: {e:#}"),
                }
            }
        }
        if sent > 0 {
            if let Some(f) = &self.on_vote {
                f();
            }
        }
        sent
    }

    /// 피드백 한 마디(사전 · 부정 분석으로 고른 의도)를 최근 5분 안에 브리핑한 기록에 나눈다. 후보가 없으면 None
    pub fn record_feedback(&mut self, intent: FeedbackIntent, negated: bool, utterance: &str, now_ms: i64) -> Option<Credited> {
        let candidates = self.recent(now_ms);
        if candidates.is_empty() {
            return None;
        }
        let u = self.analysis.embedder.embed(utterance);
        let scores: Vec<f32> = candidates
            .iter()
            .map(|b| {
                let t = if b.text.is_empty() { b.label.as_str() } else { b.text.as_str() };
                cosine(&u, &self.analysis.embedder.embed(t))
            })
            .collect();
        let params = Params::v1();
        let mut out = Credited::default();
        let mut votes: Vec<(String, i32, &str)> = Vec::new();
        for (b, credit) in candidates.iter().zip(assign_credit(&scores)) {
            if credit <= 0.0 {
                continue;
            }
            let obs = feedback_obs(intent, negated, credit, now_ms, &params);
            out.credits.push((b.id.clone(), credit, obs.sign));
            // 표는 몫이 온전한(한 기록에 몰린) 피드백만: 애매한 피드백을 서버 표로 만들지 않는다
            if credit >= 1.0 && obs.sign != 0 {
                votes.push((b.id.clone(), obs.sign, "vote"));
            }
            self.ledger.add(&b.id, obs);
        }
        self.ledger.save();
        out.votes_sent = self.send_votes(&votes);
        Some(out)
    }

    /// 결과(승리 +1)를 최근 5분 안에 브리핑한 기록에 똑같이 나눈다. 후보가 없으면 None
    pub fn record_outcome(&mut self, sign: i32, now_ms: i64) -> Option<Credited> {
        let candidates = self.recent(now_ms);
        if candidates.is_empty() || sign == 0 {
            return None;
        }
        let params = Params::v1();
        let credit = 1.0 / candidates.len() as f64;
        let mut out = Credited::default();
        let mut votes: Vec<(String, i32, &str)> = Vec::new();
        for b in &candidates {
            let obs = Obs { kind: "outcome".into(), sign: sign.signum(), weight: params.weight("outcome"), credit, at_ms: now_ms };
            out.credits.push((b.id.clone(), credit, obs.sign));
            votes.push((b.id.clone(), obs.sign, "vote_outcome"));
            self.ledger.add(&b.id, obs);
        }
        self.ledger.save();
        out.votes_sent = self.send_votes(&votes);
        Some(out)
    }
}

/// 파이프라인이 쓰는 다리 (실제 premium-discount-mcp 클라이언트)
pub type SharedBridge = Arc<std::sync::Mutex<PdBridge<premium_discount_mcp::PdSyncClient>>>;

/// 실제 다리: premium-discount-mcp 를 찾아 띄우고(그 서버가 premium-discount-app 을 띄운다), 동기화 루프를 붙이고, 기기 장부를
/// `%LOCALAPPDATA%\RaonZena\premium-discount\zena_ledger.json` 에서 연다. 서버가 없거나 뜨지 않으면 기기 장부만 쓰는 다리
/// ([`PdBridge::local`]): 제나는 장부에 남은 기록과 내 판정으로 답하고, 피드백 · 결과도 장부에 쌓는다
pub fn connect(analysis: Analysis, share: bool) -> Option<SharedBridge> {
    let ledger = LocalLedger::open(&premium_discount_mcp::default_home().join("zena_ledger.json"));
    let local = |analysis, ledger| Some(Arc::new(std::sync::Mutex::new(PdBridge::local(analysis, ledger))));
    let Some(server) = premium_discount_mcp::find_mcp_server() else {
        eprintln!("[pd] premium-discount-mcp not found ($PD_MCP, next to raon-zena, install folders); answering from the local ledger");
        return local(analysis, ledger);
    };
    let client = match premium_discount_mcp::PdSyncClient::spawn(&server, &[]) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[pd] {}: {e:#}; answering from the local ledger", server.display());
            return local(analysis, ledger);
        }
    };
    let tools = Arc::new(std::sync::Mutex::new(client));
    let sync = Arc::new(premium_discount_mcp::SyncLoop::start(tools.clone(), premium_discount_mcp::SYNC_INTERVAL, None));
    let mut bridge = PdBridge::new(tools, analysis, ledger, share);
    bridge.on_vote = Some(Box::new(move || sync.request(true)));
    eprintln!("[pd] premium-discount bridge: {} (share votes: {share})", server.display());
    Some(Arc::new(std::sync::Mutex::new(bridge)))
}

/// Lv3 질문의 맥락 단계: LLM 스레드에서 `budget` 안에 다리를 부르고, 받은 `[추천]` · `[참고]` 줄을 상황에 넣는다.
/// 늦으면 맥락 없이 답한다(다리의 호출은 끝까지 돌고 결과는 버린다)
pub fn ask_step<T: PdTools + 'static>(
    bridge: Arc<std::sync::Mutex<PdBridge<T>>>,
    cc: Option<String>,
    keys: Vec<String>,
    query: String,
    now_ms: i64,
    budget: std::time::Duration,
) -> crate::llm::PrepareStep {
    Box::new(move |sit: &mut zena_core::prompt::Situation| {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let r = match bridge.lock() {
                Ok(mut b) => b.resolve_context_for_ask(cc.as_deref(), &keys, &query, now_ms),
                Err(_) => Err(anyhow::anyhow!("premium-discount bridge lock poisoned")),
            };
            let _ = tx.send(r);
        });
        match rx.recv_timeout(budget) {
            Ok(Ok(ctx)) => {
                eprintln!("[pd] ask '{}': {} verdict line(s), {} note(s)", ctx.query, ctx.verdicts.len(), ctx.context.len());
                if let Some(s) = sit.session.as_mut() {
                    s.verdicts = ctx.verdicts;
                }
                sit.context.extend(ctx.context);
            }
            Ok(Err(e)) => eprintln!("[pd] ask: {e:#}"),
            Err(_) => eprintln!("[pd] ask: no context within {} ms", budget.as_millis()),
        }
    })
}

/// 피드백을 다리에 넘긴다 (표 보내기가 막지 않게 따로 돈다)
pub fn feedback_in_background<T: PdTools + 'static>(bridge: Option<&Arc<std::sync::Mutex<PdBridge<T>>>>, intent: FeedbackIntent, negated: bool, text: &str, now_ms: i64) {
    let Some(b) = bridge.cloned() else { return };
    let text = text.to_string();
    std::thread::spawn(move || {
        if let Ok(mut b) = b.lock() {
            if let Some(c) = b.record_feedback(intent, negated, &text, now_ms) {
                eprintln!("[pd] feedback {intent:?}{}: {:?}, {} vote(s)", if negated { " (negated)" } else { "" }, c.credits, c.votes_sent);
            }
        }
    });
}

/// 결과(승리 +1)를 다리에 넘긴다
pub fn outcome_in_background<T: PdTools + 'static>(bridge: Option<&Arc<std::sync::Mutex<PdBridge<T>>>>, sign: i32, now_ms: i64) {
    let Some(b) = bridge.cloned() else { return };
    std::thread::spawn(move || {
        if let Ok(mut b) = b.lock() {
            if let Some(c) = b.record_outcome(sign, now_ms) {
                eprintln!("[pd] outcome {sign:+}: {:?}, {} vote(s)", c.credits, c.votes_sent);
            }
        }
    });
}

fn last_app() -> &'static std::sync::Mutex<Option<String>> {
    static LAST: std::sync::OnceLock<std::sync::Mutex<Option<String>>> = std::sync::OnceLock::new();
    LAST.get_or_init(|| std::sync::Mutex::new(None))
}

/// 앞 창의 앱(제나 자신의 창은 빼고)을 기억한다. 파이프라인이 매 바퀴 부른다
pub fn note_foreground() {
    if let Some(uri) = premium_discount_window::exclude_self(premium_discount_window::foreground_window()).and_then(|w| w.app_uri()) {
        if let Ok(mut l) = last_app().lock() {
            *l = Some(uri);
        }
    }
}

/// 점 아래 창(제나 자신은 빼고)의 앱을 기억한다: 캐릭터를 다른 창 위로 끌어 놓으면 그 앱이 질문 범위가 된다(다음에 앞 창이
/// 바뀌기 전까지). 기억한 앱 URI 를 돌려준다
pub fn note_window_at(x: i32, y: i32) -> Option<String> {
    let uri = premium_discount_window::exclude_self(premium_discount_window::window_at(x, y)).and_then(|w| w.app_uri())?;
    if let Ok(mut l) = last_app().lock() {
        *l = Some(uri.clone());
    }
    Some(uri)
}

/// premium-discount 와 같은 cc (`premium_discount_schema::envelope::normalize_cc`): `app://EldenRing.exe/보스방` → `eldenring.exe`,
/// 웹 주소는 호스트. 정규화할 수 없으면 None
pub fn normalized_cc(raw: &str) -> Option<String> {
    premium_discount_schema::envelope::normalize_cc(raw).ok().map(|c| c.cc)
}

/// 질문 검색 · 코스의 범위(`cc`): 마지막으로 앞에 있던(또는 캐릭터를 끌어 놓은) 다른 앱을 정규화한 것 (`eldenring.exe`).
/// 채팅 창에 쓰는 동안은 제나가 앞이므로 그 전의 앱
pub fn foreground_cc() -> Option<String> {
    note_foreground();
    last_app().lock().ok().and_then(|l| l.clone()).and_then(|uri| normalized_cc(&uri))
}

/// 사용자의 말에서 피드백 의도를 찾는다: 언어팩의 `app.feedback_recommend` · `app.feedback_not_recommend` 글이 들어 있으면
/// 그 의도, 그 낱말이 부정 범위 안이면(분석기의 부정 표시: "안 통했어") 부정. 둘 다 걸리면 긴 글이 이긴다
pub fn detect_feedback(text: &str, analysis: &Analysis) -> Option<(FeedbackIntent, bool)> {
    let locale = zena_locale::current();
    let compact = |s: &str| s.split_whitespace().collect::<String>().to_lowercase();
    let t = compact(text);
    let mut best: Option<(usize, FeedbackIntent, String)> = None;
    for (key, intent) in [("feedback_recommend", FeedbackIntent::Recommend), ("feedback_not_recommend", FeedbackIntent::NotRecommend)] {
        for phrase in locale.app_list(key) {
            let p = compact(&phrase);
            if !p.is_empty() && t.contains(&p) && best.as_ref().is_none_or(|(len, _, _)| p.chars().count() > *len) {
                best = Some((p.chars().count(), intent, p));
            }
        }
    }
    let (_, intent, phrase) = best?;
    let mut n = normalize(text);
    analysis.analyzer.analyze(&mut n);
    let negated = n.words.iter().any(|w| w.negated && (phrase.contains(&compact(&w.surface)) || compact(&w.surface).contains(&phrase)));
    Some((intent, negated))
}

#[cfg(test)]
mod tests {
    use super::*;
    use premium_discount_nms::analyze::RuleAnalyzer;
    use premium_discount_nms::embed::HashedNgramEmbedder;

    #[derive(Default)]
    struct FakePd {
        searches: Vec<Value>,
        votes: Vec<(String, i32, String)>,
        supersedes: Vec<(String, String)>,
        reply: Value,
        /// 검색이 실패한다 (premium-discount-app 이 없음)
        down: bool,
    }

    impl PdTools for FakePd {
        fn search_merged(&mut self, args: Value) -> Result<Value> {
            self.searches.push(args.clone());
            if self.down {
                anyhow::bail!("unavailable: premium-discount-app is not running");
            }
            // 대상(to_key)으로 좁힌 검색은 비어 있다고 한다
            if args.get("to_key").is_some() {
                return Ok(json!({"matches": []}));
            }
            Ok(self.reply.clone())
        }
        fn vote(&mut self, id: &str, value: i32, kind: &str) -> Result<Value> {
            self.votes.push((id.to_string(), value, kind.to_string()));
            Ok(json!({"status": "sent"}))
        }
        fn supersede(&mut self, old: &str, new: &str) -> Result<Value> {
            self.supersedes.push((old.to_string(), new.to_string()));
            Ok(json!({"old": old, "new": new}))
        }
    }

    fn analysis() -> Analysis {
        Analysis { embedder: SharedEmbedder::new(HashedNgramEmbedder::default()), analyzer: Arc::new(RuleAnalyzer) }
    }

    fn bridge(reply: Value, share: bool) -> PdBridge<FakePd> {
        let tools = Arc::new(std::sync::Mutex::new(FakePd { reply, ..Default::default() }));
        PdBridge::new(tools, analysis(), LocalLedger::default(), share)
    }

    const ID_OLD: &str = "0190f0a0-0000-7000-8000-000000000001";
    const ID_NEW: &str = "0191f0a0-0000-7000-8000-000000000002";
    const ID_SERVER: &str = "0192f0a0-0000-7000-8000-000000000003";

    /// 장부에 남은 기록 둘 (예전에 브리핑한 것)과 그 판정
    fn remembered_ledger() -> LocalLedger {
        let mut l = LocalLedger::default();
        l.remember(ID_OLD, "말레니아 공략", "말레니아 물 새 공격은 왼쪽으로 구르면 피한다", 1, 10);
        l.remember(ID_SERVER, "인벤토리", "물약은 두 번째 가방에 있다", 2, 10);
        let params = Params::v1();
        for i in 0..6 {
            l.add(ID_OLD, feedback_obs(FeedbackIntent::Recommend, false, 1.0, 100 + i, &params));
            l.add(ID_SERVER, feedback_obs(FeedbackIntent::NotRecommend, false, 1.0, 100 + i, &params));
        }
        // 판정의 기준선이 될 또래 (원문 없는 기록 10개, 칭찬과 혹평이 섞임): 검색에는 나오지 않는다
        for r in 0..10 {
            for k in 0..6 {
                let intent = if (r + k) % 2 == 0 { FeedbackIntent::Recommend } else { FeedbackIntent::NotRecommend };
                l.add(&format!("peer{r}"), feedback_obs(intent, false, 1.0, 100 + k as i64, &params));
            }
        }
        l
    }

    #[test]
    fn without_premium_discount_the_ledger_answers_and_keeps_feedback() {
        let mut b: PdBridge<FakePd> = PdBridge::local(analysis(), remembered_ledger());
        let ctx = b.resolve_context_for_ask(Some("eldenring.exe"), &[], "말레니아 물 새 공격 어떻게 피해?", 1_000).unwrap();
        assert!(ctx.local_only);
        assert_eq!(ctx.records.first().map(String::as_str), Some(ID_OLD), "{ctx:?}");
        // 줄 이름은 원문에서 (서버 결과와 같은 규칙), 판정은 기기 장부의 것만
        assert_eq!(ctx.verdicts[0], "말레니아 물 새 공격은 왼쪽으로 구르면 피한다: 내 기록으로는 추천 (서버 판정 전)", "{ctx:?}");
        assert!(!ctx.records.contains(&ID_SERVER.to_string()) || ctx.verdicts.len() > 1);
        // 피드백은 장부에 쌓이고 표는 보내지 않는다 (보낼 곳이 없다)
        let c = b.record_feedback(FeedbackIntent::Recommend, false, "덕분에 피했다", 2_000).unwrap();
        assert_eq!(c.votes_sent, 0);
        assert_eq!(b.ledger.obs[ID_OLD].len(), 7);
        // 아무 기록도 남지 않았으면 맥락 없이 (패닉하지 않는다)
        let mut empty: PdBridge<FakePd> = PdBridge::local(analysis(), LocalLedger::default());
        let none = empty.resolve_context_for_ask(None, &[], "아무거나", 1).unwrap();
        assert!(none.verdicts.is_empty() && none.local_only);
    }

    #[test]
    fn a_failing_search_falls_back_to_the_ledger_and_remembers_what_it_briefed() {
        let reply = json!({"matches": [{"id": ID_NEW, "text": "말레니아 2페이즈는 물 새 공격을 왼쪽으로 피한다", "verdict": {"tier": "premium"}}]});
        let mut b = bridge(reply, true);
        let ok = b.resolve_context_for_ask(None, &[], "말레니아 2페이즈", 1_000).unwrap();
        assert!(!ok.local_only);
        assert!(b.ledger.records.contains_key(ID_NEW), "briefed records are remembered for offline answers");
        b.tools.as_ref().unwrap().lock().unwrap().down = true;
        let off = b.resolve_context_for_ask(None, &[], "말레니아 2페이즈 물 새", 2_000).unwrap();
        assert!(off.local_only);
        assert_eq!(off.records, [ID_NEW]);
        assert!(off.verdicts[0].contains("말레니아 2페이즈"), "{off:?}");
    }

    /// 실제 premium-discount-mcp 와의 왕복 (`--ignored`, `PD_MCP=<premium-discount-mcp 실행 파일>`). premium-discount-app 이 없는
    /// 자리를 가리켜 띄우면 도구 목록은 왕복되고(pd_search_merged · pd_vote · pd_supersede), 앱이 필요한 검색은 unavailable 이라
    /// 다리는 기기 장부로 답한다
    #[test]
    #[ignore = "PD_MCP=<premium-discount-mcp>"]
    fn a_real_mcp_server_round_trip_falls_back_to_the_ledger_without_the_app() {
        let server = std::path::PathBuf::from(std::env::var("PD_MCP").expect("PD_MCP"));
        std::env::set_var("PD_APP", std::env::temp_dir().join("no-such-dir").join("premium-discount-app.exe"));
        let mut mcp = premium_discount_mcp_protocol::McpClient::spawn(&server, &[]).expect("spawn premium-discount-mcp");
        let tools = mcp.list_tools().expect("tools/list");
        for t in ["pd_search_merged", "pd_vote", "pd_supersede", "pd_verdicts"] {
            assert!(tools.iter().any(|x| x == t), "{t} in {tools:?}");
        }
        let client = premium_discount_mcp::PdSyncClient::from_client(mcp);
        let mut b = PdBridge::new(Arc::new(std::sync::Mutex::new(client)), analysis(), remembered_ledger(), false);
        let ctx = b.resolve_context_for_ask(Some("eldenring.exe"), &[], "말레니아 물 새 공격 어떻게 피해?", 1_000).unwrap();
        assert!(ctx.local_only, "the search needs premium-discount-app: {ctx:?}");
        assert_eq!(ctx.records.first().map(String::as_str), Some(ID_OLD));
    }

    #[test]
    fn app_uris_become_premium_discount_cc() {
        assert_eq!(normalized_cc("app://EldenRing.exe").as_deref(), Some("eldenring.exe"));
        assert_eq!(normalized_cc("app://eldenring.exe/보스방").as_deref(), Some("eldenring.exe"), "the scene is not part of cc");
        assert_eq!(normalized_cc("https://Store.SteamPowered.com/app/1245620").as_deref(), Some("store.steampowered.com"));
        assert_eq!(normalized_cc("  "), None);
    }

    #[test]
    fn uuid7_time_and_query_restoring() {
        assert_eq!(uuid7_ms(ID_OLD), Some(0x0190f0a00000));
        assert_eq!(uuid7_ms("not-a-uuid"), None);
        assert_eq!(uuid7_ms("0190f0a0-0000-4000-8000-000000000001"), None, "v4 has no time");
        assert_eq!(restore_query("그럼 2페이즈는?", &["말레니아".into()]), "말레니아 그럼 2페이즈는?");
        assert_eq!(restore_query("말레니아 2페이즈는?", &["말레니아".into()]), "말레니아 2페이즈는?");
    }

    #[test]
    fn credit_goes_to_the_clear_winner_or_is_shared() {
        assert_eq!(assign_credit(&[0.4]), [1.0]);
        assert_eq!(assign_credit(&[0.9, 0.5, 0.2]), [1.0, 0.0, 0.0]);
        assert_eq!(assign_credit(&[0.70, 0.69, 0.2]), [0.5, 0.5, 0.0]);
        assert!(assign_credit(&[]).is_empty());
    }

    #[test]
    fn ask_context_lines_follow_the_stance_local_and_global() {
        let reply = json!({"matches": [
            {"id": ID_NEW, "text": "말레니아 공략: 물 새 공격 피해 15% 감소.", "verdict": {"tier": "premium"}, "my_vote": null},
            {"id": ID_OLD, "text": "말레니아 공략: 물 새 공격 피해 20% 감소.", "verdict": {"tier": "neutral"}, "my_vote": 1},
            {"id": ID_SERVER, "verdict": {"tier": "discount"}}
        ]});
        let mut b = bridge(reply, false);
        let ctx = b.resolve_context_for_ask(Some("eldenring.exe"), &["말레니아".into()], "그럼 물 새 공격은?", 1_000).unwrap();
        // 대상(to_key)으로 먼저 찾고, 없으면 대상 없이
        let searches = &b.tools.as_ref().unwrap().lock().unwrap().searches;
        assert_eq!(searches.len(), 2);
        assert_eq!(searches[0]["to_key"], "말레니아");
        assert_eq!(searches[1]["query"], "말레니아 그럼 물 새 공격은?");
        assert_eq!(searches[1]["cc"], "eldenring.exe");
        // 규칙 분석기에서는 주장(주어 · 목적어 · 뿌리)을 못 뽑아 대체를 판정하지 않는다: 옛 기록도 줄로 남는다
        assert!(ctx.superseded.is_empty());
        assert_eq!(ctx.verdicts.len(), 3, "{ctx:?}");
        assert!(ctx.verdicts[0].starts_with("말레니아 공략: 물 새 공격 피해 15% 감소.: 다른 사용자들은 추천"), "{ctx:?}");
        // 옛 기록: 내 표(1)는 추천, 서버는 보통 → 둘 다 말한다
        assert!(ctx.verdicts[1].ends_with("내 기록에선 추천인데 서버에선 보통 (둘 다 말할 것)"), "{ctx:?}");
        assert!(ctx.verdicts[2].starts_with("#0192f0a0: 다른 사용자들은 비추천"), "{ctx:?}");
        assert_eq!(ctx.records.len(), 3);
    }

    /// Stanza 대신 쓰는 시험용 분석기: 첫 낱말은 고유명사 주어, '공격' 은 목적어, 마지막 낱말은 뿌리, 숫자가 든 낱말은 수사
    struct TagAnalyzer;
    impl Analyzer for TagAnalyzer {
        fn name(&self) -> &'static str {
            "test-tags"
        }
        fn analyze(&self, n: &mut premium_discount_nms::normalize::Normalized) {
            let last = n.words.len().saturating_sub(1);
            for (i, w) in n.words.iter_mut().enumerate() {
                let (upos, deprel) = if w.surface.chars().any(|c| c.is_ascii_digit()) {
                    ("NUM", "nummod")
                } else if i == 0 {
                    ("PROPN", "nsubj")
                } else if w.surface.starts_with("공격") {
                    ("NOUN", "obj")
                } else if i == last {
                    ("VERB", "root")
                } else {
                    ("NOUN", "nmod")
                };
                w.upos = upos.into();
                w.deprel = deprel.into();
            }
        }
    }

    #[test]
    fn a_newer_record_with_a_changed_value_supersedes_the_old_one() {
        let reply = json!({"matches": [
            {"id": ID_NEW, "text": "말레니아 공격 피해 15% 감소", "verdict": {"tier": "premium"}},
            {"id": ID_OLD, "text": "말레니아 공격 피해 20% 감소", "verdict": {"tier": "premium"}, "my_vote": 1}
        ]});
        let tools = Arc::new(std::sync::Mutex::new(FakePd { reply, ..Default::default() }));
        let a = Analysis { embedder: SharedEmbedder::new(HashedNgramEmbedder::default()), analyzer: Arc::new(TagAnalyzer) };
        let mut b = PdBridge::new(tools, a, LocalLedger::default(), false);
        let ctx = b.resolve_context_for_ask(None, &[], "말레니아 공격", 5_000).unwrap();
        assert_eq!(ctx.superseded, [ID_OLD], "{ctx:?}");
        assert_eq!(ctx.verdicts.len(), 1, "the old record is not recommended any more: {ctx:?}");
        assert_eq!(ctx.context, ["말레니아 공격 피해 20% 감소: 더 새 기록이 있어 이 내용은 지난 것이다"]);
        // 장부에도 대체가 남아, 내 표가 추천이었어도 기기 판정은 비추천이다
        assert_eq!(b.ledger.tier(ID_OLD, 5_000), Tier::Discount);
    }

    #[test]
    fn feedback_and_outcome_credit_the_briefed_records_and_vote_only_with_consent() {
        let reply = json!({"matches": [
            {"id": ID_NEW, "text": "말레니아 물 새 공격은 왼쪽으로 구르면 피한다", "verdict": {"tier": "unrated"}},
            {"id": ID_SERVER, "text": "라다곤 2페이즈 빛 기둥은 뒤로 물러난다", "verdict": {"tier": "neutral"}}
        ]});
        let mut b = bridge(reply.clone(), false);
        assert!(b.record_feedback(FeedbackIntent::Recommend, false, "덕분에 깼다", 0).is_none(), "nothing briefed yet");
        b.resolve_context_for_ask(None, &[], "말레니아 공략", 1_000).unwrap();
        let c = b.record_feedback(FeedbackIntent::Recommend, false, "말레니아 물 새 공격 구르니까 덕분에 깼다", 2_000).unwrap();
        assert_eq!(c.credits.iter().map(|(id, _, s)| (id.as_str(), *s)).collect::<Vec<_>>(), [(ID_NEW, 1)], "{c:?}");
        assert_eq!(c.votes_sent, 0, "no consent, no vote");
        assert_eq!(b.ledger.obs[ID_NEW][0].kind, "feedback");
        // 결과(승리)는 브리핑한 기록에 똑같이 나눈다
        let o = b.record_outcome(1, 3_000).unwrap();
        assert_eq!(o.credits.len(), 2);
        assert!(o.credits.iter().all(|(_, c, s)| (*c - 0.5).abs() < 1e-9 && *s == 1));
        // 5분이 지나면 후보가 없다
        assert!(b.record_outcome(1, 1_000 + CREDIT_WINDOW_MS + 1).is_none());

        let mut shared = bridge(reply, true);
        shared.resolve_context_for_ask(None, &[], "공략", 0).unwrap();
        let s = shared.record_feedback(FeedbackIntent::NotRecommend, false, "라다곤 2페이즈 빛 기둥 별로였어", 10).unwrap();
        assert_eq!(s.votes_sent, 1);
        assert_eq!(shared.tools.as_ref().unwrap().lock().unwrap().votes, [(ID_SERVER.to_string(), -1, "vote".to_string())]);
        // 이중 부정(부정된 비추천)은 반만큼 추천 쪽
        let n = shared.record_feedback(FeedbackIntent::NotRecommend, true, "라다곤 빛 기둥 별로 아니었어", 20).unwrap();
        assert_eq!(n.credits[0].2, 1);
    }

    #[test]
    fn the_ledger_judges_against_my_other_records() {
        let mut l = LocalLedger::default();
        let params = Params::v1();
        // 기록 12개: 하나는 계속 칭찬, 하나는 계속 혹평, 나머지는 섞임
        for i in 0..12 {
            let id = format!("r{i}");
            for k in 0..6 {
                let sign = match i {
                    0 => 1,
                    1 => -1,
                    _ => if (i + k) % 2 == 0 { 1 } else { -1 },
                };
                l.add(&id, feedback_obs(if sign > 0 { FeedbackIntent::Recommend } else { FeedbackIntent::NotRecommend }, false, 1.0, 1_000 + k as i64, &params));
            }
        }
        assert_eq!(l.tier("r0", 10_000), Tier::Premium, "{:?}", l.tier("r0", 10_000));
        assert_eq!(l.tier("r1", 10_000), Tier::Discount);
        assert_eq!(l.tier("nobody", 10_000), Tier::Unrated);
        l.mark_superseded("r0");
        assert_eq!(l.tier("r0", 10_000), Tier::Discount, "superseded is discount");
        // 저장 · 열기
        let path = std::env::temp_dir().join(format!("zena-ledger-{}.json", std::process::id()));
        l.path = Some(path.clone());
        l.save();
        let back = LocalLedger::open(&path);
        assert_eq!(back.obs.len(), 12);
        assert!(back.superseded.contains("r0"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn feedback_words_and_negation() {
        let a = analysis();
        assert_eq!(detect_feedback("덕분에 깼다 진짜", &a), Some((FeedbackIntent::Recommend, false)));
        assert_eq!(detect_feedback("그거 별로였어", &a), Some((FeedbackIntent::NotRecommend, false)));
        assert_eq!(detect_feedback("안 통했어", &a).map(|(i, _)| i), Some(FeedbackIntent::Recommend));
        assert_eq!(detect_feedback("보스 어디 있어?", &a), None);
    }
}
