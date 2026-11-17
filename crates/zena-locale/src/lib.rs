//! # zena-locale: 제나의 언어팩
//!
//! raon-zena 는 국가별 카트리지다(premium-discount 는 언어와 상관없는 공용 코어). 모델(`LlmBackend`) · 표현 사전
//! (`data/zena_lexicon.json`)과 함께 갈아 끼우는 것이 **말의 틀**이다: 시스템 페르소나, 프롬프트의 블록 이름 · 세션 요약 ·
//! 말투 힌트 · 방향 이름, Lv2 즉답 큐, 추천 · 비추천 말투, 앱 문구. 이 crate 가 그 틀을 `data/locales/<언어>.json` 하나로
//! 모은다. 한국어(`ko`)와 영어(`en`)는 함께 실려 있고, 다른 언어는 같은 모양의 파일을 [`load_dir`] 로 읽는다. 그래서 다른 나라의
//! 소형 모델을 붙일 때 코드를 고치지 않고 `--lang` 으로 고른다.
//!
//! 현재 언어는 프로세스 전체 설정이다([`set_language`], 기본 `ko`). 시스템 페르소나는 Raon-1-nano 가 캐시하는 프롬프트 머리에
//! 들어가므로, 언어를 바꾸면 모델의 머리 캐시도 다시 만들어야 한다(apps/raon-zena 는 시작할 때 한 번 고른다).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, OnceLock, RwLock};

use serde::Deserialize;

/// 함께 실은 언어팩
pub const BUILTIN: [(&str, &str); 2] = [("ko", include_str!("../../../data/locales/ko.json")), ("en", include_str!("../../../data/locales/en.json"))];
/// 기본 언어
pub const DEFAULT_LANGUAGE: &str = "ko";

/// 언어팩 하나 (`data/locales/<code>.json` 의 모양)
#[derive(Debug, Clone, Deserialize)]
pub struct Locale {
    pub code: String,
    pub name: String,
    pub persona: String,
    pub prompt: BTreeMap<String, String>,
    pub directions: Vec<String>,
    pub cues: BTreeMap<String, Vec<String>>,
    pub speak: BTreeMap<String, String>,
    pub app: BTreeMap<String, serde_json::Value>,
}

/// 팩에 반드시 있어야 하는 열쇠 (빠지면 읽지 않는다: 반쯤 번역된 팩이 조용히 빈 줄을 내지 않게)
const REQUIRED_PROMPT: [&str; 36] = [
    "scene", "scene_minutes", "defeat_streak", "victory_streak", "affinity_high", "affinity_mid", "affinity_low", "tension_high", "join",
    "block_session", "block_tone", "block_recommend", "block_course", "block_history", "block_question", "block_event", "block_hp",
    "block_reference", "speaker_player", "speaker_zena", "tone_defeats", "tone_tense", "tone_streak", "tone_close", "dist_near", "dist_mid",
    "dist_far", "flow_inward", "flow_outward", "course_checkpoint", "course_ahead", "course_behind", "course_even", "course_next",
    "course_next_stance", "course_friction",
];
const REQUIRED_SPEAK: [&str; 8] = ["premium", "discount", "neutral", "agree", "differ", "local_only", "global_only", "facts_only"];
const REQUIRED_CUES: [&str; 4] = ["NICE_HIT", "CAREFUL", "HIT_ON_COOLDOWN", "MID_RANGE_FIGHT"];
const REQUIRED_APP: [&str; 13] = [
    "screen_now", "screen_unknown", "market_empathy", "event_close_threat", "event_monster", "event_skill", "event_motion", "superseded",
    "template_inventory", "template_tip", "template_low_hp", "template_careful", "template_enemy",
];

impl Locale {
    /// JSON 에서 읽고 빠진 열쇠를 검사한다
    pub fn parse(json: &str) -> Result<Locale, String> {
        let l: Locale = serde_json::from_str(json).map_err(|e| format!("locale JSON: {e}"))?;
        let missing: Vec<String> = REQUIRED_PROMPT
            .iter()
            .filter(|k| !l.prompt.contains_key(**k))
            .map(|k| format!("prompt.{k}"))
            .chain(REQUIRED_SPEAK.iter().filter(|k| !l.speak.contains_key(**k)).map(|k| format!("speak.{k}")))
            .chain(REQUIRED_CUES.iter().filter(|k| l.cues.get(**k).is_none_or(Vec::is_empty)).map(|k| format!("cues.{k}")))
            .chain(REQUIRED_APP.iter().filter(|k| !l.app.get(**k).is_some_and(serde_json::Value::is_string)).map(|k| format!("app.{k}")))
            .collect();
        if l.directions.len() != 8 {
            return Err(format!("locale {}: directions must name 8 directions", l.code));
        }
        if !missing.is_empty() {
            return Err(format!("locale {} is missing {}", l.code, missing.join(", ")));
        }
        Ok(l)
    }

    /// `prompt.<key>` (없으면 열쇠 그대로: 검사한 팩이라 생기지 않는다)
    pub fn prompt<'a>(&'a self, key: &'a str) -> &'a str {
        self.prompt.get(key).map(String::as_str).unwrap_or(key)
    }

    pub fn speak<'a>(&'a self, key: &'a str) -> &'a str {
        self.speak.get(key).map(String::as_str).unwrap_or(key)
    }

    pub fn app<'a>(&'a self, key: &'a str) -> &'a str {
        self.app.get(key).and_then(serde_json::Value::as_str).unwrap_or(key)
    }

    /// `app.<key>` 의 글 목록 (없으면 빈 목록)
    pub fn app_list(&self, key: &str) -> Vec<String> {
        self.app
            .get(key)
            .and_then(serde_json::Value::as_array)
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
            .unwrap_or_default()
    }

    /// Lv2 큐의 대사들 (`NICE_HIT` …)
    pub fn cue_lines(&self, kind: &str) -> &[String] {
        self.cues.get(kind).map(Vec::as_slice).unwrap_or(&[])
    }

    /// 각도(도)의 방향 이름 (45도씩 여덟 방향, 0 = 오른쪽, 반시계)
    pub fn direction(&self, theta_deg: f32) -> &str {
        &self.directions[(((theta_deg + 22.5) % 360.0) / 45.0) as usize % 8]
    }
}

/// `{이름}` 자리를 채운다. 없는 이름은 그대로 둔다
pub fn fill(template: &str, args: &[(&str, &str)]) -> String {
    let mut out = template.to_string();
    for (k, v) in args {
        out = out.replace(&format!("{{{k}}}"), v);
    }
    out
}

fn registry() -> &'static RwLock<BTreeMap<String, Arc<Locale>>> {
    static REG: OnceLock<RwLock<BTreeMap<String, Arc<Locale>>>> = OnceLock::new();
    REG.get_or_init(|| {
        let mut m = BTreeMap::new();
        for (code, json) in BUILTIN {
            let l = Locale::parse(json).unwrap_or_else(|e| panic!("built-in locale {code}: {e}"));
            m.insert(code.to_string(), Arc::new(l));
        }
        RwLock::new(m)
    })
}

fn current_slot() -> &'static RwLock<Arc<Locale>> {
    static CUR: OnceLock<RwLock<Arc<Locale>>> = OnceLock::new();
    CUR.get_or_init(|| RwLock::new(registry().read().expect("locale registry")[DEFAULT_LANGUAGE].clone()))
}

/// 쓸 수 있는 언어 코드 (함께 실은 것 + 읽어 들인 것)
pub fn languages() -> Vec<String> {
    registry().read().expect("locale registry").keys().cloned().collect()
}

/// 언어팩을 하나 더 등록한다 (같은 코드면 바꾼다)
pub fn register(locale: Locale) -> String {
    let code = locale.code.clone();
    registry().write().expect("locale registry").insert(code.clone(), Arc::new(locale));
    code
}

/// 폴더의 `*.json` 언어팩을 모두 읽어 등록한다. 읽은 코드를 돌려준다. 틀린 파일은 이유와 함께 오류
pub fn load_dir(dir: &Path) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let locale = Locale::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        out.push(register(locale));
    }
    Ok(out)
}

/// 현재 언어를 바꾼다. `ko-KR` · `Korean` 처럼 써도 앞의 두 글자로 찾는다
pub fn set_language(code: &str) -> Result<(), String> {
    let key = code.trim().to_ascii_lowercase();
    let reg = registry().read().expect("locale registry");
    let found = reg.get(&key).or_else(|| reg.get(key.get(..2).unwrap_or(""))).or_else(|| match key.as_str() {
        "korean" | "한국어" => reg.get("ko"),
        "english" => reg.get("en"),
        _ => None,
    });
    let Some(l) = found else {
        return Err(format!("no language pack for {code} (have: {})", reg.keys().cloned().collect::<Vec<_>>().join(", ")));
    };
    *current_slot().write().expect("current locale") = l.clone();
    Ok(())
}

/// 현재 언어팩
pub fn current() -> Arc<Locale> {
    current_slot().read().expect("current locale").clone()
}

/// 이름이 붙은 언어팩 (현재 언어를 바꾸지 않고 볼 때)
pub fn get(code: &str) -> Option<Arc<Locale>> {
    registry().read().expect("locale registry").get(code).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_packs_are_complete_and_share_the_keys() {
        let ko = get("ko").unwrap();
        let en = get("en").unwrap();
        assert_eq!(ko.prompt.keys().collect::<Vec<_>>(), en.prompt.keys().collect::<Vec<_>>());
        assert_eq!(ko.speak.keys().collect::<Vec<_>>(), en.speak.keys().collect::<Vec<_>>());
        assert_eq!(ko.cues.keys().collect::<Vec<_>>(), en.cues.keys().collect::<Vec<_>>());
        assert_eq!(ko.app.keys().collect::<Vec<_>>(), en.app.keys().collect::<Vec<_>>());
        assert_eq!(ko.direction(350.0), "오른쪽");
        assert_eq!(ko.direction(135.0), "왼쪽 위");
        assert_eq!(en.direction(90.0), "up");
        assert_eq!(fill(ko.prompt("scene_minutes"), &[("scene", "보스전"), ("minutes", "3")]), "화면: 보스전 (3분째)");
        assert_eq!(ko.app_list("template_inventory_words"), ["물약", "인벤", "포션"]);
    }

    #[test]
    fn a_partial_pack_is_refused() {
        let mut v: serde_json::Value = serde_json::from_str(BUILTIN[1].1).unwrap();
        v["code"] = "fr".into();
        v["speak"].as_object_mut().unwrap().remove("differ");
        let err = Locale::parse(&v.to_string()).unwrap_err();
        assert!(err.contains("speak.differ"), "{err}");
    }

    #[test]
    fn packs_load_from_a_folder_and_can_be_selected() {
        let dir = std::env::temp_dir().join(format!("zena-locale-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(BUILTIN[1].1).unwrap();
        v["code"] = "ja".into();
        v["name"] = "日本語".into();
        v["speak"]["premium"] = "おすすめ".into();
        std::fs::write(dir.join("ja.json"), v.to_string()).unwrap();
        std::fs::write(dir.join("readme.txt"), "not a pack").unwrap();
        assert_eq!(load_dir(&dir).unwrap(), ["ja"]);
        assert!(languages().contains(&"ja".to_string()));
        assert_eq!(get("ja").unwrap().speak("premium"), "おすすめ");
        assert!(set_language("xx").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
