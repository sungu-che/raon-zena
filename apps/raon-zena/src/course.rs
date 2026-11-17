//! 게임 화면의 코스 (premium-discount 타임트라이얼 `trial/1` 을 게임에 쓴 것): 마스터 기획서 3.1 · 3.3 (1)
//!
//! - **체크포인트**: 화면 인코더가 알아본 장면(`VisionKind::Scene`)이 바뀔 때마다 하나. 해시는 premium-discount-nav 의
//!   `screen_hash(앱, "scene:<장면 키>")` 라 같은 게임의 같은 장면은 누가 봐도 같다(장면 키는 제나 사전의 `vision.*`).
//! - **결승**: 승리(화면의 `vision.game_victory` 또는 "이겼다" 같은 말). 그때까지의 체크포인트가 런이 되어 기기의 기록실
//!   (`Collection`, 게임마다 코스 10개)에 들어간다. 패배는 결승이 아니라 그 구간의 마찰이다.
//! - **고스트**: 진행 중에 `Collection::guide` 로 내 최고 기록과 견준다(앞섬 · 뒤짐, 다음 체크포인트).
//!
//! 결과는 언어와 상관없는 [`CourseBrief`] 이고, 프롬프트의 `[코스]` 줄은 언어팩이 만든다. 다른 사용자의 코스(공개 고스트,
//! `/nav/trials`)는 아직 받지 않는다: 지금 고스트는 내 기록뿐이다.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use premium_discount_nav::{app_hash, screen_hash, Checkpoint, Collection, Run, StepKind};
use zena_core::prompt::{CourseBrief, NextCheckpoint};

/// 한 체크포인트에서 이만큼 지면 그 구간은 막혀 있다(마찰 큼)로 본다
pub const FRICTION_DEFEATS: u32 = 2;

pub struct CourseTracker {
    /// 지금 게임 (`app://eldenring.exe`)과 그 앱 해시
    cc: Option<String>,
    app: Option<String>,
    progress: Vec<Checkpoint>,
    /// 화면 해시 → 사람이 읽는 장면 이름
    labels: HashMap<String, String>,
    defeats_here: u32,
    pub collection: Collection,
    path: Option<PathBuf>,
}

impl CourseTracker {
    pub fn new(collection: Collection, path: Option<PathBuf>) -> CourseTracker {
        CourseTracker { cc: None, app: None, progress: Vec::new(), labels: HashMap::new(), defeats_here: 0, collection, path }
    }

    /// 기록실을 파일에서 연다 (없으면 빈 기록실)
    pub fn open(path: &Path) -> CourseTracker {
        let collection = std::fs::read_to_string(path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        CourseTracker::new(collection, Some(path.to_path_buf()))
    }

    fn save(&self) {
        if let Some(p) = &self.path {
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            if let Ok(s) = serde_json::to_string(&self.collection) {
                let _ = std::fs::write(p, s);
            }
        }
    }

    /// 장면이 보였다. 장면이 바뀌었으면 체크포인트를 하나 더한다. 게임이 바뀌면 진행을 새로 시작한다
    pub fn scene(&mut self, cc: Option<&str>, key: &str, label: &str, at_ms: i64) {
        let Some(cc) = cc.map(str::to_string).or_else(|| self.cc.clone()) else { return };
        if self.cc.as_deref() != Some(cc.as_str()) {
            self.cc = Some(cc.clone());
            self.app = app_hash(&cc);
            self.progress.clear();
            self.defeats_here = 0;
        }
        let Some(hash) = screen_hash(&cc, &format!("scene:{key}")) else { return };
        if self.progress.last().map(|c| c.hash.as_str()) == Some(hash.as_str()) {
            return;
        }
        self.labels.insert(hash.clone(), label.to_string());
        self.progress.push(Checkpoint { at_ms, kind: StepKind::Screen, hash, gesture: None });
        self.defeats_here = 0;
        // 코스는 체크포인트 10개까지: 오래된 것부터 버린다 (premium-discount-nav 와 같은 규칙)
        if self.progress.len() > premium_discount_nav::MAX_STEPS {
            self.progress.remove(0);
        }
    }

    pub fn defeat(&mut self) {
        self.defeats_here += 1;
    }

    /// 승리: 지금까지의 체크포인트를 런으로 기록실에 넣고 새로 시작한다. 체크포인트가 둘보다 적으면 넣지 않는다
    pub fn victory(&mut self, at_ms: i64) -> Option<premium_discount_nav::Added> {
        let app = self.app.clone()?;
        let mut points = self.progress.clone();
        if let Some(last) = points.last() {
            // 결승 시각: 마지막 장면에서 승리까지
            let finish = Checkpoint { at_ms, ..last.clone() };
            points.pop();
            points.push(finish);
        }
        self.progress.clear();
        self.defeats_here = 0;
        let run = Run::from_checkpoints(&app, &points, None).ok()?;
        let added = self.collection.add(&run);
        self.save();
        Some(added)
    }

    /// 지금 코스의 요약 (고스트가 없으면 체크포인트 · 마찰만). 고스트와는 구간 시간으로 견준다: 지금 체크포인트에 닿은 시각과
    /// 내 최고 기록이 그 체크포인트에 닿은 시각의 차이
    pub fn brief(&self) -> Option<CourseBrief> {
        let last = self.progress.last()?;
        let app = self.app.as_deref()?;
        let guide = self.collection.guide(app, &self.progress, &[]);
        Some(CourseBrief {
            course_id: guide.as_ref().map(|g| g.course.clone()).unwrap_or_default(),
            checkpoint_label: self.labels.get(&last.hash).cloned().unwrap_or_default(),
            ghost_delta_ms: guide.as_ref().and_then(|g| g.vs_best),
            next_recommended: guide.as_ref().and_then(|g| g.next.as_ref()).map(|n| NextCheckpoint {
                label: self.labels.get(&n.hash).cloned().unwrap_or_default(),
                local: None,
                global: guide.as_ref().map(|g| g.next_tier),
            }),
            high_friction: self.defeats_here >= FRICTION_DEFEATS,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GAME: &str = "app://eldenring.exe";

    #[test]
    fn a_won_course_becomes_the_ghost_of_the_next_run() {
        let mut t = CourseTracker::new(Collection::default(), None);
        // 첫 판: 성문 → 다리 → 보스방, 승리
        t.scene(Some(GAME), "vision.gate", "성문", 0);
        t.scene(Some(GAME), "vision.bridge", "다리", 30_000);
        t.scene(Some(GAME), "vision.bridge", "다리", 31_000);
        t.scene(Some(GAME), "vision.boss_room", "보스방", 60_000);
        assert!(t.brief().unwrap().course_id.is_empty(), "no ghost yet");
        assert_eq!(t.victory(90_000), Some(premium_discount_nav::Added::New));

        // 둘째 판: 같은 길을 더 빨리
        t.scene(Some(GAME), "vision.gate", "성문", 1_000_000);
        t.scene(Some(GAME), "vision.bridge", "다리", 1_020_000);
        let b = t.brief().unwrap();
        assert_eq!(b.checkpoint_label, "다리");
        assert_eq!(b.ghost_delta_ms, Some(-10_000), "10 s ahead of my best at the bridge");
        let next = b.next_recommended.unwrap();
        assert_eq!(next.label, "보스방");
        assert!(!b.high_friction);
        // 다리에서 두 번 지면 막힌 구간
        t.defeat();
        t.defeat();
        assert!(t.brief().unwrap().high_friction);
        // 프롬프트 줄 (한국어 팩)
        let line = t.brief().unwrap().render();
        assert_eq!(line, "체크포인트 '다리' / 고스트보다 10.0초 빠름 / 다음 추천 '보스방' / 이 구간에서 막혀 있다 (마찰 큼)");
    }

    #[test]
    fn another_game_starts_over_and_short_runs_are_not_kept() {
        let mut t = CourseTracker::new(Collection::default(), None);
        t.scene(Some(GAME), "vision.gate", "성문", 0);
        assert_eq!(t.victory(5_000), None, "one checkpoint is not a course");
        t.scene(Some(GAME), "vision.gate", "성문", 10_000);
        t.scene(Some("app://other.exe"), "vision.menu", "메뉴", 11_000);
        assert_eq!(t.brief().unwrap().checkpoint_label, "메뉴");
        assert!(t.collection.apps.is_empty());
        // 게임을 모르면 아무것도 하지 않는다
        let mut u = CourseTracker::new(Collection::default(), None);
        u.scene(None, "vision.gate", "성문", 0);
        assert!(u.brief().is_none());
    }
}
