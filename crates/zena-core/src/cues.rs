//! Lv2 pre-cached cues: short lines whose audio is synthesised once at start-up and replayed
//! with no LLM call. The lines come from the current language pack (`zena_locale`, `cues.<KIND>`); the emotion
//! and motion stay here.

use crate::tools::{BodyMotion, EmotionState};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CueKind {
    NiceHit,
    Careful,
    HitOnCooldown,
    MidRangeFight,
}

impl CueKind {
    /// The language pack key (`cues.NICE_HIT` …)
    pub fn key(self) -> &'static str {
        match self {
            CueKind::NiceHit => "NICE_HIT",
            CueKind::Careful => "CAREFUL",
            CueKind::HitOnCooldown => "HIT_ON_COOLDOWN",
            CueKind::MidRangeFight => "MID_RANGE_FIGHT",
        }
    }

    /// Alternatives rotated round-robin so repeats do not sound robotic (current language pack)
    pub fn lines(self) -> Vec<String> {
        zena_locale::current().cue_lines(self.key()).to_vec()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cue {
    pub kind: CueKind,
    pub emotion: EmotionState,
    pub motion: BodyMotion,
}

pub const CUES: &[Cue] = &[
    Cue {
        kind: CueKind::NiceHit,
        emotion: EmotionState::Joy,
        motion: BodyMotion::Cheer,
    },
    Cue {
        kind: CueKind::Careful,
        emotion: EmotionState::Surprised,
        motion: BodyMotion::Flinch,
    },
    Cue {
        kind: CueKind::HitOnCooldown,
        emotion: EmotionState::Worried,
        motion: BodyMotion::Flinch,
    },
    Cue {
        kind: CueKind::MidRangeFight,
        emotion: EmotionState::Focused,
        motion: BodyMotion::Nod,
    },
];

pub fn cue(kind: CueKind) -> &'static Cue {
    CUES.iter().find(|c| c.kind == kind).expect("every CueKind has an entry")
}

/// Round-robin line picker.
#[derive(Debug, Default, Clone)]
pub struct CuePlayer {
    counters: [usize; 4],
}

impl CuePlayer {
    pub fn next_line(&mut self, kind: CueKind) -> String {
        let lines = kind.lines();
        let slot = &mut self.counters[kind as usize];
        let line = lines.get(*slot % lines.len().max(1)).cloned().unwrap_or_default();
        *slot += 1;
        line
    }
}

/// Pick the cue for an Lv2 decision reason (see `reaction::ReactionRouter::classify`).
pub fn cue_for_reason(reason: &str) -> CueKind {
    match reason {
        "own skill landed" => CueKind::NiceHit,
        "hit while skills on cooldown" => CueKind::HitOnCooldown,
        "close inward threat" | "sharp HP drop" | "HP critical and falling" => CueKind::Careful,
        _ => CueKind::MidRangeFight,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_robin() {
        let mut p = CuePlayer::default();
        assert_eq!(p.next_line(CueKind::NiceHit), "나이스!");
        assert_eq!(p.next_line(CueKind::NiceHit), "좋았어!");
        assert_eq!(p.next_line(CueKind::Careful), "조심!");
        assert_eq!(cue_for_reason("own skill landed"), CueKind::NiceHit);
    }
}
