//! What Zena may say about a recommendation: the local verdict (my own record) and the global one (other
//! users, premium-discount's server) side by side. They are never mixed into one number: "it worked for me but
//! the others say it does not" is the most useful thing to say, and a blend would hide it.
//!
//! | local | global | Zena says |
//! |---|---|---|
//! | judged | same tier | both agree |
//! | judged | other tier | both, explicitly |
//! | judged | unrated | the local verdict only |
//! | unrated | judged | "the others …" |
//! | unrated | unrated | facts only: never "recommended" / "not recommended" |

use crate::Tier;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stance {
    Agree(Tier),
    Differ {
        local: Tier,
        global: Tier,
    },
    LocalOnly(Tier),
    GlobalOnly(Tier),
    /// Neither side has enough evidence.
    FactsOnly,
}

fn judged(t: Option<Tier>) -> Option<Tier> {
    t.filter(|t| *t != Tier::Unrated)
}

pub fn stance(local: Option<Tier>, global: Option<Tier>) -> Stance {
    match (judged(local), judged(global)) {
        (Some(l), Some(g)) if l == g => Stance::Agree(l),
        (Some(l), Some(g)) => Stance::Differ { local: l, global: g },
        (Some(l), None) => Stance::LocalOnly(l),
        (None, Some(g)) => Stance::GlobalOnly(g),
        (None, None) => Stance::FactsOnly,
    }
}

/// 추천 / 비추천 / 보통 (판정 전은 말하지 않으므로 빈 문자열). 현재 언어팩(`zena_locale`)의 말
pub fn word(t: Tier) -> String {
    let l = zena_locale::current();
    match t {
        Tier::Premium => l.speak("premium").to_string(),
        Tier::Discount => l.speak("discount").to_string(),
        Tier::Neutral => l.speak("neutral").to_string(),
        Tier::Unrated => String::new(),
    }
}

impl Stance {
    /// One context line for the model about the record `label`, in the current language pack (`zena_locale`;
    /// Korean by default, as the rest of Zena's prompt).
    pub fn line(self, label: &str) -> String {
        let l = zena_locale::current();
        let fill = zena_locale::fill;
        match self {
            Stance::Agree(t) => fill(l.speak("agree"), &[("label", label), ("tier", &word(t))]),
            Stance::Differ { local, global } => fill(l.speak("differ"), &[("label", label), ("local", &word(local)), ("global", &word(global))]),
            Stance::LocalOnly(t) => fill(l.speak("local_only"), &[("label", label), ("tier", &word(t))]),
            Stance::GlobalOnly(t) => fill(l.speak("global_only"), &[("label", label), ("tier", &word(t))]),
            Stance::FactsOnly => fill(l.speak("facts_only"), &[("label", label)]),
        }
    }

    /// The character emote the briefing opens with (`zena_core::tools::CharacterEmote` names).
    pub fn emote(self) -> Option<&'static str> {
        match self {
            Stance::Differ { .. } => Some("QUESTION"),
            Stance::Agree(t) | Stance::LocalOnly(t) | Stance::GlobalOnly(t) => match t {
                Tier::Premium => Some("STAR_EYES"),
                Tier::Discount => Some("SWEAT_PANIC"),
                _ => None,
            },
            Stance::FactsOnly => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_five_rows_of_the_table() {
        use Tier::*;
        assert_eq!(stance(Some(Premium), Some(Premium)), Stance::Agree(Premium));
        assert_eq!(
            stance(Some(Premium), Some(Discount)),
            Stance::Differ {
                local: Premium,
                global: Discount
            }
        );
        assert_eq!(stance(Some(Neutral), Some(Unrated)), Stance::LocalOnly(Neutral));
        assert_eq!(stance(None, Some(Discount)), Stance::GlobalOnly(Discount));
        assert_eq!(stance(Some(Unrated), None), Stance::FactsOnly);

        assert_eq!(
            stance(Some(Premium), Some(Premium)).line("빙결 공략"),
            "빙결 공략: 내 기록과 서버 평가 모두 추천"
        );
        assert!(stance(Some(Premium), Some(Discount)).line("x").contains("둘 다"));
        let facts = Stance::FactsOnly.line("구르기 공략");
        assert!(!facts.contains("추천이야") && facts.contains("판정 전"), "{facts}");

        assert_eq!(Stance::Agree(Premium).emote(), Some("STAR_EYES"));
        assert_eq!(Stance::GlobalOnly(Discount).emote(), Some("SWEAT_PANIC"));
        assert_eq!(
            Stance::Differ {
                local: Premium,
                global: Neutral
            }
            .emote(),
            Some("QUESTION")
        );
        assert_eq!(Stance::LocalOnly(Neutral).emote(), None);
    }
}
