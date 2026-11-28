//! # raon-vision
//!
//! Raon Zena's screen encoder: louis-vision's `lv-screen` (the SigLIP2 encoder that started here, now with the screen
//! sensor, NaFlex patches and the budgeted perforated pass), re-exported, plus the glue to the word dictionary
//! (`zena-lexicon`): the vision nodes as anchor-bank domains and the dictionary's prompt templates for the anchors.

pub use lv_screen::*;

/// The text tower and tokenizer, and [`text::compute_anchors`] over the dictionary's prompt templates.
pub mod text {
    pub use lv_screen::text::*;

    use anyhow::Result;

    /// The prompt-ensembled anchor of every phrase under the dictionary's templates (`VisionPrompting::forms`).
    pub fn compute_anchors(
        text: &TextTower,
        tok: &SiglipTokenizer,
        prompting: &zena_lexicon::VisionPrompting,
        phrases: &[String],
    ) -> Result<Vec<(String, Vec<f32>)>> {
        lv_screen::text::compute_anchors(text, tok, |p| prompting.forms(p), phrases)
    }
}

/// The dictionary's `vision` nodes as anchor-bank domains.
pub fn vision_specs(lex: &zena_lexicon::Lexicon) -> Vec<VisionDomainSpec> {
    lex.vision
        .iter()
        .map(|v| VisionDomainSpec {
            key: v.key.clone(),
            kind: v.kind,
            link: v.link.clone(),
            cooldown_s: v.cooldown_s(),
            bias: v.bias.iter().map(|p| (p.text.clone(), p.weight)).collect(),
            prejudice: v.prejudice.iter().map(|p| p.text.clone()).collect(),
        })
        .collect()
}

/// Building the [`AnchorBank`] from the word dictionary.
pub trait LexiconAnchors: Sized {
    /// The vision nodes whose bias prompts all have anchors (background prejudice prompts compete separately).
    fn build(lex: &zena_lexicon::Lexicon, anchors: &std::collections::HashMap<String, Vec<f32>>, scale: f32, bias: f32) -> Self;
}

impl LexiconAnchors for AnchorBank {
    fn build(lex: &zena_lexicon::Lexicon, anchors: &std::collections::HashMap<String, Vec<f32>>, scale: f32, bias: f32) -> Self {
        AnchorBank::from_specs(&vision_specs(lex), &lex.prompting.background, |t| lex.prompting.is_background(t), anchors, scale, bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use zena_lexicon::Lexicon;

    fn unit(seed: f32) -> Vec<f32> {
        let v: Vec<f32> = (0..32).map(|i| ((i as f32 + 1.0) * seed).sin()).collect();
        lv_screen::tower::unit(v)
    }

    #[test]
    fn the_nearest_anchor_wins_and_prejudice_lowers_a_look_alike() {
        let lex = Lexicon::bundled();
        let mut anchors = HashMap::new();
        for (i, p) in lex.vision_phrases().iter().enumerate() {
            anchors.insert(p.clone(), unit(0.1 + i as f32 * 0.37));
        }
        let bank = <AnchorBank as LexiconAnchors>::build(&lex, &anchors, 10.0, -2.0);
        assert_eq!(bank.domains.len(), lex.vision.len());
        let over = &anchors["a game over screen"];
        let s = bank.score(over);
        assert_eq!(s[0].key, "vision.game_defeat");
        assert!(s[0].bias > 0.99);
        // "a game over screen" is the victory node's prejudice: its score is pulled down.
        let win = s.iter().find(|x| x.key == "vision.game_victory").unwrap();
        assert!(win.prejudice > 0.99 && win.score < win.bias);
        // Nodes without anchors are left out until the anchors exist.
        anchors.remove("a game over screen");
        assert!(<AnchorBank as LexiconAnchors>::build(&lex, &anchors, 10.0, -2.0)
            .index_of("vision.game_defeat")
            .is_none());
    }

}
