//! The expression dictionary.
//!
//! File layout (`data/zena_lexicon.json`): top-level groups of nodes in the Qntology/logis-center `bias.json` node
//! format — `{semantic, bias, prejudice, exact_match}` — plus the fields that say how Zena expresses a node:
//!
//! * `express`: `{emotion, motion, emote, lines, level}` — a pre-written reaction (Lv2 with lines, Lv1 without),
//!   no LLM; `emote` names Zena's own [`CharacterEmote`] (default: the one matching the emotion);
//! * `command` (+ `window_layer_mode`): a `manage_zena_system` call, optionally with an `express` (wave before exit);
//! * `asset`: `{action, track, part}` — a `manage_avatar_asset` call (switch the body track, put on an accessory),
//!   optionally with an `express`;
//! * `ask`: hand the request to Raon-1-nano (Lv3) under a topic, optionally with an acknowledging `express`.
//!
//! The `proactive` group holds lines Zena says on her own after a silence (`proactive.<scene>`, keyed by the name
//! of the `react` node the screen's scene links to, `proactive.idle` otherwise, `proactive.stretch` after a long
//! session); they have no phrases, so the router never catches them from text.
//!
//! The `function` group is logis-center's function bank (request endings, fillers): its phrases explain words away
//! and never form a domain. `vision` nodes (noto-sandbox `world_lexicon.json` style) carry English prompts for the
//! SigLIP2 text tower and `link` to the text node whose expression Zena plays when the screen shows them;
//! `vision_prompting` holds the prompt-ensembling templates and the background phrases.

use crate::phrase::{join_phrases, split_phrase_list, weighted_phrases, Phrase};
use anyhow::{anyhow, bail, Context, Result};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use zena_core::tools::{BodyMotion, CharacterEmote, EmotionState, SystemCommand, WindowLayerMode, ZenaAssetArgs};

/// The bundled dictionary (`data/zena_lexicon.json`).
pub const BUNDLED_JSON: &str = include_str!("../../../data/zena_lexicon.json");

const RESERVED: &[&str] = &["version", "meta", "vision", "vision_prompting"];
pub const FUNCTION_GROUP: &str = "function";
pub const PROACTIVE_GROUP: &str = "proactive";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExpressLevel {
    /// Emotion and motion only (gaze, micro-expression), no speech.
    Lv1,
    /// A pre-written line with emotion and motion (cached TTS), no LLM.
    Lv2,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Express {
    pub emotion: EmotionState,
    pub motion: BodyMotion,
    /// Zena's own emote for this reaction (`None`: the emotion's default).
    pub emote: Option<CharacterEmote>,
    pub lines: Vec<String>,
    pub level: ExpressLevel,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Function words, or context only.
    None,
    Express(Express),
    System {
        command: SystemCommand,
        mode: Option<WindowLayerMode>,
        express: Option<Express>,
    },
    /// A `manage_avatar_asset` call (track switch, accessory).
    Asset {
        asset: ZenaAssetArgs,
        express: Option<Express>,
    },
    /// Raon-1-nano answers (Lv3) under `topic` ("coach", "screen", "market", "chat", ...).
    Ask {
        topic: String,
        express: Option<Express>,
    },
}

impl Action {
    pub fn express(&self) -> Option<&Express> {
        match self {
            Action::Express(e) => Some(e),
            Action::System { express, .. } | Action::Asset { express, .. } | Action::Ask { express, .. } => express.as_ref(),
            Action::None => None,
        }
    }
    /// Commands (system and asset) dedupe on their own words.
    pub fn is_command(&self) -> bool {
        matches!(self, Action::System { .. } | Action::Asset { .. })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// `group.name`.
    pub key: String,
    pub group: String,
    pub name: String,
    pub semantic: String,
    pub bias: Vec<Phrase>,
    pub prejudice: Vec<Phrase>,
    pub exact_match: Vec<String>,
    pub action: Action,
}

/// The screen kinds (scene · event · privacy): premium-discount-sensor's, shared with the vision anchor bank.
pub use premium_discount_sensor::VisionKind;

#[derive(Clone, Debug, PartialEq)]
pub struct VisionNode {
    /// `vision.name`.
    pub key: String,
    pub name: String,
    pub kind: VisionKind,
    /// Key of the text node whose expression Zena plays.
    pub link: String,
    pub semantic: String,
    pub bias: Vec<Phrase>,
    pub prejudice: Vec<Phrase>,
    pub cooldown_s: Option<u32>,
}

impl VisionNode {
    pub fn cooldown_s(&self) -> u32 {
        self.cooldown_s.unwrap_or_else(|| self.kind.default_cooldown_s())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VisionPrompting {
    pub semantic: String,
    pub templates: Vec<String>,
    pub background: Vec<String>,
}

impl VisionPrompting {
    /// `template` with `{}` replaced by `phrase` (appended when the template has no `{}`).
    pub fn apply(template: &str, phrase: &str) -> String {
        if template.contains("{}") {
            template.replace("{}", phrase)
        } else {
            format!("{template} {phrase}")
        }
    }
    pub fn is_background(&self, phrase: &str) -> bool {
        self.background.iter().any(|b| b.eq_ignore_ascii_case(phrase.trim()))
    }
    /// The prompts embedded for `phrase` (one per template, or the phrase itself without templates).
    pub fn forms(&self, phrase: &str) -> Vec<String> {
        if self.templates.is_empty() {
            vec![phrase.to_string()]
        } else {
            self.templates.iter().map(|t| VisionPrompting::apply(t, phrase)).collect()
        }
    }
    /// FNV-1a over the templates: anchors computed under other templates are stale.
    pub fn signature(&self) -> String {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for t in &self.templates {
            for b in t.bytes().chain(std::iter::once(0)) {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        format!("{h:016x}")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Lexicon {
    pub version: u64,
    pub meta: Value,
    /// Text nodes, sorted by key.
    pub nodes: Vec<Node>,
    /// Vision nodes, sorted by key.
    pub vision: Vec<VisionNode>,
    pub prompting: VisionPrompting,
}

/// Parses a zena-core enum name strictly (the enums fall back to NEUTRAL / IDLE / KEEP_ACTIVE on unknown names).
fn strict<T: DeserializeOwned + Serialize>(s: &str, what: &str) -> Result<T> {
    let v: T = serde_json::from_value(Value::String(s.to_string())).with_context(|| format!("{what} '{s}'"))?;
    if serde_json::to_value(&v)? != Value::String(s.to_string()) {
        bail!("unknown {what} '{s}'");
    }
    Ok(v)
}

fn enum_name<T: Serialize>(v: &T) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

fn phrases_of(v: Option<&Value>) -> Result<Vec<Phrase>> {
    Ok(match v {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => weighted_phrases(s),
        Some(Value::Array(a)) => {
            let joined: Vec<&str> = a.iter().filter_map(Value::as_str).collect();
            weighted_phrases(&joined.join(", "))
        }
        Some(other) => bail!("expected a phrase list, got {other}"),
    })
}

fn strings_of(v: Option<&Value>) -> Result<Vec<String>> {
    Ok(match v {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => split_phrase_list(s),
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(Value::as_str)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        Some(other) => bail!("expected a list of strings, got {other}"),
    })
}

fn express_of(v: &Value) -> Result<Express> {
    let o = v.as_object().ok_or_else(|| anyhow!("express must be an object"))?;
    let emotion = strict(o.get("emotion").and_then(Value::as_str).unwrap_or("NEUTRAL"), "emotion")?;
    let motion = strict(o.get("motion").and_then(Value::as_str).unwrap_or("IDLE"), "motion")?;
    let emote = match o.get("emote") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(CharacterEmote::parse(s).ok_or_else(|| anyhow!("unknown emote '{s}'"))?),
        Some(other) => bail!("express.emote must be a string, got {other}"),
    };
    let lines = match o.get("lines") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(Value::as_str)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        Some(Value::String(s)) if !s.trim().is_empty() => vec![s.trim().to_string()],
        Some(Value::String(_)) => Vec::new(),
        Some(other) => bail!("express.lines must be an array of strings, got {other}"),
    };
    let level = match o.get("level").and_then(Value::as_str) {
        Some("lv1") => ExpressLevel::Lv1,
        Some("lv2") => ExpressLevel::Lv2,
        Some(l) => bail!("express.level '{l}' (lv1, lv2)"),
        None if lines.is_empty() => ExpressLevel::Lv1,
        None => ExpressLevel::Lv2,
    };
    Ok(Express {
        emotion,
        motion,
        emote,
        lines,
        level,
    })
}

fn asset_of(v: &Value) -> Result<ZenaAssetArgs> {
    let o = v.as_object().ok_or_else(|| anyhow!("asset must be an object"))?;
    let action = strict(o.get("action").and_then(Value::as_str).unwrap_or(""), "asset action")?;
    let track = o
        .get("track")
        .and_then(Value::as_str)
        .map(|t| strict(t, "avatar track"))
        .transpose()?;
    let part = o
        .get("part")
        .and_then(Value::as_str)
        .map(|p| strict(p, "avatar part"))
        .transpose()?;
    Ok(ZenaAssetArgs { action, track, part })
}

fn express_json(e: &Express) -> Value {
    let mut o = json!({ "emotion": enum_name(&e.emotion), "motion": enum_name(&e.motion), "lines": e.lines });
    if let Some(m) = e.emote {
        o["emote"] = json!(m.name());
    }
    let implied = if e.lines.is_empty() { ExpressLevel::Lv1 } else { ExpressLevel::Lv2 };
    if e.level != implied {
        o["level"] = json!(if e.level == ExpressLevel::Lv1 { "lv1" } else { "lv2" });
    }
    o
}

pub(crate) fn node_of(group: &str, name: &str, v: &Value) -> Result<Node> {
    let o = v.as_object().ok_or_else(|| anyhow!("node must be an object"))?;
    let express = o.get("express").map(express_of).transpose()?;
    let action = if let Some(c) = o.get("command").and_then(Value::as_str) {
        let mode = o
            .get("window_layer_mode")
            .and_then(Value::as_str)
            .map(|m| strict(m, "window_layer_mode"))
            .transpose()?;
        Action::System {
            command: strict(c, "command")?,
            mode,
            express,
        }
    } else if let Some(a) = o.get("asset") {
        Action::Asset {
            asset: asset_of(a)?,
            express,
        }
    } else if let Some(topic) = o.get("ask").and_then(Value::as_str) {
        Action::Ask {
            topic: topic.to_string(),
            express,
        }
    } else if let Some(e) = express {
        Action::Express(e)
    } else {
        Action::None
    };
    Ok(Node {
        key: format!("{group}.{name}"),
        group: group.to_string(),
        name: name.to_string(),
        semantic: o.get("semantic").and_then(Value::as_str).unwrap_or("").to_string(),
        bias: phrases_of(o.get("bias"))?,
        prejudice: phrases_of(o.get("prejudice"))?,
        exact_match: strings_of(o.get("exact_match"))?,
        action,
    })
}

pub(crate) fn vision_of(name: &str, v: &Value) -> Result<VisionNode> {
    let o = v.as_object().ok_or_else(|| anyhow!("vision node must be an object"))?;
    Ok(VisionNode {
        key: format!("vision.{name}"),
        name: name.to_string(),
        kind: VisionKind::parse(o.get("kind").and_then(Value::as_str).unwrap_or("scene")).map_err(anyhow::Error::msg)?,
        link: o.get("link").and_then(Value::as_str).unwrap_or("").to_string(),
        semantic: o.get("semantic").and_then(Value::as_str).unwrap_or("").to_string(),
        bias: phrases_of(o.get("bias"))?,
        prejudice: phrases_of(o.get("prejudice"))?,
        cooldown_s: o.get("cooldown_s").and_then(Value::as_u64).map(|c| c as u32),
    })
}

/// One text node in the bias.json form.
pub fn node_json(n: &Node) -> Value {
    let mut o = Map::new();
    if !n.semantic.is_empty() {
        o.insert("semantic".into(), json!(n.semantic));
    }
    if !n.bias.is_empty() {
        o.insert("bias".into(), json!(join_phrases(n.bias.iter().map(|p| p.text.as_str()))));
    }
    if !n.prejudice.is_empty() {
        o.insert("prejudice".into(), json!(join_phrases(n.prejudice.iter().map(|p| p.text.as_str()))));
    }
    if !n.exact_match.is_empty() {
        o.insert("exact_match".into(), json!(n.exact_match));
    }
    match &n.action {
        Action::None => {}
        Action::Express(e) => {
            o.insert("express".into(), express_json(e));
        }
        Action::System { command, mode, express } => {
            o.insert("command".into(), json!(enum_name(command)));
            if let Some(m) = mode {
                o.insert("window_layer_mode".into(), json!(enum_name(m)));
            }
            if let Some(e) = express {
                o.insert("express".into(), express_json(e));
            }
        }
        Action::Asset { asset, express } => {
            o.insert("asset".into(), serde_json::to_value(asset).unwrap_or(Value::Null));
            if let Some(e) = express {
                o.insert("express".into(), express_json(e));
            }
        }
        Action::Ask { topic, express } => {
            o.insert("ask".into(), json!(topic));
            if let Some(e) = express {
                o.insert("express".into(), express_json(e));
            }
        }
    }
    Value::Object(o)
}

/// One vision node in the world_lexicon form.
pub fn vision_json(v: &VisionNode) -> Value {
    let mut o = Map::new();
    o.insert("kind".into(), json!(v.kind.as_str()));
    o.insert("link".into(), json!(v.link));
    if !v.semantic.is_empty() {
        o.insert("semantic".into(), json!(v.semantic));
    }
    o.insert("bias".into(), json!(join_phrases(v.bias.iter().map(|p| p.text.as_str()))));
    if !v.prejudice.is_empty() {
        o.insert("prejudice".into(), json!(join_phrases(v.prejudice.iter().map(|p| p.text.as_str()))));
    }
    if let Some(c) = v.cooldown_s {
        o.insert("cooldown_s".into(), json!(c));
    }
    Value::Object(o)
}

/// Lowercase, spaces removed: the form exact-match entries are compared in.
pub fn compact(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).flat_map(char::to_lowercase).collect()
}

impl Lexicon {
    pub fn from_json(raw: &str) -> Result<Lexicon> {
        let root: Value = serde_json::from_str(raw).context("parsing the expression lexicon")?;
        let obj = root.as_object().ok_or_else(|| anyhow!("the lexicon must be a JSON object"))?;
        let mut nodes = Vec::new();
        for (group, members) in obj.iter().filter(|(k, _)| !RESERVED.contains(&k.as_str())) {
            let members = members
                .as_object()
                .ok_or_else(|| anyhow!("group '{group}' must be an object of nodes"))?;
            for (name, v) in members {
                nodes.push(node_of(group, name, v).with_context(|| format!("node {group}.{name}"))?);
            }
        }
        nodes.sort_by(|a, b| a.key.cmp(&b.key));
        let mut vision = Vec::new();
        if let Some(vs) = obj.get("vision").and_then(Value::as_object) {
            for (name, v) in vs {
                vision.push(vision_of(name, v).with_context(|| format!("vision node {name}"))?);
            }
        }
        vision.sort_by(|a, b| a.key.cmp(&b.key));
        let prompting = match obj.get("vision_prompting") {
            Some(p) => VisionPrompting {
                semantic: p.get("semantic").and_then(Value::as_str).unwrap_or("").to_string(),
                templates: strings_of(p.get("templates"))?,
                background: strings_of(p.get("background"))?,
            },
            None => VisionPrompting::default(),
        };
        let lex = Lexicon {
            version: obj.get("version").and_then(Value::as_u64).unwrap_or(1),
            meta: obj.get("meta").cloned().unwrap_or(Value::Null),
            nodes,
            vision,
            prompting,
        };
        lex.validate()?;
        Ok(lex)
    }

    /// The dictionary's own wake words (`meta.wake_words`, e.g. another language's dictionary naming Zena differently).
    /// Empty when the dictionary has none.
    pub fn wake_words(&self) -> Vec<String> {
        self.meta
            .get("wake_words")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).map(str::trim).filter(|w| !w.is_empty()).map(str::to_string).collect())
            .unwrap_or_default()
    }

    /// The dictionary shipped in the binary.
    pub fn bundled() -> Lexicon {
        Lexicon::from_json(BUNDLED_JSON).expect("the bundled lexicon is valid (checked by tests)")
    }

    pub fn node(&self, key: &str) -> Option<&Node> {
        self.nodes
            .binary_search_by(|n| n.key.as_str().cmp(key))
            .ok()
            .map(|i| &self.nodes[i])
    }

    pub fn vision_node(&self, key: &str) -> Option<&VisionNode> {
        self.vision
            .binary_search_by(|n| n.key.as_str().cmp(key))
            .ok()
            .map(|i| &self.vision[i])
    }

    /// The expression a vision node plays (its linked node's).
    pub fn vision_express(&self, key: &str) -> Option<&Express> {
        self.vision_node(key)
            .and_then(|v| self.node(&v.link))
            .and_then(|n| n.action.express())
    }

    /// Nodes that can be caught from user text (an action and at least one phrase), excluding the function bank
    /// and the proactive lines.
    pub fn text_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter().filter(|n| {
            n.group != FUNCTION_GROUP
                && n.group != PROACTIVE_GROUP
                && n.action != Action::None
                && (!n.bias.is_empty() || !n.exact_match.is_empty())
        })
    }

    /// What Zena says on her own: `proactive.stretch` for a break, else `proactive.<name>` for the scene whose
    /// expression node is `scene_link` (`react.coding` → `proactive.coding`), else `proactive.idle`.
    pub fn proactive(&self, scene_link: Option<&str>, take_break: bool) -> Option<(&str, &Express)> {
        let scene = scene_link.and_then(|l| l.rsplit('.').next());
        let keys = [
            take_break.then(|| format!("{PROACTIVE_GROUP}.stretch")),
            scene.map(|s| format!("{PROACTIVE_GROUP}.{s}")),
            Some(format!("{PROACTIVE_GROUP}.idle")),
        ];
        keys.into_iter()
            .flatten()
            .find_map(|k| self.node(&k))
            .and_then(|n| n.action.express().map(|e| (n.key.as_str(), e)))
    }

    /// A coaching line when the course section turns high-friction: `proactive.coach`, else the scene's line
    /// ([`Lexicon::proactive`]).
    pub fn proactive_coach(&self, scene_link: Option<&str>) -> Option<(&str, &Express)> {
        self.node(&format!("{PROACTIVE_GROUP}.coach"))
            .and_then(|n| n.action.express().map(|e| (n.key.as_str(), e)))
            .or_else(|| self.proactive(scene_link, false))
    }

    /// The function bank's phrases (request endings, fillers), lowercased.
    pub fn function_phrases(&self) -> Vec<String> {
        self.nodes
            .iter()
            .filter(|n| n.group == FUNCTION_GROUP)
            .flat_map(|n| n.bias.iter().map(|p| p.text.to_lowercase()))
            .collect()
    }

    /// Every vision phrase that gets an anchor (bias, and prejudice that is not background), deduplicated, in order.
    pub fn vision_phrases(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for v in &self.vision {
            for p in v
                .bias
                .iter()
                .chain(v.prejudice.iter().filter(|p| !self.prompting.is_background(&p.text)))
            {
                if seen.insert(p.text.clone()) {
                    out.push(p.text.clone());
                }
            }
        }
        out
    }

    /// Every phrase that needs an anchor: the vision phrases plus the background phrases (which never act as
    /// prejudice, but compete with the labels in the per-cell contrast heat).
    pub fn anchor_phrases(&self) -> Vec<String> {
        let mut out = self.vision_phrases();
        for b in &self.prompting.background {
            if !out.contains(b) {
                out.push(b.clone());
            }
        }
        out
    }

    pub fn validate(&self) -> Result<()> {
        let mut problems = Vec::new();
        // An exact-match entry confirms a node without vectors, so two nodes may not claim the same one.
        let mut claimed: BTreeMap<String, &str> = BTreeMap::new();
        for n in self.nodes.iter().filter(|n| n.group != FUNCTION_GROUP) {
            for e in &n.exact_match {
                let c = compact(e);
                if c.is_empty() {
                    continue;
                }
                match claimed.get(&c) {
                    Some(other) if *other != n.key => problems.push(format!("exact_match '{e}' is claimed by {other} and {}", n.key)),
                    _ => {
                        claimed.insert(c, &n.key);
                    }
                }
            }
            if let Some(e) = n.action.express() {
                if e.level == ExpressLevel::Lv2 && e.lines.is_empty() {
                    problems.push(format!("{}: an lv2 express needs lines", n.key));
                }
            }
        }
        for v in &self.vision {
            match self.node(&v.link) {
                None => problems.push(format!("{} links to {} which does not exist", v.key, v.link)),
                Some(n) if n.action.express().is_none() => problems.push(format!("{} links to {} which has no express", v.key, v.link)),
                _ => {}
            }
            if v.bias.is_empty() {
                problems.push(format!("{} has no bias prompts", v.key));
            }
        }
        if let Some(t) = self.prompting.templates.iter().find(|t| !t.contains("{}")) {
            problems.push(format!("vision_prompting template '{t}' has no {{}}"));
        }
        if !problems.is_empty() {
            bail!("invalid lexicon:\n  {}", problems.join("\n  "));
        }
        Ok(())
    }

    /// Back to the bias.json form (groups and nodes in key order).
    pub fn to_json(&self) -> Value {
        let mut root = Map::new();
        root.insert("version".into(), json!(self.version));
        if !self.meta.is_null() {
            root.insert("meta".into(), self.meta.clone());
        }
        for n in &self.nodes {
            root.entry(n.group.clone()).or_insert_with(|| Value::Object(Map::new()))[&n.name] = node_json(n);
        }
        if !self.prompting.templates.is_empty() || !self.prompting.background.is_empty() {
            let mut p = Map::new();
            if !self.prompting.semantic.is_empty() {
                p.insert("semantic".into(), json!(self.prompting.semantic));
            }
            p.insert("templates".into(), json!(self.prompting.templates));
            p.insert("background".into(), json!(self.prompting.background));
            root.insert("vision_prompting".into(), Value::Object(p));
        }
        if !self.vision.is_empty() {
            let mut vs = Map::new();
            for v in &self.vision {
                vs.insert(v.name.clone(), vision_json(v));
            }
            root.insert("vision".into(), Value::Object(vs));
        }
        Value::Object(root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bundled_lexicon_loads_validates_and_round_trips() {
        let lex = Lexicon::bundled();
        assert!(lex.nodes.len() >= 25, "{}", lex.nodes.len());
        let exit = lex.node("system.exit_app").unwrap();
        assert!(matches!(
            &exit.action,
            Action::System {
                command: SystemCommand::ExitApp,
                mode: None,
                express: Some(_)
            }
        ));
        let ghost = lex.node("system.ghost_mode").unwrap();
        assert!(matches!(
            ghost.action,
            Action::System {
                command: SystemCommand::KeepActive,
                mode: Some(WindowLayerMode::GhostClickThrough),
                ..
            }
        ));
        let win = lex.node("emotion.celebrate").unwrap();
        match &win.action {
            Action::Express(e) => {
                assert_eq!(
                    (e.emotion, e.motion, e.level),
                    (EmotionState::Excited, BodyMotion::Cheer, ExpressLevel::Lv2)
                );
                assert!(!e.lines.is_empty());
            }
            a => panic!("{a:?}"),
        }
        assert!(matches!(&lex.node("coach.game_tip").unwrap().action, Action::Ask { topic, .. } if topic == "coach"));
        assert_eq!(lex.vision_express("vision.game_victory"), win.action.express());
        assert_eq!(lex.vision_node("vision.private").unwrap().kind, VisionKind::Privacy);
        assert!(lex.function_phrases().contains(&"해 줘".to_string()));
        assert!(lex.text_nodes().all(|n| n.group != "function" && n.group != "react"));
        // export → import gives the same dictionary
        let again = Lexicon::from_json(&lex.to_json().to_string()).unwrap();
        assert_eq!(again, lex);
    }

    #[test]
    fn unknown_names_dangling_links_and_double_claims_are_rejected() {
        let bad_emotion = r#"{"e": {"x": {"bias": "a", "express": {"emotion": "HAPPYISH", "motion": "NOD", "lines": ["hi"]}}}}"#;
        assert!(format!("{:#}", Lexicon::from_json(bad_emotion).unwrap_err()).contains("unknown emotion 'HAPPYISH'"));
        let dangling = r#"{"e": {"x": {"bias": "a", "express": {"emotion": "JOY", "motion": "NOD", "lines": ["hi"]}}},
                          "vision": {"v": {"kind": "event", "link": "e.y", "bias": "a photo"}}}"#;
        assert!(format!("{:#}", Lexicon::from_json(dangling).unwrap_err()).contains("links to e.y which does not exist"));
        let twice = r#"{"e": {"x": {"exact_match": ["대박"], "express": {"emotion": "JOY", "motion": "NOD", "lines": ["a"]}},
                               "y": {"exact_match": ["대 박"], "express": {"emotion": "JOY", "motion": "NOD", "lines": ["b"]}}}}"#;
        assert!(format!("{:#}", Lexicon::from_json(twice).unwrap_err()).contains("claimed by e.x and e.y"));
        let no_lines = r#"{"e": {"x": {"bias": "a", "express": {"emotion": "JOY", "motion": "NOD", "level": "lv2"}}}}"#;
        assert!(Lexicon::from_json(no_lines).is_err());
        let bad_emote = r#"{"e": {"x": {"bias": "a", "express": {"emotion": "JOY", "motion": "NOD", "emote": "PARTY", "lines": ["hi"]}}}}"#;
        assert!(format!("{:#}", Lexicon::from_json(bad_emote).unwrap_err()).contains("unknown emote 'PARTY'"));
        let bad_part = r#"{"e": {"x": {"bias": "a", "asset": {"action": "EQUIP_PART", "part": "JETPACK"}}}}"#;
        assert!(format!("{:#}", Lexicon::from_json(bad_part).unwrap_err()).contains("avatar part 'JETPACK'"));
    }

    #[test]
    fn emotes_assets_and_proactive_lines() {
        use zena_core::tools::{AssetAction, AvatarPart, AvatarTrack};
        let lex = Lexicon::bundled();
        let e = lex.node("emotion.praise").unwrap().action.express().unwrap();
        assert_eq!(e.emote, Some(CharacterEmote::ChibiHeart));
        match &lex.node("avatar.robot_mode").unwrap().action {
            Action::Asset { asset, express } => {
                assert_eq!(
                    (asset.action, asset.track),
                    (AssetAction::SwitchTrack, Some(AvatarTrack::Solidcraft))
                );
                assert!(express.is_some());
            }
            a => panic!("{a:?}"),
        }
        assert!(matches!(
            &lex.node("avatar.glasses").unwrap().action,
            Action::Asset { asset, .. } if asset.part == Some(AvatarPart::Glasses)
        ));
        assert!(lex.node("avatar.glasses").unwrap().action.is_command());
        let (k, _) = lex.proactive(Some("react.coding"), false).unwrap();
        assert_eq!(k, "proactive.coding");
        assert_eq!(lex.proactive(Some("react.coding"), true).unwrap().0, "proactive.stretch");
        assert_eq!(lex.proactive(Some("react.nowhere"), false).unwrap().0, "proactive.idle");
        assert_eq!(lex.proactive(None, false).unwrap().0, "proactive.idle");
        let (k, e) = lex.proactive_coach(Some("react.combat")).unwrap();
        assert_eq!(k, "proactive.coach");
        assert!(!e.lines.is_empty());
        assert!(lex.text_nodes().all(|n| n.group != PROACTIVE_GROUP));
    }

    #[test]
    fn vision_phrases_skip_background_and_templates_are_signed() {
        let lex = Lexicon::bundled();
        let ps = lex.vision_phrases();
        assert!(ps.contains(&"a game over screen".to_string()));
        assert!(
            !ps.iter().any(|p| lex.prompting.is_background(p)),
            "background phrases get no anchor"
        );
        assert_eq!(lex.prompting.forms("a chart").len(), lex.prompting.templates.len());
        assert_eq!(VisionPrompting::apply("a screenshot of {}.", "a chart"), "a screenshot of a chart.");
        let mut other = lex.prompting.clone();
        other.templates.pop();
        assert_ne!(other.signature(), lex.prompting.signature());
    }
}
