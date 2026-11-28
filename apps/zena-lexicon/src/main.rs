//! `zena-lexicon` — edit Raon Zena's word dictionary (LanceDB).
//!
//! ```text
//! zena-lexicon [--db DIR] import [data/zena_lexicon.json]       seed / re-seed (hand-added and learned phrases stay)
//! zena-lexicon [--db DIR] export [out.json]                     the dictionary back in bias.json form
//! zena-lexicon [--db DIR] add    <group.name> <bias|prejudice|exact|line> <text>
//! zena-lexicon [--db DIR] remove <group.name> <bias|prejudice|exact|line> <text>
//! zena-lexicon [--db DIR] search <text>                         BM25 over phrases and screen prompts (n-grams)
//! zena-lexicon [--db DIR] show   <group.name>
//! zena-lexicon [--db DIR] candidates                            phrases being learned from use
//! zena-lexicon [--db DIR] history | rollback <revision>         Lance versions of the dictionary
//! zena-lexicon [--db DIR] anchors --model models/KRAFTON/Raon-VisionEncoder --tokenizer tokenizer.json
//!                                 [--format f32|f16|bf16|q8] [--device auto|cpu|cuda|...] [--all]
//! zena-lexicon [--db DIR] nearest --model ... --tokenizer ... <text>   anchored prompts nearest to a text
//! zena-lexicon [--db DIR] look --model ... [--patches 256] [--pass full|checker|fovea|checker-x|fovea-x] <shot.ppm>
//!                                 what the screen watcher would see in a screenshot (binary PPM)
//! zena-lexicon [--db DIR] compact [--keep-days N]             merge small files; drop versions older than N days
//! zena-lexicon [--db DIR] stats
//! ```
//! A running Zena picks every change up within a few seconds (it watches the revision).

use anyhow::{anyhow, bail, Result};
use std::path::PathBuf;
use zena_lexicon::store::Role;
use zena_lexicon::{Lexicon, LexiconStore};

fn default_db() -> PathBuf {
    if cfg!(windows) {
        if let Some(d) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(d).join("RaonZena").join("lexicon.lance");
        }
    }
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("raon-zena").join("lexicon.lance")
}

fn flag(args: &mut Vec<String>, name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == name)?;
    args.remove(i);
    (i < args.len()).then(|| args.remove(i))
}

fn switch(args: &mut Vec<String>, name: &str) -> bool {
    match args.iter().position(|a| a == name) {
        Some(i) => {
            args.remove(i);
            true
        }
        None => false,
    }
}

/// A binary PPM (P6, 8-bit) → RGB8.
fn read_ppm(b: &[u8]) -> Result<(Vec<u8>, usize, usize)> {
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 && i < b.len() {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < b.len() && b[i] == b'#' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        while i < b.len() && !b[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(String::from_utf8_lossy(&b[start..i]).to_string());
    }
    if fields.len() < 4 || fields[0] != "P6" || fields[3] != "255" {
        bail!("not a binary 8-bit PPM (P6)");
    }
    let (w, h): (usize, usize) = (fields[1].parse()?, fields[2].parse()?);
    let data = &b[i + 1..];
    if data.len() < w * h * 3 {
        bail!("PPM data is short");
    }
    Ok((data[..w * h * 3].to_vec(), w, h))
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let db = flag(&mut args, "--db").map(PathBuf::from).unwrap_or_else(default_db);
    let model = flag(&mut args, "--model").map(PathBuf::from);
    let tokenizer = flag(&mut args, "--tokenizer").map(PathBuf::from);
    let format: raon_vision::WeightFormat = flag(&mut args, "--format")
        .map(|f| f.parse())
        .transpose()?
        .unwrap_or(raon_vision::WeightFormat::F32);
    let device: raon_infer::device::Backend = flag(&mut args, "--device")
        .map(|d| d.parse())
        .transpose()?
        .unwrap_or(raon_infer::device::Backend::Auto);
    let all = switch(&mut args, "--all");
    let patches: usize = flag(&mut args, "--patches").map(|p| p.parse()).transpose()?.unwrap_or(256);
    let pass = flag(&mut args, "--pass").unwrap_or_else(|| "full".into());
    let keep_days: Option<u64> = flag(&mut args, "--keep-days").map(|d| d.parse()).transpose()?;
    let cmd = if args.is_empty() { "stats".to_string() } else { args.remove(0) };
    let store = LexiconStore::open(&db)?;
    match cmd.as_str() {
        "import" => {
            let lex = match args.first() {
                Some(p) => Lexicon::from_json(&std::fs::read_to_string(p)?)?,
                None => Lexicon::bundled(),
            };
            let r = store.import(&lex)?;
            println!(
                "imported {} domains ({} phrases), {} screen domains ({} prompts, {} anchors kept); {} hand-added / learned phrases merged back",
                r.domains, r.phrases, r.vision, r.prompts, r.anchors_kept, r.merged
            );
        }
        "export" => {
            let json = serde_json::to_string_pretty(&store.load()?.to_json())?;
            match args.first() {
                Some(p) => raon_infer::device::write_file_atomic(std::path::Path::new(p), (json + "\n").as_bytes())?,
                None => println!("{json}"),
            }
        }
        "add" | "remove" => {
            let [key, role, text] = <[String; 3]>::try_from(args.clone())
                .map_err(|_| anyhow!("usage: {cmd} <group.name> <bias|prejudice|exact|line> <text>"))?;
            let role = Role::parse(&role)?;
            let changed = if cmd == "add" {
                store.add_phrase(&key, role, &text, "user")?
            } else {
                store.remove_phrase(&key, role, &text)?
            };
            println!(
                "{} (revision {})",
                if changed { "done" } else { "nothing to change" },
                store.revision()?
            );
        }
        "search" => {
            let q = args.join(" ");
            for h in store.search(&q, 20)? {
                println!("{:>7.3}  {:<26} {:<16} {}", h.score, h.domain, h.role, h.text);
            }
        }
        "show" => {
            let key = args.first().ok_or_else(|| anyhow!("usage: show <group.name>"))?;
            let lex = store.load()?;
            match (lex.node(key), lex.vision_node(key)) {
                (Some(n), _) => println!("{}", serde_json::to_string_pretty(&zena_lexicon::model::node_json(n))?),
                (_, Some(v)) => println!("{}", serde_json::to_string_pretty(&zena_lexicon::model::vision_json(v))?),
                _ => bail!("no node {key}"),
            }
        }
        "candidates" => {
            for (d, t, h) in store.candidates()? {
                println!("{h:>3}  {d:<26} {t}");
            }
        }
        "scoped" => {
            let cc = args.first().map(String::as_str);
            for b in store.scoped_bindings(cc)? {
                let status = if b.promoted { "promoted" } else { "pending" };
                println!(
                    "{:<22} {:<24} {:<10} sessions={} rejects={} {}",
                    b.cc,
                    b.domain,
                    status,
                    b.sessions.len(),
                    b.rejects,
                    b.text
                );
            }
        }
        "audit" => {
            let limit: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(20);
            for r in store.audit_log(limit)? {
                println!("{:>5}  {}  {}", r.revision, r.updated_ms, r.note);
            }
        }
        "history" => {
            for r in store.history()? {
                println!("{:>5}  {}  {}", r.revision, r.updated_ms, r.note);
            }
        }
        "rollback" => {
            let rev: i64 = args.first().ok_or_else(|| anyhow!("usage: rollback <revision>"))?.parse()?;
            println!("restored revision {rev} as revision {}", store.rollback(rev)?);
        }
        "anchors" | "nearest" => {
            let model = model.ok_or_else(|| anyhow!("--model <Raon-VisionEncoder or google/siglip2-so400m-patch16-naflex dir> is required"))?;
            let tokenizer = tokenizer.ok_or_else(|| anyhow!("--tokenizer <timm/ViT-SO400M-16-SigLIP2-256 tokenizer.json> is required"))?;
            let dev = raon_infer::device::select_device(device, 0)?;
            let t = std::time::Instant::now();
            let text = raon_vision::text::TextTower::load(&raon_vision::checkpoint(&model), &dev, format)?;
            let tok = raon_vision::text::SiglipTokenizer::load(&tokenizer)?;
            eprintln!(
                "text tower loaded in {:.1}s on {}",
                t.elapsed().as_secs_f32(),
                raon_infer::device::describe(&dev)
            );
            if cmd == "nearest" {
                let v = text.embed(&tok, &[args.join(" ")], 1)?.remove(0);
                for (p, c) in store.nearest_prompts(&v, 10)? {
                    println!("{c:>7.4}  {p}");
                }
                return Ok(());
            }
            let lex = store.load()?;
            let todo = if all { lex.anchor_phrases() } else { store.missing_anchors()? };
            if todo.is_empty() {
                println!("every screen prompt is anchored ({})", store.stats()?);
                return Ok(());
            }
            let t = std::time::Instant::now();
            let anchors = raon_vision::text::compute_anchors(&text, &tok, &lex.prompting, &todo)?;
            let n = store.set_anchors(&raon_vision::fingerprint(&model), &anchors)?;
            println!(
                "anchored {n} prompts × {} templates in {:.1}s → {}",
                lex.prompting.templates.len().max(1),
                t.elapsed().as_secs_f32(),
                store.stats()?
            );
        }
        "look" => {
            let model = model.ok_or_else(|| anyhow!("--model <Raon-VisionEncoder dir> is required"))?;
            let path = args.first().ok_or_else(|| anyhow!("usage: look --model DIR <screenshot.ppm>"))?;
            let (rgb, w, h) = read_ppm(&std::fs::read(path)?)?;
            let lex = store.load()?;
            let anchors = store.anchors()?;
            let dev = raon_infer::device::select_device(device, 0)?;
            let (tower, scale, bias) = raon_vision::load_vision(&model, &dev, format)?;
            let bank = <raon_vision::AnchorBank as raon_vision::LexiconAnchors>::build(&lex, &anchors.vectors, scale, bias);
            if bank.is_empty() {
                bail!("no screen domain is anchored yet: run `zena-lexicon anchors` first");
            }
            let perforation = match pass.as_str() {
                "full" => raon_vision::PerforationPolicy::Off,
                "auto" => raon_vision::PerforationPolicy::Auto,
                p => raon_vision::PerforationPolicy::Forced(
                    raon_vision::PerforatedCfg::parse(p).ok_or_else(|| anyhow!("unknown --pass {pass}"))?,
                ),
            };
            let cfg = raon_vision::ScreenConfig {
                max_patches: patches,
                perforation,
                seed_labels: 3,
            };
            let grid = raon_vision::naflex::patchify_frame_with(&rgb, w, h, patches, &raon_vision::preprocess_for(&model));
            let r = raon_vision::screen::read_pass(&tower, &cfg, &grid, &bank)?;
            println!(
                "{w}x{h} → {}x{} grid, {} of {} tokens through the encoder, {:.0} ms",
                r.grid.0, r.grid.1, r.report.tokens_run, r.report.cells, r.encode_ms
            );
            for s in r.scores.iter().take(5) {
                let express = lex
                    .vision_express(&s.key)
                    .map(|e| format!("{:?}/{:?} {:?}", e.emotion, e.motion, e.lines.first()))
                    .unwrap_or_default();
                println!(
                    "{:>6.3}  {:<22} {:<8} bias {:.3} prej {:.3} cos {:.3} → {} {}",
                    s.score,
                    s.key,
                    s.kind.as_str(),
                    s.bias,
                    s.prejudice,
                    s.cos,
                    s.link,
                    express
                );
            }
            match raon_vision::Acceptance::default().accept(&r.scores) {
                Some(s) => println!("→ Zena reacts with {} ({})", s.link, s.kind.as_str()),
                None => println!("→ no reaction (no domain is clearly on screen)"),
            }
            if let Some((x, y)) = r.peak {
                println!("gaze peak at ({x:.2}, {y:.2})");
            }
        }
        "compact" => {
            store.compact(keep_days.map(|d| std::time::Duration::from_secs(d * 86_400)))?;
            println!("compacted ({})", store.stats()?);
        }
        "stats" => println!("{}", store.stats()?),
        other => bail!(
            "unknown command {other} (import, export, add, remove, search, show, candidates, history, rollback, anchors, nearest, stats)"
        ),
    }
    raon_infer::device::rocm_exit_now_if_used(0);
    Ok(())
}
