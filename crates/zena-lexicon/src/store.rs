//! The expression dictionary kept in a LanceDB database (a directory of Lance tables).
//!
//! ```text
//! meta        key='zena', revision, meta, prompting, anchor_model, anchor_signature, tables, note, updated_ms
//! domains     key, group, name, record           record = the node in bias.json form (JSON text)
//! phrases     domain, role, text, weight, source  every bias / prejudice / exact phrase and vision prompt;
//!                                                 BM25 index on `text` with the n-gram tokenizer (Korean words
//!                                                 carry josa, so whole-word tokens would miss "물약을" → "물약")
//! vision      key, name, kind, link, record
//! anchors     text, model, signature, vector[1152] prompt-ensembled SigLIP2 anchors (cosine vector search)
//! candidates  domain, text, hits, last_ms         phrases learned from use, promoted to bias after N hits
//! ```
//!
//! A domain's `record` is the source of truth (it round-trips through `Lexicon::from_json`); its `phrases` rows are
//! rewritten from it on every edit. Every logical update bumps `meta.revision` and records the Lance version of
//! each table, so a running Zena notices the change (hot reload) and any revision can be restored: Lance keeps the
//! old versions (time travel), `rollback` checks them out and restores them as the newest.
//! The anchors are computed once per dictionary change by the Raon-VisionEncoder text tower (`zena-lexicon
//! anchors`); at run time only the vision tower is loaded and the anchors are read from here.

use crate::model::{node_json, node_of, vision_json, vision_of, Action, Express, Lexicon, Node, VisionPrompting};
use crate::phrase::{split_phrase_list, weighted_phrases};
use anyhow::{anyhow, bail, Context, Result};
use futures::TryStreamExt;
use lancedb::arrow::arrow_array::{
    types::Float32Type, Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray,
};
use lancedb::arrow::arrow_schema::{DataType, Field, Schema, SchemaRef};
use lancedb::database::CreateTableMode;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{Connection, DistanceType, Table};
use serde_json::json;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Width of the anchors: Raon-VisionEncoder's embedding (SigLIP2 So400m).
pub const ANCHOR_DIMS: usize = 1152;

const META: &str = "meta";
const DOMAINS: &str = "domains";
const PHRASES: &str = "phrases";
const VISION: &str = "vision";
const ANCHORS: &str = "anchors";
const CANDIDATES: &str = "candidates";
const TABLES: [&str; 5] = [DOMAINS, PHRASES, VISION, ANCHORS, CANDIDATES];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Bias,
    Prejudice,
    Exact,
    /// A line of the node's `express`.
    Line,
}

impl Role {
    pub fn parse(s: &str) -> Result<Role> {
        Ok(match s {
            "bias" => Role::Bias,
            "prejudice" => Role::Prejudice,
            "exact" | "exact_match" => Role::Exact,
            "line" | "lines" => Role::Line,
            _ => bail!("unknown role '{s}' (bias, prejudice, exact, line)"),
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Role::Bias => "bias",
            Role::Prejudice => "prejudice",
            Role::Exact => "exact",
            Role::Line => "line",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImportReport {
    pub domains: usize,
    pub phrases: usize,
    pub vision: usize,
    pub prompts: usize,
    /// Anchors kept from before (same prompt text, same templates).
    pub anchors_kept: usize,
    /// Phrases added by hand or learned that were merged back into the new seed.
    pub merged: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    pub text: String,
    /// `bias` / `prejudice` / `exact`, or `vision-bias` / `vision-prejudice` for screen prompts.
    pub role: String,
    pub domain: String,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Learned {
    /// Counted; `hits` so far.
    Counted(u32),
    /// Became a bias phrase of the domain.
    Promoted,
    /// Already a phrase of the domain.
    Known,
}

/// One logical revision of the dictionary.
#[derive(Clone, Debug, PartialEq)]
pub struct Revision {
    pub revision: i64,
    pub updated_ms: i64,
    pub note: String,
    /// Lance version of each table at this revision.
    pub tables: BTreeMap<String, u64>,
}

/// Vision anchors: prompt text → unit vector.
#[derive(Clone, Debug, Default)]
pub struct Anchors {
    pub model: String,
    pub signature: String,
    pub vectors: HashMap<String, Vec<f32>>,
}

/// One phrase row.
#[derive(Clone, Debug)]
struct PhraseRow {
    domain: String,
    role: String,
    text: String,
    weight: f32,
    source: String,
}

/// The meta row: (revision, meta, prompting, anchor model, anchor signature).
type MetaRow = (i64, String, String, String, String);

pub struct LexiconStore {
    rt: tokio::runtime::Runtime,
    conn: Connection,
    pub path: PathBuf,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// A SQL string literal.
fn lit(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

fn utf8(name: &str) -> Field {
    Field::new(name, DataType::Utf8, false)
}

fn schema(table: &str) -> SchemaRef {
    let fields = match table {
        META => vec![
            utf8("key"),
            Field::new("revision", DataType::Int64, false),
            utf8("meta"),
            utf8("prompting"),
            utf8("anchor_model"),
            utf8("anchor_signature"),
            utf8("tables"),
            utf8("note"),
            Field::new("updated_ms", DataType::Int64, false),
        ],
        DOMAINS => vec![utf8("key"), utf8("group"), utf8("name"), utf8("record")],
        PHRASES => vec![
            utf8("domain"),
            utf8("role"),
            utf8("text"),
            Field::new("weight", DataType::Float32, false),
            utf8("source"),
        ],
        VISION => vec![utf8("key"), utf8("name"), utf8("kind"), utf8("link"), utf8("record")],
        ANCHORS => vec![
            utf8("text"),
            utf8("model"),
            utf8("signature"),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), ANCHOR_DIMS as i32),
                false,
            ),
        ],
        CANDIDATES => vec![
            utf8("domain"),
            utf8("text"),
            Field::new("hits", DataType::Int64, false),
            Field::new("last_ms", DataType::Int64, false),
        ],
        _ => unreachable!("unknown table {table}"),
    };
    Arc::new(Schema::new(fields))
}

fn strings(v: Vec<String>) -> ArrayRef {
    Arc::new(StringArray::from(v))
}

fn col_str(b: &RecordBatch, name: &str) -> Result<Vec<String>> {
    let c = b.column_by_name(name).ok_or_else(|| anyhow!("no column {name}"))?;
    let a = c
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("column {name} is not utf8"))?;
    Ok((0..a.len()).map(|i| a.value(i).to_string()).collect())
}

fn col_i64(b: &RecordBatch, name: &str) -> Result<Vec<i64>> {
    let c = b.column_by_name(name).ok_or_else(|| anyhow!("no column {name}"))?;
    let a = c
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| anyhow!("column {name} is not int64"))?;
    Ok(a.values().to_vec())
}

fn col_f32(b: &RecordBatch, name: &str) -> Result<Vec<f32>> {
    let c = b.column_by_name(name).ok_or_else(|| anyhow!("no column {name}"))?;
    let a = c
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| anyhow!("column {name} is not float32"))?;
    Ok(a.values().to_vec())
}

fn col_vectors(b: &RecordBatch, name: &str) -> Result<Vec<Vec<f32>>> {
    let c = b.column_by_name(name).ok_or_else(|| anyhow!("no column {name}"))?;
    let a = c
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| anyhow!("column {name} is not a vector"))?;
    (0..a.len())
        .map(|i| {
            let v = a.value(i);
            let f = v
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow!("vector items are not float32"))?;
            Ok(f.values().to_vec())
        })
        .collect()
}

fn phrase_batch(rows: &[PhraseRow]) -> Result<RecordBatch> {
    Ok(RecordBatch::try_new(
        schema(PHRASES),
        vec![
            strings(rows.iter().map(|r| r.domain.clone()).collect()),
            strings(rows.iter().map(|r| r.role.clone()).collect()),
            strings(rows.iter().map(|r| r.text.clone()).collect()),
            Arc::new(Float32Array::from(rows.iter().map(|r| r.weight).collect::<Vec<f32>>())),
            strings(rows.iter().map(|r| r.source.clone()).collect()),
        ],
    )?)
}

fn anchor_batch(rows: &[(String, String, String, Vec<f32>)]) -> Result<RecordBatch> {
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        rows.iter().map(|r| Some(r.3.iter().map(|x| Some(*x)).collect::<Vec<_>>())),
        ANCHOR_DIMS as i32,
    );
    Ok(RecordBatch::try_new(
        schema(ANCHORS),
        vec![
            strings(rows.iter().map(|r| r.0.clone()).collect()),
            strings(rows.iter().map(|r| r.1.clone()).collect()),
            strings(rows.iter().map(|r| r.2.clone()).collect()),
            Arc::new(vectors),
        ],
    )?)
}

/// The phrase rows of a text node (`sources` keeps the origin of phrases that existed before).
fn node_rows(n: &Node, sources: &HashMap<(String, String, String), String>) -> Vec<PhraseRow> {
    let src = |role: &str, text: &str| {
        sources
            .get(&(n.key.clone(), role.to_string(), text.to_string()))
            .cloned()
            .unwrap_or_else(|| "seed".into())
    };
    let mut rows = Vec::new();
    for p in &n.bias {
        rows.push(PhraseRow {
            domain: n.key.clone(),
            role: "bias".into(),
            text: p.text.clone(),
            weight: p.weight,
            source: src("bias", &p.text),
        });
    }
    for p in &n.prejudice {
        rows.push(PhraseRow {
            domain: n.key.clone(),
            role: "prejudice".into(),
            text: p.text.clone(),
            weight: p.weight,
            source: src("prejudice", &p.text),
        });
    }
    for e in &n.exact_match {
        rows.push(PhraseRow {
            domain: n.key.clone(),
            role: "exact".into(),
            text: e.clone(),
            weight: 1.0,
            source: src("exact", e),
        });
    }
    rows
}

fn express_mut(a: &mut Action) -> Option<&mut Express> {
    match a {
        Action::Express(e) => Some(e),
        Action::System { express, .. } | Action::Asset { express, .. } | Action::Ask { express, .. } => express.as_mut(),
        Action::None => None,
    }
}

impl LexiconStore {
    /// Opens the dictionary directory (created on first write).
    pub fn open(path: &Path) -> Result<LexiconStore> {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
        let uri = path.to_string_lossy().to_string();
        let conn = rt
            .block_on(lancedb::connect(&uri).execute())
            .with_context(|| format!("opening LanceDB at {}", path.display()))?;
        Ok(LexiconStore {
            rt,
            conn,
            path: path.to_path_buf(),
        })
    }

    async fn table(&self, name: &str) -> Result<Option<Table>> {
        let names = self.conn.table_names().execute().await?;
        if !names.iter().any(|n| n == name) {
            return Ok(None);
        }
        Ok(Some(self.conn.open_table(name).execute().await?))
    }

    async fn rows(&self, name: &str, filter: Option<String>) -> Result<Vec<RecordBatch>> {
        let Some(t) = self.table(name).await? else { return Ok(Vec::new()) };
        let mut q = t.query();
        if let Some(f) = filter {
            q = q.only_if(f);
        }
        Ok(q.execute().await?.try_collect::<Vec<_>>().await?)
    }

    async fn overwrite(&self, name: &str, batch: RecordBatch) -> Result<Table> {
        Ok(self
            .conn
            .create_table(name, batch)
            .mode(CreateTableMode::Overwrite)
            .execute()
            .await?)
    }

    /// Opens `name`, creating it empty when missing.
    async fn table_or_create(&self, name: &str) -> Result<Table> {
        match self.table(name).await? {
            Some(t) => Ok(t),
            None => self.overwrite(name, RecordBatch::new_empty(schema(name))).await,
        }
    }

    /// (Re)builds the BM25 index over the phrase text: n-grams of 2–3 characters, lowercased.
    async fn index_phrases(&self) -> Result<()> {
        let t = self.table_or_create(PHRASES).await?;
        if t.count_rows(None).await? == 0 {
            return Ok(());
        }
        let params = FtsIndexBuilder::default()
            .base_tokenizer("ngram".to_string())
            .ngram_min_length(2)
            .ngram_max_length(3)
            .lower_case(true);
        t.create_index(&["text"], Index::FTS(params)).replace(true).execute().await?;
        Ok(())
    }

    async fn meta_row(&self) -> Result<Option<MetaRow>> {
        for b in self.rows(META, None).await? {
            if b.num_rows() > 0 {
                return Ok(Some((
                    col_i64(&b, "revision")?[0],
                    col_str(&b, "meta")?[0].clone(),
                    col_str(&b, "prompting")?[0].clone(),
                    col_str(&b, "anchor_model")?[0].clone(),
                    col_str(&b, "anchor_signature")?[0].clone(),
                )));
            }
        }
        Ok(None)
    }

    /// Writes the next revision: the meta row with every table's Lance version.
    async fn commit(&self, note: &str, meta: Option<&str>, prompting: Option<&str>, anchors: Option<(&str, &str)>) -> Result<i64> {
        let old = self.meta_row().await?;
        let revision = old.as_ref().map_or(0, |m| m.0) + 1;
        let mut tables = BTreeMap::new();
        for name in TABLES {
            tables.insert(name.to_string(), self.table_or_create(name).await?.version().await?);
        }
        let pick = |new: Option<&str>, old: Option<&String>| new.map(str::to_string).or_else(|| old.cloned()).unwrap_or_default();
        let batch = RecordBatch::try_new(
            schema(META),
            vec![
                strings(vec!["zena".into()]),
                Arc::new(Int64Array::from(vec![revision])),
                strings(vec![pick(meta, old.as_ref().map(|m| &m.1))]),
                strings(vec![pick(prompting, old.as_ref().map(|m| &m.2))]),
                strings(vec![pick(anchors.map(|a| a.0), old.as_ref().map(|m| &m.3))]),
                strings(vec![pick(anchors.map(|a| a.1), old.as_ref().map(|m| &m.4))]),
                strings(vec![serde_json::to_string(&tables)?]),
                strings(vec![note.to_string()]),
                Arc::new(Int64Array::from(vec![now_ms()])),
            ],
        )?;
        // One version of the meta table per revision, so its history lists every revision.
        self.overwrite(META, batch).await?;
        Ok(revision)
    }

    /// The dictionary's revision (0 when empty); it moves with every update, which is how Zena hot-reloads.
    pub fn revision(&self) -> Result<i64> {
        self.rt.block_on(async { Ok(self.meta_row().await?.map_or(0, |m| m.0)) })
    }

    /// Replaces the dictionary with `lex` (the seed). Phrases added by hand (`user`) or learned from use (`learned`)
    /// are merged back into their domains when those still exist, so re-seeding keeps the updates; learned
    /// candidates and the anchors of unchanged prompts stay.
    pub fn import(&self, lex: &Lexicon) -> Result<ImportReport> {
        lex.validate()?;
        self.rt.block_on(async {
            let mut rep = ImportReport::default();
            let mut sources: HashMap<(String, String, String), String> = HashMap::new();
            for b in self.rows(PHRASES, None).await? {
                let (d, r, t, s) = (
                    col_str(&b, "domain")?,
                    col_str(&b, "role")?,
                    col_str(&b, "text")?,
                    col_str(&b, "source")?,
                );
                for i in 0..b.num_rows() {
                    sources.insert((d[i].clone(), r[i].clone(), t[i].clone()), s[i].clone());
                }
            }
            let mut lex = lex.clone();
            let mut extras: Vec<_> = sources
                .iter()
                .filter(|(_, src)| src.as_str() != "seed" && src.as_str() != "vision")
                .collect();
            extras.sort();
            let node_idx: HashMap<String, usize> = lex
                .nodes
                .iter()
                .enumerate()
                .map(|(i, n)| (n.key.clone(), i))
                .collect();
            for ((domain, role, text), _) in extras {
                let Some(&i) = node_idx.get(domain.as_str()) else {
                    continue;
                };
                let mut node = lex.nodes[i].clone();
                match role.as_str() {
                    "bias" if !node.bias.iter().any(|p| &p.text == text) => node.bias.extend(weighted_phrases(text)),
                    "prejudice" if !node.prejudice.iter().any(|p| &p.text == text) => node.prejudice.extend(weighted_phrases(text)),
                    "exact" if !node.exact_match.contains(text) => node.exact_match.push(text.clone()),
                    _ => continue,
                }
                let before = std::mem::replace(&mut lex.nodes[i], node);
                if lex.validate().is_err() {
                    lex.nodes[i] = before; // e.g. an exact word the new seed gives to another node
                } else {
                    rep.merged += 1;
                }
            }

            let records: Vec<String> = lex.nodes.iter().map(|n| node_json(n).to_string()).collect();
            let domains = RecordBatch::try_new(
                schema(DOMAINS),
                vec![
                    strings(lex.nodes.iter().map(|n| n.key.clone()).collect()),
                    strings(lex.nodes.iter().map(|n| n.group.clone()).collect()),
                    strings(lex.nodes.iter().map(|n| n.name.clone()).collect()),
                    strings(records),
                ],
            )?;
            self.overwrite(DOMAINS, domains).await?;
            rep.domains = lex.nodes.len();

            let mut rows: Vec<PhraseRow> = lex.nodes.iter().flat_map(|n| node_rows(n, &sources)).collect();
            rep.phrases = rows.len();
            for v in &lex.vision {
                for (role, list) in [("vision-bias", &v.bias), ("vision-prejudice", &v.prejudice)] {
                    for p in list {
                        rows.push(PhraseRow {
                            domain: v.key.clone(),
                            role: role.into(),
                            text: p.text.clone(),
                            weight: p.weight,
                            source: "vision".into(),
                        });
                    }
                }
            }
            self.overwrite(PHRASES, phrase_batch(&rows)?).await?;
            self.index_phrases().await?;

            let vision = RecordBatch::try_new(
                schema(VISION),
                vec![
                    strings(lex.vision.iter().map(|v| v.key.clone()).collect()),
                    strings(lex.vision.iter().map(|v| v.name.clone()).collect()),
                    strings(lex.vision.iter().map(|v| v.kind.as_str().to_string()).collect()),
                    strings(lex.vision.iter().map(|v| v.link.clone()).collect()),
                    strings(lex.vision.iter().map(|v| vision_json(v).to_string()).collect()),
                ],
            )?;
            self.overwrite(VISION, vision).await?;
            rep.vision = lex.vision.len();

            // Anchors survive when the prompt text and the templates are unchanged.
            let wanted: HashSet<String> = lex.anchor_phrases().into_iter().collect();
            rep.prompts = wanted.len();
            let signature = lex.prompting.signature();
            let anchors = self.table_or_create(ANCHORS).await?;
            let mut stale = Vec::new();
            for b in self.rows(ANCHORS, None).await? {
                let (t, s) = (col_str(&b, "text")?, col_str(&b, "signature")?);
                for i in 0..b.num_rows() {
                    if wanted.contains(&t[i]) && s[i] == signature {
                        rep.anchors_kept += 1;
                    } else {
                        stale.push(lit(&t[i]));
                    }
                }
            }
            if !stale.is_empty() {
                anchors.delete(format!("text IN ({})", stale.join(", ")).as_str()).await?;
            }
            self.table_or_create(CANDIDATES).await?;
            let prompting =
                json!({"semantic": lex.prompting.semantic, "templates": lex.prompting.templates, "background": lex.prompting.background})
                    .to_string();
            let (old_model, old_sig) = self.meta_row().await?.map(|m| (m.3, m.4)).unwrap_or_default();
            let model = if old_sig == signature { old_model } else { String::new() };
            self.commit("import", Some(&lex.meta.to_string()), Some(&prompting), Some((&model, &signature)))
                .await?;
            Ok(rep)
        })
    }

    async fn load_async(&self) -> Result<Lexicon> {
        let (revision, meta, prompting, _, _) = self
            .meta_row()
            .await?
            .ok_or_else(|| anyhow!("{} holds no lexicon (run `zena-lexicon import`)", self.path.display()))?;
        let mut nodes = Vec::new();
        for b in self.rows(DOMAINS, None).await? {
            let (g, n, r) = (col_str(&b, "group")?, col_str(&b, "name")?, col_str(&b, "record")?);
            for i in 0..b.num_rows() {
                let rec: serde_json::Value = serde_json::from_str(&r[i]).context("domain record")?;
                nodes.push(node_of(&g[i], &n[i], &rec).with_context(|| format!("stored node {}.{}", g[i], n[i]))?);
            }
        }
        nodes.sort_by(|a, b| a.key.cmp(&b.key));
        let mut vision = Vec::new();
        for b in self.rows(VISION, None).await? {
            let (n, r) = (col_str(&b, "name")?, col_str(&b, "record")?);
            for i in 0..b.num_rows() {
                let rec: serde_json::Value = serde_json::from_str(&r[i]).context("vision record")?;
                vision.push(vision_of(&n[i], &rec)?);
            }
        }
        vision.sort_by(|a, b| a.key.cmp(&b.key));
        let p: serde_json::Value = serde_json::from_str(&prompting).unwrap_or(serde_json::Value::Null);
        let list = |k: &str| -> Vec<String> {
            p.get(k)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect())
                .unwrap_or_default()
        };
        let lex = Lexicon {
            version: revision as u64,
            meta: serde_json::from_str(&meta).unwrap_or(serde_json::Value::Null),
            nodes,
            vision,
            prompting: VisionPrompting {
                semantic: p.get("semantic").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                templates: list("templates"),
                background: list("background"),
            },
        };
        lex.validate()?;
        Ok(lex)
    }

    /// The dictionary as stored (`version` = the revision).
    pub fn load(&self) -> Result<Lexicon> {
        self.rt.block_on(self.load_async())
    }

    /// The anchors computed under the current templates.
    pub fn anchors(&self) -> Result<Anchors> {
        self.rt.block_on(async {
            let mut a = Anchors::default();
            if let Some(m) = self.meta_row().await? {
                a.model = m.3;
                a.signature = m.4;
            }
            for b in self.rows(ANCHORS, Some(format!("signature = {}", lit(&a.signature)))).await? {
                let (t, v) = (col_str(&b, "text")?, col_vectors(&b, "vector")?);
                a.vectors.extend(t.into_iter().zip(v));
            }
            Ok(a)
        })
    }

    /// Vision prompts without an anchor yet.
    pub fn missing_anchors(&self) -> Result<Vec<String>> {
        let lex = self.load()?;
        let have = self.anchors()?;
        Ok(lex.anchor_phrases().into_iter().filter(|t| !have.vectors.contains_key(t)).collect())
    }

    /// Stores anchors computed by `model` under the current templates.
    pub fn set_anchors(&self, model: &str, anchors: &[(String, Vec<f32>)]) -> Result<usize> {
        if let Some((t, v)) = anchors.iter().find(|(_, v)| v.len() != ANCHOR_DIMS) {
            bail!("anchor for '{t}' has {} dimensions, the table holds {ANCHOR_DIMS}", v.len());
        }
        self.rt.block_on(async {
            let (_, _, _, _, signature) = self.meta_row().await?.ok_or_else(|| anyhow!("no lexicon to anchor"))?;
            let t = self.table_or_create(ANCHORS).await?;
            if !anchors.is_empty() {
                let texts: Vec<String> = anchors.iter().map(|(t, _)| lit(t)).collect();
                t.delete(format!("text IN ({})", texts.join(", ")).as_str()).await?;
                let rows: Vec<(String, String, String, Vec<f32>)> = anchors
                    .iter()
                    .map(|(t, v)| (t.clone(), model.to_string(), signature.clone(), v.clone()))
                    .collect();
                t.add(anchor_batch(&rows)?).execute().await?;
            }
            self.commit("anchors", None, None, Some((model, &signature))).await?;
            Ok(anchors.len())
        })
    }

    async fn domain_node(&self, key: &str) -> Result<Node> {
        for b in self.rows(DOMAINS, Some(format!("key = {}", lit(key)))).await? {
            if b.num_rows() > 0 {
                let (g, n, r) = (col_str(&b, "group")?, col_str(&b, "name")?, col_str(&b, "record")?);
                let rec: serde_json::Value = serde_json::from_str(&r[0])?;
                return node_of(&g[0], &n[0], &rec);
            }
        }
        bail!("no domain '{key}' in the dictionary")
    }

    /// Writes an edited node: its record, its phrase rows (keeping their sources), the index and a revision.
    async fn rewrite(&self, node: &Node, edit: (&str, Role, &str), note: &str) -> Result<()> {
        // The whole dictionary must stay valid (an exact word claimed twice is refused before anything is written).
        let mut lex = self.load_async().await?;
        let i = lex
            .nodes
            .iter()
            .position(|n| n.key == node.key)
            .ok_or_else(|| anyhow!("no domain '{}'", node.key))?;
        lex.nodes[i] = node.clone();
        lex.validate()?;
        let phrases = self.table_or_create(PHRASES).await?;
        let mut sources = HashMap::new();
        for b in self.rows(PHRASES, Some(format!("domain = {}", lit(&node.key)))).await? {
            let (r, t, s) = (col_str(&b, "role")?, col_str(&b, "text")?, col_str(&b, "source")?);
            for j in 0..b.num_rows() {
                sources.insert((node.key.clone(), r[j].clone(), t[j].clone()), s[j].clone());
            }
        }
        sources.insert(
            (node.key.clone(), edit.1.name().to_string(), edit.2.to_string()),
            edit.0.to_string(),
        );
        phrases.delete(format!("domain = {}", lit(&node.key)).as_str()).await?;
        let rows = node_rows(node, &sources);
        if !rows.is_empty() {
            phrases.add(phrase_batch(&rows)?).execute().await?;
        }
        self.index_phrases().await?;
        let domains = self.table_or_create(DOMAINS).await?;
        domains.delete(format!("key = {}", lit(&node.key)).as_str()).await?;
        let batch = RecordBatch::try_new(
            schema(DOMAINS),
            vec![
                strings(vec![node.key.clone()]),
                strings(vec![node.group.clone()]),
                strings(vec![node.name.clone()]),
                strings(vec![node_json(node).to_string()]),
            ],
        )?;
        domains.add(batch).execute().await?;
        self.commit(note, None, None, None).await?;
        Ok(())
    }

    /// Adds a phrase (or an express line) to domain `key`; `false` when it was already there.
    pub fn add_phrase(&self, key: &str, role: Role, text: &str, source: &str) -> Result<bool> {
        let text = text.trim();
        if text.is_empty() {
            bail!("empty phrase");
        }
        self.rt.block_on(async {
            let mut node = self.domain_node(key).await?;
            let added = match role {
                Role::Bias | Role::Prejudice => {
                    let list = if role == Role::Bias { &mut node.bias } else { &mut node.prejudice };
                    let fresh: Vec<_> = weighted_phrases(text)
                        .into_iter()
                        .filter(|p| !list.iter().any(|q| q.text == p.text))
                        .collect();
                    let any = !fresh.is_empty();
                    list.extend(fresh);
                    any
                }
                Role::Exact => {
                    let fresh: Vec<String> = split_phrase_list(text)
                        .into_iter()
                        .filter(|e| !node.exact_match.contains(e))
                        .collect();
                    let any = !fresh.is_empty();
                    node.exact_match.extend(fresh);
                    any
                }
                Role::Line => match express_mut(&mut node.action) {
                    Some(e) if !e.lines.iter().any(|l| l == text) => {
                        e.lines.push(text.to_string());
                        true
                    }
                    Some(_) => false,
                    None => bail!("{key} has no express to add a line to"),
                },
            };
            if added {
                self.rewrite(&node, (source, role, text), &format!("add {} {key}: {text}", role.name()))
                    .await?;
            }
            Ok(added)
        })
    }

    /// Removes a phrase (or a line) from domain `key`; `false` when it was not there.
    pub fn remove_phrase(&self, key: &str, role: Role, text: &str) -> Result<bool> {
        self.rt.block_on(async {
            let mut node = self.domain_node(key).await?;
            let removed = match role {
                Role::Bias => {
                    let n = node.bias.len();
                    node.bias.retain(|p| p.text != text);
                    n != node.bias.len()
                }
                Role::Prejudice => {
                    let n = node.prejudice.len();
                    node.prejudice.retain(|p| p.text != text);
                    n != node.prejudice.len()
                }
                Role::Exact => {
                    let n = node.exact_match.len();
                    node.exact_match.retain(|e| e != text);
                    n != node.exact_match.len()
                }
                Role::Line => match express_mut(&mut node.action) {
                    Some(e) => {
                        let n = e.lines.len();
                        e.lines.retain(|l| l != text);
                        n != e.lines.len()
                    }
                    None => false,
                },
            };
            if removed {
                self.rewrite(&node, ("seed", role, text), &format!("remove {} {key}: {text}", role.name()))
                    .await?;
            }
            Ok(removed)
        })
    }

    /// Counts a phrase seen for domain `key`; after `promote_at` sightings it becomes a bias phrase (source `learned`).
    pub fn learn(&self, key: &str, text: &str, promote_at: u32) -> Result<Learned> {
        let text = text.trim().to_lowercase();
        if text.is_empty() {
            bail!("empty phrase");
        }
        let hits = self.rt.block_on(async {
            let node = self.domain_node(key).await?;
            if node.bias.iter().any(|p| p.text.to_lowercase() == text) || node.exact_match.iter().any(|e| e.to_lowercase() == text) {
                return Ok::<_, anyhow::Error>(None);
            }
            let filter = format!("domain = {} AND text = {}", lit(key), lit(&text));
            let mut hits = 0i64;
            for b in self.rows(CANDIDATES, Some(filter.clone())).await? {
                hits = hits.max(col_i64(&b, "hits")?.first().copied().unwrap_or(0));
            }
            hits += 1;
            let t = self.table_or_create(CANDIDATES).await?;
            t.delete(filter.as_str()).await?;
            if hits < promote_at as i64 {
                let batch = RecordBatch::try_new(
                    schema(CANDIDATES),
                    vec![
                        strings(vec![key.to_string()]),
                        strings(vec![text.clone()]),
                        Arc::new(Int64Array::from(vec![hits])),
                        Arc::new(Int64Array::from(vec![now_ms()])),
                    ],
                )?;
                t.add(batch).execute().await?;
                self.commit(&format!("learn {key}: {text} ({hits})"), None, None, None).await?;
            }
            Ok(Some(hits))
        })?;
        match hits {
            None => Ok(Learned::Known),
            Some(h) if h >= promote_at as i64 => {
                self.add_phrase(key, Role::Bias, &text, "learned")?;
                Ok(Learned::Promoted)
            }
            Some(h) => Ok(Learned::Counted(h as u32)),
        }
    }

    /// Learned candidates: `(domain, text, hits)`.
    pub fn candidates(&self) -> Result<Vec<(String, String, u32)>> {
        self.rt.block_on(async {
            let mut out = Vec::new();
            for b in self.rows(CANDIDATES, None).await? {
                let (d, t, h) = (col_str(&b, "domain")?, col_str(&b, "text")?, col_i64(&b, "hits")?);
                for i in 0..b.num_rows() {
                    out.push((d[i].clone(), t[i].clone(), h[i] as u32));
                }
            }
            out.sort();
            Ok(out)
        })
    }

    /// BM25 search (2–3 character n-grams) over dictionary phrases and vision prompts.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        self.rt.block_on(async {
            let Some(t) = self.table(PHRASES).await? else {
                return Ok(Vec::new());
            };
            let batches: Vec<RecordBatch> = t
                .query()
                .full_text_search(FullTextSearchQuery::new(query.to_string()))
                .select(Select::columns(&["domain", "role", "text"]))
                .limit(limit)
                .execute()
                .await?
                .try_collect()
                .await?;
            let mut hits = Vec::new();
            for b in batches {
                let (d, r, x, s) = (
                    col_str(&b, "domain")?,
                    col_str(&b, "role")?,
                    col_str(&b, "text")?,
                    col_f32(&b, "_score")?,
                );
                for i in 0..b.num_rows() {
                    hits.push(SearchHit {
                        text: x[i].clone(),
                        role: r[i].clone(),
                        domain: d[i].clone(),
                        score: s[i],
                    });
                }
            }
            hits.sort_by(|a, b| b.score.total_cmp(&a.score));
            Ok(hits)
        })
    }

    /// The anchored prompts nearest to `v` (cosine): `(text, cosine similarity)`.
    pub fn nearest_prompts(&self, v: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        self.rt.block_on(async {
            let Some(t) = self.table(ANCHORS).await? else {
                return Ok(Vec::new());
            };
            let batches: Vec<RecordBatch> = t
                .query()
                .nearest_to(v)?
                .distance_type(DistanceType::Cosine)
                .select(Select::columns(&["text"]))
                .limit(k)
                .execute()
                .await?
                .try_collect()
                .await?;
            let mut out = Vec::new();
            for b in batches {
                let (x, d) = (col_str(&b, "text")?, col_f32(&b, "_distance")?);
                out.extend(x.into_iter().zip(d.into_iter().map(|d| 1.0 - d)));
            }
            out.sort_by(|a, b| b.1.total_cmp(&a.1));
            Ok(out)
        })
    }

    /// Every revision still held by Lance, newest first.
    pub fn history(&self) -> Result<Vec<Revision>> {
        self.rt.block_on(async {
            let Some(t) = self.table(META).await? else { return Ok(Vec::new()) };
            let mut seen = BTreeMap::new();
            for v in t.list_versions().await? {
                t.checkout(v.version).await?;
                let batches: Vec<RecordBatch> = t.query().execute().await?.try_collect().await?;
                for b in batches.iter().filter(|b| b.num_rows() > 0) {
                    let rev = col_i64(b, "revision")?[0];
                    let tables: BTreeMap<String, u64> = serde_json::from_str(&col_str(b, "tables")?[0]).unwrap_or_default();
                    seen.insert(
                        rev,
                        Revision {
                            revision: rev,
                            updated_ms: col_i64(b, "updated_ms")?[0],
                            note: col_str(b, "note")?[0].clone(),
                            tables,
                        },
                    );
                }
            }
            t.checkout_latest().await?;
            Ok(seen.into_values().rev().collect())
        })
    }

    /// Restores every table to its state at `revision` (Lance time travel) and records that as a new revision.
    pub fn rollback(&self, revision: i64) -> Result<i64> {
        self.rt.block_on(async {
            // Single reverse pass over META versions finds both the table versions and the
            // prompting/anchor metadata of `revision`, avoiding two full history scans.
            let m = self
                .table(META)
                .await?
                .ok_or_else(|| anyhow!("revision {revision} is not held any more"))?;
            let mut found: Option<(BTreeMap<String, u64>, (String, String, String, String))> = None;
            for v in m.list_versions().await?.into_iter().rev() {
                m.checkout(v.version).await?;
                let batches: Vec<RecordBatch> = m.query().execute().await?.try_collect().await?;
                if let Some(b) = batches
                    .iter()
                    .find(|b| b.num_rows() > 0 && col_i64(b, "revision").map(|r| r[0] == revision).unwrap_or(false))
                {
                    let tables: BTreeMap<String, u64> =
                        serde_json::from_str(&col_str(b, "tables")?[0]).unwrap_or_default();
                    let old = (
                        col_str(b, "meta")?[0].clone(),
                        col_str(b, "prompting")?[0].clone(),
                        col_str(b, "anchor_model")?[0].clone(),
                        col_str(b, "anchor_signature")?[0].clone(),
                    );
                    found = Some((tables, old));
                    break;
                }
            }
            m.checkout_latest().await?;
            let (tables, (meta, prompting, model, sig)) =
                found.ok_or_else(|| anyhow!("revision {revision} is not held any more"))?;
            for (name, version) in &tables {
                let t = self.table_or_create(name).await?;
                if t.version().await? != *version {
                    t.checkout(*version).await?;
                    t.restore().await?;
                }
            }
            if self.table(PHRASES).await?.is_some() {
                self.index_phrases().await?;
            }
            self.commit(
                &format!("rollback to {revision}"),
                Some(&meta),
                Some(&prompting),
                Some((&model, &sig)),
            )
            .await
        })
    }

    /// Merges the small files every edit leaves behind; with `keep`, also drops the versions older than that (they
    /// can no longer be rolled back to). Learning writes often, so a long-running dictionary wants this now and then.
    pub fn compact(&self, keep: Option<std::time::Duration>) -> Result<()> {
        use lancedb::table::{CompactionOptions, OptimizeAction};
        self.rt.block_on(async {
            for name in TABLES.iter().copied().chain(std::iter::once(META)) {
                let Some(t) = self.table(name).await? else { continue };
                t.optimize(OptimizeAction::Compact {
                    options: CompactionOptions::default(),
                    remap_options: None,
                })
                .await?;
                if let Some(k) = keep {
                    let older_than = lancedb::table::optimize::Duration::from_std(k).map_err(|e| anyhow!("{e}"))?;
                    t.optimize(OptimizeAction::Prune {
                        older_than: Some(older_than),
                        delete_unverified: None,
                        error_if_tagged_old_versions: None,
                    })
                    .await?;
                }
            }
            if self.table(PHRASES).await?.is_some() {
                self.index_phrases().await?;
            }
            Ok(())
        })
    }

    /// One line about the store.
    pub fn stats(&self) -> Result<String> {
        let lex = self.load()?;
        let anchors = self.anchors()?;
        let candidates = self.candidates()?.len();
        let phrases = self.rt.block_on(async {
            Ok::<_, anyhow::Error>(match self.table(PHRASES).await? {
                Some(t) => t.count_rows(None).await?,
                None => 0,
            })
        })?;
        Ok(format!(
            "revision {} · {} domains · {} phrase rows · {} vision nodes · {} prompts ({} anchored{}) · {} candidates · LanceDB {}",
            lex.version,
            lex.nodes.len(),
            phrases,
            lex.vision.len(),
            lex.anchor_phrases().len(),
            anchors.vectors.len(),
            if anchors.model.is_empty() {
                String::new()
            } else {
                format!(" by {}", anchors.model)
            },
            candidates,
            self.path.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(name: &str) -> (LexiconStore, PathBuf) {
        let dir = std::env::temp_dir().join(format!("zena-lexicon-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        (LexiconStore::open(&dir).unwrap(), dir)
    }

    #[test]
    fn import_load_round_trips_and_edits_are_revisions() {
        let (s, dir) = store("roundtrip");
        assert_eq!(s.revision().unwrap(), 0);
        let lex = Lexicon::bundled();
        let rep = s.import(&lex).unwrap();
        assert_eq!(rep.domains, lex.nodes.len());
        assert_eq!(rep.prompts, lex.anchor_phrases().len());
        let back = s.load().unwrap();
        assert_eq!(back.nodes, lex.nodes);
        assert_eq!(back.vision, lex.vision);
        assert_eq!(back.prompting, lex.prompting);
        let r1 = s.revision().unwrap();
        assert!(s.add_phrase("emotion.celebrate", Role::Bias, "개꿀이다", "user").unwrap());
        assert!(
            !s.add_phrase("emotion.celebrate", Role::Bias, "개꿀이다", "user").unwrap(),
            "already there"
        );
        assert!(s.revision().unwrap() > r1, "an edit is a new revision");
        assert!(s
            .load()
            .unwrap()
            .node("emotion.celebrate")
            .unwrap()
            .bias
            .iter()
            .any(|p| p.text == "개꿀이다"));
        // Two nodes may not claim one exact word: the edit is refused and nothing changes.
        let before = s.revision().unwrap();
        assert!(s.add_phrase("emotion.thanks", Role::Exact, "대박", "user").is_err());
        assert_eq!(s.revision().unwrap(), before);
        assert!(s.add_phrase("emotion.thanks", Role::Line, "별말씀을!", "user").unwrap());
        // Re-seeding from the JSON keeps the user's phrase (merged back into its domain).
        let rep = s.import(&lex).unwrap();
        assert_eq!(rep.merged, 1, "{rep:?}");
        assert!(s
            .load()
            .unwrap()
            .node("emotion.celebrate")
            .unwrap()
            .bias
            .iter()
            .any(|p| p.text == "개꿀이다"));
        assert!(s.remove_phrase("emotion.celebrate", Role::Bias, "개꿀이다").unwrap());
        assert!(!s
            .load()
            .unwrap()
            .node("emotion.celebrate")
            .unwrap()
            .bias
            .iter()
            .any(|p| p.text == "개꿀이다"));
        // Full-text search over n-grams finds a Korean word inside "인벤토리에서 물약 어디 있어" and a misspelt prompt.
        let hits = s.search("물약", 10).unwrap();
        assert!(hits.iter().any(|h| h.domain == "coach.inventory"), "{hits:?}");
        let hits = s.search("candlestik", 10).unwrap();
        assert!(
            hits.iter()
                .any(|h| h.role.starts_with("vision") && h.domain == "vision.stock_chart"),
            "{hits:?}"
        );
        drop(s);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn anchors_learning_and_rollback() {
        let (s, dir) = store("anchors");
        let lex = Lexicon::bundled();
        s.import(&lex).unwrap();
        let missing = s.missing_anchors().unwrap();
        assert_eq!(missing.len(), lex.anchor_phrases().len());
        let unit = |seed: usize| {
            let v: Vec<f32> = (0..ANCHOR_DIMS).map(|i| ((i * 7 + seed * 13) as f32 * 0.37).sin()).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / n).collect::<Vec<f32>>()
        };
        let anchors: Vec<(String, Vec<f32>)> = missing.iter().enumerate().map(|(i, t)| (t.clone(), unit(i))).collect();
        assert_eq!(s.set_anchors("test-model", &anchors).unwrap(), anchors.len());
        let a = s.anchors().unwrap();
        assert_eq!(a.model, "test-model");
        assert_eq!(a.vectors.len(), anchors.len());
        let (t0, v0) = &anchors[3];
        let cos: f32 = a.vectors[t0].iter().zip(v0).map(|(x, y)| x * y).sum();
        assert!(cos > 0.9999, "{cos}");
        let near = s.nearest_prompts(v0, 3).unwrap();
        assert_eq!(&near[0].0, t0);
        assert!(near[0].1 > 0.999);
        // Re-importing the same dictionary keeps every anchor.
        assert_eq!(s.import(&lex).unwrap().anchors_kept, anchors.len());
        assert!(s.missing_anchors().unwrap().is_empty());
        // A phrase seen twice for a domain becomes one of its bias phrases.
        let before_learning = s.revision().unwrap();
        assert_eq!(s.learn("emotion.celebrate", "드디어 깼다", 2).unwrap(), Learned::Counted(1));
        assert_eq!(s.candidates().unwrap(), vec![("emotion.celebrate".into(), "드디어 깼다".into(), 1)]);
        assert_eq!(s.learn("emotion.celebrate", "드디어 깼다", 2).unwrap(), Learned::Promoted);
        assert_eq!(s.learn("emotion.celebrate", "드디어 깼다", 2).unwrap(), Learned::Known);
        assert!(s
            .load()
            .unwrap()
            .node("emotion.celebrate")
            .unwrap()
            .bias
            .iter()
            .any(|p| p.text == "드디어 깼다"));
        // Lance keeps every version: roll the dictionary back to before the learning.
        let hist = s.history().unwrap();
        assert!(hist.iter().any(|r| r.revision == before_learning), "{hist:?}");
        let r = s.rollback(before_learning).unwrap();
        assert!(r > before_learning);
        let back = s.load().unwrap();
        assert!(!back.node("emotion.celebrate").unwrap().bias.iter().any(|p| p.text == "드디어 깼다"));
        assert!(s.candidates().unwrap().is_empty() || s.candidates().unwrap().iter().all(|c| c.1 != "드디어 깼다"));
        assert_eq!(
            s.anchors().unwrap().vectors.len(),
            anchors.len(),
            "anchors of that revision are back too"
        );
        assert!(s.stats().unwrap().contains("anchored by test-model"));
        s.compact(None).unwrap();
        assert_eq!(s.load().unwrap().nodes, back.nodes, "compaction changes files, not contents");
        drop(s);
        let _ = std::fs::remove_dir_all(dir);
    }
}
