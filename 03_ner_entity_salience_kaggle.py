# V2_20_NER_ENTITY_SALIENCE_KAGGLE.py
# Reference-paper-aligned NER/entity salience analysis using spaCy transformer NER.

import os, json, re
from pathlib import Path
import pandas as pd
import spacy

OUT = Path("/kaggle/working/V2_NER_ENTITY_SALIENCE")
OUT.mkdir(parents=True, exist_ok=True)
MENTIONS = OUT / "entity_mentions.csv"
ARTICLE_SUMMARY = OUT / "article_entity_summary.csv"
ENTITY_BY_SOURCE = OUT / "entity_salience_by_media_house.csv"
TOP_PERSONS = OUT / "top_persons_by_media_house.csv"
CHECKPOINT = OUT / "ner_checkpoint.csv"

candidates = [
    "/kaggle/input/datasets/adityanegi517/media-05/final_clean_cjp_protest_corpus.csv",
    "/kaggle/input/datasets/adityanegi517/media-04/final_clean_cjp_protest_corpus.csv",
]
INPUT = next((p for p in candidates if os.path.exists(p)), None)
if INPUT is None:
    raise FileNotFoundError("Final 3,857 corpus not found.")

df = pd.read_csv(INPUT)
print("="*72)
print("V2 NER + ENTITY SALIENCE")
print("="*72)
print("Input:", INPUT)
print("Articles:", len(df))

def pick(names):
    for n in names:
        if n in df.columns:
            return n
    return None

body_col = pick(["article_text","body","text","content"])
headline_col = pick(["headline","title"])
source_col = pick(["source","media_house","outlet"])
url_col = pick(["article_url","url"])
date_col = pick(["publication_day","publication_date","date"])

if body_col is None or source_col is None:
    raise ValueError(f"Required columns missing: {list(df.columns)}")

print("Loading spaCy transformer model: en_core_web_trf")
nlp = spacy.load("en_core_web_trf")
nlp.max_length = 2_000_000

# Disable components not needed for NER; transformer + ner remain active.
keep = {"transformer","ner"}
disable = [p for p in nlp.pipe_names if p not in keep]
print("Pipeline:", nlp.pipe_names, "| Disabled:", disable)

def norm_entity(text):
    x = re.sub(r"\s+", " ", str(text)).strip()
    x = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", x)
    return x

# Process in batches; checkpoint stores article completion.
completed = set()
if CHECKPOINT.exists():
    c = pd.read_csv(CHECKPOINT)
    completed = set(c["row_id"].astype(int))
    print("Resume checkpoint:", len(completed))

todo = []
meta = []
for idx, row in df.iterrows():
    rid = idx + 1
    if rid in completed:
        continue
    body = "" if pd.isna(row[body_col]) else str(row[body_col])
    todo.append(body)
    meta.append((rid, row))

mention_rows = []
article_rows = []

# If resuming, retain previous outputs.
if MENTIONS.exists():
    mention_rows = pd.read_csv(MENTIONS).to_dict("records")
if ARTICLE_SUMMARY.exists():
    article_rows = pd.read_csv(ARTICLE_SUMMARY).to_dict("records")

processed_now = 0
for doc, (rid, row) in zip(
    nlp.pipe(todo, batch_size=8, disable=disable),
    meta
):
    source = "" if pd.isna(row[source_col]) else str(row[source_col])
    headline = "" if headline_col is None or pd.isna(row[headline_col]) else str(row[headline_col])
    url = "" if url_col is None or pd.isna(row[url_col]) else str(row[url_col])
    date = "" if date_col is None or pd.isna(row[date_col]) else str(row[date_col])

    counts = {}
    for ent in doc.ents:
        if ent.label_ not in {"PERSON","ORG","GPE","NORP","FAC"}:
            continue
        txt = norm_entity(ent.text)
        if len(txt) < 2:
            continue
        key = (txt, ent.label_)
        counts[key] = counts.get(key, 0) + 1

    for (entity, label), count in counts.items():
        mention_rows.append({
            "row_id": rid,
            "source": source,
            "date": date,
            "headline": headline,
            "article_url": url,
            "entity": entity,
            "entity_type": label,
            "mentions_in_article": count,
        })

    article_rows.append({
        "row_id": rid,
        "source": source,
        "date": date,
        "headline": headline,
        "article_url": url,
        "unique_entities": len(counts),
        "total_entity_mentions": sum(counts.values()),
    })

    completed.add(rid)
    processed_now += 1

    if processed_now % 25 == 0 or len(completed) == len(df):
        pd.DataFrame(mention_rows).to_csv(MENTIONS, index=False)
        pd.DataFrame(article_rows).to_csv(ARTICLE_SUMMARY, index=False)
        pd.DataFrame({"row_id": sorted(completed)}).to_csv(CHECKPOINT, index=False)
        print(f"{len(completed)}/{len(df)} processed; outputs saved")

mentions = pd.DataFrame(mention_rows)
articles = pd.DataFrame(article_rows)

# Case-insensitive canonicalization within exact surface forms.
mentions["entity_key"] = mentions["entity"].str.lower().str.strip()

# Aggregate by outlet/entity/type.
agg = (
    mentions.groupby(["source","entity_key","entity_type"], as_index=False)
    .agg(
        entity=("entity","first"),
        total_mentions=("mentions_in_article","sum"),
        articles_mentioning=("row_id","nunique"),
    )
)

source_article_counts = df.groupby(source_col).size().to_dict()
agg["source_articles"] = agg["source"].map(source_article_counts)
agg["mentions_per_article"] = agg["total_mentions"] / agg["source_articles"]
agg["article_presence_pct"] = 100 * agg["articles_mentioning"] / agg["source_articles"]
agg = agg.sort_values(["source","total_mentions"], ascending=[True,False])
agg.to_csv(ENTITY_BY_SOURCE, index=False)

persons = agg[agg["entity_type"]=="PERSON"].copy()
persons["rank_within_source"] = persons.groupby("source")["total_mentions"].rank(
    method="first", ascending=False
)
top_persons = persons[persons["rank_within_source"] <= 30].sort_values(
    ["source","rank_within_source"]
)
top_persons.to_csv(TOP_PERSONS, index=False)

print("\nTOP 10 PERSONS PER MEDIA HOUSE")
for source, g in top_persons.groupby("source"):
    print("\n", source)
    print(g[["entity","total_mentions","articles_mentioning","mentions_per_article"]].head(10).to_string(index=False))

print("\nSaved:")
print(MENTIONS)
print(ARTICLE_SUMMARY)
print(ENTITY_BY_SOURCE)
print(TOP_PERSONS)
print("\nIMPORTANT: Download outputs or Save Version before ending the Kaggle session.")
