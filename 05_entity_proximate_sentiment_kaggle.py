# V2_30_ENTITY_PROXIMATE_SENTIMENT_KAGGLE.py
# Reference-paper-aligned entity-proximate sentence sentiment.
# Uses the same j-hartmann RoBERTa-Large sentiment model as V2 article sentiment.

import os, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "j-hartmann/sentiment-roberta-large-english-3-classes"
OUT = Path("/kaggle/working/V2_ENTITY_PROXIMATE_SENTIMENT")
OUT.mkdir(parents=True, exist_ok=True)
SENT_OUT = OUT/"entity_sentence_sentiment.csv"
SUMMARY_OUT = OUT/"entity_sentiment_by_media_house.csv"
CHECK = OUT/"checkpoint.csv"

# Final corpus
corpus_candidates = [
 "/kaggle/input/datasets/adityanegi517/media-05/final_clean_cjp_protest_corpus.csv",
 "/kaggle/input/datasets/adityanegi517/media-04/final_clean_cjp_protest_corpus.csv",
]
CORPUS = next((x for x in corpus_candidates if os.path.exists(x)), None)
if CORPUS is None: raise FileNotFoundError("Final corpus not found.")

# Canonical entity file must be uploaded to Kaggle.
import glob
entity_files = glob.glob("/kaggle/input/**/*entity_mentions_canonicalized.csv", recursive=True)
if not entity_files:
    raise FileNotFoundError("Upload entity_mentions_canonicalized.csv to Kaggle input first.")
ENTITY_FILE = entity_files[0]

df = pd.read_csv(CORPUS)
ents = pd.read_csv(ENTITY_FILE)
print("Corpus:", CORPUS, len(df))
print("Entities:", ENTITY_FILE, len(ents))
print("Model:", MODEL, "CUDA:", torch.cuda.is_available())

def pick(names):
    for n in names:
        if n in df.columns: return n
body_col=pick(["article_text","body","text","content"])
source_col=pick(["source","media_house","outlet"])
url_col=pick(["article_url","url"])
if body_col is None or source_col is None: raise ValueError(list(df.columns))

# Focus on PERSON entities, as in the paper's named-figure sentiment analysis.
persons=ents[ents["entity_type"].astype(str).str.upper()=="PERSON"].copy()
# Top canonical persons corpus-wide; enough for stable cross-outlet comparison.
top=(persons.groupby("canonical_entity")["mentions_in_article"].sum()
     .sort_values(ascending=False).head(20).index.tolist())
persons=persons[persons["canonical_entity"].isin(top)]
print("Top persons:", top)

# Aliases per canonical entity from NER surface forms.
alias_map={}
for canon,g in persons.groupby("canonical_entity"):
    aliases=set(str(x).strip() for x in g["entity"].dropna())
    aliases.add(str(canon))
    alias_map[canon]=sorted(aliases,key=len,reverse=True)

tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForSequenceClassification.from_pretrained(MODEL)
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(dev).eval()
labs={int(k):str(v).lower() for k,v in model.config.id2label.items()}
pos={}
for i,l in labs.items():
    if "neg" in l: pos["negative"]=i
    elif "neu" in l: pos["neutral"]=i
    elif "pos" in l: pos["positive"]=i

# Sentence splitter without requiring spaCy runtime.
sentence_re=re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“‘])')

def split_sentences(text):
    return [s.strip() for s in sentence_re.split(str(text)) if len(s.strip())>=10]

def score_batch(texts,batch=32):
    rows=[]
    for i in range(0,len(texts),batch):
        enc=tok(texts[i:i+batch],padding=True,truncation=True,max_length=512,return_tensors="pt")
        enc={k:v.to(dev) for k,v in enc.items()}
        with torch.inference_mode():
            probs=torch.softmax(model(**enc).logits,-1).cpu().numpy()
        for p in probs:
            n=float(p[pos["negative"]]);u=float(p[pos["neutral"]]);q=float(p[pos["positive"]])
            d={"negative":n,"neutral":u,"positive":q}
            rows.append((n,u,q,q-n,max(d,key=d.get)))
    return rows

# Resume article IDs
done=set()
records=[]
if CHECK.exists():
    ck=pd.read_csv(CHECK);done=set(ck["row_id"].astype(int))
if SENT_OUT.exists():
    records=pd.read_csv(SENT_OUT).to_dict("records")
print("Resume articles:",len(done))

for idx,row in df.iterrows():
    rid=idx+1
    if rid in done: continue
    text="" if pd.isna(row[body_col]) else str(row[body_col])
    sents=split_sentences(text)
    source=str(row[source_col])
    url="" if url_col is None or pd.isna(row[url_col]) else str(row[url_col])

    targets=persons[persons["row_id"]==rid]["canonical_entity"].dropna().unique().tolist()
    pending=[]
    meta=[]
    for canon in targets:
        aliases=alias_map.get(canon,[canon])
        for si,s in enumerate(sents):
            low=s.lower()
            matched=next((a for a in aliases if a.lower() in low),None)
            if matched:
                pending.append(s)
                meta.append((canon,matched,si))

    if pending:
        scores=score_batch(pending)
        for (canon,alias,si),sent,(n,u,p,c,l) in zip(meta,pending,scores):
            records.append({
                "row_id":rid,"source":source,"article_url":url,
                "canonical_entity":canon,"matched_alias":alias,
                "sentence_index":si,"sentence":sent,
                "negative_prob":n,"neutral_prob":u,"positive_prob":p,
                "compound":c,"sentiment_label":l
            })

    done.add(rid)
    if rid%25==0 or rid==len(df):
        pd.DataFrame(records).to_csv(SENT_OUT,index=False)
        pd.DataFrame({"row_id":sorted(done)}).to_csv(CHECK,index=False)
        print(f"{rid}/{len(df)} articles processed; sentence rows {len(records)}")

res=pd.DataFrame(records)
if res.empty: raise RuntimeError("No entity sentences extracted.")

# Aggregate per entity x outlet.
rows=[]
for (source,entity),g in res.groupby(["source","canonical_entity"]):
    q=g["sentiment_label"].value_counts(normalize=True)*100
    rows.append({
        "source":source,"canonical_entity":entity,
        "sentence_instances":len(g),
        "articles":g["row_id"].nunique(),
        "negative_pct":q.get("negative",0),
        "neutral_pct":q.get("neutral",0),
        "positive_pct":q.get("positive",0),
        "mean_entity_sentence_compound":g["compound"].mean()
    })
summary=pd.DataFrame(rows).sort_values(["canonical_entity","sentence_instances"],ascending=[True,False])
summary.to_csv(SUMMARY_OUT,index=False)

print("\nENTITY-PROXIMATE SENTIMENT COMPLETE")
print("Sentence instances:",len(res))
print("Entities:",res["canonical_entity"].nunique())
print("\nTop corpus-wide entity sentiment:")
overall=(res.groupby("canonical_entity")
         .agg(sentence_instances=("sentence","size"),articles=("row_id","nunique"),
              mean_compound=("compound","mean"))
         .sort_values("sentence_instances",ascending=False))
print(overall.head(20).round(3).to_string())
print("\nSaved:",SENT_OUT)
print("Saved:",SUMMARY_OUT)
print("IMPORTANT: Download outputs before ending Kaggle session.")
