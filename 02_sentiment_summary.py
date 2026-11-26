import sys, importlib.util, json
from pathlib import Path

for pkg in ["torch","transformers"]:
    if importlib.util.find_spec(pkg) is None:
        print("Missing:",pkg)
        print("Install with: pip install torch transformers")
        sys.exit(1)

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

INPUT = Path("data/research/cjp/final_clean/final_clean_cjp_protest_corpus.csv")
OUT = Path("analysis/cjp_protest/FINAL/02_sentiment")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
MAX_LEN = 512
BODY_CHUNKS = 4
BATCH_SIZE = 16
LABELS = ["negative","neutral","positive"]

d = pd.read_csv(INPUT, low_memory=False)
d["headline"] = d["headline"].fillna("").astype(str)
d["article_text"] = d["article_text"].fillna("").astype(str)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*76)
print("FINAL ANALYSIS 02 - SENTIMENT")
print("="*76)
print("Articles:",len(d))
print("Model:",MODEL_NAME)
print("Device:",device)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

def predict(texts):
    out=[]
    with torch.no_grad():
        for i in range(0,len(texts),BATCH_SIZE):
            b=texts[i:i+BATCH_SIZE]
            enc=tok(b,padding=True,truncation=True,max_length=MAX_LEN,return_tensors="pt").to(device)
            out.append(torch.softmax(model(**enc).logits,dim=-1).cpu().numpy())
    return np.vstack(out)

print("\nScoring headlines...")
hp=predict(d["headline"].tolist())
for j,l in enumerate(LABELS): d[f"headline_{l}_prob"]=hp[:,j]
d["headline_sentiment"]=[LABELS[i] for i in hp.argmax(1)]
d["headline_sentiment_confidence"]=hp.max(1)

print("Preparing body chunks...")
texts=[]; owners=[]
for i,text in enumerate(d["article_text"]):
    ids=tok(text,add_special_tokens=False,truncation=False,verbose=False)["input_ids"]
    payload=MAX_LEN-2
    chunks=[ids[j:j+payload] for j in range(0,len(ids),payload)]
    if len(chunks)>BODY_CHUNKS:
        pos=np.linspace(0,len(chunks)-1,BODY_CHUNKS).round().astype(int)
        chunks=[chunks[p] for p in pos]
    for c in chunks:
        texts.append(tok.decode(c,skip_special_tokens=True)); owners.append(i)

print("Body chunks:",len(texts))
print("Scoring body chunks...")
cp=predict(texts)
sm=np.zeros((len(d),3)); n=np.zeros(len(d),dtype=int)
for o,p in zip(owners,cp): sm[o]+=p; n[o]+=1
bp=sm/n[:,None]

for j,l in enumerate(LABELS): d[f"body_{l}_prob"]=bp[:,j]
d["body_sentiment"]=[LABELS[i] for i in bp.argmax(1)]
d["body_sentiment_confidence"]=bp.max(1)
d["body_chunks_scored"]=n
d["body_valence"]=d["body_positive_prob"]-d["body_negative_prob"]
d["headline_valence"]=d["headline_positive_prob"]-d["headline_negative_prob"]

d.to_csv(OUT/"article_level_sentiment.csv",index=False)

def dist(col):
    c=pd.crosstab(d["source"],d[col])
    for l in LABELS:
        if l not in c.columns: c[l]=0
    return (c[LABELS].div(c[LABELS].sum(axis=1),axis=0)*100).reset_index()

body=dist("body_sentiment"); head=dist("headline_sentiment")
body.to_csv(OUT/"body_sentiment_by_source_pct.csv",index=False)
head.to_csv(OUT/"headline_sentiment_by_source_pct.csv",index=False)

scores=d.groupby("source").agg(
    articles=("article_url","size"),
    mean_body_valence=("body_valence","mean"),
    mean_headline_valence=("headline_valence","mean"),
    mean_body_confidence=("body_sentiment_confidence","mean"),
    mean_headline_confidence=("headline_sentiment_confidence","mean")
).reset_index()
scores.to_csv(OUT/"sentiment_scores_by_source.csv",index=False)

dt=pd.to_datetime(d["publication_date"],format="mixed",utc=True,errors="coerce")
d["_month"]=dt.dt.strftime("%Y-%m")
d.groupby(["_month","source","body_sentiment"]).size().reset_index(name="articles").to_csv(
    OUT/"monthly_body_sentiment_counts.csv",index=False
)

for data,title,name in [
    (body,"Final corpus: body sentiment by media outlet","body_sentiment_by_source.png"),
    (head,"Final corpus: headline sentiment by media outlet","headline_sentiment_by_source.png")
]:
    ax=data.set_index("source")[LABELS].plot(kind="barh",stacked=True,figsize=(11,7))
    ax.set_xlabel("Percent"); ax.set_ylabel("Media outlet"); ax.set_title(title)
    ax.legend(title="Sentiment")
    plt.tight_layout(); plt.savefig(OUT/name,dpi=300,bbox_inches="tight"); plt.close()

s=scores.sort_values("mean_body_valence")
plt.figure(figsize=(10,6)); plt.barh(s["source"],s["mean_body_valence"]); plt.axvline(0,linewidth=1)
plt.xlabel("Mean valence (positive probability - negative probability)")
plt.ylabel("Media outlet"); plt.title("Final corpus: mean body sentiment valence")
plt.tight_layout(); plt.savefig(OUT/"mean_body_valence_by_source.png",dpi=300,bbox_inches="tight"); plt.close()

summary={
    "corpus_version":"final_clean_v1",
    "articles":len(d),
    "model":MODEL_NAME,
    "body_chunks_max":BODY_CHUNKS,
    "methodological_status":"exploratory model-based sentiment; manual validation required before paper-level claims"
}
(OUT/"analysis_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
(OUT/"README.txt").write_text(
"""FINAL ANALYSIS 02 - SENTIMENT
This supersedes the earlier exploratory sentiment run on 4,584 candidates.
Negative sentiment is NOT equivalent to media bias.
The model is Twitter/TweetEval-domain rather than news-specific; manual validation is required before final paper claims.
International outlet samples are small and must be reported with N.
""",encoding="utf-8")

# Reproducible validation sample
parts=[]
for src,g in d.groupby("source"):
    parts.append(g.sample(min(20,len(g)),random_state=20261024))
pd.concat(parts,ignore_index=True).to_csv(OUT/"manual_validation_sample.csv",index=False)

print("\nBODY SENTIMENT % BY SOURCE")
print(body.round(2).to_string(index=False))
print("\nHEADLINE SENTIMENT % BY SOURCE")
print(head.round(2).to_string(index=False))
print("\nSaved:",OUT)
