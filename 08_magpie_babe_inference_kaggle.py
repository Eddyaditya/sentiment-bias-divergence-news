import os,re,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

MODEL="mediabiasgroup/magpie-babe-ft"
OUT=Path("/kaggle/working/V2_MAGPIE_BABE");OUT.mkdir(parents=True,exist_ok=True)
SENT=OUT/"sentence_level_magpie.csv";ART=OUT/"article_level_magpie.csv";SRC=OUT/"magpie_by_media_house.csv";CK=OUT/"checkpoint.csv"
CANDS=["/kaggle/input/datasets/adityanegi517/media-05/final_clean_cjp_protest_corpus.csv","/kaggle/input/datasets/adityanegi517/media-04/final_clean_cjp_protest_corpus.csv"]
INPUT=next((x for x in CANDS if os.path.exists(x)),None)
if not INPUT:raise FileNotFoundError("Final 3,857 corpus not found")
d=pd.read_csv(INPUT)
def pick(ns):
 for n in ns:
  if n in d.columns:return n
tc=pick(["article_text","body","text","content"]);sc=pick(["source","media_house","outlet"]);uc=pick(["article_url","url"]);hc=pick(["headline","title"])
if not tc or not sc:raise ValueError(list(d.columns))
print("Model:",MODEL,"Articles:",len(d),"CUDA:",torch.cuda.is_available())

tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForSequenceClassification.from_pretrained(MODEL)
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");m.to(dev).eval()
print("Device:",dev,"Labels:",m.config.id2label)
labs={int(k):str(v).lower() for k,v in m.config.id2label.items()}
# infer biased class; fallback class 1 for binary BABE checkpoints
bias_idx=next((i for i,l in labs.items() if any(x in l for x in ["bias","biased"])),1)
print("Bias class index:",bias_idx,labs.get(bias_idx))

splitter=re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“‘])')
def sentences(x):
 return [s.strip() for s in splitter.split(str(x)) if 8<=len(s.strip().split())<=250]
def score(xs,bs=64):
 out=[]
 for i in range(0,len(xs),bs):
  e=tok(xs[i:i+bs],padding=True,truncation=True,max_length=512,return_tensors="pt")
  e={k:v.to(dev) for k,v in e.items()}
  with torch.inference_mode():p=torch.softmax(m(**e).logits,-1).cpu().numpy()
  for z in p:
   bp=float(z[bias_idx]);out.append((bp,int(np.argmax(z)==bias_idx)))
 return out

done=set();rows=[]
if CK.exists():done=set(pd.read_csv(CK).row_id.astype(int))
if SENT.exists():rows=pd.read_csv(SENT).to_dict("records")
print("Resume:",len(done))
for i,r in d.iterrows():
 rid=i+1
 if rid in done:continue
 ss=sentences("" if pd.isna(r[tc]) else r[tc])
 if ss:
  scores=score(ss)
  for j,(s,(bp,bl)) in enumerate(zip(ss,scores)):
   rows.append({"row_id":rid,"source":str(r[sc]),"headline":"" if hc is None or pd.isna(r[hc]) else str(r[hc]),
    "article_url":"" if uc is None or pd.isna(r[uc]) else str(r[uc]),"sentence_index":j,"sentence":s,
    "bias_probability":bp,"biased_label":bl})
 done.add(rid)
 if rid%25==0 or rid==len(d):
  pd.DataFrame(rows).to_csv(SENT,index=False);pd.DataFrame({"row_id":sorted(done)}).to_csv(CK,index=False)
  print(f"{rid}/{len(d)} articles; sentence rows {len(rows)}")
s=pd.DataFrame(rows)
a=(s.groupby(["row_id","source"],as_index=False)
   .agg(sentences=("sentence","size"),biased_sentences=("biased_label","sum"),mean_bias_probability=("bias_probability","mean")))
a["biased_sentence_pct"]=100*a.biased_sentences/a.sentences
a.to_csv(ART,index=False)
src=(a.groupby("source",as_index=False)
     .agg(articles=("row_id","nunique"),total_sentences=("sentences","sum"),biased_sentences=("biased_sentences","sum"),
          mean_article_bias_probability=("mean_bias_probability","mean"),mean_article_biased_sentence_pct=("biased_sentence_pct","mean")))
src["pooled_biased_sentence_pct"]=100*src.biased_sentences/src.total_sentences
src=src.sort_values("articles",ascending=False);src.to_csv(SRC,index=False)
print("\nMAGPIE/BABE BY MEDIA HOUSE")
print(src.round(3).to_string(index=False))
print("\nSaved:",SENT);print("Saved:",ART);print("Saved:",SRC)
print("IMPORTANT: Download outputs before ending session. Interpret as sentence-level linguistic/media-bias signal, not universal outlet bias.")
