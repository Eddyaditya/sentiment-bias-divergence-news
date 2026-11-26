import os,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
MODEL="j-hartmann/sentiment-roberta-large-english-3-classes"
OUT=Path("/kaggle/working/V2_SENTIMENT_PAPER_METHOD");OUT.mkdir(parents=True,exist_ok=True)
FINAL=OUT/"article_level_sentiment.csv";CHECK=OUT/"sentiment_checkpoint.csv";BY=OUT/"sentiment_by_media_house.csv"
cands=["/kaggle/input/datasets/adityanegi517/media-05/final_clean_cjp_protest_corpus.csv","/kaggle/input/datasets/adityanegi517/media-04/final_clean_cjp_protest_corpus.csv"]
INPUT=next((x for x in cands if os.path.exists(x)),None)
if not INPUT: raise FileNotFoundError("3,857 corpus not found.")
df=pd.read_csv(INPUT);print("Articles:",len(df),"Model:",MODEL,"CUDA:",torch.cuda.is_available())
def pick(ns):
 for n in ns:
  if n in df.columns:return n
body=pick(["article_text","body","text","content"]);head=pick(["headline","title"]);source=pick(["source","media_house","outlet"]);url=pick(["article_url","url"])
if not body or not source:raise ValueError(list(df.columns))
tok=AutoTokenizer.from_pretrained(MODEL);model=AutoModelForSequenceClassification.from_pretrained(MODEL)
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");model.to(dev).eval()
labs={int(k):str(v).lower() for k,v in model.config.id2label.items()};pos={}
for i,l in labs.items():
 if "neg" in l:pos["negative"]=i
 elif "neu" in l:pos["neutral"]=i
 elif "pos" in l:pos["positive"]=i
WIN,OVER=400,50
def score(txt):
 ids=tok.encode(str(txt),add_special_tokens=False); chunks=[]
 for s in range(0,max(1,len(ids)),WIN-OVER):
  chunks.append(ids[s:s+WIN])
  if s+WIN>=len(ids):break
 ps=[]
 for s in range(0,len(chunks),16):
  e=tok.pad({"input_ids":[tok.build_inputs_with_special_tokens(x) for x in chunks[s:s+16]]},padding=True,return_tensors="pt")
  e={k:v.to(dev) for k,v in e.items()}
  with torch.inference_mode(): ps.extend(torch.softmax(model(**e).logits,-1).cpu().numpy())
 m=np.asarray(ps).mean(0);n=float(m[pos["negative"]]);u=float(m[pos["neutral"]]);p=float(m[pos["positive"]]);d={"negative":n,"neutral":u,"positive":p}
 return n,u,p,p-n,max(d,key=d.get),len(chunks)
done={}
if CHECK.exists():
 for _,r in pd.read_csv(CHECK).iterrows():done[int(r.row_id)]=r.to_dict()
rec=list(done.values())
for i,r in df.iterrows():
 rid=i+1
 if rid in done:continue
 n,u,p,c,l,k=score("" if pd.isna(r[body]) else r[body])
 h="" if head is None or pd.isna(r[head]) else str(r[head]);hn,hu,hp,hc,hl,_=score(h)
 rec.append({"row_id":rid,"source":str(r[source]),"headline":h,"article_url":"" if url is None or pd.isna(r[url]) else str(r[url]),"body_negative_prob":n,"body_neutral_prob":u,"body_positive_prob":p,"body_compound":c,"body_sentiment_label":l,"body_chunks":k,"headline_negative_prob":hn,"headline_neutral_prob":hu,"headline_positive_prob":hp,"headline_compound":hc,"headline_sentiment_label":hl,"title_content_compound_gap":hc-c})
 if rid%25==0 or rid==len(df):
  pd.DataFrame(rec).sort_values("row_id").to_csv(CHECK,index=False);print(f"{rid}/{len(df)} processed")
res=pd.DataFrame(rec).sort_values("row_id");res.to_csv(FINAL,index=False)
rows=[]
for s,g in res.groupby("source"):
 q=g.body_sentiment_label.value_counts(normalize=True)*100
 rows.append({"source":s,"articles":len(g),"negative_pct":q.get("negative",0),"neutral_pct":q.get("neutral",0),"positive_pct":q.get("positive",0),"mean_body_compound":g.body_compound.mean(),"mean_headline_compound":g.headline_compound.mean(),"mean_title_content_gap":g.title_content_compound_gap.mean()})
by=pd.DataFrame(rows).sort_values("articles",ascending=False);by.to_csv(BY,index=False)
print(by.round(3).to_string(index=False));print("FINAL:",FINAL);print("BY SOURCE:",BY)
print("Download outputs before ending the Kaggle session.")
