from pathlib import Path
import re, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sktext

R=Path.cwd(); I=R/"data"/"research"/"cjp"/"final_clean"/"final_clean_cjp_protest_corpus.csv"
O=R/"V2_RESEARCH"/"07_TFIDF_LEXICAL_PROFILING"/"FINAL_CLEAN"; O.mkdir(parents=True,exist_ok=True)
if not I.exists(): raise SystemExit(f"MISSING: {I}")
d=pd.read_csv(I)
tc=next((x for x in ["article_text","body","text","content"] if x in d.columns),None)
sc=next((x for x in ["source","media_house","outlet"] if x in d.columns),None)
if not tc or not sc: raise SystemExit("Required columns missing")

# Analysis-only exclusions: standard English stopwords + publisher/web/social boilerplate.
EXTRA={
"read","follow","latest","stay","post","posts","posted","view","views","shared","share","video","videos",
"twitter","instagram","whatsapp","telegram","pic","com","app","apps","download","click","updates","update",
"published","publish","headline","picture","pictures","feed","feeds","reworked","staff","content","syndicated",
"auto","generated","important","news","live","wrote","website","web","channel","channels","photo","photos",
"getty","images","image","copyright","subscribe","subscription","newsletter","report","reports",
"bbc","cnn","ht","tnie","express","hindustan","standard","business",
"ist","pm","am","jul","june","july","aug","august","sep","september",
"pti","ani","inputs","input"
}
STOP=set(sktext.ENGLISH_STOP_WORDS)|EXTRA

# Remove URLs/handles and normalize whitespace only. Original corpus remains unchanged.
def clean(x):
    s="" if pd.isna(x) else str(x)
    s=re.sub(r'https?://\S+|www\.\S+|@\w+'," ",s,flags=re.I)
    s=re.sub(r'\b\d{1,2}:\d{2}\s*(?:am|pm|ist)?\b'," ",s,flags=re.I)
    s=re.sub(r'\s+'," ",s).strip()
    return s

texts=d[tc].map(clean); sources=d[sc].astype(str)
v=TfidfVectorizer(lowercase=True,stop_words=list(STOP),min_df=2,max_df=.90,ngram_range=(1,2),
                  sublinear_tf=True,norm="l2",max_features=60000,
                  token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]{1,}\b")
X=v.fit_transform(texts); terms=np.asarray(v.get_feature_names_out())

rows=[]
for src in sorted(sources.unique()):
    own=(sources==src).to_numpy(); other=~own
    a=np.asarray(X[own].mean(axis=0)).ravel()
    b=np.asarray(X[other].mean(axis=0)).ravel()
    diff=a-b
    rank=0
    for j in np.argsort(diff)[::-1]:
        if diff[j] <= 0: continue
        rank+=1
        rows.append({"source":src,"term":terms[j],"outlet_mean_tfidf":a[j],
                     "other_outlets_mean_tfidf":b[j],"distinctiveness":diff[j],
                     "rank_within_source":rank})
        if rank>=50: break

res=pd.DataFrame(rows)
res.to_csv(O/"distinctive_terms_top50_FINAL.csv",index=False)

G=O/"graphs";G.mkdir(exist_ok=True)
for src,g in res.groupby("source"):
    q=g.head(15).sort_values("distinctiveness")
    fig,ax=plt.subplots(figsize=(9,6))
    ax.barh(q["term"],q["distinctiveness"])
    ax.set_xlabel("TF-IDF distinctiveness vs other outlets")
    ax.set_title(f"Characteristic CJP vocabulary — {src}")
    fig.tight_layout()
    safe=re.sub(r'\W+','_',src).strip('_')
    fig.savefig(G/f"{safe}_distinctive_terms_FINAL.png",dpi=220)
    plt.close(fig)

# Compact comparison table: top 10 terms per outlet.
wide=(res[res.rank_within_source<=10]
      .pivot(index="rank_within_source",columns="source",values="term")
      .reset_index())
wide.to_csv(O/"top10_terms_cross_outlet.csv",index=False)

print("="*72);print("FINAL CLEAN TF-IDF COMPLETE")
print("Articles:",len(d),"Features:",len(terms))
for src,g in res.groupby("source"):
    print("\n",src)
    print(g[["term","distinctiveness"]].head(10).round(5).to_string(index=False))
print("\nSaved:",O)
print("Original corpus NOT modified.")
print("Interpretation: characteristic vocabulary only; NOT a direct bias score.")
