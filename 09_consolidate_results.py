from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil

R=Path.cwd()
OUT=R/"V2_RESEARCH"/"08_FINAL_RESULTS"
T=OUT/"TABLES"; G=OUT/"GRAPHS"
T.mkdir(parents=True,exist_ok=True); G.mkdir(parents=True,exist_ok=True)

# ---------- locate completed V2 outputs ----------
paths={
"sent":R/"V2_RESEARCH"/"03_SENTIMENT"/"sentiment_by_media_house.csv",
"art_sent":R/"V2_RESEARCH"/"03_SENTIMENT"/"article_level_sentiment.csv",
"sal":R/"V2_RESEARCH"/"04_NER_ENTITY_SALIENCE"/"canonical_person_salience_by_media_house.csv",
"entity":R/"V2_RESEARCH"/"05_ENTITY_SENTIMENT"/"entity_sentiment_by_media_house.csv",
"div":R/"V2_RESEARCH"/"06_ARTICLE_VS_ENTITY_SENTIMENT"/"article_vs_entity_sentiment_overall.csv",
"div_source":R/"V2_RESEARCH"/"06_ARTICLE_VS_ENTITY_SENTIMENT"/"article_vs_entity_sentiment_by_media_house.csv",
"tfidf":R/"V2_RESEARCH"/"07_TFIDF_LEXICAL_PROFILING"/"FINAL_CLEAN"/"distinctive_terms_top50_FINAL.csv",
"corpus":R/"data"/"research"/"cjp"/"final_clean"/"final_clean_cjp_protest_corpus.csv",
}
missing=[str(p) for p in paths.values() if not p.exists()]
if missing:
    raise SystemExit("MISSING FILES:\n"+"\n".join(missing))

sent=pd.read_csv(paths["sent"])
art=pd.read_csv(paths["art_sent"])
sal=pd.read_csv(paths["sal"])
ent=pd.read_csv(paths["entity"])
div=pd.read_csv(paths["div"])
divs=pd.read_csv(paths["div_source"])
tf=pd.read_csv(paths["tfidf"])
corpus=pd.read_csv(paths["corpus"])

# ---------- coverage ----------
date_col=next((c for c in ["publication_day","publication_date","date"] if c in corpus.columns),None)
source_col=next((c for c in ["source","media_house","outlet"] if c in corpus.columns),None)
coverage=corpus.groupby(source_col).size().reset_index(name="articles").rename(columns={source_col:"source"})
coverage["share_pct"]=100*coverage["articles"]/len(corpus)
coverage=coverage.sort_values("articles",ascending=False)
coverage.to_csv(T/"01_coverage_by_media_house.csv",index=False)

if date_col:
    dt=pd.to_datetime(corpus[date_col],errors="coerce")
    monthly=(pd.DataFrame({"source":corpus[source_col],"month":dt.dt.to_period("M").astype(str)})
             .groupby(["month","source"]).size().reset_index(name="articles"))
    monthly.to_csv(T/"02_monthly_coverage.csv",index=False)

# ---------- sentiment tables ----------
sent.to_csv(T/"03_sentiment_by_media_house.csv",index=False)

# ---------- top canonical persons ----------
top_sal=(sal.sort_values(["source","total_mentions"],ascending=[True,False])
         .groupby("source",group_keys=False).head(10))
top_sal.to_csv(T/"04_top10_person_salience_by_media_house.csv",index=False)

# ---------- entity sentiment ----------
ent.to_csv(T/"05_entity_sentiment_by_media_house.csv",index=False)
div.to_csv(T/"06_article_vs_entity_sentiment_overall.csv",index=False)
divs.to_csv(T/"07_article_vs_entity_sentiment_by_media_house.csv",index=False)

# ---------- TF-IDF: top 10 only ----------
tf10=tf[tf["rank_within_source"]<=10].copy()
tf10.to_csv(T/"08_top10_distinctive_terms_by_media_house.csv",index=False)

# ---------- graphs ----------
# 1 coverage
p=coverage.sort_values("articles")
fig,ax=plt.subplots(figsize=(10,6))
ax.barh(p["source"],p["articles"])
ax.set_xlabel("Articles"); ax.set_title("CJP protest coverage by media house")
fig.tight_layout(); fig.savefig(G/"01_Coverage_by_Media_House.png",dpi=220); plt.close(fig)

# 2 sentiment stacked
s=sent.copy()
fig,ax=plt.subplots(figsize=(11,6))
left=np.zeros(len(s))
for col,label in [("negative_pct","Negative"),("neutral_pct","Neutral"),("positive_pct","Positive")]:
    ax.barh(s["source"],s[col],left=left,label=label)
    left+=s[col].to_numpy()
ax.set_xlabel("Articles (%)");ax.set_title("Body sentiment by media house — reference-paper model")
ax.legend();fig.tight_layout();fig.savefig(G/"02_Body_Sentiment_by_Media_House.png",dpi=220);plt.close(fig)

# 3 headline vs body compound
s2=sent.sort_values("mean_body_compound")
x=np.arange(len(s2));w=.38
fig,ax=plt.subplots(figsize=(12,6))
ax.bar(x-w/2,s2["mean_body_compound"],w,label="Body")
ax.bar(x+w/2,s2["mean_headline_compound"],w,label="Headline")
ax.axhline(0,linewidth=.8);ax.set_xticks(x);ax.set_xticklabels(s2["source"],rotation=45,ha="right")
ax.set_ylabel("Mean compound sentiment");ax.set_title("Headline vs body sentiment")
ax.legend();fig.tight_layout();fig.savefig(G/"03_Headline_vs_Body_Sentiment.png",dpi=220);plt.close(fig)

# 4 article vs entity sentiment: supported entities
d=div[div["articles"]>=100].head(12).copy().sort_values("articles",ascending=False)
x=np.arange(len(d));w=.38
fig,ax=plt.subplots(figsize=(13,7))
ax.bar(x-w/2,d["mean_article_compound"],w,label="Whole article")
ax.bar(x+w/2,d["mean_entity_sentence_compound"],w,label="Entity sentences")
ax.axhline(0,linewidth=.8);ax.set_xticks(x);ax.set_xticklabels(d["entity"],rotation=45,ha="right")
ax.set_ylabel("Mean compound sentiment");ax.set_title("Whole-article vs entity-proximate sentiment")
ax.legend();fig.tight_layout();fig.savefig(G/"04_Article_vs_Entity_Sentiment.png",dpi=220);plt.close(fig)

# 5 person salience heatmap-style matrix: top 8 globally supported canonical persons
global_person=(sal.groupby("canonical_entity",as_index=False)["articles_mentioning"].sum()
               .sort_values("articles_mentioning",ascending=False).head(8)["canonical_entity"].tolist())
mat=(sal[sal["canonical_entity"].isin(global_person)]
     .pivot_table(index="canonical_entity",columns="source",values="mentions_per_article",aggfunc="sum",fill_value=0))
fig,ax=plt.subplots(figsize=(13,6))
im=ax.imshow(mat.values,aspect="auto")
ax.set_xticks(range(len(mat.columns)));ax.set_xticklabels(mat.columns,rotation=45,ha="right")
ax.set_yticks(range(len(mat.index)));ax.set_yticklabels(mat.index)
ax.set_title("Person salience — mentions per article")
fig.colorbar(im,ax=ax,label="Mentions per article")
fig.tight_layout();fig.savefig(G/"05_Person_Salience_Heatmap.png",dpi=220);plt.close(fig)

# Copy final TF-IDF outlet graphs
tf_graph_dir=paths["tfidf"].parent/"graphs"
if tf_graph_dir.exists():
    for f in tf_graph_dir.glob("*.png"):
        shutil.copy2(f,G/f"TFIDF_{f.name}")

# ---------- compact machine-readable summary ----------
summary=coverage.merge(sent,on=["source","articles"],how="left")
summary.to_csv(OUT/"FINAL_RESULTS_SUMMARY.csv",index=False)

# ---------- README ----------
readme=f"""V2 FINAL RESULTS — REFERENCE-PAPER-ALIGNED ANALYSIS

Working corpus: {len(corpus):,} CJP protest articles
Media houses: {coverage['source'].nunique()}

CORE COMPLETED ANALYSES
1. Coverage / temporal context
2. RoBERTa-Large article sentiment using 400-token windows and 50-token overlap
3. Headline vs body sentiment
4. Transformer NER and entity salience
5. Entity-proximate sentence sentiment
6. Whole-article vs entity-proximate sentiment divergence
7. TF-IDF lexical profiling

INTERPRETATION RULES
- Negative sentiment does NOT equal media bias.
- Entity salience measures prominence, not endorsement.
- TF-IDF terms show characteristic vocabulary, not political bias.
- Small international outlet samples must be interpreted descriptively.
- Residual publisher boilerplate remains a limitation for lexical profiling.
- Ambiguous surname-only entities should not be used for strong person-specific conclusions.

This folder consolidates completed V2 outputs. Original datasets and analysis outputs were not modified.
"""
(OUT/"README_RESULTS.txt").write_text(readme,encoding="utf-8")

print("="*72)
print("V2 FINAL RESULTS CONSOLIDATION COMPLETE")
print("="*72)
print("Corpus:",len(corpus))
print("Media houses:",coverage["source"].nunique())
print("Tables:",len(list(T.glob("*.csv"))))
print("Graphs:",len(list(G.glob("*.png"))))
print("Saved:",OUT)
print("\nCoverage:")
print(coverage.to_string(index=False))
print("\nNo models were rerun; existing V2 outputs were consolidated only.")
