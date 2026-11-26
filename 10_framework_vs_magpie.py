from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt

R=Path.cwd()
FW=R/"V2_RESEARCH"/"08_FINAL_RESULTS"/"FINAL_RESULTS_SUMMARY.csv"
MG=R/"V2_RESEARCH"/"09_MAGPIE_BABE"/"magpie_by_media_house.csv"
OUT=R/"V2_RESEARCH"/"10_FINAL_COMPARISON"
G=OUT/"GRAPHS";T=OUT/"TABLES";G.mkdir(parents=True,exist_ok=True);T.mkdir(parents=True,exist_ok=True)
for x in [FW,MG]:
 if not x.exists(): raise SystemExit(f"MISSING: {x}")

fw=pd.read_csv(FW); mg=pd.read_csv(MG)
m=fw.merge(mg,on="source",how="left",suffixes=("_corpus","_magpie"))
m.to_csv(T/"framework_vs_magpie_by_media_house.csv",index=False)

# Main Indian outlets: avoid tiny international samples in comparative inference.
indian=["Hindustan Times","Economic Times","The Indian Express","Deccan Herald","The New Indian Express","Business Standard"]
mi=m[m.source.isin(indian)].copy()
mi.to_csv(T/"indian_outlets_primary_comparison.csv",index=False)

# Correlations are descriptive only (n=6), not inferential proof.
metrics=[c for c in ["negative_pct","neutral_pct","mean_body_compound","mean_headline_compound","mean_title_content_gap"] if c in mi.columns]
corr=[]
for c in metrics:
 x=pd.to_numeric(mi[c],errors="coerce"); y=pd.to_numeric(mi["pooled_biased_sentence_pct"],errors="coerce")
 ok=x.notna()&y.notna()
 if ok.sum()>=3:
  corr.append({"framework_metric":c,"n_outlets":int(ok.sum()),"pearson_r":float(np.corrcoef(x[ok],y[ok])[0,1])})
pd.DataFrame(corr).to_csv(T/"descriptive_correlations_indian_outlets.csv",index=False)

# Graph 1: MAGPIE signal
q=m.sort_values("pooled_biased_sentence_pct")
fig,ax=plt.subplots(figsize=(10,6));ax.barh(q.source,q.pooled_biased_sentence_pct)
ax.set_xlabel("MAGPIE/BABE biased-labelled sentences (%)")
ax.set_title("Sentence-level linguistic bias signal by media house")
fig.tight_layout();fig.savefig(G/"01_MAGPIE_Bias_Signal_by_Media_House.png",dpi=220);plt.close(fig)

# Graph 2: body negativity vs MAGPIE, Indian outlets
fig,ax=plt.subplots(figsize=(8,6))
ax.scatter(mi.negative_pct,mi.pooled_biased_sentence_pct)
for _,r in mi.iterrows(): ax.annotate(r.source,(r.negative_pct,r.pooled_biased_sentence_pct),xytext=(4,4),textcoords="offset points",fontsize=8)
ax.set_xlabel("Negative body sentiment (%)");ax.set_ylabel("MAGPIE biased-labelled sentences (%)")
ax.set_title("Textual negativity vs sentence-level bias signal — Indian outlets")
fig.tight_layout();fig.savefig(G/"02_Sentiment_vs_MAGPIE_Indian_Outlets.png",dpi=220);plt.close(fig)

# Graph 3: headline/body gap vs MAGPIE
if "mean_title_content_gap" in mi:
 fig,ax=plt.subplots(figsize=(8,6));ax.scatter(mi.mean_title_content_gap,mi.pooled_biased_sentence_pct)
 for _,r in mi.iterrows(): ax.annotate(r.source,(r.mean_title_content_gap,r.pooled_biased_sentence_pct),xytext=(4,4),textcoords="offset points",fontsize=8)
 ax.set_xlabel("Mean headline minus body compound sentiment")
 ax.set_ylabel("MAGPIE biased-labelled sentences (%)")
 ax.set_title("Headline–body tone gap vs linguistic bias signal")
 fig.tight_layout();fig.savefig(G/"03_Headline_Body_Gap_vs_MAGPIE.png",dpi=220);plt.close(fig)

# Coverage of MAGPIE analysis vs corpus
m["magpie_article_coverage_pct"]=100*m["articles_magpie"]/m["articles_corpus"]
m[["source","articles_corpus","articles_magpie","magpie_article_coverage_pct"]].to_csv(T/"magpie_analysis_coverage.csv",index=False)

readme="""FINAL TWO-APPROACH COMPARISON

APPROACH 1 — Reference-paper-aligned framework
Coverage, RoBERTa-Large sentiment, headline/body comparison, transformer NER/entity salience,
entity-proximate sentiment, article-vs-entity divergence, and TF-IDF lexical profiling.

APPROACH 2 — MAGPIE/BABE
Sentence-level linguistic/media-bias classification.

INTERPRETATION
The two approaches measure different constructs. Agreement is converging evidence of a pattern;
disagreement is also substantive because editorial selection/framing and sentence-level linguistic
bias need not move together. Negative sentiment is not equivalent to bias. MAGPIE percentage is
not an outlet's universal 'percent biased' score.

Primary cross-outlet comparison should emphasize the six Indian outlets because international
samples are very small (especially CNN, Guardian, BBC). Correlations across six outlets are
descriptive only and should not be presented as statistically established relationships.

ANALYSIS STATUS: FREEZE AFTER REVIEWING THESE OUTPUTS.
"""
(OUT/"README_FINAL_COMPARISON.txt").write_text(readme,encoding="utf-8")

print("="*72);print("FINAL FRAMEWORK vs MAGPIE COMPARISON COMPLETE");print("="*72)
cols=[c for c in ["source","articles_corpus","negative_pct","mean_body_compound","pooled_biased_sentence_pct","magpie_article_coverage_pct"] if c in m.columns]
print(m[cols].round(3).to_string(index=False))
print("\nDescriptive correlations — Indian outlets:")
if corr: print(pd.DataFrame(corr).round(3).to_string(index=False))
print("\nSaved:",OUT)
print("No model was rerun.")
