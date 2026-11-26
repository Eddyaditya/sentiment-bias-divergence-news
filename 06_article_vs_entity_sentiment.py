from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
ARTICLE_FILE = ROOT/"V2_RESEARCH"/"03_SENTIMENT"/"article_level_sentiment.csv"
ENTITY_FILE = ROOT/"V2_RESEARCH"/"05_ENTITY_SENTIMENT"/"entity_sentence_sentiment.csv"
OUT = ROOT/"V2_RESEARCH"/"06_ARTICLE_VS_ENTITY_SENTIMENT"
OUT.mkdir(parents=True, exist_ok=True)

for p in [ARTICLE_FILE, ENTITY_FILE]:
    if not p.exists():
        raise SystemExit(f"MISSING: {p}")

a = pd.read_csv(ARTICLE_FILE)
e = pd.read_csv(ENTITY_FILE)

# Only high-confidence alias cleanup. Ambiguous surnames are intentionally NOT merged.
ALIASES = {
    "Dharmendra Pradhan's": "Dharmendra Pradhan",
    "Nadda": "JP Nadda",
}
e["entity_final"] = e["canonical_entity"].replace(ALIASES)

# Article sentiment is joined to every entity-sentence instance by original article row_id.
need = ["row_id","source","body_compound","body_sentiment_label"]
m = e.merge(a[need], on=["row_id","source"], how="left", validate="many_to_one")
if m["body_compound"].isna().any():
    print("WARNING: unmatched entity sentence rows:", int(m["body_compound"].isna().sum()))
m = m.dropna(subset=["body_compound"]).copy()

# First aggregate each entity within each article so articles with many repeated mentions
# do not dominate the entity-level mean.
per_article = (
    m.groupby(["source","entity_final","row_id"], as_index=False)
     .agg(
         entity_sentence_compound=("compound","mean"),
         article_compound=("body_compound","first"),
         sentence_instances=("sentence","size")
     )
)
per_article.to_csv(OUT/"entity_article_level_pairs.csv", index=False)

def divergence(article_mean, entity_mean):
    # Paper-style relative increase in negative intensity.
    # Meaningful only when both aggregate means are negative and entity mean != 0.
    if article_mean < 0 and entity_mean < 0 and abs(entity_mean) > 1e-12:
        return (abs(article_mean)-abs(entity_mean))/abs(entity_mean)*100
    return np.nan

# Corpus-wide entity comparison.
rows=[]
for ent,g in per_article.groupby("entity_final"):
    am=g["article_compound"].mean()
    em=g["entity_sentence_compound"].mean()
    rows.append({
        "entity":ent,
        "articles":g["row_id"].nunique(),
        "sentence_instances":int(g["sentence_instances"].sum()),
        "mean_article_compound":am,
        "mean_entity_sentence_compound":em,
        "article_minus_entity_compound":am-em,
        "delta_negativity_pct":divergence(am,em)
    })
overall=pd.DataFrame(rows).sort_values(["articles","sentence_instances"],ascending=False)
overall.to_csv(OUT/"article_vs_entity_sentiment_overall.csv",index=False)

# Outlet x entity comparison.
rows=[]
for (src,ent),g in per_article.groupby(["source","entity_final"]):
    am=g["article_compound"].mean()
    em=g["entity_sentence_compound"].mean()
    rows.append({
        "source":src,"entity":ent,
        "articles":g["row_id"].nunique(),
        "sentence_instances":int(g["sentence_instances"].sum()),
        "mean_article_compound":am,
        "mean_entity_sentence_compound":em,
        "article_minus_entity_compound":am-em,
        "delta_negativity_pct":divergence(am,em)
    })
by=pd.DataFrame(rows).sort_values(["entity","articles"],ascending=[True,False])
by.to_csv(OUT/"article_vs_entity_sentiment_by_media_house.csv",index=False)

# Focus chart on entities with enough article support.
plot=overall[overall["articles"]>=100].head(12).copy()
if len(plot):
    x=np.arange(len(plot)); w=0.38
    fig,ax=plt.subplots(figsize=(13,7))
    ax.bar(x-w/2,plot["mean_article_compound"],w,label="Whole article")
    ax.bar(x+w/2,plot["mean_entity_sentence_compound"],w,label="Entity-containing sentences")
    ax.axhline(0,linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot["entity"],rotation=45,ha="right")
    ax.set_ylabel("Mean compound sentiment")
    ax.set_title("Whole-article vs entity-proximate sentiment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT/"article_vs_entity_sentiment.png",dpi=220)
    plt.close(fig)

# Alias audit
pd.DataFrame(
    [{"original":k,"final":v} for k,v in ALIASES.items()]
).to_csv(OUT/"second_round_alias_audit.csv",index=False)

print("="*72)
print("ARTICLE VS ENTITY-PROXIMATE SENTIMENT COMPLETE")
print("="*72)
print("Entity sentence rows:",len(e))
print("Matched rows:",len(m))
print("Entity/article pairs:",len(per_article))
print("\nTOP SUPPORTED ENTITIES")
print(overall.head(15).round(3).to_string(index=False))
print("\nSaved:",OUT)
print("NOTE: delta_negativity_pct is reported only where both aggregate means are negative.")
