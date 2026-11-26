from pathlib import Path
import pandas as pd,re
R=Path.cwd(); D=R/"V2_RESEARCH"/"04_NER_ENTITY_SALIENCE"; I=D/"entity_mentions.csv"
if not I.exists(): raise SystemExit(f"MISSING: {I}")
df=pd.read_csv(I)
A={"wangchuk":"Sonam Wangchuk","sonam wangchuk":"Sonam Wangchuk","pradhan":"Dharmendra Pradhan","dharmendra pradhan":"Dharmendra Pradhan","dipke":"Abhijeet Dipke","abhijeet dipke":"Abhijeet Dipke","modi":"Narendra Modi","narendra modi":"Narendra Modi","rahul gandhi":"Rahul Gandhi"}
def c(x):
 x=re.sub(r"\s+"," ",str(x)).strip();return A.get(x.lower(),x)
df["canonical_entity"]=df.entity.map(c);df["was_canonicalized"]=df.entity.astype(str)!=df.canonical_entity
df.to_csv(D/"entity_mentions_canonicalized.csv",index=False)
ppl=df[df.entity_type.astype(str).str.upper()=="PERSON"].copy()
agg=ppl.groupby(["source","canonical_entity"],as_index=False).agg(total_mentions=("mentions_in_article","sum"),articles_mentioning=("row_id","nunique"))
art=pd.read_csv(D/"article_entity_summary.csv"); den=art.groupby("source").row_id.nunique().to_dict()
agg["source_articles"]=agg.source.map(den);agg["mentions_per_article"]=agg.total_mentions/agg.source_articles;agg["article_presence_pct"]=100*agg.articles_mentioning/agg.source_articles
agg=agg.sort_values(["source","total_mentions"],ascending=[True,False]);agg.to_csv(D/"canonical_person_salience_by_media_house.csv",index=False)
agg.groupby("source",group_keys=False).head(30).to_csv(D/"top_canonical_persons_by_media_house.csv",index=False)
audit=df[df.was_canonicalized].groupby(["entity","canonical_entity"],as_index=False).agg(rows=("row_id","count"),mentions=("mentions_in_article","sum")).sort_values("mentions",ascending=False)
audit.to_csv(D/"entity_alias_merge_audit.csv",index=False)
print("ENTITY CANONICALIZATION COMPLETE");print("Mention rows:",len(df));print("Rows canonicalized:",int(df.was_canonicalized.sum()));print(audit.to_string(index=False));print("Saved:",D)
