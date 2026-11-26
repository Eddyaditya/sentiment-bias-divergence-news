from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SENTIMENT_FILE = Path(
    r"V2_RESEARCH\08_FINAL_RESULTS\TABLES\03_sentiment_by_media_house.csv"
)

MAGPIE_FILE = Path(
    r"V2_RESEARCH\09_MAGPIE_BABE\magpie_by_media_house.csv"
)

OUTPUT_DIR = Path(
    r"V2_RESEARCH\11_PUBLICATION_FIGURES"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUTPUT_DIR / "FIG_NEGATIVITY_VS_MAGPIE.png"
PDF_OUT = OUTPUT_DIR / "FIG_NEGATIVITY_VS_MAGPIE.pdf"


# ============================================================
# OUTLETS
# ============================================================

indian_outlets = [
    "Hindustan Times",
    "Economic Times",
    "The Indian Express",
    "Deccan Herald",
    "The New Indian Express",
    "Business Standard",
]

abbr = {
    "Hindustan Times": "HT",
    "Economic Times": "ET",
    "The Indian Express": "IE",
    "Deccan Herald": "DH",
    "The New Indian Express": "NIE",
    "Business Standard": "BS",
}


# ============================================================
# LOAD VERIFIED DATA
# ============================================================

sent = pd.read_csv(SENTIMENT_FILE)
mag = pd.read_csv(MAGPIE_FILE)

sent = sent[
    sent["source"].isin(indian_outlets)
][["source", "negative_pct"]]

mag = mag[
    mag["source"].isin(indian_outlets)
][["source", "pooled_biased_sentence_pct"]]

d = sent.merge(
    mag,
    on="source",
    how="inner",
)

d = (
    d.set_index("source")
    .loc[indian_outlets]
    .reset_index()
)


# ============================================================
# CORRELATIONS
# ============================================================

pearson_all = d[
    ["negative_pct", "pooled_biased_sentence_pct"]
].corr(method="pearson").iloc[0, 1]

spearman_all = d[
    ["negative_pct", "pooled_biased_sentence_pct"]
].corr(method="spearman").iloc[0, 1]

without_bs = d[
    d["source"] != "Business Standard"
]

pearson_without_bs = without_bs[
    ["negative_pct", "pooled_biased_sentence_pct"]
].corr(method="pearson").iloc[0, 1]


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.2, 4.1)
)


# ============================================================
# NORMAL OUTLETS
# ============================================================

normal = d[
    d["source"] != "Business Standard"
]

ax.scatter(
    normal["negative_pct"],
    normal["pooled_biased_sentence_pct"],
    s=58,
    marker="o",
    edgecolor="black",
    linewidth=0.7,
    zorder=3,
)


# ============================================================
# BUSINESS STANDARD HIGHLIGHT
# ============================================================

bs = d[
    d["source"] == "Business Standard"
]

ax.scatter(
    bs["negative_pct"],
    bs["pooled_biased_sentence_pct"],
    s=72,
    marker="D",
    edgecolor="black",
    linewidth=0.8,
    zorder=4,
)


# ============================================================
# LABEL POSITIONS
# ============================================================

offsets = {
    "Hindustan Times": (5, 7),
    "Economic Times": (5, 7),
    "The Indian Express": (5, -13),
    "Deccan Herald": (5, 7),
    "The New Indian Express": (5, -13),
    "Business Standard": (7, 6),
}

for _, row in d.iterrows():

    dx, dy = offsets[row["source"]]

    ax.annotate(
        abbr[row["source"]],
        (
            row["negative_pct"],
            row["pooled_biased_sentence_pct"],
        ),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
    )


# ============================================================
# AXES
# ============================================================

ax.set_xlabel(
    "Negative body sentiment share (%)"
)

ax.set_ylabel(
    "Biased-labelled sentences (%)"
)

ax.set_xlim(15, 40)
ax.set_ylim(7, 17)

ax.grid(
    linewidth=0.5,
    alpha=0.20,
)

ax.set_axisbelow(True)


# ============================================================
# CORRELATION NOTE
# ============================================================

note = (
    f"All outlets: r = {pearson_all:.2f}\n"
    f"Without BS: r = {pearson_without_bs:.2f}"
)

ax.text(
    0.04,
    0.94,
    note,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=7.8,
    bbox={
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "0.65",
        "linewidth": 0.6,
    },
)


# ============================================================
# LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.16,
    right=0.97,
    top=0.96,
    bottom=0.16,
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    PNG_OUT,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# VERIFICATION
# ============================================================

print("=" * 72)
print("NEGATIVITY VS MAGPIE SCATTER CREATED")
print("=" * 72)

print("PNG :", PNG_OUT.resolve())
print("PDF :", PDF_OUT.resolve())

print("\nValues plotted:")

print(
    d.round(4).to_string(index=False)
)

print("\nCorrelations:")
print(f"Pearson all six   : {pearson_all:.4f}")
print(f"Spearman all six  : {spearman_all:.4f}")
print(f"Pearson without BS: {pearson_without_bs:.4f}")