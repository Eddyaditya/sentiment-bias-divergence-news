from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

INPUT = Path(
    r"V2_RESEARCH\08_FINAL_RESULTS\TABLES\03_sentiment_by_media_house.csv"
)

OUTPUT_DIR = Path(
    r"V2_RESEARCH\11_PUBLICATION_FIGURES"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUTPUT_DIR / "FIG_SENTIMENT_DISTRIBUTION.png"
PDF_OUT = OUTPUT_DIR / "FIG_SENTIMENT_DISTRIBUTION.pdf"


# ============================================================
# 2. LOAD FINAL VERIFIED DATA
# ============================================================

df = pd.read_csv(INPUT)

indian_outlets = [
    "Hindustan Times",
    "Economic Times",
    "The Indian Express",
    "Deccan Herald",
    "The New Indian Express",
    "Business Standard",
]

d = (
    df[df["source"].isin(indian_outlets)]
    .set_index("source")
    .loc[indian_outlets]
    .reset_index()
)

negative = d["negative_pct"].to_numpy()
neutral = d["neutral_pct"].to_numpy()
positive = d["positive_pct"].to_numpy()

labels = d["source"].tolist()
y = np.arange(len(d))


# ============================================================
# 3. PUBLICATION STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 4. CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 3.8))


# ============================================================
# 5. 100% STACKED HORIZONTAL BARS
# ============================================================

ax.barh(
    y,
    negative,
    label="Negative",
    edgecolor="black",
    linewidth=0.45,
)

ax.barh(
    y,
    neutral,
    left=negative,
    label="Neutral",
    edgecolor="black",
    linewidth=0.45,
)

ax.barh(
    y,
    positive,
    left=negative + neutral,
    label="Positive",
    edgecolor="black",
    linewidth=0.45,
)


# ============================================================
# 6. PERCENTAGE LABELS
# ============================================================

for i in range(len(d)):

    # Negative share
    if negative[i] >= 5:
        ax.text(
            negative[i] / 2,
            i,
            f"{negative[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
        )

    # Neutral share
    if neutral[i] >= 5:
        ax.text(
            negative[i] + neutral[i] / 2,
            i,
            f"{neutral[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
        )

    # Positive labels only when large enough to be readable
    if positive[i] >= 2:
        ax.text(
            negative[i] + neutral[i] + positive[i] / 2,
            i,
            f"{positive[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=7,
        )


# ============================================================
# 7. SAMPLE SIZE LABELS
# ============================================================

for i, n in enumerate(d["articles"]):
    ax.text(
        100.8,
        i,
        f"n={int(n):,}",
        ha="left",
        va="center",
        fontsize=7.5,
        clip_on=False,
    )


# ============================================================
# 8. AXES
# ============================================================

ax.set_yticks(y)
ax.set_yticklabels(labels)

# Hindustan Times at top
ax.invert_yaxis()

ax.set_xlim(0, 100)

ax.set_xlabel("Share of articles (%)")
ax.set_ylabel("")

ax.set_xticks(np.arange(0, 101, 20))

ax.grid(
    axis="x",
    linewidth=0.5,
    alpha=0.20,
)

ax.set_axisbelow(True)


# ============================================================
# 9. LEGEND
# ============================================================

ax.legend(
    frameon=False,
    ncol=3,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.015),
    columnspacing=1.6,
    handlelength=1.8,
)


# ============================================================
# 10. CLEAN FRAME
# ============================================================

ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)


# ============================================================
# 11. MANUAL JOURNAL LAYOUT
# ============================================================

# No graph title:
# the LaTeX figure caption will provide the title/explanation.

fig.subplots_adjust(
    left=0.29,
    right=0.89,
    top=0.86,
    bottom=0.16,
)


# ============================================================
# 12. SAVE
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
# 13. CONSOLE VERIFICATION
# ============================================================

print("=" * 72)
print("FINAL SENTIMENT DISTRIBUTION FIGURE CREATED")
print("=" * 72)

print("Input :", INPUT.resolve())
print("PNG   :", PNG_OUT.resolve())
print("PDF   :", PDF_OUT.resolve())

print("\nValues plotted:")

print(
    d[
        [
            "source",
            "articles",
            "negative_pct",
            "neutral_pct",
            "positive_pct",
        ]
    ]
    .round(3)
    .to_string(index=False)
)