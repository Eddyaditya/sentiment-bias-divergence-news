from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT = Path(
    r"V2_RESEARCH\06_ARTICLE_VS_ENTITY_SENTIMENT\article_vs_entity_sentiment_overall.csv"
)

OUTPUT_DIR = Path(
    r"V2_RESEARCH\11_PUBLICATION_FIGURES"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUTPUT_DIR / "FIG_ARTICLE_VS_ENTITY_SENTIMENT.png"
PDF_OUT = OUTPUT_DIR / "FIG_ARTICLE_VS_ENTITY_SENTIMENT.pdf"


# ============================================================
# LOAD VERIFIED DATA
# ============================================================

df = pd.read_csv(INPUT)


# ============================================================
# SELECT UNAMBIGUOUS NAMED PEOPLE
# ============================================================

selected_people = [
    "Dharmendra Pradhan",
    "Sonam Wangchuk",
    "Abhijeet Dipke",
    "Narendra Modi",
    "Rahul Gandhi",
    "Saurav Das",
    "Surya Kant",
    "Ashutosh Ranka",
    "JP Nadda",
    "Amit Shah",
    "Jitendra Singh",
]

d = (
    df[df["entity"].isin(selected_people)]
    .set_index("entity")
    .loc[selected_people]
    .reset_index()
)


# ============================================================
# VALUES
# ============================================================

article = d["mean_article_compound"].to_numpy()
entity = d["mean_entity_sentence_compound"].to_numpy()
divergence = d["delta_negativity_pct"].to_numpy()

labels = d["entity"].tolist()
y = np.arange(len(d))


# ============================================================
# PUBLICATION STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 5.2)
)


# ============================================================
# CONNECTING LINES
# ============================================================

for i in range(len(d)):
    ax.plot(
        [article[i], entity[i]],
        [i, i],
        color="0.68",
        linewidth=1.4,
        zorder=1,
    )


# ============================================================
# ARTICLE POINTS
# ============================================================

ax.scatter(
    article,
    y,
    s=48,
    marker="o",
    label="Whole article",
    edgecolor="black",
    linewidth=0.55,
    zorder=3,
)


# ============================================================
# ENTITY-SENTENCE POINTS
# ============================================================

ax.scatter(
    entity,
    y,
    s=48,
    marker="s",
    label="Entity-proximate sentences",
    edgecolor="black",
    linewidth=0.55,
    zorder=3,
)


# ============================================================
# DIVERGENCE LABELS
# ============================================================

label_x = -0.115

for i, value in enumerate(divergence):

    sign = "+" if value > 0 else ""

    ax.text(
        label_x,
        i,
        f"{sign}{value:.1f}%",
        ha="left",
        va="center",
        fontsize=7.4,
    )


# ============================================================
# DIVERGENCE HEADING
# ============================================================

ax.text(
    label_x,
    -0.75,
    r"$\Delta$ negativity",
    ha="left",
    va="center",
    fontsize=7.5,
    fontweight="bold",
)


# ============================================================
# AXES
# ============================================================

ax.set_yticks(y)
ax.set_yticklabels(labels)

ax.invert_yaxis()

ax.set_xlim(-0.49, -0.08)

ax.set_xlabel(
    "Mean compound sentiment score"
)

ax.set_ylabel("")

ax.grid(
    axis="x",
    linewidth=0.5,
    alpha=0.20,
)

ax.set_axisbelow(True)


# ============================================================
# LEGEND
# ============================================================

ax.legend(
    frameon=False,
    ncol=2,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.015),
    columnspacing=1.6,
)


# ============================================================
# LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.30,
    right=0.96,
    top=0.89,
    bottom=0.12,
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
# CONSOLE VERIFICATION
# ============================================================

print("=" * 72)
print("ARTICLE VS ENTITY SENTIMENT FIGURE CREATED")
print("=" * 72)

print("Input :", INPUT.resolve())
print("PNG   :", PNG_OUT.resolve())
print("PDF   :", PDF_OUT.resolve())

print("\nValues plotted:")

print(
    d[
        [
            "entity",
            "articles",
            "sentence_instances",
            "mean_article_compound",
            "mean_entity_sentence_compound",
            "delta_negativity_pct",
        ]
    ]
    .round(4)
    .to_string(index=False)
)