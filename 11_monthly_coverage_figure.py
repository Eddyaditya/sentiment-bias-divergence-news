from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = Path(
    r"V2_RESEARCH\11_PUBLICATION_FIGURES"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUTPUT_DIR / "FIG_MONTHLY_COVERAGE.png"
PDF_OUT = OUTPUT_DIR / "FIG_MONTHLY_COVERAGE.pdf"


# ============================================================
# VERIFIED FINAL CORPUS COUNTS
# ============================================================

months = [
    "May",
    "June",
    "July",
    "August",
    "September",
]

articles = np.array([
    25,
    299,
    2504,
    723,
    306,
])

total = articles.sum()
shares = articles / total * 100

x = np.arange(len(months))


# ============================================================
# PUBLICATION STYLE
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
    figsize=(6.4, 3.3)
)


# ============================================================
# BARS
# ============================================================

bars = ax.bar(
    x,
    articles,
    width=0.62,
    edgecolor="black",
    linewidth=0.6,
)


# ============================================================
# COUNT + SHARE LABELS
# ============================================================

for bar, count, share in zip(
    bars,
    articles,
    shares,
):

    # July gets count + percentage because it is
    # the major temporal concentration in the corpus.
    if share >= 50:

        label = f"{count:,}\n({share:.1f}%)"

    else:

        label = f"{count:,}"

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        count + 55,
        label,
        ha="center",
        va="bottom",
        fontsize=7.8,
    )


# ============================================================
# AXES
# ============================================================

ax.set_xticks(x)
ax.set_xticklabels(months)

ax.set_ylabel("Number of articles")
ax.set_xlabel("")

ax.set_ylim(0, 2900)

ax.grid(
    axis="y",
    linewidth=0.5,
    alpha=0.20,
)

ax.set_axisbelow(True)


# ============================================================
# CLEAN FRAME
# ============================================================

ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)


# ============================================================
# LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.15,
    right=0.97,
    top=0.94,
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
# VERIFY
# ============================================================

print("=" * 72)
print("MONTHLY COVERAGE FIGURE CREATED")
print("=" * 72)

print("PNG :", PNG_OUT.resolve())
print("PDF :", PDF_OUT.resolve())

print("\nMonthly counts:")
for month, count, share in zip(
    months,
    articles,
    shares,
):
    print(
        f"{month:10s} | {count:4d} | {share:5.2f}%"
    )

print("\nTotal:", total)