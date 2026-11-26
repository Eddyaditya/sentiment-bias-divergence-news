from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT = Path(
    r"V2_RESEARCH\09_MAGPIE_BABE\magpie_by_media_house.csv"
)

OUTPUT_DIR = Path(
    r"V2_RESEARCH\11_PUBLICATION_FIGURES"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUTPUT_DIR / "FIG_MAGPIE_BIASED_SENTENCE_SHARE.png"
PDF_OUT = OUTPUT_DIR / "FIG_MAGPIE_BIASED_SENTENCE_SHARE.pdf"


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
# LOAD VERIFIED MAGPIE OUTPUT
# ============================================================

df = pd.read_csv(INPUT)

d = (
    df[df["source"].isin(indian_outlets)]
    .set_index("source")
    .loc[indian_outlets]
    .reset_index()
)


# ============================================================
# WILSON 95% CONFIDENCE INTERVAL
# ============================================================

def wilson_interval(k, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan

    p = k / n
    z2 = z ** 2

    denominator = 1 + z2 / n

    centre = (
        p + z2 / (2 * n)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (p * (1 - p) / n)
            + (z2 / (4 * n ** 2))
        )
        / denominator
    )

    return centre - margin, centre + margin


ci_low = []
ci_high = []

for _, row in d.iterrows():

    low, high = wilson_interval(
        int(row["biased_sentences"]),
        int(row["total_sentences"]),
    )

    ci_low.append(low * 100)
    ci_high.append(high * 100)

d["ci_low"] = ci_low
d["ci_high"] = ci_high


# ============================================================
# VALUES
# ============================================================

share = d["pooled_biased_sentence_pct"].to_numpy()

lower_error = (
    share - d["ci_low"].to_numpy()
)

upper_error = (
    d["ci_high"].to_numpy() - share
)

asymmetric_error = np.vstack(
    [lower_error, upper_error]
)

x = np.arange(len(d))


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
    figsize=(7.0, 3.8)
)


# ============================================================
# BARS + EXACT ASYMMETRIC WILSON CI
# ============================================================

bars = ax.bar(
    x,
    share,
    width=0.62,
    yerr=asymmetric_error,
    capsize=4,
    edgecolor="black",
    linewidth=0.6,
    error_kw={
        "elinewidth": 0.9,
        "capthick": 0.9,
    },
)


# ============================================================
# VALUE LABELS
# ============================================================

for i, (bar, value, high) in enumerate(
    zip(
        bars,
        share,
        d["ci_high"].to_numpy(),
    )
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        high + 0.32,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )


# ============================================================
# AXES
# ============================================================

ax.set_xticks(x)

ax.set_xticklabels(
    [abbr[s] for s in d["source"]]
)

ax.set_ylabel(
    "Biased-labelled sentences (%)"
)

ax.set_xlabel("")

ax.set_ylim(
    0,
    max(d["ci_high"]) + 2.0,
)

ax.grid(
    axis="y",
    linewidth=0.5,
    alpha=0.20,
)

ax.set_axisbelow(True)


# ============================================================
# SAMPLE SIZE UNDER X LABELS
# ============================================================

for i, n in enumerate(d["total_sentences"]):

    ax.annotate(
        f"{int(n):,}",
        xy=(i, 0),
        xytext=(0, -21),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=6.8,
    )

ax.text(
    -0.07,
    -0.17,
    "Sentences:",
    transform=ax.transAxes,
    ha="right",
    va="center",
    fontsize=6.8,
)


# ============================================================
# LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.14,
    right=0.97,
    top=0.95,
    bottom=0.23,
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
print("MAGPIE WILSON FIGURE CREATED")
print("=" * 72)

print("Input :", INPUT.resolve())
print("PNG   :", PNG_OUT.resolve())
print("PDF   :", PDF_OUT.resolve())

print("\nExact values and Wilson intervals:")

print(
    d[
        [
            "source",
            "articles",
            "total_sentences",
            "biased_sentences",
            "pooled_biased_sentence_pct",
            "ci_low",
            "ci_high",
        ]
    ]
    .round(3)
    .to_string(index=False)
)