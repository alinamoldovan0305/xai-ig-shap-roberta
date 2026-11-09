"""Data preparation checks for SST-2 and EmoBank.

Run: python -m src.data_check
Output: results/data_summary.json and results/figures/*.png
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr

from src.data import ROOT, load_emobank, load_sst2, sample_emobank, sample_sst2

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MAX_LENGTH = 128

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"


def n_words(texts):
    return pd.Series(texts).astype(str).str.split().str.len()


def length_stats(texts):
    w = n_words(texts)
    return {
        "mean_words": round(float(w.mean()), 1),
        "median_words": int(w.median()),
        "min_words": int(w.min()),
        "max_words": int(w.max()),
    }


def token_stats(texts, tokenizer):
    lens = np.array([len(tokenizer(t)["input_ids"]) for t in texts])
    return {
        "mean_tokens": round(float(lens.mean()), 1),
        "max_tokens": int(lens.max()),
        "pct_over_max_length": round(float((lens > MAX_LENGTH).mean() * 100), 2),
    }



def check_sst2(train, val, sample):
    """train, val, sample: DataFrames with 'sentence' and 'label'."""
    s = {}
    for name, df in [("train", train), ("validation", val)]:
        counts = df["label"].value_counts().sort_index()
        s[name] = {
            "n": len(df),
            "labels": {"negative": int(counts.get(0, 0)), "positive": int(counts.get(1, 0))},
            "pct_positive": round(float(df["label"].mean() * 100), 1),
            "empty": int((df["sentence"].str.strip() == "").sum()),
            "duplicates": int(df["sentence"].str.strip().duplicated().sum()),
            **length_stats(df["sentence"]),
        }

    train_set = set(train["sentence"].str.strip())
    s["validation_in_train"] = int(val["sentence"].str.strip().isin(train_set).sum())

    s["sample"] = {
        "n": len(sample),
        "pct_positive": round(float(sample["label"].mean() * 100), 1),
        **length_stats(sample["sentence"]),
        # p > 0.05: no evidence the sample length differs from the full validation set
        "ks_p_length_vs_validation": round(
            float(ks_2samp(n_words(sample["sentence"]), n_words(val["sentence"])).pvalue), 3
        ),
    }
    return s


def check_emobank(df, sample):
    s = {"n": len(df), "splits": df["split"].value_counts().to_dict()}

    s["out_of_range_1_5"] = int(((df[["V", "A"]] < 1) | (df[["V", "A"]] > 5)).any(axis=1).sum())
    s["empty"] = int((df["text"].str.strip() == "").sum())
    s["duplicates"] = int(df["text"].str.strip().duplicated().sum())
    s.update(length_stats(df["text"]))

    s["per_split"] = {
        split: {
            "n": len(d),
            "V_mean": round(float(d["V"].mean()), 2), "V_std": round(float(d["V"].std()), 2),
            "A_mean": round(float(d["A"].mean()), 2), "A_std": round(float(d["A"].std()), 2),
        }
        for split, d in df.groupby("split")
    }

    # share of near-neutral texts (within 0.25 of the scale midpoint 3)
    s["pct_near_neutral"] = {
        dim: round(float(((df[dim] - 3).abs() <= 0.25).mean() * 100), 1) for dim in ("V", "A")
    }

    r, p = pearsonr(df["V"], df["A"])
    s["pearson_V_A"] = {"r": round(float(r), 3), "p": float(f"{p:.2e}")}

    test = df[df["split"] == "test"]
    s["sample"] = {
        "n": len(sample),
        "V_mean": round(float(sample["V"].mean()), 2), "V_std": round(float(sample["V"].std()), 2),
        "A_mean": round(float(sample["A"].mean()), 2), "A_std": round(float(sample["A"].std()), 2),
        "ks_p_V_vs_test": round(float(ks_2samp(sample["V"], test["V"]).pvalue), 3),
        "ks_p_A_vs_test": round(float(ks_2samp(sample["A"], test["A"]).pvalue), 3),
    }
    return s



def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_sst2(train, val, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    ax = axes[0]
    x = np.arange(2)
    w = 0.38
    for i, (name, df, color) in enumerate([("train", train, BLUE), ("validation", val, ORANGE)]):
        pct = df["label"].value_counts(normalize=True).sort_index().reindex([0, 1], fill_value=0) * 100
        bars = ax.bar(x + (i - 0.5) * w, pct.values, w * 0.95, color=color, label=name)
        for b, v in zip(bars, pct.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, ["negative", "positive"])
    ax.set_ylabel("share of texts (%)", color=MUTED)
    ax.set_ylim(0, 70)
    ax.set_title("Label balance", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)

    ax = axes[1]
    bins = np.arange(0, 56, 2)
    ax.hist(n_words(train["sentence"]).clip(upper=54), bins=bins, density=True,
            color=BLUE, alpha=0.55, label="train (phrases)")
    ax.hist(n_words(val["sentence"]).clip(upper=54), bins=bins, density=True,
            histtype="step", linewidth=2, color=ORANGE, label="validation (sentences)")
    ax.set_xlabel("words per text", color=MUTED)
    ax.set_ylabel("density", color=MUTED)
    ax.set_title("Text length", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)

    fig.suptitle("SST-2", x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_emobank(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    bins = np.arange(1, 5.01, 0.1)

    for ax, dim, name, color in [(axes[0], "V", "Valence", BLUE), (axes[1], "A", "Arousal", ORANGE)]:
        ax.hist(df[dim], bins=bins, color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(3, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xlim(1, 5)
        ax.set_xlabel(f"{name.lower()} (1-5)", color=MUTED)
        ax.set_ylabel("texts", color=MUTED)
        ax.set_title(f"{name}: mean {df[dim].mean():.2f}, sd {df[dim].std():.2f}",
                     loc="left", fontsize=11, color=INK)
        _style(ax)

    ax = axes[2]
    ax.scatter(df["V"], df["A"], s=8, alpha=0.25, color=BLUE, edgecolors="none")
    r = df["V"].corr(df["A"])
    ax.set_xlim(1, 5)
    ax.set_ylim(1, 5)
    ax.set_xlabel("valence", color=MUTED)
    ax.set_ylabel("arousal", color=MUTED)
    ax.set_title(f"Valence vs. arousal (r = {r:.2f})", loc="left", fontsize=11, color=INK)
    _style(ax)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)

    fig.suptitle("EmoBank", x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    train, val = load_sst2()
    train, val = train.to_pandas(), val.to_pandas()
    sst_sample = sample_sst2().to_pandas()
    eb = load_emobank()
    eb_sample = sample_emobank()

    summary = {
        "sst2": check_sst2(train, val, sst_sample),
        "emobank": check_emobank(eb, eb_sample),
    }

    # token lengths with the RoBERTa tokenizer (are we cutting texts at 128 tokens?)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("roberta-base")
        summary["sst2"]["validation"].update(token_stats(val["sentence"].tolist(), tok))
        summary["emobank"].update(token_stats(eb["text"].tolist(), tok))
    except Exception as e:
        print("Token stats skipped:", e)

    plot_sst2(train, val, FIGURES / "sst2_data.png")
    plot_emobank(eb, FIGURES / "emobank_data.png")

    with open(RESULTS / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {RESULTS / 'data_summary.json'}")
    print(f"Saved: {FIGURES / 'sst2_data.png'}, {FIGURES / 'emobank_data.png'}")


if __name__ == "__main__":
    main()