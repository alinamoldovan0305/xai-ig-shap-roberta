"""Load SST-2 and EmoBank."""

from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EMOBANK_URL = (
    "https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv"
)
SST2_LABELS = {0: "negative", 1: "positive"}


def load_sst2(cache=True):
    # validation = 872 sentences; GLUE test set has no labels
    # saved once as data/sst2_train.csv and data/sst2_validation.csv
    paths = {s: DATA_DIR / f"sst2_{s}.csv" for s in ("train", "validation")}

    if not (cache and all(p.exists() for p in paths.values())):
        ds = load_dataset("nyu-mll/glue", "sst2")
        if cache:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            for s, p in paths.items():
                ds[s].to_pandas().to_csv(p, index=False)
        if not cache:
            return ds["train"], ds["validation"]

    out = []
    for p in paths.values():
        df = pd.read_csv(p, keep_default_na=False)
        df["sentence"] = df["sentence"].astype(str)
        out.append(Dataset.from_pandas(df, preserve_index=False))
    return out[0], out[1]


def sample_sst2(n=150, seed=SEED):
    _, val = load_sst2()
    n = min(n, len(val))
    return val.shuffle(seed=seed).select(range(n))


def load_emobank(cache=True):
    local = DATA_DIR / "emobank.csv"
    if cache and local.exists():
        df = pd.read_csv(local)
    else:
        df = pd.read_csv(EMOBANK_URL)
        if cache:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(local, index=False)

    expected = {"id", "split", "text", "V", "A"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"EmoBank: missing columns {missing}. Found: {list(df.columns)}")

    df = df[["id", "split", "text", "V", "A"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    return df.reset_index(drop=True)


def emobank_splits():
    df = load_emobank()
    splits = {s: d.reset_index(drop=True) for s, d in df.groupby("split")}
    return splits.get("train"), splits.get("dev"), splits.get("test")


def sample_emobank(n=400, split="test", seed=SEED):
    df = load_emobank()
    df = df[df["split"] == split]
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    # quick check: python -m src.data
    train, val = load_sst2()
    print(f"SST-2  train={len(train)}  validation={len(val)}")
    print("Example:", val[0])

    s = sample_sst2()
    print(f"SST-2 sample: {len(s)}, labels:",
          pd.Series(s["label"]).value_counts().to_dict())

    eb = load_emobank()
    print(f"\nEmoBank total={len(eb)}  splits:", eb["split"].value_counts().to_dict())
    print(eb[["V", "A"]].describe().round(2))
    print("Example:", eb.iloc[0].to_dict())