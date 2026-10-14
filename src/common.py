"""Shared data loading utilities."""
from pathlib import Path
import pandas as pd

LABELS = ["Favour", "Against", "None"]

def load_dataset(path, text_col="text", label_col="label", language_col=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Put the course CSV at data/train.csv "
            "or pass its path using --input."
        )
    df = pd.read_csv(path)
    missing = [c for c in (text_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}; available columns: {list(df.columns)}")
    cols = [text_col, label_col]
    if language_col and language_col in df.columns:
        cols.append(language_col)
    df = df[cols].copy()
    rename = {text_col: "text", label_col: "label"}
    if language_col and language_col in df.columns:
        rename[language_col] = "language"
    df.rename(columns=rename, inplace=True)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.strip().ne("") & df["label"].ne("")].copy()
    normalized = {x.lower(): x for x in LABELS}
    df["label"] = df["label"].map(lambda x: normalized.get(x.lower(), x))
    invalid = sorted(set(df["label"]) - set(LABELS))
    if invalid:
        raise ValueError(f"Unexpected labels {invalid}; expected {LABELS}")
    return df.reset_index(drop=True)
