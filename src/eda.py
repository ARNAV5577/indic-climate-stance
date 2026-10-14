"""Dataset statistics and visualizations."""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from common import load_dataset, LABELS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/train.csv")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--language-col", default="language")
    p.add_argument("--output-dir", default="reports")
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(a.input)
    lang_col = a.language_col if a.language_col in raw.columns else None
    df = load_dataset(a.input, a.text_col, a.label_col, lang_col)
    df["word_count"] = df["text"].str.split().str.len()
    df["character_count"] = df["text"].str.len()
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
    table = counts.rename_axis("label").reset_index(name="count")
    table["percentage"] = (table["count"] / max(len(df), 1) * 100).round(2)
    table.to_csv(out / "class_distribution.csv", index=False)
    summary = {
        "rows_analyzed": int(len(df)),
        "exact_duplicate_texts": int(df["text"].duplicated().sum()),
        "unique_texts": int(df["text"].nunique()),
        "word_count": df["word_count"].describe().to_dict(),
        "character_count": df["character_count"].describe().to_dict(),
        "label_counts": {str(k): int(v) for k, v in counts.items()},
    }
    if lang_col:
        lang_counts = df["language"].fillna("Unknown").value_counts()
        lang_counts.rename_axis("language").reset_index(name="count").to_csv(out/"language_distribution.csv", index=False)
        pd.crosstab(df["language"].fillna("Unknown"), df["label"]).to_csv(out/"class_by_language.csv")
        summary["language_counts"] = {str(k): int(v) for k, v in lang_counts.items()}
    (out/"eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.figure(figsize=(7,4)); counts.plot(kind="bar")
    plt.title("Class distribution"); plt.xlabel("Label"); plt.ylabel("Comments")
    plt.tight_layout(); plt.savefig(out/"class_distribution.png", dpi=160); plt.close()
    plt.figure(figsize=(7,4)); df["word_count"].plot(kind="hist", bins=25)
    plt.title("Comment length distribution"); plt.xlabel("Words per comment"); plt.ylabel("Frequency")
    plt.tight_layout(); plt.savefig(out/"text_length_distribution.png", dpi=160); plt.close()
    print(f"Analyzed {len(df)} rows. Reports saved to {out.resolve()}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
