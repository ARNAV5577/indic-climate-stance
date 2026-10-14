"""Majority, TF-IDF + Logistic Regression, and TF-IDF + Linear SVM baselines."""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from common import load_dataset, LABELS

def tfidf_model(classifier):
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1,2), sublinear_tf=True, max_features=150000)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(2,5), sublinear_tf=True, max_features=200000)),
    ])
    return Pipeline([("features", features), ("classifier", classifier)])

def evaluate(name, model, x_train, x_val, y_train, y_val, out):
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    metrics = {
        "accuracy": float(accuracy_score(y_val, pred)),
        "macro_f1": float(f1_score(y_val, pred, labels=LABELS, average="macro", zero_division=0)),
        "report": classification_report(y_val, pred, labels=LABELS, output_dict=True, zero_division=0)
    }
    (out/f"classification_report_{name}.txt").write_text(
        classification_report(y_val, pred, labels=LABELS, zero_division=0), encoding="utf-8")
    cm = confusion_matrix(y_val, pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5,4)); im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion matrix: {name}"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks(range(3), LABELS, rotation=20); ax.set_yticks(range(3), LABELS)
    for i in range(3):
        for j in range(3): ax.text(j, i, str(cm[i,j]), ha="center", va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(out/f"confusion_matrix_{name}.png", dpi=160); plt.close(fig)
    return metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/train.csv")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--language-col", default="language")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(a.input)
    lang_col = a.language_col if a.language_col in raw.columns else None
    df = load_dataset(a.input, a.text_col, a.label_col, lang_col)
    if df["label"].nunique() < 2: raise ValueError("At least two classes are required.")
    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    if stratify is None: print("WARNING: tiny class detected; split is not stratified.")
    xtr, xva, ytr, yva = train_test_split(
        df["text"], df["label"], test_size=a.test_size, random_state=a.random_state, stratify=stratify)
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "tfidf_logreg": tfidf_model(LogisticRegression(max_iter=2000, class_weight="balanced")),
        "tfidf_linearsvm": tfidf_model(LinearSVC(class_weight="balanced")),
    }
    result = {"dataset_rows": int(len(df)), "train_rows": len(xtr), "validation_rows": len(xva),
              "random_state": a.random_state, "labels": LABELS}
    for name, model in models.items():
        print(f"Training {name}...")
        result[name] = evaluate(name, model, xtr, xva, ytr, yva, out)
        print(f"{name}: Macro F1={result[name]['macro_f1']:.4f}; accuracy={result[name]['accuracy']:.4f}")
    (out/"metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to {out.resolve()}")

if __name__ == "__main__": main()
