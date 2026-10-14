"""Optional XLM-R fine-tuning using Hugging Face Transformers."""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from common import load_dataset, LABELS

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/train.csv")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    p.add_argument("--output-dir", default="models/xlmr-climate-stance")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--random-state", type=int, default=42)
    a = p.parse_args()
    df = load_dataset(a.input, a.text_col, a.label_col, None)
    counts = df["label"].value_counts()
    stratify = df["label"] if counts.min() >= 2 else None
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=a.random_state, stratify=stratify)
    train_df = train_df.copy(); val_df = val_df.copy()
    train_df["labels"] = train_df["label"].map(LABEL2ID)
    val_df["labels"] = val_df["label"].map(LABEL2ID)

    tokenizer = AutoTokenizer.from_pretrained(a.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model_name, num_labels=3, label2id=LABEL2ID, id2label=ID2LABEL)
    train_ds = Dataset.from_pandas(train_df[["text","labels"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["text","labels"]], preserve_index=False)
    def tokenize(batch): return tokenizer(batch["text"], truncation=True, max_length=a.max_length)
    train_ds = train_ds.map(tokenize, batched=True).remove_columns(["text"])
    val_ds = val_ds.map(tokenize, batched=True).remove_columns(["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds),
                "macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out), evaluation_strategy="epoch", save_strategy="epoch",
        learning_rate=a.learning_rate, per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size, num_train_epochs=a.epochs,
        weight_decay=0.01, load_best_model_at_end=True, metric_for_best_model="eval_macro_f1",
        greater_is_better=True, seed=a.random_state, report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      tokenizer=tokenizer, data_collator=collator, compute_metrics=compute_metrics)
    trainer.train()
    result = trainer.evaluate()
    pred = np.argmax(trainer.predict(val_ds).predictions, axis=-1)
    print("Validation:", result)
    print(classification_report(val_df["labels"], pred, labels=[0,1,2], target_names=LABELS, zero_division=0))
    trainer.save_model(str(out)); tokenizer.save_pretrained(str(out))
    print(f"Saved model to {out.resolve()}")

if __name__ == "__main__": main()
