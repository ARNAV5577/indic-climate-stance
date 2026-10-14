# Indic Climate Stance Challenge — Baseline Code

CSET346 NLP project: classify Hindi/Bengali social-media comments toward the claim
**“Climate change and global warming is a serious concern.”**

Labels: `Favour`, `Against`, `None`.

## Includes
- Dataset EDA and plots
- Majority-class baseline
- TF-IDF word + character n-gram models (Logistic Regression and Linear SVM)
- Macro F1, per-class report, confusion matrices
- Optional XLM-R transformer fine-tuning

**No actual course dataset was included. Do not report statistics or model scores until you run the code on your real dataset.**

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.

## Dataset
Put your course-approved CSV at `data/train.csv`. By default it should contain:
- `text`: comment
- `label`: `Favour`, `Against`, or `None`
- `language`: optional, e.g. Hindi/Bengali

If your columns differ, use command-line flags. Do not publish course data to GitHub unless redistribution is permitted.

## EDA
```bash
python src/eda.py --input data/train.csv --text-col text --label-col label --language-col language
```

## Traditional ML baselines
```bash
python src/train_baselines.py --input data/train.csv --text-col text --label-col label
```

Outputs go into `reports/`. Review the split and class counts before trusting the metrics.

## Optional transformer
```bash
pip install -r requirements-transformers.txt
python src/train_transformer.py --input data/train.csv --text-col text --label-col label --model-name FacebookAI/xlm-roberta-base
```

Transformer training can be slow on CPU and may require substantial memory.

## Research cautions
- Macro F1 is the primary metric; report class-wise precision/recall/F1 too.
- Check duplicates and related comments crossing splits.
- Preserve negation, punctuation and emojis unless experiments justify changing them.
- Never use hidden test data for model selection.
- Synthetic augmentation can create incorrect labels; validate generated examples.
- All performance values remain unknown until measured on the actual dataset.

## Structure
```text
indic-climate-stance-code/
├── README.md
├── requirements.txt
├── requirements-transformers.txt
├── .gitignore
├── data/README.md
├── src/common.py
├── src/eda.py
├── src/train_baselines.py
└── src/train_transformer.py
```
