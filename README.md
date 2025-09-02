# Aviation Safety Narrative Triage (C964)

A small ML app that scores aviation incident narratives for human-factors risk using TF-IDF + Logistic Regression. It trains a model, saves metrics and images, and lets a user score one narrative or a whole CSV.

## What problem this solves

Safety teams read a lot of incident reports. Manual triage is slow and inconsistent. This app ranks new reports by estimated human-factors risk so analysts start with the highest-risk items.

## Supported Python version

This project targets **Python 3.11 only**. It was built and tested with 3.11.x.

## Features

- Train a baseline model and save artifacts  
- Metrics: precision, recall, F1, PR-AUC  
- Visuals: confusion matrix, PR curve, calibration curve  
- Predict one narrative from the command line  
- Batch score a CSV and write predictions  
- One-command quickstart through `main.py`

---
### Sample files
You can test quickly with the sample CSVs in `data/samples/`:
- `Table1WithLabels.csv`
- `Table2NoLabels.csv`
- `Table3WithExtraColumns.csv`

Go to Batch CSV, upload a file, and review the results. If prompted to pick a narrative column, choose the one with text.

## 1) Setup

### Mac or Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
### Windows

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```
---

## 2) One Command Quickstart

```python
python main.py quickstart
```

This will:

* generate a tiny sample CSV if missing at data/raw/asrs_sample.csv

* run EDA and save two pictures in visuals/

* train the model and save models/baseline_logreg.joblib and models/metrics.txt

* batch score the sample CSV to data/processed/predictions.csv

* do a demo single prediction

## 3) Launch the GUI

First train the model (if you have not already done so) so the app can load it:

```bash
python main.py train
```
Start the GUI:
```bash
streamlit run app.py
```

In the browser:

* Use Single narrative to paste text and get a probability and label.

* Use Batch CSV to upload a file with a narrative column.

* Use Visuals to see the training confusion matrix, PR curve, and calibration curve.

Example batch input columns:

* required: narrative

* optional: report_id, hf_label (kept as hf_label_true in output), and any other columns you want preserved.
---

## 4) Common commands

Train the model:
```python
python main.py train
```

Create EDA plots:
```python
python main.py eda
```

Predict one narrative:
```python
python main.py predict-one --text "Crew reported fatigue and high workload during approach." --threshold 0.5
```

Batch predict from a CSV:
```python
python main.py predict-batch --in data/raw/asrs_sample.csv --out data/processed/predictions.csv --threshold 0.5
```

---


## 5) What artifacts you should see

Metrics file:
```
models/metrics.txt
```

Model file:
```
models/baseline_logreg.joblib
```

Images:
```
visuals/confusion_matrix.png
visuals/pr_curve.png
visuals/calibration_curve.png
```

Batch results:
```
data/processed/predictions.csv
```

Note: If your input CSV already has a true label column named `hf_label`, the batch script will keep it as `hf_label_true` and add `hf_prob_pred` and `hf_label_pred`.

## 6) Data

For development the project uses a tiny synthetic sample created by:
```bash
python src/generate_sample_data.py
```

Replace data/raw/asrs_sample.csv with your ASRS export that includes columns:
```
report_id, narrative, hf_label
```

`hf_label` is optional for batch scoring but required for training and evaluation.

## 7) Troubleshooting

Interpreter issues in an IDE:
```
Point the interpreter to:
- Mac:     .venv/bin/python
- Windows: .venv\Scripts\python.exe
```

Import errors:
```
Activate the venv and run:
pip install -r requirements.txt
```

Plots are missing:
```bash
python main.py train
```

CSV format errors:
```
Your input CSV must have a column named narrative
If it also has hf_label it will be preserved as hf_label_true
Predictions will be written as hf_prob_pred and hf_label_pred
```
## 8) Requirements

Pinned versions:
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
matplotlib==3.8.4
joblib==1.4.2
streamlit==1.36.0
```

Install all with:
```bash
pip install -r requirements.txt
```
