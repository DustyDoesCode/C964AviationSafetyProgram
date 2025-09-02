"""
Train a baseline text classifier and save artifacts.

Model:
  Pipeline(TfidfVectorizer, LogisticRegression)

Reads:
  data/raw/asrs_sample.csv

Writes:
  models/baseline_logreg.joblib   trained pipeline (vectorizer + classifier)
  models/metrics.txt              precision, recall, F1, PR-AUC, confusion matrix text
  visuals/confusion_matrix.png    confusion matrix image
  visuals/pr_curve.png            precision–recall curve
  visuals/calibration_curve.png   reliability (calibration) curve

Usage:
  python src/train.py

Notes:
  - Uses a stratified train/test split.
  - Logistic Regression has interpretable word weights for simple explanations.
"""
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve

from utils import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "asrs_sample.csv"
MODELS_DIR = ROOT / "models"
VIS_DIR = ROOT / "visuals"

RANDOM_SEED = 42

def plot_confusion_matrix(cm: np.ndarray, out_path: Path, labels=("Not HF", "HF")):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

def plot_pr_curve(prec, rec, ap: float, out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(rec, prec, label=f"Model (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_calibration_curve(y_true, y_prob, out_path: Path, n_bins: int = 10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    print(f"Loading: {DATA}")
    df = load_dataset(str(DATA))

    X = df["narrative"].values
    y = df["hf_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED
    )

    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("logreg", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            n_jobs=None,
            solver="liblinear"
        ))
    ])

    pipe.fit(X_train, y_train)

    # Hard labels and probabilities
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # Threshold metrics (default 0.5 used by predict)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Curve-based metrics
    rec_curve, prec_curve, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = MODELS_DIR / "baseline_logreg.joblib"
    joblib.dump(pipe, model_path)

    # Save metrics summary
    metrics_path = MODELS_DIR / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"precision: {precision:.3f}\n")
        f.write(f"recall:    {recall:.3f}\n")
        f.write(f"f1:        {f1:.3f}\n")
        f.write(f"pr_auc:    {ap:.3f}\n")
        f.write(f"confusion_matrix:\n{cm}\n")

    # Save images
    plot_confusion_matrix(cm, VIS_DIR / "confusion_matrix.png", labels=("Not HF", "HF"))
    plot_pr_curve(prec_curve, rec_curve, ap, VIS_DIR / "pr_curve.png")
    plot_calibration_curve(y_test, y_prob, VIS_DIR / "calibration_curve.png")

    print(f"Saved model   → {model_path}")
    print(f"Saved metrics → {metrics_path}")
    print(f"Saved images  → {VIS_DIR / 'confusion_matrix.png'}, {VIS_DIR / 'pr_curve.png'}, {VIS_DIR / 'calibration_curve.png'}")

if __name__ == "__main__":
    main()
