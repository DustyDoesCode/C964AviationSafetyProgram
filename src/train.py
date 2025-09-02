"""
Train a baseline text classifier and save artifacts.

Model:
  Pipeline(TfidfVectorizer, LogisticRegression)

Reads:
  data/raw/asrs_sample.csv

Writes:
  models/baseline_logreg.joblib   trained pipeline (vectorizer + classifier)
  models/metrics.txt              precision, recall, F1, confusion matrix text
  visuals/confusion_matrix.png    confusion matrix image

Usage:
  python src/train.py

Notes:
  - Uses a stratified train/test split.
  - Logistic Regression has interpretable word weights for simple explanations.
"""


from pathlib import Path
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

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
    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    vec = pipe.named_steps["tfidf"]
    feat_names = vec.get_feature_names_out()
    print("vocab size:", len(feat_names))
    print("example features:", list(feat_names[:10]))

    y_pred = pipe.predict(X_test)
    import numpy as np
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["logreg"]
    feat_names = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    top_pos_idx = np.argsort(coefs)[-10:][::-1]
    top_neg_idx = np.argsort(coefs)[:10]
    print("\nTop words indicating HF:")
    for i in top_pos_idx:
        print(f"{feat_names[i]:20s}  {coefs[i]:+.3f}")
    print("\nTop words indicating Not HF:")
    for i in top_neg_idx:
        print(f"{feat_names[i]:20s}  {coefs[i]:+.3f}")


    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # save model
    model_path = MODELS_DIR / "baseline_logreg.joblib"
    joblib.dump(pipe, model_path)

    # save metrics
    metrics_path = MODELS_DIR / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"precision: {precision:.3f}\n")
        f.write(f"recall:    {recall:.3f}\n")
        f.write(f"f1:        {f1:.3f}\n")
        f.write(f"confusion_matrix:\n{cm}\n")

    # save confusion matrix plot
    cm_img = VIS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, cm_img, labels=("Not HF", "HF"))

    print(f"Saved model   → {model_path}")
    print(f"Saved metrics → {metrics_path}")
    print(f"Saved image   → {cm_img}")

if __name__ == "__main__":
    main()
