"""
Batch-score a CSV of narratives with the saved model.

Reads:
  models/baseline_logreg.joblib

Input CSV:
  Must have a column named 'narrative'. Other columns are kept and passed through.

Writes:
  data/processed/predictions.csv  with columns:
    [ ...original columns..., hf_prob, hf_label ]

Usage:
  python src/predict_batch.py --in data/raw/asrs_sample.csv --out data/processed/predictions.csv --threshold 0.5
"""
from pathlib import Path
import argparse
import pandas as pd
import joblib
from utils import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "baseline_logreg.joblib"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV with 'narrative' column")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path")
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for HF label")
    args = ap.parse_args()

    in_path = ROOT / args.in_path if not Path(args.in_path).is_absolute() else Path(args.in_path)
    out_path = ROOT / args.out_path if not Path(args.out_path).is_absolute() else Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Train first with: python src/train.py")
    model = joblib.load(MODEL_PATH)
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["logreg"]

    df = pd.read_csv(in_path)
    if "narrative" not in df.columns:
        raise SystemExit("Input CSV must contain a 'narrative' column.")
    df["_n_clean"] = df["narrative"].apply(clean_text)

    X = tfidf.transform(df["_n_clean"].tolist())
    probs = clf.predict_proba(X)[:, 1]
    labels = (probs >= args.threshold).astype(int)

    # keep any existing truth as hf_label_true
    if "hf_label" in df.columns:
        df = df.rename(columns={"hf_label": "hf_label_true"})

    df["hf_prob_pred"] = probs
    df["hf_label_pred"] = labels
    df.drop(columns=["_n_clean"], inplace=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  rows={len(df)}  threshold={args.threshold}")

if __name__ == "__main__":
    main()
