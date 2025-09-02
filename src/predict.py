from pathlib import Path
import argparse
import numpy as np
import joblib

from utils import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "baseline_logreg.joblib"

def predict_one(text: str, threshold: float = 0.5, top_n: int = 8):
    """Return label, probability, and top positive/negative word contributions."""
    pipe = joblib.load(MODEL_PATH)
    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["logreg"]

    text_clean = clean_text(text)
    X = tfidf.transform([text_clean])           # sparse vector
    prob = float(clf.predict_proba(X)[0, 1])    # probability of HF=1
    label = int(prob >= threshold)

    # contribution of each feature = weight * value
    coefs = clf.coef_[0]                        # shape [n_features]
    # convert sparse row to coo to get indices that are present
    row = X.tocoo()
    contrib = {}
    for i, v in zip(row.col, row.data):
        contrib[i] = coefs[i] * v

    feat_names = tfidf.get_feature_names_out()
    # top positive pushes toward HF, top negative pushes away
    items = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    top_pos = [(feat_names[i], contrib[i]) for i, _ in items[:top_n]]
    top_neg = [(feat_names[i], contrib[i]) for i, _ in sorted(contrib.items(), key=lambda kv: kv[1])[:top_n]]

    return label, prob, top_pos, top_neg

def main():
    parser = argparse.ArgumentParser(description="Predict human-factors risk from a narrative.")
    parser.add_argument("--text", type=str, help="Narrative text to score")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for HF label")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Train first with: python src/train.py")

    if not args.text:
        print("Usage example:")
        print('  python src/predict.py --text "Crew reported fatigue and high workload during approach."')
        return

    label, prob, top_pos, top_neg = predict_one(args.text, threshold=args.threshold)
    print("\nInput:")
    print(args.text)
    print("\nPrediction:")
    print(f"  HF label: {label}   probability: {prob:.3f}   threshold: {args.threshold}")
    print("\nTop words pushing toward HF:")
    for w, c in top_pos:
        print(f"  {w:20s}  {c:+.4f}")
    print("\nTop words pushing toward Not HF:")
    for w, c in top_neg:
        print(f"  {w:20s}  {c:+.4f}")

if __name__ == "__main__":
    main()
