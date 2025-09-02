#!/usr/bin/env python3
"""
One entry point to run the pipeline.

Examples:
  python main.py quickstart
  python main.py train
  python main.py eda
  python main.py predict-one --text "Crew reported fatigue and high workload during approach."
  python main.py predict-batch --in data/raw/asrs_sample.csv --out data/processed/predictions.csv --threshold 0.7
"""
from pathlib import Path
import argparse
import subprocess
import sys
import shlex

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PY = sys.executable  # use the exact Python from your venv

def run_script(name: str, *args: str) -> None:
    """Call one of our scripts as a separate process using the current interpreter."""
    script = SRC / f"{name}.py"
    if not script.exists():
        raise SystemExit(f"Cannot find {script}")
    cmd = [PY, str(script), *args]
    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)

def ensure_data() -> Path:
    """Make sure we have a CSV to work with. If missing, create the sample."""
    data = ROOT / "data" / "raw" / "asrs_sample.csv"
    if data.exists():
        print(f"Data present: {data}")
    else:
        print("No data found. Generating sample dataset...")
        run_script("generate_sample_data")
    return data

def quickstart(threshold: float) -> None:
    """End to end: ensure data, EDA, train, batch predict, single demo predict."""
    ensure_data()
    run_script("eda")
    run_script("train")
    run_script(
        "predict_batch",
        "--in", "data/raw/asrs_sample.csv",
        "--out", "data/processed/predictions.csv",
        "--threshold", str(threshold),
    )
    run_script(
        "predict",
        "--text", "Crew reported fatigue and high workload during approach.",
        "--threshold", str(threshold),
    )
    print("\nQuickstart complete. Artifacts:")
    for p in [
        "models/baseline_logreg.joblib",
        "models/metrics.txt",
        "visuals/confusion_matrix.png",
        "visuals/pr_curve.png",
        "visuals/calibration_curve.png",
        "data/processed/predictions.csv",
    ]:
        print(" ", p)

def main():
    ap = argparse.ArgumentParser(description="Run the aviation safety ML pipeline.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_qs = sub.add_parser("quickstart", help="Run EDA, train, then batch and single predicts")
    p_qs.add_argument("--threshold", type=float, default=0.5)

    sub.add_parser("gen-sample", help="Create the tiny sample CSV")
    sub.add_parser("eda", help="Make EDA plots")
    sub.add_parser("train", help="Train and save model")

    p_one = sub.add_parser("predict-one", help="Score a single narrative")
    p_one.add_argument("--text", required=True)
    p_one.add_argument("--threshold", type=float, default=0.5)

    p_batch = sub.add_parser("predict-batch", help="Score a CSV of narratives")
    p_batch.add_argument("--in", dest="in_path", required=True)
    p_batch.add_argument("--out", dest="out_path", required=True)
    p_batch.add_argument("--threshold", type=float, default=0.5)

    args = ap.parse_args()

    if args.cmd == "quickstart":
        quickstart(args.threshold)
    elif args.cmd == "gen-sample":
        run_script("generate_sample_data")
    elif args.cmd == "eda":
        ensure_data()
        run_script("eda")
    elif args.cmd == "train":
        ensure_data()
        run_script("train")
    elif args.cmd == "predict-one":
        run_script("predict", "--text", args.text, "--threshold", str(args.threshold))
    elif args.cmd == "predict-batch":
        run_script(
            "predict_batch",
            "--in", args.in_path,
            "--out", args.out_path,
            "--threshold", str(args.threshold),
        )

if __name__ == "__main__":
    main()
