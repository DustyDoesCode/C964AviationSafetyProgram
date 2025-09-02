"""
Exploratory data analysis for the narratives.

Reads:
  data/raw/asrs_sample.csv

Writes:
  visuals/class_balance.png       bar chart of label counts
  visuals/length_hist.png         histogram of narrative character lengths

Usage:
  python src/eda.py
"""


from pathlib import Path
import os
import matplotlib.pyplot as plt
from utils import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "asrs_sample.csv"
OUT_DIR = ROOT / "visuals"

def main():
    print(f"Reading dataset from: {DATA}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA.exists():
        # optional: generate a sample if missing
        from generate_sample_data import main as gen
        gen(out_csv=DATA)

    df = load_dataset(str(DATA))

    ax = df["hf_label"].value_counts().sort_index().rename(index={0:"Not HF",1:"HF"}).plot(kind="bar")
    ax.set_xlabel("Class"); ax.set_ylabel("Count"); ax.set_title("Class Balance")
    plt.tight_layout(); plt.savefig(OUT_DIR / "class_balance.png"); plt.close()

    ax = df["n_char"].plot(kind="hist", bins=20)
    ax.set_xlabel("Characters"); ax.set_title("Narrative Length Distribution")
    plt.tight_layout(); plt.savefig(OUT_DIR / "length_hist.png"); plt.close()

if __name__ == "__main__":
    main()
