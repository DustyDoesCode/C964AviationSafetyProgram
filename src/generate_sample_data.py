"""
Generate a tiny synthetic ASRS-like dataset for quick testing.

Creates:
  data/raw/asrs_sample.csv  with columns [report_id, narrative, hf_label]

Usage:
  python src/generate_sample_data.py
Notes:
  - hf_label = 1 means the narrative mentions human-factors themes.
  - This is just seed data so we can practice the pipeline before using real ASRS.
"""


from pathlib import Path
import random, csv

HF = [
    "During approach the crew reported fatigue after extended duty time and miscommunication with ATC.",
    "Pilot experienced high workload and distraction while programming the FMS leading to altitude deviation.",
    "Captain and FO had a misunderstanding about checklist items which delayed configuring for landing.",
    "Ground crew reported time pressure and coordination issues causing a near pushback incident.",
    "ATC phraseology was unclear and the crew requested repeat which increased workload in busy airspace.",
    "First officer was task saturated during departure and missed a callout."
]
NONHF = [
    "During taxi a small piece of FOD was observed near the centerline and removed without further issue.",
    "Maintenance discovered a worn tire during routine inspection and replaced it per manual.",
    "Bird strike noted on left wing leading edge with minor damage no injuries reported.",
    "Autopilot disconnected due to mild turbulence system performed as expected.",
    "Weather required a short hold vectors were given and approach continued uneventfully.",
    "Runway lights were briefly dim no operational impact noted."
]

ROOT = Path(__file__).resolve().parents[1]  # project root

def main(n_rows=400, out_csv: Path | str = ROOT / "data" / "raw" / "asrs_sample.csv"):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["report_id", "narrative", "hf_label"])
        w.writeheader()
        for i in range(n_rows):
            if i % 2 == 0:
                text = random.choice(HF); label = 1
            else:
                text = random.choice(NONHF); label = 0
            w.writerow({"report_id": 1000 + i, "narrative": text, "hf_label": label})
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()
