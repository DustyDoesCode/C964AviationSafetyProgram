"""
Small helpers used across the project.

Functions:
  clean_text(s: str) -> str
      Lowercase, collapse whitespace, and trim. Safe on non-strings.

  load_dataset(path: str) -> pandas.DataFrame
      Read a CSV that has 'narrative' and 'hf_label', apply clean_text,
      and add a simple length column 'n_char'.
"""


from pathlib import Path
import re
import pandas as pd

def load_asrs_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    # Try reading when the real header is on line 2
    try:
        df = pd.read_csv(p, header=1, engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(p, engine="python", on_bad_lines="skip")

    # Normalize and deduplicate column names (keep original text for display)
    seen = {}
    newcols = []
    for c in df.columns:
        name = str(c).strip()
        key = name.lower()
        seen[key] = seen.get(key, 0) + 1
        newcols.append(name if seen[key] == 1 else f"{name}_{seen[key]}")
    df.columns = newcols

    # Standardize narrative text column
    candidates = [
        c for c in df.columns
        if c.lower() in ("narrative", "callback", "synopsis", "report text", "callback conversation", "narrative text")
    ]

    if "narrative" not in df.columns:
        if candidates:
            df["narrative"] = df[candidates[0]].astype(str)
        else:
            raise ValueError("Expected a text column: Narrative, Callback, Synopsis, or Report Text.")

    df["narrative"] = (
        df["narrative"].astype(str)
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df


def infer_hf_label(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Primary problem columns that explicitly say "Human Factors"
    primary_like = [c for c in df.columns if ("primary" in c.lower() and "problem" in c.lower())]
    for c in primary_like:
        vals = df[c].astype(str).str.lower()
        mask = vals.str.contains("human") & vals.str.contains("factor")
        if mask.any():
            out = df.copy()
            out["hf_label"] = mask.astype(int)
            return out

    # 2) Generic human-factors signal columns
    name_patterns = [
        r"human\s*factors",
        r"\bhf\b",  # columns like "HF: Fatigue"
        r"crew\s*resource\s*management",
        r"\bfatigue\b",
        r"\bcommunication\b",
        r"\bdistraction\b",
        r"\bworkload\b",
        r"\bprocedure\b|\bprocedural\b",
        r"\btraining\b",
        r"\bcomplacency\b",
    ]
    signal_cols = [c for c in df.columns if any(re.search(p, c.lower()) for p in name_patterns)]

    if signal_cols:
        out = df.copy()
        label = pd.Series(False, index=out.index)
        NEG = {"", "nan", "none", "n/a", "na", "no", "false", "0", "."}
        POS = {"y", "yes", "true", "1"}
        for c in signal_cols:
            s = out[c].astype(str).str.strip().str.lower()
            signal = s.isin(POS) | (~s.isin(NEG))
            label = label | signal
        out["hf_label"] = label.astype(int)
        return out

    raise ValueError(
        "Could not infer `hf_label` from this export. "
        "Please supply a mapping or share a few column names so we can extend the patterns."
    )

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "narrative" not in df.columns or "hf_label" not in df.columns:
        raise ValueError("CSV must contain 'narrative' and 'hf_label' columns.")
    df["narrative"] = df["narrative"].apply(clean_text)
    df["n_char"] = df["narrative"].str.len()
    return df
