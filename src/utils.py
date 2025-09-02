"""
Small helpers used across the project.

Functions:
  clean_text(s: str) -> str
      Lowercase, collapse whitespace, and trim. Safe on non-strings.

  load_dataset(path: str) -> pandas.DataFrame
      Read a CSV that has 'narrative' and 'hf_label', apply clean_text,
      and add a simple length column 'n_char'.
"""


import re
import pandas as pd

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
