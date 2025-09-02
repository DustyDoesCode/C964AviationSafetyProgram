import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "baseline_logreg.joblib"
VIS_DIR = ROOT / "visuals"

st.set_page_config(page_title="Aviation Safety Triage", page_icon="✈️")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python main.py train` first."
        )
    pipe = joblib.load(MODEL_PATH)
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["logreg"]
    vocab = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]
    return pipe, vocab, coefs

def score_text(pipe, text: str) -> float:
    prob = float(pipe.predict_proba([text])[:, 1][0])
    return prob

def top_contributions(pipe, vocab: np.ndarray, coefs: np.ndarray, text: str, k: int = 10):
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    X = vec.transform([text])            # 1 x V sparse row
    # Contribution per feature is term tfidf value times model weight
    contrib = (X.multiply(coefs)).toarray()[0]  # shape (V,)
    # Positive pushes toward HF, negative away
    pos_idx = np.argsort(-contrib)[:k]
    neg_idx = np.argsort(contrib)[:k]
    pos = [(vocab[i], float(contrib[i])) for i in pos_idx if contrib[i] > 0]
    neg = [(vocab[i], float(contrib[i])) for i in neg_idx if contrib[i] < 0]
    return pos, neg

st.title("Aviation Safety Narrative Triage")
st.write("Paste a narrative and get a human-factors risk score.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.05)
    st.caption("Label = 1 if probability ≥ threshold, else 0")

    st.divider()
    st.subheader("Model status")
    if MODEL_PATH.exists():
        st.success("Model found")
    else:
        st.error("Model not found. Run training from terminal:")
        st.code("python main.py train", language="bash")

tab_one, tab_batch, tab_viz = st.tabs(["Single narrative", "Batch CSV", "Visuals"])

with tab_one:
    text = st.text_area(
        "Narrative",
        height=160,
        placeholder="Example: During approach the crew reported fatigue after extended duty time and miscommunication with ATC.",
    )
    if st.button("Score narrative", type="primary") and text.strip():
        try:
            pipe, vocab, coefs = load_model()
            prob = score_text(pipe, text)
            label = int(prob >= threshold)
            st.metric(label="Human Factor probability", value=f"{prob:.3f}")
            st.write(f"Predicted label at threshold {threshold:.2f}: **{label}**")

            with st.expander("Why this score? Top terms"):
                pos, neg = top_contributions(pipe, vocab, coefs, text, k=8)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Push toward HF**")
                    if pos:
                        for term, val in pos:
                            st.write(f"{term}: {val:+.4f}")
                    else:
                        st.write("No strong positive terms.")
                with c2:
                    st.markdown("**Push away from HF**")
                    if neg:
                        for term, val in neg:
                            st.write(f"{term}: {val:+.4f}")
                    else:
                        st.write("No strong negative terms.")
        except FileNotFoundError as e:
            st.error(str(e))

with tab_batch:
    st.write("Upload a CSV with a `narrative` column. Optional `hf_label` is kept as `hf_label_true`.")
    file = st.file_uploader("CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        if "narrative" not in df.columns:
            st.error("CSV must contain a 'narrative' column.")
        else:
            try:
                pipe, _, _ = load_model()
                probs = pipe.predict_proba(df["narrative"])[:, 1]
                labels = (probs >= threshold).astype(int)
                out = df.copy()
                if "hf_label" in out.columns:
                    out = out.rename(columns={"hf_label": "hf_label_true"})
                out["hf_prob_pred"] = probs
                out["hf_label_pred"] = labels
                st.dataframe(out.head(20))
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except FileNotFoundError as e:
                st.error(str(e))

with tab_viz:
    st.write("Training visuals if available:")
    imgs = [
        ("Confusion matrix", VIS_DIR / "confusion_matrix.png"),
        ("PR curve", VIS_DIR / "pr_curve.png"),
        ("Calibration curve", VIS_DIR / "calibration_curve.png"),
    ]
    any_found = False
    for title, path in imgs:
        if path.exists():
            any_found = True
            st.markdown(f"**{title}**")
            st.image(str(path))
    if not any_found:
        st.info("No images found. Run training first to generate visuals.")
