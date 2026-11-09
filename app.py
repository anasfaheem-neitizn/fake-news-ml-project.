"""Streamlit app. Chalane ke liye:  streamlit run app.py
Pehle train.py chalana zaroori hai taaki model.pkl aur tfidf.pkl ban jayein."""
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from utils import clean_text, strip_source

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

MODEL_PATH, TFIDF_PATH = Path("model.pkl"), Path("tfidf.pkl")

SAMPLES = {
    "Sample 1 (wire-style)": "The city council approved a road repair budget of 4.2 million on Tuesday, "
                             "officials said in a statement. Work will begin in March.",
    "Sample 2 (sensational)": "SHOCKING secret miracle cure THEY don't want you to know about! "
                              "Share before it gets deleted!!!",
}


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(TFIDF_PATH)


def real_probability(model, X) -> float:
    """Model ke hisaab se 'Real' hone ki probability (0 se 1)."""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 1])
    score = float(model.decision_function(X)[0])
    return 1.0 / (1.0 + np.exp(-score))


st.title("📰 Fake News Detector")
st.caption("TF-IDF + Machine Learning se news ko Real ya Fake classify karta hai.")

if not (MODEL_PATH.exists() and TFIDF_PATH.exists()):
    st.error("model.pkl ya tfidf.pkl nahi mili. Pehle terminal mein `python train.py --data data` chalao.")
    st.stop()

model, tfidf = load_artifacts()

choice = st.selectbox("Sample chuno (optional)", ["Apna text likhunga"] + list(SAMPLES))
default_text = SAMPLES.get(choice, "")
text = st.text_area("News text yahan paste karo", value=default_text, height=220)

if st.button("Check karo", type="primary"):
    if len(text.split()) < 5:
        st.warning("Kam se kam 5 shabd ka text daalo.")
    else:
        cleaned = clean_text(strip_source(text))
        p_real = real_probability(model, tfidf.transform([cleaned]))
        is_real = p_real >= 0.5
        confidence = p_real if is_real else 1 - p_real

        if is_real:
            st.success(f"✅ Real news lag rahi hai ({confidence:.0%} confidence)")
        else:
            st.error(f"🚩 Fake news lag rahi hai ({confidence:.0%} confidence)")
        st.progress(float(p_real), text=f"Real hone ki probability: {p_real:.0%}")
        st.caption("Ye ek ML model ka andaza hai, final saboot nahi. Khabar ko hamesha dusre source se verify karo.")
