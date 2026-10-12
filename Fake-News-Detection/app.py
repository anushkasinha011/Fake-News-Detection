"""app.py - Streamlit web app for Fake News Detection.  Run: streamlit run app.py"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from utils import clean_text

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_PATH = Path(__file__).parent / "models" / "fake_news_pipeline.joblib"
MIN_WORDS = 15

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection")
st.write("Paste a news headline and article below to see what the model predicts.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

text = st.text_area("News headline / article", height=250,
                    placeholder="Paste the full news text here...")

if st.button("Check news", type="primary"):
    cleaned = clean_text(text)
    if not text.strip():
        st.warning("Please enter some text.")
    elif len(cleaned.split()) < MIN_WORDS:
        st.warning(f"Text is too short. Please enter at least {MIN_WORDS} meaningful words "
                   "(a full headline plus a few sentences works best).")
    else:
        try:
            pred = int(model.predict([cleaned])[0])
            clf = model.named_steps["clf"]
            if hasattr(clf, "predict_proba"):
                conf = float(model.predict_proba([cleaned])[0][pred])
            else:  # Passive Aggressive has no probabilities; squash the score instead
                score = float(model.decision_function([cleaned])[0])
                conf = float(1 / (1 + np.exp(-abs(score))))
            if pred == 1:
                st.success(f"✅ Likely REAL news (model confidence ≈ {conf:.0%})")
            else:
                st.error(f"🚨 Likely FAKE news (model confidence ≈ {conf:.0%})")
            st.progress(min(max(conf, 0.0), 1.0))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.divider()
st.caption("⚠️ This prediction is based on writing patterns learned from the training "
           "dataset. It does NOT verify facts. Always check trusted sources before "
           "believing or sharing news.")
