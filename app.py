import streamlit as st
import pandas as pd

from core.rules import rule_based_check
from core.ml import ml_anomaly_check
from core.dl import dl_anomaly_score
from core.risk import compute_verdict

st.set_page_config(page_title="PDPE", layout="centered")

st.title("🫀 Physiological Data Plausibility Engine (PDPE)")
st.caption("Pre-training validation for healthcare machine learning")

st.header("🔹 Single Patient Validation")

age = st.slider("Age", 0, 120, 55)
bp = st.slider("Resting Blood Pressure", 50, 300, 120)
chol = st.slider("Cholesterol", 100, 500, 200)
hr = st.slider("Max Heart Rate", 50, 300, 150)
oldpeak = st.slider("Oldpeak", 0.0, 10.0, 1.5)

if st.button("Validate Data"):
    row = {
        "age": age,
        "resting_bp": bp,
        "cholesterol": chol,
        "max_heart_rate": hr,
        "oldpeak": oldpeak
    }

    rule_v, issues = rule_based_check(row)
    ml_v = ml_anomaly_check(row)
    dl_score = dl_anomaly_score(row)

    verdict, explanation = compute_verdict(rule_v, ml_v, dl_score)

    st.subheader("📋 Clinical Validation Report")
    st.write("**Rule-based Check:**", rule_v)
    st.write("**ML Check:**", ml_v)
    st.write("**DL Reconstruction Score:**", round(dl_score, 4))

    if issues:
        st.warning("Clinical Issues Detected:")
        for i in issues:
            st.write("-", i)

    st.markdown("---")
    st.success(verdict)
    st.info(explanation)

st.header("🔹 CSV Dataset Validation")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required = ["age", "resting_bp", "cholesterol", "max_heart_rate", "oldpeak"]
    if any(c not in df.columns for c in required):
        st.error("Missing required columns")
    else:
        verdicts = []
        
        # Add progress bar for CSV processing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_rows = len(df)
        
        for idx, r in df.iterrows():
            # Update progress
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing row {idx + 1}/{total_rows}...")
            
            row = r[required].to_dict()
            rule_v, _ = rule_based_check(row)
            ml_v = ml_anomaly_check(row)
            dl_score = dl_anomaly_score(row)
            verdict, _ = compute_verdict(rule_v, ml_v, dl_score)
            verdicts.append(verdict)

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        df["PDPE_Verdict"] = verdicts
        st.dataframe(df)

        unsafe_pct = sum("IMPLAUSIBLE" in v for v in verdicts) / len(verdicts) * 100

        st.markdown("---")
        if unsafe_pct > 20:
            st.error("❌ Dataset NOT suitable for training")
        else:
            st.success("✅ Dataset suitable for training")
