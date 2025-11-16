# app.py
import io

import joblib
import pandas as pd
import streamlit as st

from inference import preprocess_for_inference


@st.cache_resource
def load_artifact(path="artifacts/fraud_model.joblib"):
    artifact = joblib.load(path)
    return artifact["model"], artifact["encoders"], artifact["features"]


st.title("💳 Fraud Detection Demo")

st.write(
    "Upload a CSV file of transactions (same structure as training data, "
    "but `TX_FRAUD` column is optional). The app will predict whether each "
    "transaction is fraudulent or not."
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    try:
        model, encoders, feature_names = load_artifact()
    except Exception as e:
        st.error(f"Could not load model artifact: {e}")
        st.stop()

    try:
        X, _ = preprocess_for_inference(df, encoders, feature_names)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    with st.spinner("Running predictions..."):
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

    df_out = df.copy()
    df_out["FRAUD_PROB"] = probs
    df_out["PRED_FRAUD"] = preds

    st.subheader("Predictions")
    st.dataframe(df_out.head(20))

    st.download_button(
        "Download full predictions as CSV",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="fraud_predictions.csv",
        mime="text/csv",
    )
