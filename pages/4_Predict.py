import streamlit as st
import pandas as pd
from utils.state import init_state
from utils.ui import inject_css, sidebar_status, require_dataset, require_model
from utils import prediction as PR
from utils.data_io import load_dataframe

st.set_page_config(page_title="Predict — ML Platform", page_icon="🔮", layout="wide")
init_state()
inject_css()
sidebar_status()
require_dataset()
require_model()

st.title("🔮 Predict")
st.caption(f"Using **{st.session_state.model_name}** trained to predict **{st.session_state.target_col}**")

tab_single, tab_batch = st.tabs(["🎯 Single Prediction", "📦 Batch Prediction"])

# ---------------------------------------------------------------------------
# SINGLE
# ---------------------------------------------------------------------------
with tab_single:
    st.markdown("Enter values for each feature:")
    df = st.session_state.df
    feature_cols = st.session_state.feature_cols

    input_values = {}
    n_cols = 3
    cols = st.columns(n_cols)
    for i, feat in enumerate(feature_cols):
        col = cols[i % n_cols]
        default_val = float(df[feat].median()) if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]) else 0.0
        input_values[feat] = col.number_input(feat, value=default_val, key=f"single_{feat}")

    if st.button("🔮 Predict", type="primary"):
        try:
            result = PR.predict_single(
                st.session_state.model, feature_cols, st.session_state.task,
                input_values, st.session_state.label_classes,
            )
            st.success(f"### Prediction: `{result['prediction']}`")
            if result["confidence"] is not None:
                st.metric("Confidence", f"{result['confidence']}%")
            if result["probabilities"]:
                st.markdown("**Class probabilities:**")
                proba_df = pd.DataFrame(
                    list(result["probabilities"].items()), columns=["Class", "Probability"]
                ).sort_values("Probability", ascending=False)
                st.bar_chart(proba_df.set_index("Class"))

            # log to prediction history (capped at 50, mirroring the original)
            st.session_state.prediction_history.append({**input_values, "prediction": result["prediction"]})
            st.session_state.prediction_history = st.session_state.prediction_history[-50:]
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    if st.session_state.prediction_history:
        with st.expander(f"📜 Prediction History ({len(st.session_state.prediction_history)})"):
            st.dataframe(pd.DataFrame(st.session_state.prediction_history), use_container_width=True)

# ---------------------------------------------------------------------------
# BATCH
# ---------------------------------------------------------------------------
with tab_batch:
    st.markdown(f"Upload a CSV containing at least these columns: `{', '.join(st.session_state.feature_cols)}`")
    batch_file = st.file_uploader("Batch CSV", type=["csv", "xlsx", "xls"], key="batch_uploader")

    if batch_file is not None:
        try:
            batch_df = load_dataframe(batch_file)
            st.dataframe(batch_df.head(5), use_container_width=True)

            if st.button("📦 Run Batch Prediction", type="primary"):
                with st.spinner("Predicting..."):
                    out_df, summary = PR.predict_batch(
                        st.session_state.model, st.session_state.feature_cols, st.session_state.task,
                        batch_df, st.session_state.target_col, st.session_state.label_classes,
                    )
                    st.session_state.batch_pred_df = out_df
                    st.success(f"Predicted {len(out_df)} rows.")

                    st.markdown("**Summary**")
                    st.json(summary)

                    st.markdown("**Preview**")
                    st.dataframe(out_df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

    if st.session_state.batch_pred_df is not None:
        csv_bytes = st.session_state.batch_pred_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions CSV", csv_bytes, "predictions.csv", "text/csv")
