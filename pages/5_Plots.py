import streamlit as st
import numpy as np
from utils.state import init_state
from utils.ui import inject_css, sidebar_status, require_dataset
from utils import plotting as P

st.set_page_config(page_title="Plots — ML Platform", page_icon="📊", layout="wide")
init_state()
inject_css()
sidebar_status()
require_dataset()

st.title("📊 Visualisations")
df = st.session_state.df

tab_data, tab_model = st.tabs(["📈 Data Plots", "🎯 Model Plots"])

# ---------------------------------------------------------------------------
# DATA PLOTS
# ---------------------------------------------------------------------------
with tab_data:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    kind = st.selectbox("Chart type", [
        "Distribution", "Box Plot", "Bar Counts (categorical)", "Correlation Heatmap", "Missing Values"
    ])

    if kind == "Distribution":
        col = st.selectbox("Column (numeric)", numeric_cols)
        if col:
            st.pyplot(P.plot_distribution(df, col))

    elif kind == "Box Plot":
        col = st.selectbox("Column (numeric)", numeric_cols)
        group_col = st.selectbox("Group by (optional)", ["(none)"] + [c for c in all_cols if c not in numeric_cols])
        if col:
            st.pyplot(P.plot_box(df, col, None if group_col == "(none)" else group_col))

    elif kind == "Bar Counts (categorical)":
        cat_cols = [c for c in all_cols if c not in numeric_cols] or all_cols
        col = st.selectbox("Column", cat_cols)
        top_n = st.slider("Top N categories", 5, 30, 15)
        if col:
            st.pyplot(P.plot_bar_counts(df, col, top_n))

    elif kind == "Correlation Heatmap":
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            st.pyplot(P.plot_heatmap(df))

    elif kind == "Missing Values":
        st.pyplot(P.plot_missing(df))

# ---------------------------------------------------------------------------
# MODEL PLOTS
# ---------------------------------------------------------------------------
with tab_model:
    if st.session_state.model is None:
        st.warning("⚠️ Train a model first on the **Train** page to unlock these charts.")
    else:
        task = st.session_state.task
        options = ["Confusion Matrix", "Feature Importance", "ROC Curve"] if task == "classification" else \
                  ["Actual vs Predicted", "Residuals", "Feature Importance"]
        if st.session_state.comparison:
            options.append("Model Comparison")

        kind = st.selectbox("Chart type", options)

        if kind == "Confusion Matrix":
            st.pyplot(P.plot_confusion_matrix(st.session_state.y_test, st.session_state.y_pred, st.session_state.label_classes))

        elif kind == "Feature Importance":
            from utils.training import extract_importances
            importances = extract_importances(st.session_state.model, st.session_state.feature_cols)
            if importances is not None:
                st.pyplot(P.plot_feature_importance(importances))
            else:
                st.info("This model type doesn't expose feature importances (e.g. KNN, Naive Bayes, non-linear SVM).")

        elif kind == "ROC Curve":
            if st.session_state.y_proba is not None and len(set(st.session_state.y_test)) == 2:
                st.pyplot(P.plot_roc(st.session_state.y_test, st.session_state.y_proba))
            else:
                st.info("ROC curve is only available for binary classification models with probability output.")

        elif kind == "Actual vs Predicted":
            st.pyplot(P.plot_actual_vs_predicted(st.session_state.y_test, st.session_state.y_pred))

        elif kind == "Residuals":
            st.pyplot(P.plot_residuals(st.session_state.y_test, st.session_state.y_pred))

        elif kind == "Model Comparison":
            results_df_cols = [k for k in st.session_state.comparison[0].keys() if k not in ("Model", "error")]
            metric = st.selectbox("Metric", results_df_cols)
            st.pyplot(P.plot_comparison(st.session_state.comparison, metric))
