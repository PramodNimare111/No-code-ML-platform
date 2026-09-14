import streamlit as st
import numpy as np
import pandas as pd
from utils.state import init_state
from utils.ui import inject_css, sidebar_status, require_dataset
from utils import training as T
from utils import plotting as P

st.set_page_config(page_title="Train — ML Platform", page_icon="🤖", layout="wide")
init_state()
inject_css()
sidebar_status()
require_dataset()

st.title("🤖 Train a Model")
df = st.session_state.df
all_cols = df.columns.tolist()


def render_param_widgets(model_name, key_prefix):
    """Dynamically renders sliders/selects from PARAM_SCHEMA — mirrors the
    original React Train page building its hyper-parameter form from
    GET /train/models' schema instead of hard-coding a form per model."""
    schema = T.PARAM_SCHEMA.get(model_name, {})
    params = {}
    if not schema:
        st.caption("This model has no tunable hyper-parameters exposed.")
        return params
    for pname, spec in schema.items():
        widget_key = f"{key_prefix}_{model_name}_{pname}"
        if spec["type"] == "int":
            params[pname] = st.slider(pname, spec["min"], spec["max"], spec["default"], spec["step"],
                                       help=spec.get("help"), key=widget_key)
        elif spec["type"] == "float":
            params[pname] = st.slider(pname, spec["min"], spec["max"], spec["default"], spec["step"],
                                       help=spec.get("help"), key=widget_key)
        elif spec["type"] == "select":
            params[pname] = st.selectbox(pname, spec["options"],
                                          index=spec["options"].index(spec["default"]),
                                          help=spec.get("help"), key=widget_key)
    return params


# ---------------------------------------------------------------------------
# Task / Target / Features
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)
task = c1.radio("Task", ["classification", "regression"], horizontal=True)
target_col = c2.selectbox("Target column", all_cols)

if st.checkbox("💡 Suggest relevant features for this target"):
    with st.spinner("Scoring feature relevance..."):
        suggestions = T.suggest_features(df, target_col)
    st.dataframe(suggestions, use_container_width=True, hide_index=True)

feature_candidates = [c for c in all_cols if c != target_col]
feature_cols = st.multiselect("Feature columns", feature_candidates, default=feature_candidates)

col1, col2 = st.columns(2)
test_size = col1.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
seed = col2.number_input("Random seed", value=42, step=1)

st.divider()

registry = T.get_registry(task)
model_names = list(registry.keys())

tab_train, tab_cv, tab_compare = st.tabs(["🎯 Train a Single Model", "🔁 Cross-Validation", "⚖️ Compare Models"])

# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
with tab_train:
    model_name = st.selectbox("Model", model_names, key="train_model_select")
    with st.expander("Hyper-parameters", expanded=True):
        params = render_param_widgets(model_name, "train")

    if st.button("🚀 Train Model", type="primary"):
        if not feature_cols:
            st.error("Select at least one feature column.")
        else:
            with st.spinner(f"Training {model_name}..."):
                try:
                    result = T.train_model(df, feature_cols, target_col, task, model_name, params, test_size, seed)
                    st.session_state.model = result["model"]
                    st.session_state.model_name = model_name
                    st.session_state.task = task
                    st.session_state.feature_cols = feature_cols
                    st.session_state.target_col = target_col
                    st.session_state.label_classes = result["label_classes"]
                    st.session_state.X_test = result["X_test"]
                    st.session_state.y_test = result["y_test"]
                    st.session_state.y_pred = result["y_pred"]
                    st.session_state.y_proba = result["y_proba"]
                    st.session_state.comparison = None
                    st.success(f"✅ {model_name} trained successfully!")

                    st.markdown("### Metrics")
                    metric_cols = st.columns(len(result["metrics"]))
                    for col, (k, v) in zip(metric_cols, result["metrics"].items()):
                        col.metric(k, v)

                    viz_col1, viz_col2 = st.columns(2)
                    with viz_col1:
                        if task == "classification":
                            st.pyplot(P.plot_confusion_matrix(result["y_test"], result["y_pred"], result["label_classes"]))
                        else:
                            st.pyplot(P.plot_actual_vs_predicted(result["y_test"], result["y_pred"]))
                    with viz_col2:
                        if result["importances"] is not None:
                            st.pyplot(P.plot_feature_importance(result["importances"]))
                        else:
                            st.caption("This model type doesn't expose feature importances.")
                except Exception as e:
                    st.error(f"Training failed: {e}")

# ---------------------------------------------------------------------------
# CROSS-VALIDATION
# ---------------------------------------------------------------------------
with tab_cv:
    cv_model_name = st.selectbox("Model", model_names, key="cv_model_select")
    with st.expander("Hyper-parameters", expanded=True):
        cv_params = render_param_widgets(cv_model_name, "cv")
    k = st.slider("Number of folds (k)", 2, 10, 5)

    if st.button("🔁 Run Cross-Validation", type="primary"):
        if not feature_cols:
            st.error("Select at least one feature column.")
        else:
            with st.spinner(f"Running {k}-fold cross-validation..."):
                try:
                    cv_result = T.cross_validate(df, feature_cols, target_col, task, cv_model_name, cv_params, k, seed)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(f"Mean {cv_result['scoring']}", cv_result["mean"])
                    c2.metric("Std Dev", cv_result["std"])
                    c3.metric("Min", cv_result["min"])
                    c4.metric("Max", cv_result["max"])
                    fold_df = pd.DataFrame({"Fold": range(1, k + 1), "Score": cv_result["scores"]})
                    st.bar_chart(fold_df.set_index("Fold"))
                except Exception as e:
                    st.error(f"Cross-validation failed: {e}")

# ---------------------------------------------------------------------------
# COMPARE
# ---------------------------------------------------------------------------
with tab_compare:
    compare_choices = st.multiselect("Models to compare", model_names, default=model_names[:3])

    if st.button("⚖️ Compare Models", type="primary"):
        if not feature_cols:
            st.error("Select at least one feature column.")
        elif not compare_choices:
            st.error("Select at least 2 models to compare.")
        else:
            with st.spinner("Training all selected models..."):
                try:
                    # use each model's default hyper-parameters for a fair, simple comparison
                    params_map = {name: {} for name in compare_choices}
                    comp = T.compare_models(df, feature_cols, target_col, task, compare_choices, params_map, test_size, seed)
                    st.session_state.comparison = comp["results"]
                    results_df = pd.DataFrame(comp["results"])
                    st.dataframe(results_df, use_container_width=True, hide_index=True)

                    metric_options = [c for c in results_df.columns if c not in ("Model", "error")]
                    if metric_options:
                        chosen_metric = st.selectbox("Chart metric", metric_options)
                        st.pyplot(P.plot_comparison(comp["results"], chosen_metric))
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
