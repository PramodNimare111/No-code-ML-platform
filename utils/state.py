"""
Central session-state management.

In the original MERN version, all of this lived in the Express `req.session`
object (because the Python engine was stateless and re-invoked per request).
In Streamlit, the whole app is one long-running Python process per user
session, so `st.session_state` plays exactly the same role — a dict-like
object that persists across reruns for a single browser tab/session.
"""

import streamlit as st

DEFAULTS = {
    "df": None,                 # current (possibly cleaned) DataFrame
    "df_raw": None,              # original, untouched DataFrame
    "filename": None,
    "snapshots": [],             # undo stack of previous DataFrames
    "operations_log": [],        # human-readable cleaning history

    "model": None,                # fitted sklearn estimator
    "model_name": None,
    "task": None,                 # 'classification' | 'regression'
    "feature_cols": [],
    "target_col": None,
    "label_classes": None,        # original class labels for classification (for display)

    "X_test": None,
    "y_test": None,
    "y_pred": None,
    "y_proba": None,

    "comparison": None,           # list of dicts from the last model comparison
    "prediction_history": [],     # log of single-row predictions (capped)

    "batch_pred_df": None,        # result of the last batch prediction
}

def init_state():
    """Call once at the top of every page — sets any missing keys to their default."""
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            # use a fresh copy for mutable defaults (lists)
            st.session_state[key] = default.copy() if isinstance(default, list) else default


def has_dataset() -> bool:
    return st.session_state.get("df") is not None


def has_model() -> bool:
    return st.session_state.get("model") is not None


def reset_model_state():
    """Called whenever a new file is uploaded — a model trained on the old
    dataset's columns is no longer valid."""
    for key in ["model", "model_name", "task", "feature_cols", "target_col",
                "label_classes", "X_test", "y_test", "y_pred", "y_proba",
                "comparison", "prediction_history", "batch_pred_df"]:
        st.session_state[key] = DEFAULTS[key].copy() if isinstance(DEFAULTS[key], list) else DEFAULTS[key]


def push_snapshot():
    """Save the current df onto the undo stack before mutating it."""
    if st.session_state.df is not None:
        st.session_state.snapshots.append(st.session_state.df.copy())
        # cap the undo stack so memory doesn't grow unbounded in a long session
        if len(st.session_state.snapshots) > 20:
            st.session_state.snapshots.pop(0)


def undo():
    if st.session_state.snapshots:
        st.session_state.df = st.session_state.snapshots.pop()
        if st.session_state.operations_log:
            st.session_state.operations_log.pop()
        return True
    return False


def reset_to_raw():
    if st.session_state.df_raw is not None:
        st.session_state.df = st.session_state.df_raw.copy()
        st.session_state.operations_log = []
        st.session_state.snapshots = []
