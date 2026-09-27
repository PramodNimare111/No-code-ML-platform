import streamlit as st
import numpy as np
from utils.state import init_state, push_snapshot, undo, reset_to_raw
from utils.ui import inject_css, sidebar_status, require_dataset
from utils.cleaning import apply_operation, OPERATION_GROUPS

st.set_page_config(page_title="Clean — ML Platform", page_icon="", layout="wide")
init_state()
inject_css()
sidebar_status()
require_dataset()

st.title("🧹 Clean Your Data")
st.caption("Apply cleaning operations. Every step is logged and can be undone.")

df = st.session_state.df

# ---------------------------------------------------------------------------
# Undo / Reset controls
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("↩️ Undo last operation", disabled=not st.session_state.snapshots, use_container_width=True):
        if undo():
            st.success("Undone.")
            st.rerun()
with col2:
    if st.button("🔄 Reset to original", use_container_width=True):
        reset_to_raw()
        st.success("Reset to the originally uploaded data.")
        st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Operation picker, grouped exactly like the original Clean.jsx UI
# ---------------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

group_name = st.selectbox("Category", list(OPERATION_GROUPS.keys()))
op_options = OPERATION_GROUPS[group_name]
op_labels = [label for _, label in op_options]
op_choice_label = st.selectbox("Operation", op_labels)
op_key = dict((label, key) for key, label in op_options)[op_choice_label]

st.markdown(f"**Configure:** {op_choice_label}")
opts = {}

# ---- Dynamic options form per operation ----
if op_key == "drop_missing_rows":
    opts["how"] = st.radio("Drop rows where...", ["any", "all"], horizontal=True,
                            help="'any' = drop if ANY column is missing, 'all' = drop only if ALL columns are missing")

elif op_key == "drop_missing_cols":
    opts["threshold"] = st.slider("Missing % threshold", 0, 100, 50)

elif op_key in ("fill_mean", "fill_median"):
    opts["columns"] = st.multiselect("Columns (numeric)", numeric_cols, default=numeric_cols)

elif op_key in ("fill_mode", "fill_ffill", "fill_bfill"):
    opts["columns"] = st.multiselect("Columns", all_cols, default=all_cols)

elif op_key == "fill_constant":
    opts["columns"] = st.multiselect("Columns", all_cols, default=all_cols)
    opts["value"] = st.text_input("Fill value", "0")

elif op_key == "remove_outliers" or op_key == "cap_outliers":
    opts["columns"] = st.multiselect("Columns (numeric)", numeric_cols, default=numeric_cols)
    opts["mult"] = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)

elif op_key == "drop_columns":
    opts["columns"] = st.multiselect("Columns to drop", all_cols)

elif op_key == "rename_columns":
    st.caption("Enter one `old_name -> new_name` mapping per line")
    text = st.text_area("Renames", placeholder="old_col -> new_col")
    mapping = {}
    for line in text.splitlines():
        if "->" in line:
            old, new = line.split("->", 1)
            mapping[old.strip()] = new.strip()
    opts["mapping"] = mapping

elif op_key in ("to_numeric", "to_datetime", "to_category", "to_string"):
    opts["columns"] = st.multiselect("Columns", all_cols)

elif op_key == "label_encode":
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    opts["columns"] = st.multiselect("Categorical columns", cat_cols)

elif op_key == "onehot_encode":
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    opts["columns"] = st.multiselect("Categorical columns", cat_cols)
    opts["drop_first"] = st.checkbox("Drop first category (avoid dummy trap)", value=False)

elif op_key in ("standard_scale", "minmax_scale"):
    opts["columns"] = st.multiselect("Columns (numeric)", numeric_cols, default=numeric_cols)
    if op_key == "minmax_scale":
        c1, c2 = st.columns(2)
        opts["min"] = c1.number_input("Range min", value=0.0)
        opts["max"] = c2.number_input("Range max", value=1.0)

st.write("")
if st.button("✅ Apply Operation", type="primary"):
    try:
        push_snapshot()
        new_df, message = apply_operation(df, op_key, opts)
        st.session_state.df = new_df
        st.session_state.operations_log.append(message)
        st.success(message)
        st.rerun()
    except Exception as e:
        # undo the snapshot push since the operation failed
        if st.session_state.snapshots:
            st.session_state.snapshots.pop()
        st.error(f"Operation failed: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Live preview + operations log
# ---------------------------------------------------------------------------
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(f"**Live Data Preview** — {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(15), use_container_width=True)
with c2:
    st.markdown(f"**Operations Log** ({len(st.session_state.operations_log)})")
    if st.session_state.operations_log:
        for i, msg in enumerate(reversed(st.session_state.operations_log), 1):
            st.markdown(f"{len(st.session_state.operations_log) - i + 1}. {msg}")
    else:
        st.caption("No operations applied yet.")
