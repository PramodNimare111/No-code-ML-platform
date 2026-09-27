import streamlit as st
from utils.state import init_state, reset_model_state
from utils.ui import inject_css, sidebar_status
from utils.data_io import load_dataframe
from utils import eda

st.set_page_config(page_title="Upload — ML Platform", page_icon="", layout="wide")
st.set_page_config(page_title="Upload — ML Platform", page_icon="", layout="wide")
init_state()
inject_css()
sidebar_status()

st.title("Upload & Explore")
st.caption("Upload a CSV or Excel file to get an instant exploratory data analysis (EDA) report.")

uploaded_file = st.file_uploader("Drag and drop a file here, or click to browse", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Only re-process if it's a genuinely new file (avoid re-running on every rerun)
    if st.session_state.get("filename") != uploaded_file.name:
        with st.spinner("Reading file and computing EDA..."):
            try:
                df = load_dataframe(uploaded_file)
                st.session_state.df = df
                st.session_state.df_raw = df.copy()
                st.session_state.filename = uploaded_file.name
                st.session_state.operations_log = []
                st.session_state.snapshots = []
                reset_model_state()  # a new dataset invalidates any previously trained model
                st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Failed to load file: {e}")
                st.stop()

if st.session_state.df is None:
    st.info("Upload your file")
    st.info("Upload your file.")
    st.stop()

# ---------------------------------------------------------------------------
# EDA view — toggle between the current (possibly cleaned) and raw dataset
# ---------------------------------------------------------------------------
view_choice = st.radio("View EDA for:", ["Current (cleaned)", "Raw (original upload)"], horizontal=True)
active_df = st.session_state.df if view_choice.startswith("Current") else st.session_state.df_raw

report = eda.compute_full_eda(active_df)

tabs = st.tabs(["Overview", "Columns", "Numeric Stats", "Categorical", "Missing Values", "Correlations", "Data Preview"])

with tabs[0]:
    ov = report["overview"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", ov["rows"])
    c2.metric("Columns", ov["cols"])
    c3.metric("Missing %", f"{ov['missing_pct']}%")
    c4.metric("Duplicate Rows", ov["duplicates"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Numeric Columns", ov["numeric_cols"])
    c2.metric("Categorical Columns", ov["categorical_cols"])
    c3.metric("Memory Usage", f"{ov['memory_kb']} KB")

with tabs[1]:
    st.dataframe(report["column_summary"], use_container_width=True, hide_index=True)

with tabs[2]:
    if report["numeric_stats"].empty:
        st.info("No numeric columns found.")
    else:
        st.dataframe(report["numeric_stats"], use_container_width=True, hide_index=True)

with tabs[3]:
    if report["categorical_stats"].empty:
        st.info("No categorical columns found.")
    else:
        st.dataframe(report["categorical_stats"], use_container_width=True, hide_index=True)

with tabs[4]:
    if report["missing_table"].empty:
        st.success("No missing values in this dataset! 🎉")
    else:
        st.dataframe(report["missing_table"], use_container_width=True, hide_index=True)

with tabs[5]:
    if report["top_correlations"].empty:
        st.info("Need at least 2 numeric columns to compute correlations.")
    else:
        st.markdown("**Top correlated feature pairs**")
        st.dataframe(report["top_correlations"], use_container_width=True, hide_index=True)
        if report["correlation_matrix"] is not None:
            from utils.plotting import plot_heatmap
            st.pyplot(plot_heatmap(active_df))

with tabs[6]:
    st.markdown(f"Showing first 20 of {len(active_df)} rows")
    st.dataframe(active_df.head(20), use_container_width=True)
