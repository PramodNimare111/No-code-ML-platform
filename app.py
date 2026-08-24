"""
ML Platform — Streamlit Edition
================================
A single-process rewrite of the original MERN + Python-bridge project.
Everything that used to require Node.js, Express, React, and a subprocess
IPC bridge now happens directly in one Python process, with Streamlit
handling both the UI and the "web server" part.

Run with:  streamlit run app.py
"""

import streamlit as st
from utils.state import init_state, has_dataset, has_model
from utils.ui import inject_css, sidebar_status

st.set_page_config(
    page_title="ML Platform — Streamlit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()
inject_css()
sidebar_status()

st.title("🤖 ML Platform")
st.caption("A no-code, end-to-end machine learning workflow — upload, clean, train, predict, visualise.")

st.markdown("""
Welcome! This app walks you through a complete classical machine-learning
pipeline without writing a single line of Python yourself. Use the pages in
the sidebar to move through each step:
""")

col1, col2, col3, col4, col5 = st.columns(5)
steps = [
    ("📁", "1. Upload", "Upload a CSV/Excel file and get an instant EDA report."),
    ("🧹", "2. Clean", "Fix missing values, outliers, encoding, and scaling."),
    ("🤖", "3. Train", "Train one of 13 models, tune hyper-parameters, cross-validate, or compare."),
    ("🔮", "4. Predict", "Predict a single row or a whole batch CSV with your trained model."),
    ("📊", "5. Plots", "Visualise your data and your model's performance."),
]
for col, (icon, title, desc) in zip([col1, col2, col3, col4, col5], steps):
    with col:
        st.markdown(f"### {icon} {title}")
        st.caption(desc)

st.divider()

status_col1, status_col2 = st.columns(2)
with status_col1:
    if has_dataset():
        df = st.session_state.df
        st.success(f"Dataset loaded: **{st.session_state.filename}** ({df.shape[0]} rows × {df.shape[1]} cols)")
    else:
        st.info("!!! No dataset loaded yet — start with the **Upload** page in the sidebar.")
with status_col2:
    if has_model():
        st.success(f"Model trained: **{st.session_state.model_name}** ({st.session_state.task})")
    else:
        st.info("!!! No model trained yet — visit **Train** once your data is cleaned.")

with st.expander("!!! About this rewrite"):
    st.markdown("""
This is a Streamlit port of a project originally built with **MongoDB, Express,
React, and Node.js**, which called out to a separate Python subprocess for every
ML operation via a custom JSON-over-stdio bridge.

Because Streamlit apps are plain Python, that entire bridge — and the Node/React
layer around it — is no longer needed. `st.session_state` now plays the role the
Express session used to play (remembering the current dataset and trained model
between interactions), except it holds **real Python objects** (an actual
DataFrame, an actual fitted scikit-learn model) instead of CSV strings and
base64-encoded pickles.
""")
