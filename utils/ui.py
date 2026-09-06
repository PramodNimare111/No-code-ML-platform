"""
Small shared UI helpers so every page looks/feels consistent — a lightweight
stand-in for the original React project's <Layout> component (sidebar
status panel, operations log, custom styling).
"""

import streamlit as st

CUSTOM_CSS = """
<style>
    .stApp { background-color: #0F1117; }
    div[data-testid="stMetricValue"] { color: #4F8EF7; }
    .op-log-item {
        background: #161A23; border-left: 3px solid #4F8EF7;
        padding: 6px 10px; margin-bottom: 6px; border-radius: 4px; font-size: 0.85rem;
    }
    .status-badge {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.78rem; font-weight: 600; margin-right: 6px;
    }
    .status-ok { background: #123524; color: #2DCE89; }
    .status-missing { background: #3a1f1f; color: #F75F4F; }
</style>
"""


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def sidebar_status():
    """Mirrors the original sidebar's dataset/model status + operations log."""
    st.sidebar.markdown("### 📌 Session Status")

    has_df = st.session_state.get("df") is not None
    has_model = st.session_state.get("model") is not None

    df_badge = '<span class="status-badge status-ok">✓ Dataset loaded</span>' if has_df \
        else '<span class="status-badge status-missing">✗ No dataset</span>'
    model_badge = '<span class="status-badge status-ok">✓ Model trained</span>' if has_model \
        else '<span class="status-badge status-missing">✗ No model</span>'
    st.sidebar.markdown(df_badge, unsafe_allow_html=True)
    st.sidebar.markdown(model_badge, unsafe_allow_html=True)

    if has_df:
        df = st.session_state.df
        st.sidebar.caption(f"**{st.session_state.get('filename', 'dataset')}** — {df.shape[0]} rows × {df.shape[1]} cols")
    if has_model:
        st.sidebar.caption(f"**{st.session_state.get('model_name')}** ({st.session_state.get('task')})")

    log = st.session_state.get("operations_log", [])
    if log:
        with st.sidebar.expander(f"🧹 Operations Log ({len(log)})", expanded=False):
            for i, msg in enumerate(reversed(log[-15:]), 1):
                st.markdown(f'<div class="op-log-item">{msg}</div>', unsafe_allow_html=True)

    st.sidebar.divider()
    st.sidebar.caption("ML Platform · Streamlit Edition")


def require_dataset():
    """Call at the top of Clean/Train/Predict/Plots pages — stops the page
    early with a friendly message if no dataset has been uploaded yet."""
    if st.session_state.get("df") is None:
        st.warning("⚠️ No dataset loaded yet. Head to the **Upload** page first.")
        st.stop()


def require_model():
    if st.session_state.get("model") is None:
        st.warning("⚠️ No trained model yet. Head to the **Train** page first.")
        st.stop()
