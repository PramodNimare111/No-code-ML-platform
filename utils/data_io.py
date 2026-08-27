"""
File loading — ported from the file-reading portion of `handle_load()`.

In the original project this had to base64-decode bytes that traveled
through Node -> JSON -> Python. Streamlit's `st.file_uploader` already
hands us a file-like buffer directly, so that whole detour disappears.
"""

import pandas as pd


def load_dataframe(uploaded_file) -> pd.DataFrame:
    """uploaded_file is a Streamlit UploadedFile (file-like + .name)."""
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type — please upload a .csv, .xlsx, or .xls file")
