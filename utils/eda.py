"""
Exploratory Data Analysis helpers.

This is a direct port of the EDA logic that used to live in the Node
project's `ml_engine.py` (handle_load / handle_eda). Because Streamlit is a
single Python process, these functions now operate directly on a live
pandas DataFrame instead of a CSV string passed over a subprocess pipe.
"""

import numpy as np
import pandas as pd


def get_basic_overview(df: pd.DataFrame) -> dict:
    rows, cols = df.shape
    missing_total = int(df.isnull().sum().sum())
    missing_pct = round(100 * missing_total / (rows * cols), 2) if rows * cols else 0.0
    duplicates = int(df.duplicated().sum())
    numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
    datetime_cols = df.select_dtypes(include=["datetime64"]).shape[1]
    categorical_cols = cols - numeric_cols - datetime_cols
    memory_kb = round(df.memory_usage(deep=True).sum() / 1024, 2)
    return {
        "rows": rows, "cols": cols,
        "missing_total": missing_total, "missing_pct": missing_pct,
        "duplicates": duplicates,
        "duplicates_pct": round(100 * duplicates / rows, 2) if rows else 0.0,
        "numeric_cols": int(numeric_cols),
        "categorical_cols": int(categorical_cols),
        "datetime_cols": int(datetime_cols),
        "memory_kb": memory_kb,
    }


def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = int(s.notnull().sum())
        null = int(s.isnull().sum())
        sample = s.dropna().iloc[0] if non_null else None
        rows.append({
            "Column": col,
            "Dtype": str(s.dtype),
            "Non-Null": non_null,
            "Null": null,
            "Null %": round(100 * null / len(s), 2) if len(s) else 0.0,
            "Unique": int(s.nunique()),
            "Sample": str(sample)[:40] if sample is not None else "",
        })
    return pd.DataFrame(rows)


def get_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    rows = []
    for col in numeric_df.columns:
        s = numeric_df[col].dropna()
        if len(s) == 0:
            continue
        desc = s.describe()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = int(((s < lo) | (s > hi)).sum())
        rows.append({
            "Column": col,
            "Count": int(desc["count"]), "Mean": round(desc["mean"], 3),
            "Std": round(desc["std"], 3) if not np.isnan(desc["std"]) else 0.0,
            "Min": round(desc["min"], 3), "25%": round(desc["25%"], 3),
            "Median": round(desc["50%"], 3), "75%": round(desc["75%"], 3),
            "Max": round(desc["max"], 3),
            "Skew": round(float(s.skew()), 3) if len(s) > 2 else 0.0,
            "Kurtosis": round(float(s.kurt()), 3) if len(s) > 2 else 0.0,
            "Outliers (IQR)": outliers,
        })
    return pd.DataFrame(rows)


def get_categorical_stats(df: pd.DataFrame) -> pd.DataFrame:
    cat_df = df.select_dtypes(exclude=[np.number, "datetime64"])
    rows = []
    for col in cat_df.columns:
        s = cat_df[col].dropna()
        if len(s) == 0:
            rows.append({"Column": col, "Unique": 0, "Top Value": "", "Top Count": 0, "Top %": 0.0})
            continue
        vc = s.value_counts()
        top_val, top_count = vc.index[0], int(vc.iloc[0])
        rows.append({
            "Column": col,
            "Unique": int(s.nunique()),
            "Top Value": str(top_val)[:30],
            "Top Count": top_count,
            "Top %": round(100 * top_count / len(s), 2),
        })
    return pd.DataFrame(rows)


def get_missing_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        return pd.DataFrame(columns=["Column", "Missing", "Missing %"])
    return pd.DataFrame({
        "Column": miss.index,
        "Missing": miss.values,
        "Missing %": [round(100 * v / len(df), 2) for v in miss.values],
    })


def get_top_correlations(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame(columns=["Feature 1", "Feature 2", "Correlation"])
    corr = numeric_df.corr()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if not np.isnan(val):
                pairs.append((cols[i], cols[j], val))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    pairs = pairs[:top_n]
    return pd.DataFrame(pairs, columns=["Feature 1", "Feature 2", "Correlation"]).round(3)


def get_correlation_matrix(df: pd.DataFrame, max_cols: int = 20):
    numeric_df = df.select_dtypes(include=[np.number]).iloc[:, :max_cols]
    if numeric_df.shape[1] < 2:
        return None
    return numeric_df.corr()


def compute_full_eda(df: pd.DataFrame) -> dict:
    """Bundles everything the Upload page needs in one call — mirrors what
    the original `handle_load` returned in a single JSON payload."""
    return {
        "overview": get_basic_overview(df),
        "column_summary": get_column_summary(df),
        "numeric_stats": get_numeric_stats(df),
        "categorical_stats": get_categorical_stats(df),
        "missing_table": get_missing_table(df),
        "top_correlations": get_top_correlations(df),
        "correlation_matrix": get_correlation_matrix(df),
    }
