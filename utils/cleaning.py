"""
Data-cleaning operations — a direct port of the big if/elif dispatch that
used to live in `handle_clean()` inside the Node project's ml_engine.py.

Every function takes the current DataFrame plus an `opts` dict and returns
(new_df, message) — `message` is exactly what gets appended to the
Operations Log in the UI.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


def iqr_bounds(series: pd.Series, mult: float = 1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - mult * iqr, q3 + mult * iqr


def apply_operation(df: pd.DataFrame, op: str, opts: dict):
    """Dispatches on `op` and returns (new_df, message)."""
    d = df.copy()
    opts = opts or {}

    # ---------- Missing values ----------
    if op == "drop_missing_rows":
        how = opts.get("how", "any")  # 'any' | 'all'
        before = len(d)
        d = d.dropna(how=how)
        return d, f"Dropped {before - len(d)} rows with missing values ({how})"

    if op == "drop_missing_cols":
        threshold = float(opts.get("threshold", 50))  # % missing
        miss_pct = d.isnull().mean() * 100
        drop_cols = miss_pct[miss_pct > threshold].index.tolist()
        d = d.drop(columns=drop_cols)
        return d, f"Dropped {len(drop_cols)} columns with >{threshold}% missing: {', '.join(drop_cols) or 'none'}"

    if op == "fill_mean":
        cols = opts.get("columns") or d.select_dtypes(include=[np.number]).columns.tolist()
        for c in cols:
            if c in d.columns and pd.api.types.is_numeric_dtype(d[c]):
                d[c] = d[c].fillna(d[c].mean())
        return d, f"Filled missing values with mean in: {', '.join(cols)}"

    if op == "fill_median":
        cols = opts.get("columns") or d.select_dtypes(include=[np.number]).columns.tolist()
        for c in cols:
            if c in d.columns and pd.api.types.is_numeric_dtype(d[c]):
                d[c] = d[c].fillna(d[c].median())
        return d, f"Filled missing values with median in: {', '.join(cols)}"

    if op == "fill_mode":
        cols = opts.get("columns") or d.columns.tolist()
        for c in cols:
            if c in d.columns and not d[c].mode().empty:
                d[c] = d[c].fillna(d[c].mode().iloc[0])
        return d, f"Filled missing values with mode in: {', '.join(cols)}"

    if op == "fill_constant":
        cols = opts.get("columns") or d.columns.tolist()
        value = opts.get("value", 0)
        for c in cols:
            if c in d.columns:
                d[c] = d[c].fillna(value)
        return d, f"Filled missing values with '{value}' in: {', '.join(cols)}"

    if op == "fill_ffill":
        cols = opts.get("columns") or d.columns.tolist()
        d[cols] = d[cols].ffill()
        return d, f"Forward-filled missing values in: {', '.join(cols)}"

    if op == "fill_bfill":
        cols = opts.get("columns") or d.columns.tolist()
        d[cols] = d[cols].bfill()
        return d, f"Backward-filled missing values in: {', '.join(cols)}"

    # ---------- Duplicates & outliers ----------
    if op == "remove_duplicates":
        before = len(d)
        d = d.drop_duplicates()
        return d, f"Removed {before - len(d)} duplicate rows"

    if op == "remove_outliers":
        cols = opts.get("columns") or d.select_dtypes(include=[np.number]).columns.tolist()
        mult = float(opts.get("mult", 1.5))
        before = len(d)
        for c in cols:
            if c in d.columns and pd.api.types.is_numeric_dtype(d[c]):
                lo, hi = iqr_bounds(d[c].dropna(), mult)
                d = d[(d[c] >= lo) & (d[c] <= hi) | d[c].isna()]
        return d, f"Removed {before - len(d)} outlier rows (IQR x{mult}) from: {', '.join(cols)}"

    if op == "cap_outliers":
        cols = opts.get("columns") or d.select_dtypes(include=[np.number]).columns.tolist()
        mult = float(opts.get("mult", 1.5))
        for c in cols:
            if c in d.columns and pd.api.types.is_numeric_dtype(d[c]):
                lo, hi = iqr_bounds(d[c].dropna(), mult)
                d[c] = d[c].clip(lower=lo, upper=hi)
        return d, f"Capped outliers (IQR x{mult}) in: {', '.join(cols)}"

    # ---------- Column ops ----------
    if op == "drop_columns":
        cols = opts.get("columns", [])
        d = d.drop(columns=[c for c in cols if c in d.columns])
        return d, f"Dropped columns: {', '.join(cols)}"

    if op == "rename_columns":
        mapping = opts.get("mapping", {})
        d = d.rename(columns=mapping)
        pairs = ", ".join(f"{k}→{v}" for k, v in mapping.items())
        return d, f"Renamed columns: {pairs}"

    # ---------- Type casting ----------
    if op == "to_numeric":
        cols = opts.get("columns", [])
        for c in cols:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        return d, f"Cast to numeric: {', '.join(cols)}"

    if op == "to_datetime":
        cols = opts.get("columns", [])
        for c in cols:
            if c in d.columns:
                d[c] = pd.to_datetime(d[c], errors="coerce")
        return d, f"Cast to datetime: {', '.join(cols)}"

    if op == "to_category":
        cols = opts.get("columns", [])
        for c in cols:
            if c in d.columns:
                d[c] = d[c].astype("category")
        return d, f"Cast to category: {', '.join(cols)}"

    if op == "to_string":
        cols = opts.get("columns", [])
        for c in cols:
            if c in d.columns:
                d[c] = d[c].astype(str)
        return d, f"Cast to string: {', '.join(cols)}"

    # ---------- Encoding ----------
    if op == "label_encode":
        cols = opts.get("columns", [])
        for c in cols:
            if c in d.columns:
                le = LabelEncoder()
                mask = d[c].notnull()
                encoded = pd.Series(np.nan, index=d.index, dtype="float64")
                encoded.loc[mask] = le.fit_transform(d.loc[mask, c].astype(str)).astype(float)
                d[c] = encoded  # assign as a fresh numeric Series rather than mutating in place
        return d, f"Label-encoded: {', '.join(cols)}"

    if op == "onehot_encode":
        cols = opts.get("columns", [])
        drop_first = bool(opts.get("drop_first", False))
        d = pd.get_dummies(d, columns=cols, drop_first=drop_first)
        return d, f"One-hot encoded: {', '.join(cols)} (drop_first={drop_first})"

    # ---------- Scaling ----------
    if op == "standard_scale":
        cols = opts.get("columns") or d.select_dtypes(include=[np.number]).columns.tolist()
        scaler = StandardScaler()
        d[cols] = scaler.fit_transform(d[cols])
        return d, f"Standard-scaled (z-score): {', '.join(cols)}"

    if op == "minmax_scale":
        cols = opts.get("columns") or d.select_dtypes(include=[np.number]).columns.tolist()
        feature_range = (float(opts.get("min", 0)), float(opts.get("max", 1)))
        scaler = MinMaxScaler(feature_range=feature_range)
        d[cols] = scaler.fit_transform(d[cols])
        return d, f"Min-max scaled to {feature_range}: {', '.join(cols)}"

    raise ValueError(f"Unknown cleaning operation: {op}")


# Category groupings used to render the Clean page's UI sections
OPERATION_GROUPS = {
    "Missing Values": [
        ("drop_missing_rows", "Drop rows with missing values"),
        ("drop_missing_cols", "Drop columns above a missing % threshold"),
        ("fill_mean", "Fill with mean (numeric)"),
        ("fill_median", "Fill with median (numeric)"),
        ("fill_mode", "Fill with mode"),
        ("fill_constant", "Fill with a constant value"),
        ("fill_ffill", "Forward fill"),
        ("fill_bfill", "Backward fill"),
    ],
    "Duplicates & Outliers": [
        ("remove_duplicates", "Remove duplicate rows"),
        ("remove_outliers", "Remove outliers (IQR rule)"),
        ("cap_outliers", "Cap outliers (winsorize, IQR rule)"),
    ],
    "Column Operations": [
        ("drop_columns", "Drop columns"),
        ("rename_columns", "Rename columns"),
    ],
    "Type Casting": [
        ("to_numeric", "Cast to numeric"),
        ("to_datetime", "Cast to datetime"),
        ("to_category", "Cast to category"),
        ("to_string", "Cast to string"),
    ],
    "Encoding": [
        ("label_encode", "Label encode"),
        ("onehot_encode", "One-hot encode"),
    ],
    "Scaling": [
        ("standard_scale", "Standard scale (z-score)"),
        ("minmax_scale", "Min-max scale"),
    ],
}
