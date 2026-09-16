"""
Single-row and batch prediction — ported from `handle_predict()`.
"""

import numpy as np
import pandas as pd


def predict_single(model, feature_cols, task, input_values: dict, label_classes=None):
    row = {}
    for c in feature_cols:
        v = input_values.get(c, 0)
        try:
            row[c] = float(v)
        except (TypeError, ValueError):
            row[c] = 0.0
    X = pd.DataFrame([row])[feature_cols].values

    pred = model.predict(X)[0]
    result = {"prediction": None, "confidence": None, "probabilities": None}

    if task == "classification":
        pred_label = label_classes[int(pred)] if label_classes else pred
        result["prediction"] = pred_label
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = label_classes if label_classes else [str(c) for c in model.classes_]
            result["probabilities"] = dict(zip(classes, [round(float(p), 4) for p in proba]))
            result["confidence"] = round(float(proba.max()) * 100, 2)
    else:
        result["prediction"] = round(float(pred), 4)

    return result


def predict_batch(model, feature_cols, task, batch_df: pd.DataFrame, target_col: str, label_classes=None):
    missing = [c for c in feature_cols if c not in batch_df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing required feature columns: {', '.join(missing)}")

    X = batch_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    preds = model.predict(X)

    out = batch_df.copy()
    pred_col = f"predicted_{target_col}"

    if task == "classification":
        if label_classes:
            out[pred_col] = [label_classes[int(p)] for p in preds]
        else:
            out[pred_col] = preds
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            classes = label_classes if label_classes else [str(c) for c in model.classes_]
            for i, cls_name in enumerate(classes):
                out[f"prob_{cls_name}"] = np.round(proba[:, i], 4)
        summary = out[pred_col].value_counts().to_dict()
    else:
        out[pred_col] = np.round(preds, 4)
        summary = {
            "mean": round(float(np.mean(preds)), 4),
            "median": round(float(np.median(preds)), 4),
            "std": round(float(np.std(preds)), 4),
            "min": round(float(np.min(preds)), 4),
            "max": round(float(np.max(preds)), 4),
        }

    return out, summary
