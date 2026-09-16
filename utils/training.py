"""
Model registries + training / cross-validation / comparison logic.

Ported from `handle_train()` and `handle_compare()` in the original
ml_engine.py. Because everything now runs in one process, models are
returned as live fitted objects (not joblib-pickled + base64-encoded) —
Streamlit's `st.session_state` can hold arbitrary Python objects directly.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, r2_score, mean_absolute_error, mean_squared_error,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Model registries: name -> (class, default_kwargs)
# ---------------------------------------------------------------------------
CLF = {
    "Logistic Regression": (LogisticRegression, {"max_iter": 1000, "random_state": RANDOM_STATE}),
    "Decision Tree": (DecisionTreeClassifier, {"random_state": RANDOM_STATE}),
    "Random Forest": (RandomForestClassifier, {"random_state": RANDOM_STATE, "n_jobs": -1}),
    "KNN": (KNeighborsClassifier, {}),
    "Naive Bayes": (GaussianNB, {}),
    "SVM": (SVC, {"probability": True, "random_state": RANDOM_STATE}),
    "Gradient Boosting": (GradientBoostingClassifier, {"random_state": RANDOM_STATE}),
}

REG = {
    "Linear Regression": (LinearRegression, {}),
    "Ridge Regression": (Ridge, {"random_state": RANDOM_STATE}),
    "Lasso Regression": (Lasso, {"random_state": RANDOM_STATE}),
    "Decision Tree Regressor": (DecisionTreeRegressor, {"random_state": RANDOM_STATE}),
    "Random Forest Regressor": (RandomForestRegressor, {"random_state": RANDOM_STATE, "n_jobs": -1}),
    "SVR": (SVR, {}),
}

# Hyper-parameter schema per model — drives the dynamically-rendered sliders
# in the Train page, exactly like the original GET /train/models response.
PARAM_SCHEMA = {
    "Logistic Regression": {"C": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01,
                                   "help": "Inverse regularization strength — smaller = stronger regularization"}},
    "Decision Tree": {"max_depth": {"type": "int", "min": 1, "max": 30, "default": 5, "step": 1,
                                     "help": "Maximum tree depth (limits overfitting)"},
                       "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2, "step": 1,
                                              "help": "Minimum samples required to split a node"}},
    "Random Forest": {"n_estimators": {"type": "int", "min": 10, "max": 500, "default": 100, "step": 10,
                                        "help": "Number of trees in the forest"},
                       "max_depth": {"type": "int", "min": 1, "max": 30, "default": 10, "step": 1,
                                     "help": "Maximum depth of each tree"}},
    "KNN": {"n_neighbors": {"type": "int", "min": 1, "max": 30, "default": 5, "step": 1,
                             "help": "Number of neighbors to vote"}},
    "Naive Bayes": {},
    "SVM": {"C": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01,
                  "help": "Regularization strength"},
            "kernel": {"type": "select", "options": ["rbf", "linear", "poly"], "default": "rbf",
                       "help": "Kernel type"}},
    "Gradient Boosting": {"n_estimators": {"type": "int", "min": 10, "max": 500, "default": 100, "step": 10,
                                            "help": "Number of boosting stages"},
                           "learning_rate": {"type": "float", "min": 0.001, "max": 1.0, "default": 0.1, "step": 0.001,
                                              "help": "Shrinks each tree's contribution"}},
    "Linear Regression": {},
    "Ridge Regression": {"alpha": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01,
                                    "help": "L2 regularization strength"}},
    "Lasso Regression": {"alpha": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01,
                                    "help": "L1 regularization strength"}},
    "Decision Tree Regressor": {"max_depth": {"type": "int", "min": 1, "max": 30, "default": 5, "step": 1,
                                               "help": "Maximum tree depth"}},
    "Random Forest Regressor": {"n_estimators": {"type": "int", "min": 10, "max": 500, "default": 100, "step": 10,
                                                  "help": "Number of trees in the forest"},
                                 "max_depth": {"type": "int", "min": 1, "max": 30, "default": 10, "step": 1,
                                               "help": "Maximum depth of each tree"}},
    "SVR": {"C": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01,
                  "help": "Regularization strength"},
            "kernel": {"type": "select", "options": ["rbf", "linear", "poly"], "default": "rbf",
                       "help": "Kernel type"}},
}


def get_registry(task: str):
    return CLF if task == "classification" else REG


# ---------------------------------------------------------------------------
# Feature suggestion (ported from op == 'suggest')
# ---------------------------------------------------------------------------
def suggest_features(df: pd.DataFrame, target: str, top_n: int = 10):
    candidates = [c for c in df.columns if c != target]
    target_numeric = pd.api.types.is_numeric_dtype(df[target])
    scores = []
    for c in candidates:
        try:
            feature_numeric = pd.api.types.is_numeric_dtype(df[c])

            if feature_numeric and target_numeric:
                # numeric vs numeric: Pearson correlation
                sub = df[[c, target]].dropna()
                if len(sub) < 3:
                    continue
                score = abs(float(np.corrcoef(sub[c], sub[target])[0, 1]))

            elif feature_numeric and not target_numeric:
                # numeric feature vs categorical target: correlate the feature
                # against the target's integer-encoded categories (a simple,
                # fast proxy for "does this feature separate the classes?")
                sub = df[[c, target]].dropna()
                if len(sub) < 3 or sub[target].nunique() < 2:
                    continue
                target_codes = sub[target].astype("category").cat.codes
                score = abs(float(np.corrcoef(sub[c], target_codes)[0, 1]))

            elif not feature_numeric and target_numeric:
                # categorical feature vs numeric target: ratio of between-group
                # std to overall std (higher = groups have very different means)
                sub = df[[c, target]].dropna()
                overall_std = sub[target].std()
                if overall_std == 0 or len(sub) < 3:
                    continue
                group_means = sub.groupby(c)[target].mean()
                score = min(1.0, float(group_means.std() / overall_std)) if len(group_means) > 1 else 0.0

            else:
                # categorical vs categorical: chi-square-based association (Cramér's-V-like)
                sub = df[[c, target]].dropna()
                if len(sub) < 3:
                    continue
                ct = pd.crosstab(sub[c], sub[target])
                chi2 = _chi_square_stat(ct)
                n = ct.values.sum()
                k = min(ct.shape) - 1
                score = min(1.0, float(np.sqrt(chi2 / (n * max(k, 1))))) if n > 0 else 0.0

            scores.append((c, score))
        except Exception:
            continue
    scores.sort(key=lambda x: x[1], reverse=True)
    scores = scores[:top_n]
    results = []
    for col, score in scores:
        label = "high" if score > 0.5 else ("medium" if score > 0.2 else "low")
        results.append({"Feature": col, "Score": round(score, 3), "Relevance": label})
    return pd.DataFrame(results)


def _chi_square_stat(ct: pd.DataFrame) -> float:
    observed = ct.values.astype(float)
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    if total == 0:
        return 0.0
    expected = row_sums @ col_sums / total
    expected[expected == 0] = 1e-9
    return float(((observed - expected) ** 2 / expected).sum())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _coerce_params(model_name: str, raw_params: dict) -> dict:
    schema = PARAM_SCHEMA.get(model_name, {})
    out = {}
    for k, v in raw_params.items():
        if k not in schema:
            continue
        ptype = schema[k]["type"]
        if ptype == "int":
            out[k] = int(v)
        elif ptype == "float":
            out[k] = float(v)
        else:
            out[k] = v
    return out


def train_model(df, features, target, task, model_name, params, test_size=0.2, seed=RANDOM_STATE):
    data = df[features + [target]].dropna()
    X = data[features].values
    y_raw = data[target].values

    label_classes = None
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        label_classes = le.classes_.tolist()
    else:
        y = y_raw.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=y if task == "classification" else None,
    )

    registry = get_registry(task)
    cls, defaults = registry[model_name]
    model_params = {**defaults, **_coerce_params(model_name, params)}
    model = cls(**model_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_proba = None
    if task == "classification" and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)

    metrics = compute_metrics(task, y_test, y_pred, y_proba)
    importances = extract_importances(model, features)

    return {
        "model": model, "metrics": metrics, "importances": importances,
        "X_test": X_test, "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba,
        "label_classes": label_classes,
    }


def compute_metrics(task, y_test, y_pred, y_proba=None):
    if task == "classification":
        m = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1 (weighted)": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "Precision (weighted)": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "Recall (weighted)": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        }
        n_classes = len(np.unique(y_test))
        if n_classes == 2 and y_proba is not None:
            try:
                m["AUC-ROC"] = round(roc_auc_score(y_test, y_proba[:, 1]), 4)
            except Exception:
                pass
        return m
    else:
        mse = mean_squared_error(y_test, y_pred)
        return {
            "R2 Score": round(r2_score(y_test, y_pred), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "RMSE": round(float(np.sqrt(mse)), 4),
            "MSE": round(mse, 4),
        }


def extract_importances(model, feature_names):
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        vals = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
    else:
        return None
    return pd.DataFrame({"Feature": feature_names, "Importance": vals}).sort_values(
        "Importance", ascending=False).reset_index(drop=True)


def cross_validate(df, features, target, task, model_name, params, k=5, seed=RANDOM_STATE):
    data = df[features + [target]].dropna()
    X = data[features].values
    y_raw = data[target].values

    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        scoring = "accuracy"
    else:
        y = y_raw.astype(float)
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        scoring = "r2"

    registry = get_registry(task)
    cls, defaults = registry[model_name]
    model_params = {**defaults, **_coerce_params(model_name, params)}
    model = cls(**model_params)

    scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring, n_jobs=-1)
    return {
        "scores": scores.tolist(),
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "min": round(float(scores.min()), 4),
        "max": round(float(scores.max()), 4),
        "scoring": scoring,
    }


def compare_models(df, features, target, task, model_names, params_map, test_size=0.2, seed=RANDOM_STATE):
    results = []
    data = df[features + [target]].dropna()
    X = data[features].values
    y_raw = data[target].values

    label_classes = None
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        label_classes = le.classes_.tolist()
    else:
        y = y_raw.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=y if task == "classification" else None,
    )

    registry = get_registry(task)
    for name in model_names:
        try:
            cls, defaults = registry[name]
            model_params = {**defaults, **_coerce_params(name, params_map.get(name, {}))}
            model = cls(**model_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if (task == "classification" and hasattr(model, "predict_proba")) else None
            metrics = compute_metrics(task, y_test, y_pred, y_proba)
            results.append({"Model": name, **metrics})
        except Exception as e:
            results.append({"Model": name, "error": str(e)})

    return {
        "results": results, "X_test": X_test, "y_test": y_test,
        "label_classes": label_classes,
    }
