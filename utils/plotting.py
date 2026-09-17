"""
Chart rendering — ported from `handle_plot()`.

The original Node version had to render each chart to a PNG and base64-encode
it (since it had to travel back through Node/JSON to a React <img> tag).
In Streamlit, matplotlib Figures can be handed directly to `st.pyplot(fig)`,
so no encoding step is needed at all — one of the biggest simplifications
of moving to a single-process Python app.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend — same reason as the original project
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.metrics import confusion_matrix, roc_curve, auc

BG = "#0F1117"
FG = "#E2E8F0"
ACCENT = "#4F8EF7"
GRID = "#2A2F3A"


def dark_fig(figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(color=GRID, alpha=0.4)
    return fig, ax


# ---------------------------------------------------------------------------
# Data plots
# ---------------------------------------------------------------------------
def plot_distribution(df, col):
    fig, ax = dark_fig()
    s = df[col].dropna()
    ax.hist(s, bins=30, color=ACCENT, alpha=0.75, edgecolor=BG)
    if len(s) > 1 and s.std() > 0:
        try:
            kde = scipy_stats.gaussian_kde(s)
            xs = np.linspace(s.min(), s.max(), 200)
            ax2 = ax.twinx()
            ax2.plot(xs, kde(xs), color="#F0A500", linewidth=2)
            ax2.set_yticks([])
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
        except Exception:
            pass
    ax.axvline(s.mean(), color="#2DCE89", linestyle="--", linewidth=1.5, label=f"Mean: {s.mean():.2f}")
    ax.axvline(s.median(), color="#F75F4F", linestyle="--", linewidth=1.5, label=f"Median: {s.median():.2f}")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def plot_box(df, col, group_col=None):
    fig, ax = dark_fig()
    if group_col and group_col in df.columns:
        groups = df[[group_col, col]].dropna()
        cats = groups[group_col].unique()[:15]
        data = [groups[groups[group_col] == c][col].values for c in cats]
        bp = ax.boxplot(data, patch_artist=True, labels=[str(c)[:10] for c in cats])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    else:
        bp = ax.boxplot(df[col].dropna(), patch_artist=True, labels=[col])
    for patch in bp["boxes"]:
        patch.set_facecolor(ACCENT)
        patch.set_alpha(0.6)
    for element in ["whiskers", "medians", "caps"]:
        plt.setp(bp[element], color=FG)
    ax.set_title(f"Box Plot: {col}" + (f" by {group_col}" if group_col else ""))
    fig.tight_layout()
    return fig


def plot_bar_counts(df, col, top_n=15):
    fig, ax = dark_fig()
    vc = df[col].value_counts().head(top_n)
    ax.bar([str(x)[:15] for x in vc.index], vc.values, color=ACCENT)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(f"Top {len(vc)} Values in {col}")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number]).iloc[:, :20]
    fig, ax = dark_fig(figsize=(7, 6))
    corr = numeric_df.corr()
    annot = corr.shape[0] <= 12
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, annot=annot, fmt=".2f",
                annot_kws={"size": 7, "color": FG}, cbar_kws={"label": "Correlation"},
                linewidths=0.5, linecolor=BG)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cbar.ax.get_yticklabels(), color=FG)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", color=FG, fontsize=8)
    plt.setp(ax.get_yticklabels(), color=FG, fontsize=8)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig


def plot_missing(df):
    fig, ax = dark_fig()
    miss_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    miss_pct = miss_pct[miss_pct > 0]
    if miss_pct.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", color=FG, transform=ax.transAxes)
    else:
        ax.bar(miss_pct.index.astype(str), miss_pct.values, color="#F75F4F")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel("Missing %")
    ax.set_title("Missing Values by Column")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_test, y_pred, class_names=None):
    fig, ax = dark_fig(figsize=(5.5, 5))
    cm = confusion_matrix(y_test, y_pred)
    labels = class_names if class_names else [str(i) for i in sorted(set(y_test))]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 9}, cbar=False, linewidths=0.5, linecolor=BG)
    plt.setp(ax.get_xticklabels(), color=FG, rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), color=FG, rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_feature_importance(importances_df):
    fig, ax = dark_fig(figsize=(6, max(3, 0.35 * len(importances_df))))
    top = importances_df.head(20).iloc[::-1]
    ax.barh(top["Feature"], top["Importance"], color=ACCENT)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = dark_fig()
    ax.scatter(y_test, y_pred, alpha=0.5, color=ACCENT, s=20)
    lo, hi = min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))
    ax.plot([lo, hi], [lo, hi], color="#F75F4F", linestyle="--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs. Predicted")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8)
    fig.tight_layout()
    return fig


def plot_residuals(y_test, y_pred):
    fig, ax = dark_fig()
    residuals = np.array(y_test) - np.array(y_pred)
    ax.scatter(y_pred, residuals, alpha=0.5, color=ACCENT, s=20)
    ax.axhline(0, color="#F75F4F", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Plot")
    fig.tight_layout()
    return fig


def plot_roc(y_test, y_proba):
    fig, ax = dark_fig()
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_comparison(comparison_results, metric):
    fig, ax = dark_fig(figsize=(7, 4))
    names, values = [], []
    for r in comparison_results:
        if "error" not in r and metric in r:
            names.append(r["Model"])
            values.append(r[metric])
    ax.bar(names, values, color=ACCENT)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title(f"Model Comparison — {metric}")
    ax.set_ylabel(metric)
    fig.tight_layout()
    return fig
