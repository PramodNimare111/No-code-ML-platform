# ML Platform — Streamlit Edition

A single-process rewrite of the original **MERN + Python-bridge** ML Platform project, using
[Streamlit](https://streamlit.io) instead of MongoDB / Express / React / Node.js.

No Flask, no Django, no Node.js — just Python.

---

## What changed vs. the original MERN version

| Original (MERN + Python bridge) | This version (Streamlit) |
|---|---|
| React SPA (5 pages) | Streamlit multipage app (5 pages, in `pages/`) |
| Express REST API (7 route files) | Not needed — Streamlit *is* the app server |
| `bridge.js` spawning Python per request | Not needed — everything runs in one Python process |
| JSON-over-stdio contract | Not needed — Python functions are called directly |
| Express session (`req.session`) storing CSV strings + base64 models | `st.session_state` storing **real** DataFrames + fitted model objects |
| matplotlib → base64 PNG → `<img>` | matplotlib `Figure` → `st.pyplot(fig)` directly |
| MongoDB (optional, session persistence) | Not used — Streamlit session state lives for the browser session |

All the actual data-science logic (EDA, the 15+ cleaning operations, the 13 models, metrics,
cross-validation, model comparison, prediction, and every chart type) has been preserved —
it's just been ported from the old `ml_engine.py` into a set of importable modules under `utils/`.

---

## Project Structure

```
ml-platform-streamlit/
├── app.py                    # Home page (multipage app entry point)
├── requirements.txt
├── .streamlit/
│   └── config.toml           # dark theme config
├── utils/
│   ├── state.py               # st.session_state management (undo stack, model state, etc.)
│   ├── data_io.py             # CSV/Excel file loading
│   ├── eda.py                 # EDA statistics (overview, column summary, correlations, ...)
│   ├── cleaning.py            # 15+ cleaning operations (fill, drop, encode, scale, ...)
│   ├── training.py            # model registries, feature suggestion, train/CV/compare
│   ├── prediction.py          # single-row and batch prediction
│   ├── plotting.py            # all matplotlib/seaborn chart functions
│   └── ui.py                  # shared sidebar/status/CSS helpers
└── pages/
    ├── 1_Upload.py
    ├── 2_Clean.py
    ├── 3_Train.py
    ├── 4_Predict.py
    └── 5_Plots.py
```

---

## Setup

```bash
cd ml-platform-streamlit

# (recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually **http://localhost:8501**).

---

## Features

1. **Upload** — CSV/Excel upload with a full EDA report (overview, column summary, numeric
   stats, categorical stats, missing values, correlations + heatmap, data preview). Toggle
   between the raw upload and the currently-cleaned dataset.
2. **Clean** — 15+ operations grouped into Missing Values, Duplicates & Outliers, Column
   Operations, Type Casting, Encoding, and Scaling — with a snapshot-based Undo stack and a
   one-click Reset to the original upload.
3. **Train** — pick a task (classification/regression), a target, and features (with an
   optional feature-relevance suggestion tool); choose from 13 scikit-learn models with
   dynamically-rendered hyper-parameter sliders; run a single training, k-fold
   cross-validation, or a side-by-side comparison of multiple models.
4. **Predict** — single-row prediction with confidence/probabilities, or batch prediction on
   an uploaded CSV with a downloadable results file.
5. **Plots** — data visualisations (distribution, box plot, bar counts, correlation heatmap,
   missing-values chart) and model visualisations (confusion matrix, ROC curve, feature
   importance, actual-vs-predicted, residuals, multi-model comparison chart).

---

## Why Streamlit instead of Flask/Django?

Streamlit is a self-contained Python app framework — it runs its own server, handles routing
via the `pages/` folder convention, and gives you `st.session_state` for free instead of
needing to build session management yourself (as Flask/Django would require). For a
data-facing internal tool or portfolio project like this one, it removes an enormous amount
of boilerplate compared to either a Flask/Django + HTML-templates approach or the original
full MERN-stack approach — at the cost of less UI control and a "rerun the whole script on
every interaction" execution model you have to design around.

## Known limitations (same spirit as the original project's README)

- **Single-user-per-process mental model**: Streamlit gives each browser tab its own
  `session_state`, but it isn't designed for very high concurrent user counts on a single
  instance the way a horizontally-scaled Express+MongoDB deployment would be.
- **No persistence across restarts**: like the original project's default in-memory session
  store, closing the Streamlit server loses all in-progress session state. Add your own
  persistence (e.g. saving to disk/S3/a database) if you need datasets/models to survive
  restarts.
- **No authentication** — same as the original project, this is a no-login demo/internal-tool
  pattern, not a multi-tenant production system.
