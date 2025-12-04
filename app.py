# app.py
# Robust EV Battery SOH & RUL: A Missing-Data–Aware Analytics and Visualization Framework

import math
import itertools

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

from joblib import Parallel, delayed

# -----------------------------------------------------------------------------------
# PAGE CONFIG & THEME
# -----------------------------------------------------------------------------------
st.set_page_config(
    page_title="EV Battery SOH & RUL Dashboard",
    page_icon="🔋",
    layout="wide",
)

st.sidebar.title("🔧 Controls & Settings")

theme_choice = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
PLOTLY_TEMPLATE = "plotly_dark" if theme_choice == "Dark" else "plotly_white"
if theme_choice == "Dark":
    plt.style.use("dark_background")
else:
    plt.style.use("default")

CSS = """
<style>
.block-container {
    padding-top: 0.6rem;
    padding-bottom: 0.6rem;
    max-width: 1400px;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 0.6rem;
}
.kpi-card {
    border-radius: 10px;
    padding: 12px 14px;
    border: 1px solid rgba(255,255,255,0.15);
}
.kpi-label {
    font-size: 0.8rem;
    opacity: 0.8;
}
.kpi-value {
    font-size: 1.4rem;
    font-weight: 800;
}
.kpi-sub {
    font-size: 0.75rem;
    opacity: 0.75;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

COLOR_SEQ = px.colors.qualitative.Set2 + px.colors.qualitative.Set1


def kpi(label, value, sub=""):
    """Nice little KPI card."""
    if isinstance(value, float):
        vtxt = f"{value:,.3f}"
    else:
        vtxt = f"{value:,}" if isinstance(value, (int, np.integer)) else str(value)
    html = f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{vtxt}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def explain(title, bullets):
    """Expandable explanation for each tab."""
    with st.expander(f"ℹ️ What this tab does — {title}", expanded=True):
        for b in bullets:
            st.write(f"- {b}")


# -----------------------------------------------------------------------------------
# SYNTHETIC DEMO DATA (3 DATASETS) + METADATA SOURCES
# -----------------------------------------------------------------------------------
@st.cache_data
def generate_ev_demo(n_cells_per=10, n_cycles=180, seed=7):
    """
    Generate THREE synthetic battery-cycle datasets:
    - Urban, Highway, Mixed
    + Cell metadata (2nd source)
    + Environment profile (3rd source)
    """
    rng = np.random.default_rng(seed)

    def make_one(kind, n_cells, n_cycles, base_temp, base_crate, base_dod, region):
        rows = []
        for cidx in range(n_cells):
            cell_id = f"{kind[0]}{cidx:03d}"
            # cell-specific offsets
            start_soh = rng.uniform(0.96, 0.995)
            if kind == "Urban":
                slope = rng.uniform(0.0006, 0.0012)
            elif kind == "Highway":
                slope = rng.uniform(0.0003, 0.0007)
            else:
                slope = rng.uniform(0.0004, 0.0009)

            for cyc in range(n_cycles):
                cycle = cyc
                # features
                ambient_temp = rng.normal(base_temp, 4)
                c_rate = np.clip(rng.normal(base_crate, 0.3), 0.2, 3.5)
                dod = np.clip(rng.normal(base_dod, 0.15), 0.1, 1.0)  # depth of discharge
                avg_voltage = rng.normal(3.7, 0.05) + (0.1 if kind == "Highway" else 0.0)
                calendar_age = rng.uniform(6, 60)
                hour_of_day = rng.integers(0, 24)

                # usage pattern & stress
                if kind == "Urban":
                    usage_pattern = rng.choice(["aggressive", "moderate"], p=[0.7, 0.3])
                elif kind == "Highway":
                    usage_pattern = rng.choice(["gentle", "moderate"], p=[0.6, 0.4])
                else:
                    usage_pattern = rng.choice(["gentle", "moderate", "aggressive"])

                if usage_pattern == "aggressive":
                    stress_level = "high"
                elif usage_pattern == "moderate":
                    stress_level = "medium"
                else:
                    stress_level = "low"

                # SOH model
                noise = rng.normal(0, 0.003)
                soh = start_soh - slope * cycle - 0.02 * (dod - 0.7) + noise
                soh = float(np.clip(soh, 0.6, 1.02))

                note = f"{kind.lower()} driving, {usage_pattern} usage in {region} climate"

                # inject missingness
                if rng.random() < 0.04:
                    ambient_temp = np.nan
                if rng.random() < 0.04:
                    c_rate = np.nan
                if rng.random() < 0.04:
                    dod = np.nan
                if rng.random() < 0.03:
                    soh = np.nan

                rows.append(
                    dict(
                        dataset=kind,
                        cell_id=cell_id,
                        cycle=cycle,
                        ambient_temp=ambient_temp,
                        c_rate=c_rate,
                        depth_of_discharge=dod,
                        avg_voltage=avg_voltage,
                        calendar_age_months=calendar_age,
                        hour_of_day=hour_of_day,
                        usage_pattern=usage_pattern,
                        stress_level=stress_level,
                        region=region,
                        soh=soh,
                        note=note,
                    )
                )
        return pd.DataFrame(rows)

    df_urban = make_one(
        "Urban", n_cells_per, n_cycles, base_temp=30, base_crate=2.0, base_dod=0.85, region="Hot-city"
    )
    df_highway = make_one(
        "Highway", n_cells_per, n_cycles, base_temp=20, base_crate=1.2, base_dod=0.65, region="Temperate-highway"
    )
    df_mixed = make_one(
        "Mixed", n_cells_per, n_cycles, base_temp=25, base_crate=1.6, base_dod=0.75, region="Mixed-suburban"
    )

    # Second data source: cell metadata
    def build_cell_metadata(df, kind):
        cells = sorted(df["cell_id"].unique().tolist())
        manufacturers = ["Panasonic", "CATL", "LG", "Samsung"]
        cooling_opts = ["air", "liquid"]
        seg = {"Urban": "city-EV", "Highway": "long-range-EV", "Mixed": "multi-use-EV"}[kind]
        rng_local = np.random.default_rng(seed + hash(kind) % 10_000)
        rows = []
        for c in cells:
            rows.append(
                dict(
                    cell_id=c,
                    manufacturer=rng_local.choice(manufacturers),
                    cooling=rng_local.choice(cooling_opts, p=[0.7, 0.3]),
                    vehicle_segment=seg,
                    capacity_nominal_ah=rng_local.uniform(45, 80),
                )
            )
        return pd.DataFrame(rows)

    cell_meta = pd.concat(
        [
            build_cell_metadata(df_urban, "Urban"),
            build_cell_metadata(df_highway, "Highway"),
            build_cell_metadata(df_mixed, "Mixed"),
        ],
        ignore_index=True,
    )

    # Third data source: environment profile
    env_profile = pd.DataFrame(
        [
            dict(dataset="Urban", climate_zone="Hot", road_type="Stop-and-go city"),
            dict(dataset="Highway", climate_zone="Temperate", road_type="Highway cruising"),
            dict(dataset="Mixed", climate_zone="Mixed", road_type="Mixed usage"),
        ]
    )

    # Per-cycle datasets as dict
    per_cycle = {
        "Urban": df_urban,
        "Highway": df_highway,
        "Mixed": df_mixed,
    }

    return per_cycle, cell_meta, env_profile


# Generate demo data once
per_cycle_data, cell_metadata, env_profile = generate_ev_demo()

# -----------------------------------------------------------------------------------
# SIDEBAR: DATA SOURCES & OPTIONS
# -----------------------------------------------------------------------------------
st.sidebar.markdown("### Data sources")

use_demo = st.sidebar.checkbox("Use built-in EV demo datasets", value=True)

uploaded_files = st.sidebar.file_uploader(
    "Optional: upload additional CSVs (multiple allowed)",
    type=["csv"],
    accept_multiple_files=True,
)

uploaded_datasets = {}
if uploaded_files:
    for i, f in enumerate(uploaded_files, start=1):
        try:
            df_u = pd.read_csv(f)
            df_u = df_u.copy()
            df_u["dataset"] = f"Upload_{i}"
            uploaded_datasets[f"Upload_{i}"] = df_u
        except Exception as e:
            st.sidebar.warning(f"Could not read {f.name}: {e}")

# Combine per-cycle data
frames = []
if use_demo:
    for name, df in per_cycle_data.items():
        frames.append(df.copy())
for name, df in uploaded_datasets.items():
    frames.append(df.copy())

if not frames:
    st.error("No data available. Enable demo datasets or upload CSVs.")
    st.stop()

all_cycles = pd.concat(frames, ignore_index=True, sort=False)

# Merge cell_metadata and env_profile where possible
combined_all = all_cycles.merge(cell_metadata, on="cell_id", how="left")
combined_all = combined_all.merge(env_profile, on="dataset", how="left")

# Drop duplicate columns if any
combined_all = combined_all.loc[:, ~combined_all.columns.duplicated()].copy()

# Sidebar: select which datasets to analyze
available_sources = sorted(combined_all["dataset"].astype(str).unique().tolist())
selected_sources = st.sidebar.multiselect(
    "Datasets to ANALYZE (others still visible in overview)",
    options=available_sources,
    default=available_sources,
)

current_df = combined_all[combined_all["dataset"].isin(selected_sources)].copy()

# Sidebar: imputation & scaling choices
st.sidebar.markdown("### Preprocessing options")
imp_choice = st.sidebar.selectbox(
    "Imputer",
    ["Mean (Simple)", "Median (Simple)", "KNN (k=5)", "Iterative (MICE)"],
    index=1,
)
scale_choice = st.sidebar.checkbox("Standardize features (z-score)", value=True)

target_col = "soh"

# -----------------------------------------------------------------------------------
# UTILITIES: ENCODING, IMPUTATION, SCALING
# -----------------------------------------------------------------------------------
def build_encoded_matrices(df, target="soh", imputer="Median (Simple)", scale=True):
    """
    Returns:
      pre_enc_sample: original feature view (10 rows)
      X_train, X_test, y_train, y_test
      X_encoded: full encoded feature matrix (for inspection)
    """
    if df.empty or target not in df.columns:
        return None, None, None, None, None, None

    # Keep only rows with target defined
    dfm = df.copy()
    dfm = dfm[dfm[target].notna()].copy()
    if dfm.empty:
        return None, None, None, None, None, None

    y = dfm[target].astype(float)

    # ----- numeric base (except target) -----
    num_cols = dfm.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    num_base = dfm[num_cols].copy()

    # ----- ordinal encodings -----
    ord_df = pd.DataFrame(index=dfm.index)
    if "usage_pattern" in dfm.columns:
        mapping = {"gentle": 0, "moderate": 1, "aggressive": 2}
        ord_df["usage_pattern_ord"] = dfm["usage_pattern"].map(mapping)
    if "stress_level" in dfm.columns:
        mapping = {"low": 0, "medium": 1, "high": 2}
        ord_df["stress_level_ord"] = dfm["stress_level"].map(mapping)

    # ----- cyclical encodings -----
    cyc_df = pd.DataFrame(index=dfm.index)
    if "hour_of_day" in dfm.columns:
        h = pd.to_numeric(dfm["hour_of_day"], errors="coerce")
        cyc_df["hour_sin"] = np.sin(2 * math.pi * h / 24.0)
        cyc_df["hour_cos"] = np.cos(2 * math.pi * h / 24.0)

    # ----- one-hot encodings -----
    cat_cols = dfm.select_dtypes(include=["object", "category"]).columns.tolist()
    # remove text-like column from one-hot
    text_cols = []
    if "note" in dfm.columns:
        text_cols.append("note")
    for c in ["note"]:
        if c in cat_cols:
            cat_cols.remove(c)

    one_hot_cols = []
    for c in cat_cols:
        # we already handled these as ordinal
        if c in ["usage_pattern", "stress_level"]:
            continue
        one_hot_cols.append(c)

    if one_hot_cols:
        onehot = pd.get_dummies(dfm[one_hot_cols], prefix=one_hot_cols, drop_first=True)
    else:
        onehot = pd.DataFrame(index=dfm.index)

    # ----- combine all features -----
    X_struct = pd.concat([num_base, ord_df, cyc_df, onehot], axis=1)
    X_struct = X_struct.loc[:, ~X_struct.columns.duplicated()].copy()

    # ----- imputation -----
    if imputer.startswith("Mean"):
        imp = SimpleImputer(strategy="mean")
    elif imputer.startswith("Median"):
        imp = SimpleImputer(strategy="median")
    elif imputer.startswith("KNN"):
        imp = KNNImputer(n_neighbors=5)
    else:
        imp = IterativeImputer(random_state=7, max_iter=10, initial_strategy="median")

    X_imp_arr = imp.fit_transform(X_struct)
    X_imp = pd.DataFrame(X_imp_arr, columns=X_struct.columns, index=X_struct.index)

    # ----- scaling -----
    if scale:
        scaler = StandardScaler()
        X_scaled_arr = scaler.fit_transform(X_imp)
        X_final = pd.DataFrame(X_scaled_arr, columns=X_imp.columns, index=X_imp.index)
    else:
        X_final = X_imp

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.25, random_state=42
    )

    pre_enc_sample = dfm[
        ["dataset", "cell_id", "cycle", "ambient_temp", "c_rate", "depth_of_discharge",
         "avg_voltage", "calendar_age_months", "usage_pattern", "stress_level", "region"]
        if set(["dataset", "cell_id", "cycle"]).issubset(dfm.columns)
        else dfm.columns
    ].head(10)

    return pre_enc_sample, X_train, X_test, y_train, y_test, X_final


# -----------------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------------
tabs = st.tabs(
    [
        "📘 Introduction",
        "📦 Data Overview",
        "🔍 IDA & EDA",
        "🧩 Missingness & Imputation",
        "🔤 Encoding",
        "📊 EDA Gallery",
        "🤖 Classical Models",
        "🧠 Advanced Models & Neural Net",
        "⏱️ Time-Series & Forecasting",
        "📝 Text & NLP",
        "🌍 Real-World Insights",
        "✅ Rubric Map",
    ]
)

# -----------------------------------------------------------------------------------
# 1. INTRODUCTION TAB
# -----------------------------------------------------------------------------------
with tabs[0]:
    explain(
        "Introduction",
        [
            "Describe the project at a high level for a **non-technical** audience.",
            "Explain what SOH (State of Health) and RUL (Remaining Useful Life) mean.",
            "Summarise what each tab in the dashboard does.",
        ],
    )

    st.title("🔋 Robust EV Battery SOH & RUL Dashboard")
    st.markdown(
        """
This app is a **full end-to-end data science project**:

- We simulate **three EV battery usage scenarios**: **Urban**, **Highway**, and **Mixed** driving.
- For each cycle we track:
  - Ambient temperature, C-rate, depth-of-discharge (DoD), voltage, calendar age, hour-of-day…
  - Usage pattern (gentle / moderate / aggressive), stress level, and a free-text note.
- We compute and model:
  - **State of Health (SOH)** – how healthy the battery is (1.0 = like new, 0.7 = near end-of-life).
  - **Remaining Useful Life (RUL)** – how many cycles remain until SOH crosses a chosen EOL threshold.

**Tab guide:**

1. **📦 Data Overview** – combined view of **all datasets** (Urban/Highway/Mixed + uploads) and their stats.
2. **🔍 IDA & EDA** – initial & exploratory data analysis for the currently selected datasets.
3. **🧩 Missingness & Imputation** – missing data patterns, MCAR/MAR simulation, and imputer comparison.
4. **🔤 Encoding** – data **before** and **after** encoding (one-hot, ordinal, cyclical).
5. **📊 EDA Gallery** – a gallery of classic data science plots (histograms, box plots, scatter plots, etc.).
6. **🤖 Classical Models** – Random Forest & Gradient Boosting models for SOH.
7. **🧠 Advanced Models & Neural Net** – an MLP (neural network) and comparison against classical models.
8. **⏱️ Time-Series & Forecasting** – SOH trajectory over cycles and simple forecasting.
9. **📝 Text & NLP** – TF–IDF analysis of text notes (specialised data type: text).
10. **🌍 Real-World Insights** – story-style synthesis: what all this means for EV design/operation.
11. **✅ Rubric Map** – how this app covers each requirement of the project rubric.
"""
    )

# -----------------------------------------------------------------------------------
# 2. DATA OVERVIEW TAB (ALL DATASETS)
# -----------------------------------------------------------------------------------
with tabs[1]:
    explain(
        "Data overview",
        [
            "Always show **ALL datasets** (Urban, Highway, Mixed, and any uploaded CSVs).",
            "Then show the subset corresponding to the **current sidebar selection**.",
            "Give per-dataset row counts, SOH sums, missingness, and type information.",
        ],
    )

    st.markdown("### Combined dataset (ALL sources: Urban, Highway, Mixed, Uploads)")

    df_all_view = combined_all.copy()
    df_all_view = df_all_view.loc[:, ~df_all_view.columns.duplicated()]

    st.dataframe(df_all_view.head(30), use_container_width=True)
    st.caption(
        "First 30 rows of the fully integrated dataset across **all sources**, "
        "including per-cycle data plus merged cell metadata and environment profile."
    )

    # Per-dataset summary
    st.markdown("#### Per-dataset summary (ALL sources)")
    per_ds_all = (
        df_all_view.groupby("dataset")
        .agg(
            n_rows=("soh", "size"),
            n_cells=("cell_id", "nunique"),
            n_cycles=("cycle", "nunique"),
            mean_soh=("soh", "mean"),
        )
        .reset_index()
    )
    st.dataframe(per_ds_all, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_rows = px.bar(
            per_ds_all,
            x="dataset",
            y="n_rows",
            color="dataset",
            color_discrete_sequence=COLOR_SEQ,
            template=PLOTLY_TEMPLATE,
            title="Row count per dataset (ALL sources)",
        )
        st.plotly_chart(fig_rows, use_container_width=True)
        st.caption(
            "Interpretation: this bar chart shows how many rows come from each dataset. "
            "If one dataset dominates, it can bias models toward that usage profile."
        )

    with c2:
        fig_soh = px.bar(
            per_ds_all,
            x="dataset",
            y="mean_soh",
            color="dataset",
            color_discrete_sequence=COLOR_SEQ,
            template=PLOTLY_TEMPLATE,
            title="Average SOH per dataset (ALL sources)",
        )
        st.plotly_chart(fig_soh, use_container_width=True)
        st.caption(
            "Interpretation: datasets with lower mean SOH reflect harsher conditions (e.g. urban stop‑and‑go)."
        )

    st.markdown("---")

    # Column info & missingness
    st.markdown("### Column info & summary statistics (ALL sources)")
    col1, col2 = st.columns([1.2, 1.4])

    with col1:
        dtype_df_all = pd.DataFrame(
            {
                "column": df_all_view.columns,
                "dtype": [str(df_all_view[c].dtype) for c in df_all_view.columns],
                "n_unique": [df_all_view[c].nunique(dropna=True) for c in df_all_view.columns],
                "pct_missing": [100 * df_all_view[c].isna().mean() for c in df_all_view.columns],
            }
        )
        st.dataframe(dtype_df_all, use_container_width=True)

        missing_sorted = dtype_df_all.sort_values("pct_missing", ascending=False)
        fig_miss = px.bar(
            missing_sorted,
            x="column",
            y="pct_missing",
            template=PLOTLY_TEMPLATE,
            color="pct_missing",
            color_continuous_scale="Reds",
            title="Percent missing by column (ALL sources)",
            height=350,
        )
        fig_miss.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_miss, use_container_width=True)
        st.caption(
            "Interpretation: tall red bars are features that need more careful imputation/handling."
        )

    with col2:
        st.dataframe(df_all_view.describe(include="all").transpose(), use_container_width=True)
        st.caption(
            "Descriptive statistics across all sources: ranges, averages, and spread of numeric variables."
        )

    st.markdown("---")

    # Current selection overview (filter used in other tabs)
    st.markdown("### Current selection overview (what the rest of the app uses)")
    st.write(
        f"Current sidebar selection: **{', '.join(selected_sources)}**. "
        "This subset is used in EDA, missingness, encoding, and modelling tabs."
    )

    cur_summary = (
        current_df.groupby("dataset")
        .agg(
            n_rows=("soh", "size"),
            n_cells=("cell_id", "nunique"),
            n_cycles=("cycle", "nunique"),
            mean_soh=("soh", "mean"),
        )
        .reset_index()
    )
    st.dataframe(cur_summary, use_container_width=True)

    st.markdown("#### Sample rows from CURRENT selection")
    st.dataframe(current_df.head(10), use_container_width=True)

    st.markdown("---")

    st.markdown("### Cell metadata (2nd data source)")
    st.dataframe(cell_metadata.head(15), use_container_width=True)

    st.markdown("### Environment profile (3rd data source)")
    st.dataframe(env_profile, use_container_width=True)

# -----------------------------------------------------------------------------------
# 3. IDA & EDA TAB
# -----------------------------------------------------------------------------------
with tabs[2]:
    explain(
        "IDA & EDA",
        [
            "Initial Data Analysis (IDA): data types, ranges, missingness.",
            "Exploratory Data Analysis (EDA): distributions and relationships for selected datasets.",
        ],
    )

    st.markdown("### Filtered dataset (current selection)")
    st.dataframe(current_df.head(20), use_container_width=True)

    num_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = current_df.select_dtypes(include=["object", "category"]).columns.tolist()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Rows (current selection)", len(current_df))
    with c2:
        kpi("Cells", current_df["cell_id"].nunique())
    with c3:
        kpi("Datasets", current_df["dataset"].nunique())
    with c4:
        kpi("Mean SOH", float(current_df["soh"].mean(skipna=True)) if "soh" in current_df.columns else np.nan)

    st.markdown("### Numeric distributions (histograms)")
    if len(num_cols) > 0:
        top_num = [c for c in num_cols if c not in ["cycle", "hour_of_day"]][:4]
        if not top_num:
            top_num = num_cols[:4]
        n_cols = len(top_num)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 3))
        if n_cols == 1:
            axes = [axes]
        for ax, col in zip(axes, top_num):
            sns.histplot(current_df[col], kde=True, ax=ax)
            ax.set_title(col)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "These histograms show the distribution of key numeric features; skewness and multimodality can influence model choice."
        )
    else:
        st.info("No numeric columns available.")

    st.markdown("### SOH vs cycle by dataset")
    if {"cycle", "soh", "dataset"}.issubset(current_df.columns):
        fig = px.scatter(
            current_df,
            x="cycle",
            y="soh",
            color="dataset",
            template=PLOTLY_TEMPLATE,
            opacity=0.6,
            title="SOH vs cycle, coloured by dataset",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "This shows degradation patterns: Urban typically declines faster than Highway, with Mixed in between."
        )

    st.markdown("### Correlation heatmap")
    if len(num_cols) >= 2:
        corr = current_df[num_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            template=PLOTLY_TEMPLATE,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation matrix (numeric features)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Correlations reveal which variables move together (e.g. higher DoD correlating with lower SOH)."
        )

# -----------------------------------------------------------------------------------
# 4. MISSINGNESS & IMPUTATION TAB
# -----------------------------------------------------------------------------------
with tabs[3]:
    explain(
        "Missingness & Imputation",
        [
            "Quantify missing data patterns.",
            "Simulate MCAR / MAR missingness.",
            "Compare different imputers on reconstruction error.",
        ],
    )

    df = current_df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.markdown("### Missing values summary")
    miss_counts = df.isna().sum()
    miss_pct = 100 * df.isna().mean()
    miss_df = pd.DataFrame({"missing": miss_counts, "pct_missing": miss_pct})
    st.dataframe(miss_df.sort_values("pct_missing", ascending=False), use_container_width=True)

    st.markdown("#### Missing values bar plot")
    miss_nonzero = miss_df[miss_df["missing"] > 0].sort_values("pct_missing", ascending=False)
    if not miss_nonzero.empty:
        fig = px.bar(
            miss_nonzero.reset_index().rename(columns={"index": "column"}),
            x="column",
            y="pct_missing",
            color="pct_missing",
            color_continuous_scale="Reds",
            template=PLOTLY_TEMPLATE,
            title="Percent missing by column (current selection)",
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No missing values in current selection.")

    st.markdown("#### Missingness heatmap (rows × columns)")
    miss_matrix = df[num_cols].isna().astype(int) if len(num_cols) > 0 else None
    if miss_matrix is not None and miss_matrix.shape[1] > 0:
        fig = px.imshow(
            miss_matrix.T,
            aspect="auto",
            color_continuous_scale="Viridis",
            template=PLOTLY_TEMPLATE,
            title="Missingness heatmap (yellow=missing)",
            labels=dict(x="Row index", y="Feature", color="Missing"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### MCAR experiment + imputer comparison")

    if len(num_cols) >= 1:
        target_mcar = st.selectbox("Numeric column to artificially mask (MCAR)", num_cols, index=0)
        rate = st.slider("Missing rate", 0.0, 0.8, 0.3, 0.05)
        seed = st.number_input("Random seed", 0, 9999, 7, 1)

        rng = np.random.default_rng(int(seed))
        df_mcar = df.copy()
        idx_all = df_mcar.index.to_numpy()
        k = int(rate * len(idx_all))
        if k > 0:
            mask_idx = rng.choice(idx_all, size=k, replace=False)
            df_mcar.loc[mask_idx, target_mcar] = np.nan

        st.caption("We masked values at random to simulate Missing Completely At Random (MCAR).")

        # Compare imputers on RMSE vs original
        original = df[target_mcar].copy()
        results = []
        for label in ["Mean (Simple)", "Median (Simple)", "KNN (k=5)", "Iterative (MICE)"]:
            cols_here = [c for c in num_cols if df_mcar[c].notna().sum() > 0]
            if target_mcar not in cols_here:
                continue
            X = df_mcar[cols_here]
            if label.startswith("Mean"):
                imp = SimpleImputer(strategy="mean")
            elif label.startswith("Median"):
                imp = SimpleImputer(strategy="median")
            elif label.startswith("KNN"):
                imp = KNNImputer(n_neighbors=5)
            else:
                imp = IterativeImputer(random_state=7, max_iter=10, initial_strategy="median")

            X_imp = imp.fit_transform(X)
            X_imp = pd.DataFrame(X_imp, columns=cols_here, index=X.index)
            rmse = float(
                np.sqrt(
                    np.nanmean(
                        (X_imp[target_mcar] - original) ** 2
                    )
                )
            )
            results.append({"imputer": label, "rmse": rmse})

        if results:
            df_res = pd.DataFrame(results).sort_values("rmse")
            st.dataframe(df_res, use_container_width=True)
            fig_cmp = px.bar(
                df_res,
                x="imputer",
                y="rmse",
                template=PLOTLY_TEMPLATE,
                title=f"Imputation RMSE for MCAR on '{target_mcar}'",
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")
    st.markdown("### MAR experiment (mask when another feature is high)")

    if len(num_cols) >= 2:
        target_mar = st.selectbox("Column to mask (MAR target)", num_cols, index=0, key="mar_tgt")
        ref_candidates = [c for c in num_cols if c != target_mar]
        ref_col = st.selectbox("Reference column", ref_candidates, index=0, key="mar_ref")
        q = st.slider("Reference quantile threshold", 0.5, 0.95, 0.8, 0.05)
        thr = float(df[ref_col].quantile(q))
        df_mar = df.copy()
        df_mar.loc[df_mar[ref_col] > thr, target_mar] = np.nan
        st.caption(
            f"We created MAR missingness by dropping {target_mar} when {ref_col} > {thr:.3f} "
            f"(quantile q={q:.2f})."
        )

        # Compare correlation structure before/after
        if len(num_cols) >= 2:
            corr_orig = df[num_cols].corr()
            corr_mar = df_mar[num_cols].corr()
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Original corr", "After MAR"),
            )
            fig.add_trace(
                go.Heatmap(
                    z=corr_orig.values,
                    x=num_cols,
                    y=num_cols,
                    zmin=-1,
                    zmax=1,
                    colorscale="RdBu",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Heatmap(
                    z=corr_mar.values,
                    x=num_cols,
                    y=num_cols,
                    zmin=-1,
                    zmax=1,
                    colorscale="RdBu",
                ),
                row=1,
                col=2,
            )
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                height=400,
                title="Correlation matrices before and after MAR",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "If correlations change a lot under MAR, naive imputation can distort your downstream relationships."
            )

# -----------------------------------------------------------------------------------
# 5. ENCODING TAB
# -----------------------------------------------------------------------------------
with tabs[4]:
    explain(
        "Encoding",
        [
            "Show **original tabular data BEFORE encoding** (for all selected datasets).",
            "Show the encoded feature matrix AFTER applying one-hot, ordinal, and cyclical encodings.",
            "Explain what is being encoded and how.",
        ],
    )

    st.markdown("### Raw features BEFORE encoding (current selection)")
    pre_sample_cols = [
        c
        for c in [
            "dataset",
            "cell_id",
            "cycle",
            "ambient_temp",
            "c_rate",
            "depth_of_discharge",
            "avg_voltage",
            "calendar_age_months",
            "hour_of_day",
            "usage_pattern",
            "stress_level",
            "region",
            target_col,
        ]
        if c in current_df.columns
    ]
    if pre_sample_cols:
        st.dataframe(
            current_df[pre_sample_cols]
            .sort_values(["dataset", "cell_id", "cycle"])
            .groupby("dataset")
            .head(5),
            use_container_width=True,
        )
        st.caption(
            "We show 5 rows per dataset (Urban / Highway / Mixed / uploads) to illustrate the **raw categorical and numeric variables**."
        )

    st.markdown("---")

    pre_enc_sample, X_tr, X_te, y_tr, y_te, X_full = build_encoded_matrices(
        current_df, target=target_col, imputer=imp_choice, scale=scale_choice
    )

    if pre_enc_sample is None:
        st.info("Not enough labelled SOH rows to build encoded matrices.")
    else:
        st.markdown("### Example rows BEFORE encoding (for reference)")
        st.dataframe(pre_enc_sample, use_container_width=True)

        st.markdown("### Encoded feature matrix AFTER encoding + imputation + scaling")
        st.dataframe(X_full.head(10), use_container_width=True)
        st.caption(
            """
Encoding details:

- **Numeric features** (e.g. `ambient_temp`, `c_rate`, `depth_of_discharge`) are kept as continuous.
- **Ordinal encoding**:
  - `usage_pattern` → `usage_pattern_ord` (gentle < moderate < aggressive).
  - `stress_level` → `stress_level_ord` (low < medium < high).
- **Cyclical encoding**:
  - `hour_of_day` → `hour_sin`, `hour_cos` to respect circular time-of-day structure.
- **One-hot encoding**:
  - `dataset`, `region`, `manufacturer`, `cooling`, `vehicle_segment`, `climate_zone`, `road_type`, etc.

Missing values are imputed using the sidebar-selected imputer, and then (optionally) standardized.
"""
        )

        st.markdown("### Feature-type summary")
        feat_types = []
        for c in X_full.columns:
            if c.endswith("_ord"):
                typ = "ordinal"
            elif c in ["hour_sin", "hour_cos"]:
                typ = "cyclical"
            elif any(c.startswith(pref + "_") for pref in ["dataset", "region", "manufacturer", "cooling", "vehicle_segment", "climate_zone", "road_type"]):
                typ = "one-hot"
            else:
                typ = "numeric"
            feat_types.append({"feature": c, "type": typ})
        ft_df = pd.DataFrame(feat_types)
        st.dataframe(ft_df, use_container_width=True)

# -----------------------------------------------------------------------------------
# 6. EDA GALLERY TAB (LOTS OF PLOTS)
# -----------------------------------------------------------------------------------
with tabs[5]:
    explain(
        "EDA Gallery",
        [
            "Showcase a variety of **classic data science visualisations**.",
            "Histogram, KDE, boxplot, violin, scatter, scatter matrix, 3D scatter, etc.",
        ],
    )

    df = current_df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # 6.1 Class imbalance-like plots: dataset distribution
    st.markdown("### Dataset distribution (class imbalance style)")
    counts = df["dataset"].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(
            pd.DataFrame(
                {
                    "dataset": counts.index,
                    "count": counts.values,
                    "pct": (counts.values / len(df) * 100).round(2),
                }
            ),
            use_container_width=True,
        )
    with col2:
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            color=counts.index,
            color_discrete_sequence=COLOR_SEQ,
            template=PLOTLY_TEMPLATE,
            title="Row count per dataset (current selection)",
            text=counts.values,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Pie chart of dataset proportions")
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        hole=0.3,
        template=PLOTLY_TEMPLATE,
        title="Dataset proportions",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Box plots & outliers (numeric features)")

    top_num = num_cols[:4]
    if len(top_num) > 0:
        fig_box = make_subplots(
            rows=1,
            cols=len(top_num),
            subplot_titles=top_num,
            shared_y=True,
        )
        for i, col in enumerate(top_num, start=1):
            fig_box.add_trace(
                go.Box(
                    y=df[col],
                    name=col,
                    marker_color=COLOR_SEQ[(i - 1) % len(COLOR_SEQ)],
                    boxpoints="outliers",
                ),
                row=1,
                col=i,
            )
        fig_box.update_layout(
            template=PLOTLY_TEMPLATE,
            height=400,
            showlegend=False,
            title="Box plots for outlier detection",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.markdown("### Violin + histogram by dataset")

    if {"soh", "dataset"}.issubset(df.columns):
        fig_violin = px.violin(
            df,
            y="soh",
            x="dataset",
            color="dataset",
            color_discrete_sequence=COLOR_SEQ,
            box=True,
            points="all",
            template=PLOTLY_TEMPLATE,
            title="SOH distribution by dataset (Violin plot)",
        )
        st.plotly_chart(fig_violin, use_container_width=True)

        fig_hist = px.histogram(
            df,
            x="soh",
            color="dataset",
            template=PLOTLY_TEMPLATE,
            barmode="overlay",
            opacity=0.6,
            title="SOH distribution by dataset (Histogram)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.markdown("### Scatter & 3D scatter")

    if len(num_cols) >= 2:
        x_axis = st.selectbox("Scatter X-axis", num_cols, index=num_cols.index("cycle") if "cycle" in num_cols else 0)
        y_axis = st.selectbox("Scatter Y-axis", num_cols, index=num_cols.index("soh") if "soh" in num_cols else 1)

        fig_scatter = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color="dataset",
            template=PLOTLY_TEMPLATE,
            opacity=0.6,
            title=f"{y_axis} vs {x_axis}",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        if len(num_cols) >= 3:
            st.markdown("#### 3D scatter (cycle, SOH, ambient_temp)")
            cols_3d = ["cycle", "soh", "ambient_temp"]
            cols_3d = [c for c in cols_3d if c in num_cols]
            if len(cols_3d) == 3:
                fig_3d = px.scatter_3d(
                    df,
                    x=cols_3d[0],
                    y=cols_3d[1],
                    z=cols_3d[2],
                    color="dataset",
                    template=PLOTLY_TEMPLATE,
                    opacity=0.7,
                    title="3D scatter: cycle vs SOH vs ambient_temp",
                )
                st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("---")
    st.markdown("### Scatter matrix (pairwise relationships)")

    small = df.sample(min(500, len(df)), random_state=7) if len(df) > 500 else df.copy()
    if len(num_cols) >= 3:
        sub = num_cols[:4]
        fig_matrix = px.scatter_matrix(
            small,
            dimensions=sub,
            color="dataset",
            template=PLOTLY_TEMPLATE,
            title="Scatter matrix (small sample)",
        )
        fig_matrix.update_traces(diagonal_visible=False)
        st.plotly_chart(fig_matrix, use_container_width=True)

# -----------------------------------------------------------------------------------
# 7. CLASSICAL MODELS TAB
# -----------------------------------------------------------------------------------
with tabs[6]:
    explain(
        "Classical models",
        [
            "Train baseline regression models for SOH using encoded features.",
            "Compare RandomForestRegressor and GradientBoostingRegressor.",
            "Demonstrate model evaluation (MAE, R²) and feature importances.",
        ],
    )

    pre_enc_sample, X_tr, X_te, y_tr, y_te, X_full = build_encoded_matrices(
        current_df, target=target_col, imputer=imp_choice, scale=scale_choice
    )
    if X_tr is None:
        st.info("Need at least some rows with SOH to train classical models.")
    else:
        st.markdown("### Model performance (hold-out split)")

        rf = RandomForestRegressor(
            n_estimators=200,
            random_state=7,
            n_jobs=-1,  # parallel / HPC
        )
        gb = GradientBoostingRegressor(random_state=7)

        rf.fit(X_tr, y_tr)
        gb.fit(X_tr, y_tr)

        preds = []
        for name, model in [("RandomForest", rf), ("GradientBoosting", gb)]:
            yhat_tr = model.predict(X_tr)
            yhat_te = model.predict(X_te)
            preds.append(
                {
                    "model": name,
                    "MAE_train": mean_absolute_error(y_tr, yhat_tr),
                    "MAE_test": mean_absolute_error(y_te, yhat_te),
                    "R2_train": r2_score(y_tr, yhat_tr),
                    "R2_test": r2_score(y_te, yhat_te),
                }
            )

        perf_df = pd.DataFrame(preds)
        st.dataframe(perf_df, use_container_width=True)

        fig_perf = px.bar(
            perf_df,
            x="model",
            y="MAE_test",
            template=PLOTLY_TEMPLATE,
            title="Test MAE by model (lower is better)",
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        st.caption(
            "RandomForest is an ensemble of decision trees (bagging). GradientBoosting builds trees sequentially to reduce residuals."
        )

        st.markdown("### Feature importance (RandomForest)")
        importances = pd.DataFrame(
            {
                "feature": X_tr.columns,
                "importance": rf.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        st.dataframe(importances.head(20), use_container_width=True)

        fig_imp = px.bar(
            importances.head(20),
            x="feature",
            y="importance",
            template=PLOTLY_TEMPLATE,
            title="Top 20 features by importance (RandomForest)",
        )
        fig_imp.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("### Predicted vs actual SOH (test set)")
        yhat = rf.predict(X_te)
        fig_sc = px.scatter(
            x=y_te,
            y=yhat,
            labels={"x": "Actual SOH", "y": "Predicted SOH"},
            template=PLOTLY_TEMPLATE,
            title="Predicted vs actual SOH (RandomForest, test set)",
        )
        fig_sc.add_shape(
            type="line",
            x0=y_te.min(),
            y0=y_te.min(),
            x1=y_te.max(),
            y1=y_te.max(),
            line=dict(color="red", dash="dash"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# -----------------------------------------------------------------------------------
# 8. ADVANCED MODELS & NEURAL NET TAB
# -----------------------------------------------------------------------------------
with tabs[7]:
    explain(
        "Advanced & Neural Net",
        [
            "Train a multi-layer perceptron (MLPRegressor) as a neural network for SOH.",
            "Compare its performance to classical models.",
            "Visualise the **neural network architecture graphically**.",
        ],
    )

    pre_enc_sample, X_tr, X_te, y_tr, y_te, X_full = build_encoded_matrices(
        current_df, target=target_col, imputer=imp_choice, scale=scale_choice
    )
    if X_tr is None:
        st.info("Need labelled SOH rows to train neural network.")
    else:
        st.markdown("### Training a neural network (MLPRegressor)")

        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            random_state=7,
            max_iter=300,
        )
        mlp.fit(X_tr, y_tr)

        yhat_tr = mlp.predict(X_tr)
        yhat_te = mlp.predict(X_te)

        mae_tr = mean_absolute_error(y_tr, yhat_tr)
        mae_te = mean_absolute_error(y_te, yhat_te)
        r2_tr = r2_score(y_tr, yhat_tr)
        r2_te = r2_score(y_te, yhat_te)

        st.write(
            f"**MLPRegressor results:**  "
            f"Train MAE = {mae_tr:.4f}, Test MAE = {mae_te:.4f}, "
            f"Train R² = {r2_tr:.3f}, Test R² = {r2_te:.3f}"
        )

        st.markdown("### Neural network architecture (schematic)")

        # Simple layered diagram
        n_in = X_tr.shape[1]
        layers = [n_in, 64, 32, 16, 1]
        layer_x_pos = [0, 1, 2, 3, 4]

        node_x = []
        node_y = []
        node_labels = []
        edge_x = []
        edge_y = []

        for li, (n_nodes, x) in enumerate(zip(layers, layer_x_pos)):
            if n_nodes == 1:
                ys = [0.0]
            else:
                ys = np.linspace(-1, 1, n_nodes)
            for y in ys:
                node_x.append(x)
                node_y.append(y)
                if li == 0:
                    node_labels.append(f"in")
                elif li == len(layers) - 1:
                    node_labels.append("out")
                else:
                    node_labels.append(f"h{li}")
            # edges to next layer
            if li < len(layers) - 1:
                next_nodes = layers[li + 1]
                if next_nodes == 1:
                    ys_next = [0.0]
                else:
                    ys_next = np.linspace(-1, 1, next_nodes)
                for y1 in ys:
                    for y2 in ys_next:
                        edge_x += [x, x + 1, None]
                        edge_y += [y1, y2, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="gray", width=1),
            hoverinfo="none",
        )
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=12,
                color=list(range(len(node_x))),
                colorscale="Viridis",
                showscale=False,
            ),
            text=node_labels,
        )
        fig_nn = go.Figure(data=[edge_trace, node_trace])
        fig_nn.update_layout(
            template=PLOTLY_TEMPLATE,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            title="MLP architecture: Input → 64 → 32 → 16 → Output",
            height=400,
        )
        st.plotly_chart(fig_nn, use_container_width=True)

        st.caption(
            "Each column is a layer: input features → three hidden layers (ReLU) → single SOH output neuron."
        )

        st.markdown("### Learning curve (loss over iterations)")
        if hasattr(mlp, "loss_curve_"):
            fig_loss = px.line(
                y=mlp.loss_curve_,
                x=list(range(1, len(mlp.loss_curve_) + 1)),
                labels={"x": "Iteration", "y": "Loss"},
                template=PLOTLY_TEMPLATE,
                title="MLP training loss curve",
            )
            st.plotly_chart(fig_loss, use_container_width=True)

# -----------------------------------------------------------------------------------
# 9. TIME-SERIES & FORECASTING TAB
# -----------------------------------------------------------------------------------
with tabs[8]:
    explain(
        "Time-Series & Forecasting",
        [
            "Treat SOH as a time series over cycle index.",
            "Fit a simple regression on cycle to forecast future SOH.",
            "Demonstrate specialised time-series reasoning (even with a simple model).",
        ],
    )

    df = current_df.copy()
    if {"cycle", "soh", "cell_id"}.issubset(df.columns) and df["soh"].notna().sum() > 20:
        cells = sorted(df["cell_id"].unique().tolist())
        cell_sel = st.selectbox("Choose a cell for time-series view", cells, index=0)
        dcell = df[df["cell_id"] == cell_sel].sort_values("cycle")
        dcell = dcell[dcell["soh"].notna()]
        if len(dcell) < 10:
            st.info("Not enough points in this cell to forecast SOH.")
        else:
            st.markdown(f"### SOH trajectory for cell {cell_sel}")
            fig_line = px.line(
                dcell,
                x="cycle",
                y="soh",
                color="dataset",
                template=PLOTLY_TEMPLATE,
                markers=True,
                title=f"SOH vs cycle for {cell_sel}",
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # Simple linear regression on cycle → soh
            X = dcell[["cycle"]].values
            y = dcell["soh"].values

            # Use scikit-learn LinearRegression via poly features by hand
            from sklearn.linear_model import LinearRegression

            deg = st.slider("Polynomial degree for trend", 1, 3, 1)
            X_poly = np.column_stack([X[:, 0] ** p for p in range(1, deg + 1)])
            lr = LinearRegression()
            lr.fit(X_poly, y)

            # Forecast next N cycles
            max_cyc = int(dcell["cycle"].max())
            n_future = st.slider("Forecast horizon (cycles)", 20, 120, 50, 10)
            future_cycles = np.arange(max_cyc + 1, max_cyc + 1 + n_future)
            Xf_poly = np.column_stack([future_cycles ** p for p in range(1, deg + 1)])
            y_future = lr.predict(Xf_poly)

            df_future = pd.DataFrame({"cycle": future_cycles, "soh": y_future})
            fig_fore = px.line(
                pd.concat([dcell[["cycle", "soh"]], df_future]),
                x="cycle",
                y="soh",
                template=PLOTLY_TEMPLATE,
                title=f"SOH forecast for {cell_sel}",
                color_discrete_sequence=["cyan"],
            )
            fig_fore.add_vline(x=max_cyc, line_dash="dash", line_color="white")
            st.plotly_chart(fig_fore, use_container_width=True)
            st.caption(
                "This simple polynomial trend illustrates the idea of RUL-type forecasting: "
                "we extend the observed SOH trajectory into the future."
            )
    else:
        st.info("Need cycle, SOH, cell_id columns with enough data to show time series.")

# -----------------------------------------------------------------------------------
# 10. TEXT & NLP TAB
# -----------------------------------------------------------------------------------
with tabs[9]:
    explain(
        "Text & NLP",
        [
            "Use the `note` column as a small text corpus.",
            "Apply TF–IDF (term frequency–inverse document frequency) to extract informative tokens.",
            "Demonstrate specialised text handling (NLP).",
        ],
    )

    if "note" not in current_df.columns:
        st.info("No text field available (note column missing).")
    else:
        df = current_df.copy()
        notes = df["note"].fillna("")
        if notes.str.len().sum() == 0:
            st.info("Text column is empty.")
        else:
            st.markdown("### Sample notes")
            st.write(notes.sample(min(5, len(notes)), random_state=7).tolist())

            vect = TfidfVectorizer(
                max_features=50,
                stop_words="english",
                ngram_range=(1, 2),
            )
            X_t = vect.fit_transform(notes)
            vocab = np.array(vect.get_feature_names_out())
            tfidf_mean = np.asarray(X_t.mean(axis=0)).ravel()

            top_idx = np.argsort(tfidf_mean)[::-1][:20]
            df_tokens = pd.DataFrame(
                {
                    "token": vocab[top_idx],
                    "mean_tfidf": tfidf_mean[top_idx],
                }
            )
            st.markdown("### Top tokens by mean TF–IDF")
            st.dataframe(df_tokens, use_container_width=True)

            fig_tok = px.bar(
                df_tokens,
                x="token",
                y="mean_tfidf",
                template=PLOTLY_TEMPLATE,
                title="Most informative tokens in notes (TF–IDF)",
            )
            fig_tok.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_tok, use_container_width=True)

            st.caption(
                "Tokens like 'urban', 'highway', 'aggressive', etc. distinguish different usage profiles "
                "and could be fed into models as additional text features."
            )

# -----------------------------------------------------------------------------------
# 11. REAL-WORLD INSIGHTS TAB
# -----------------------------------------------------------------------------------
with tabs[10]:
    explain(
        "Real-World Insights",
        [
            "Translate quantitative results into **engineering & operational recommendations**.",
            "Answer: What should EV designers, fleet managers, or drivers do differently?",
        ],
    )

    st.markdown("### Key patterns from the data (demo narrative)")

    st.markdown(
        """
**Usage patterns and degradation:**

- **Urban** dataset tends to have:
  - Higher C-rates (more aggressive acceleration/braking).
  - Higher depth-of-discharge (DoD).
  - Higher ambient temperatures.
  - Faster SOH decline per cycle.

- **Highway** dataset:
  - Lower C-rates and DoD.
  - Cooler average temperatures.
  - Slower degradation, higher mean SOH.

- **Mixed** lies between Urban and Highway.

**Model insights (from classical + neural models):**

- The most important predictors for SOH (in RandomForest feature importance) are typically:
  - `cycle` (how many cycles already used),
  - `depth_of_discharge` (DoD),
  - `ambient_temp`,
  - `usage_pattern_ord` / `stress_level_ord`,
  - and environment features (e.g. `climate_zone`, `road_type`).

- The neural network (MLP) usually matches or slightly improves MAE vs classical models
  when enough training data is available.

**Practical recommendations:**

- For **EV designers**:
  - Thermal management matters:
    - High `ambient_temp` + aggressive driving → accelerated aging.
  - Designing packs for **moderate DoD** (e.g. 20–80% SoC windows) can significantly
    reduce degradation relative to full 0–100% swings.

- For **fleet operators**:
  - Highway-dominant profiles may allow **longer warranties** or **less aggressive derating**.
  - Urban fleets might need **more frequent monitoring** and **smarter charging strategies**
    (e.g., avoiding fast charging immediately after high-stress trips).

- For **drivers**:
  - Avoid repeatedly deep discharges (high DoD).
  - Try to keep the pack within moderate temperatures (garage, shaded parking).
  - Aggressive acceleration and regenerative braking cycles show up as higher C-rates and DoD,
    which correlate with faster SOH decline.

This tab is meant to be where you **tell the story** of your data science project to
a non-technical stakeholder.
"""
    )

# -----------------------------------------------------------------------------------
# 12. RUBRIC MAP TAB
# -----------------------------------------------------------------------------------
with tabs[11]:
    explain(
        "Rubric map",
        [
            "Explicit mapping from app features to project requirements.",
            "Useful when presenting and when grading.",
        ],
    )

    st.markdown("### Rubric coverage overview")

    rubric_items = [
        ("Data Collection & Preparation", "3+ datasets (Urban/Highway/Mixed, uploads), cleaning, metadata + env merges."),
        ("EDA & Visualizations", "Multiple tab-level plots: histograms, box, violin, scatter, 3D, scatter-matrix, heatmaps."),
        ("Data Processing & Feature Engineering", "Imputation (4 types), one-hot, ordinal, cyclical, scaling."),
        ("Model Development & Evaluation", "RandomForest & GradientBoosting, metrics, feature importance, forecasting."),
        ("Streamlit App Development", "Multi-tab, many interactive widgets, caching, polished layout."),
        ("GitHub & Documentation", "README + this rubric map + in-app explanations (to be completed in repo)."),
        ("Advanced Modelling", "MLPRegressor (neural net), ensemble (RandomForest), parallelism (n_jobs=-1)."),
        ("Specialised DS", "Time-series forecasting tab + NLP (TF–IDF)."),
        ("High-Performance Computing", "RandomForest with n_jobs=-1, efficient vectorised operations & caching."),
        ("Real-world Impact", "Real-World Insights tab translating patterns into recommendations."),
        ("Exceptional Visualisation", "Publication-style multi-colour plots, architecture schematic, comparison charts."),
    ]

    rmap = pd.DataFrame(
        [
            {"Category": cat, "How this app addresses it": desc}
            for cat, desc in rubric_items
        ]
    )
    st.dataframe(rmap, use_container_width=True)

    st.markdown(
        """
In your GitHub repo and README, you can copy/adapt this table to clearly communicate
how your project satisfies both the **base requirements** and the **“Above & Beyond”** items.
"""
    )
