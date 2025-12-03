# app.py
# Robust EV Battery SOH & RUL Dashboard

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from joblib import Parallel, delayed

# optional libs
try:
    from statsmodels.tsa.ar_model import AutoReg
    STATS_OK = True
except Exception:
    STATS_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False


# ------------------------------------------------------------------
# CONFIG & STYLING
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Robust EV Battery SOH & RUL",
    page_icon="🔋",
    layout="wide",
)

DARK_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #050816;
    color: #f5f5f5;
}
[data-testid="stSidebar"] {
    background-color: #0b1020;
}
h1, h2, h3, h4, h5, h6 {
    color: #ffffff;
}
.stMetric {
    background-color: #101522;
    padding: 8px 12px;
    border-radius: 8px;
}
.kpi-box {
    border-radius: 10px;
    padding: 10px 12px;
    background: linear-gradient(145deg,#101522,#181f33);
    border: 1px solid #252b40;
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
    opacity: 0.7;
}
.section-caption {
    font-size: 0.85rem;
    opacity: 0.8;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
EOL_THRESH = 0.80
MIN_LABELS_TRAIN = 40


def kpi(label, value, sub=""):
    if isinstance(value, float):
        vtxt = f"{value:,.3f}"
    elif isinstance(value, (int, np.integer)):
        vtxt = f"{value:,}"
    else:
        vtxt = str(value)
    html = f"""
    <div class="kpi-box">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{vtxt}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def explain(title: str, bullets):
    with st.expander(f"ℹ️ What this tab shows — {title}", expanded=False):
        for b in bullets:
            st.write(f"- {b}")


def numeric_cols(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def corr_heatmap(df: pd.DataFrame, title: str, key: str):
    cols = [c for c in numeric_cols(df) if df[c].notna().sum() > 10]
    if len(cols) < 2:
        st.info("Need at least two numeric columns with enough data.")
        return
    C = df[cols].corr()
    fig = px.imshow(
        C,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        template=PLOTLY_TEMPLATE,
        title=title,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def pct_missing(df: pd.DataFrame) -> float:
    if df.size == 0:
        return 0.0
    return 100.0 * df.isna().mean().mean()


# ------------------------------------------------------------------
# SYNTHETIC EV DATA (3 DATASETS WITH MISSINGNESS)
# ------------------------------------------------------------------
@st.cache_data
def generate_ev_dataset(profile: str, n_cells=4, n_cycles=260, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    params = {
        "Urban": dict(deg_rate=0.0010, temp0=30.0, temp_slope=0.03, c_rate=0.9, ambient=30),
        "Highway": dict(deg_rate=0.0006, temp0=25.0, temp_slope=0.02, c_rate=1.3, ambient=24),
        "Mixed": dict(deg_rate=0.0008, temp0=27.0, temp_slope=0.025, c_rate=1.1, ambient=27),
    }
    p = params.get(profile, params["Urban"])

    usage_templates = {
        "Urban": [
            "urban stop-and-go traffic",
            "frequent short trips with AC",
            "dense city driving with regen braking",
        ],
        "Highway": [
            "long constant-speed highway commute",
            "high-speed freeway with moderate hills",
            "intercity highway travel",
        ],
        "Mixed": [
            "mixed city and highway usage",
            "suburban commute with occasional highway",
            "balanced driving patterns",
        ],
    }

    for cell_idx in range(n_cells):
        cell_id = f"{profile[:3].upper()}_{cell_idx+1:02d}"
        cap0 = rng.normal(2.6, 0.04)

        for cyc in range(n_cycles):
            deg = p["deg_rate"]
            soh_mean = 1.0 - deg * cyc + rng.normal(0, 0.002)
            soh = float(np.clip(soh_mean, 0.6, 1.03))
            cap = cap0 * soh

            q_abs = rng.normal(0.30 + 0.005 * cyc, 0.04)
            q_abs = float(max(q_abs, 0.05))

            v_nom = 4.15 - 0.04 * (1 - soh)
            v_mean = rng.normal(v_nom, 0.015)
            v_std = abs(rng.normal(0.08 + 0.02 * (1 - soh), 0.01))

            e_abs = q_abs * v_mean + rng.normal(0, 0.05)

            temp_mean = rng.normal(p["temp0"] + p["temp_slope"] * cyc, 3.0)
            temp_max = temp_mean + abs(rng.normal(3.0, 1.0))

            curr_rms = rng.normal(p["c_rate"], 0.15)
            ambient = rng.normal(p["ambient"], 2.0)

            usage_text = rng.choice(usage_templates[profile])

            rows.append(
                dict(
                    dataset=profile,
                    cell_id=cell_id,
                    cycle=cyc,
                    soh=soh,
                    cap_ah=cap,
                    q_abs=q_abs,
                    e_abs=e_abs,
                    temp_mean=temp_mean,
                    temp_max=temp_max,
                    v_mean=v_mean,
                    v_std=v_std,
                    current_rms=curr_rms,
                    ambient_temp=ambient,
                    usage_text=usage_text,
                )
            )

    df = pd.DataFrame(rows)

    def bucket(soh):
        if pd.isna(soh):
            return "Missing"
        if soh >= 0.9:
            return "Healthy"
        if soh >= 0.85:
            return "Monitor"
        if soh >= EOL_THRESH:
            return "Aging"
        return "EOL"

    df["bucket"] = df["soh"].apply(bucket)

    # MCAR missingness
    rng = np.random.default_rng(seed + 123)
    m_mcar_q = rng.random(len(df)) < 0.06
    df.loc[m_mcar_q, "q_abs"] = np.nan

    m_mcar_v = rng.random(len(df)) < 0.04
    df.loc[m_mcar_v, "v_std"] = np.nan

    # MAR: hotter cycles have missing SOH and capacity
    high_temp = df["temp_max"] > (p["temp0"] + 12)
    mar_mask = high_temp & (rng.random(len(df)) < 0.35)
    df.loc[mar_mask, ["soh", "cap_ah"]] = np.nan

    # duplicates to show cleaning
    dup_rows = df.sample(40, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


@st.cache_data
def get_all_datasets():
    urban = generate_ev_dataset("Urban", seed=0)
    highway = generate_ev_dataset("Highway", seed=1)
    mixed = generate_ev_dataset("Mixed", seed=2)
    return {"Urban": urban, "Highway": highway, "Mixed": mixed}


def clean_data(df: pd.DataFrame):
    d = df.copy()
    n_before = len(d)
    d = d.drop_duplicates()
    n_after = len(d)

    if "temp_mean" in d.columns:
        d["temp_mean"] = d["temp_mean"].clip(-20, 90)
    if "temp_max" in d.columns:
        d["temp_max"] = d["temp_max"].clip(-20, 120)

    for col in ["dataset", "cell_id", "bucket"]:
        if col in d.columns:
            d[col] = d[col].astype("category")

    return d, n_before, n_after


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
all_datasets = get_all_datasets()
st.sidebar.title("Controls")

ds_names = list(all_datasets.keys())
selected_sources = st.sidebar.multiselect(
    "Select dataset(s)",
    ds_names,
    default=ds_names,
    help="Analyse one source, multiple sources, or all combined.",
)
if not selected_sources:
    selected_sources = ds_names

impute_choice = st.sidebar.selectbox(
    "Imputation method (modelling)",
    ["Simple (median)", "KNN (k=5)", "Iterative (MICE)"],
)
task_type = st.sidebar.radio("Modelling task", ["SOH regression", "Bucket classification"])

use_mlp = st.sidebar.checkbox("Include MLP (neural network)", value=True)
use_xgb = st.sidebar.checkbox("Include XGBoost model (if installed)", value=XGB_OK)

st.sidebar.markdown("---")
EOL_THRESH = st.sidebar.slider("EOL threshold (SOH)", 0.6, 0.95, 0.8, 0.01)
MIN_LABELS_TRAIN = st.sidebar.slider("Min labelled rows to train", 20, 200, 40, 5)

# ------------------------------------------------------------------
# BUILD COMBINED DATASET
# ------------------------------------------------------------------
cleaned_sources = {}
raw_sources = {}
for name in ds_names:
    raw = all_datasets[name]
    clean, n_before, n_after = clean_data(raw)
    raw_sources[name] = raw
    cleaned_sources[name] = clean

combined_raw = pd.concat([raw_sources[n] for n in selected_sources], ignore_index=True)
combined_clean, n_before_combined, n_after_combined = clean_data(combined_raw)
current_df = combined_clean.copy()

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tabs = st.tabs(
    [
        "🏠 Summary",
        "📦 Data Overview",
        "📊 EDA Gallery",
        "🧩 Missingness & Imputation",
        "🔁 Encoding & Models",
        "⏱ Time-Series & Forecast",
        "🌍 Insights & Story",
        "💾 Export",
    ]
)

# ------------------------------------------------------------------
# 1. SUMMARY
# ------------------------------------------------------------------
with tabs[0]:
    explain(
        "Summary dashboard",
        [
            "High-level KPIs for the selected dataset(s).",
            "SOH curves, energy throughput, bucket distribution, and missingness.",
        ],
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Data sources selected", len(selected_sources), ", ".join(selected_sources))
    with c2:
        kpi("Unique cells", int(current_df["cell_id"].nunique()), "cell_id")
    with c3:
        kpi("Rows (after cleaning)", len(current_df), f"from {n_before_combined} raw")
    with c4:
        kpi("Avg % missing", f"{pct_missing(current_df):.1f}%", "across all columns")

    st.markdown("### SOH and energy at a glance")
    left, mid, right = st.columns([1.4, 1.1, 1.1])

    with left:
        df_plot = current_df.dropna(subset=["soh"]).copy()
        if not df_plot.empty:
            fig = px.line(
                df_plot,
                x="cycle",
                y="soh",
                color="cell_id",
                template=PLOTLY_TEMPLATE,
                title="SOH vs cycle by cell",
                height=340,
            )
            fig.add_hline(
                y=EOL_THRESH,
                line_dash="dot",
                line_color="red",
                annotation_text="EOL threshold",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No SOH labels available for selected datasets.")

    with mid:
        if "e_abs" in current_df.columns:
            g = (
                current_df.groupby("cell_id", as_index=False)["e_abs"]
                .sum()
                .sort_values("e_abs", ascending=False)
            )
            fig2 = px.bar(
                g,
                x="cell_id",
                y="e_abs",
                template=PLOTLY_TEMPLATE,
                title="Total energy throughput per cell",
                height=340,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Feature e_abs not available.")

    with right:
        if "bucket" in current_df.columns:
            counts = (
                current_df["bucket"]
                .value_counts(dropna=False)
                .rename_axis("bucket")
                .reset_index(name="count")
            )
            fig3 = px.pie(
                counts,
                values="count",
                names="bucket",
                hole=0.5,
                template=PLOTLY_TEMPLATE,
                title="Health bucket distribution",
                height=340,
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Bucket labels not found.")

# ------------------------------------------------------------------
# 2. DATA OVERVIEW
# ------------------------------------------------------------------
with tabs[1]:
    explain(
        "Data overview",
        [
            "View the combined dataset after cleaning.",
            "Look at data types, value ranges and missingness per column.",
        ],
    )

    st.markdown("### Combined dataset (after basic cleaning)")
    st.dataframe(current_df.head(20), use_container_width=True)

    st.markdown("#### Column info & summary statistics")
    col1, col2 = st.columns([1.2, 1.4])

    with col1:
        dtype_df = pd.DataFrame(
            {
                "column": current_df.columns,
                "dtype": [str(current_df[c].dtype) for c in current_df.columns],
                "n_unique": [current_df[c].nunique(dropna=True) for c in current_df.columns],
                "pct_missing": [100 * current_df[c].isna().mean() for c in current_df.columns],
            }
        )
        st.dataframe(dtype_df, use_container_width=True)

    with col2:
        st.dataframe(
            current_df.describe(include="all").transpose(), use_container_width=True
        )

# ------------------------------------------------------------------
# 3. EDA GALLERY
# ------------------------------------------------------------------
with tabs[2]:
    explain(
        "EDA gallery",
        [
            "Multiple visualisations: histograms, boxplots, correlations, scatter.",
            "Mirrors the rich penguins EDA but for EV battery data.",
        ],
    )

    st.markdown("### Histograms of key numeric features")
    numc = [c for c in numeric_cols(current_df) if c not in ["cycle"]]
    if numc:
        for col in numc[:4]:
            st.write(f"**Histogram of {col}**")
            fig = px.histogram(
                current_df,
                x=col,
                nbins=30,
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Box plots for outlier inspection")
    if numc:
        # melt to long for a single box plot figure
        box_df = current_df[numc[:4]].melt(var_name="feature", value_name="value")
        fig_box = px.box(
            box_df,
            x="feature",
            y="value",
            template=PLOTLY_TEMPLATE,
            title="Box plots for selected numeric features",
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No numeric columns available for box plots.")

    st.markdown("### Correlation matrix")
    corr_heatmap(current_df, "Correlation heatmap (numeric features)", key="eda_corr")

    st.markdown("### Scatter plot")
    if len(numc) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            x_axis = st.selectbox("X-axis", numc, index=0, key="scatter_x")
        with c2:
            y_axis = st.selectbox("Y-axis", numc, index=1, key="scatter_y")

        color_by = st.selectbox(
            "Color by",
            ["dataset", "cell_id", "bucket"],
            index=0,
        )
        fig_sc = px.scatter(
            current_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            hover_data=["cycle", "cell_id"],
            template=PLOTLY_TEMPLATE,
            height=500,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ------------------------------------------------------------------
# 4. MISSINGNESS & IMPUTATION
# ------------------------------------------------------------------
with tabs[3]:
    explain(
        "Missingness & Imputation",
        [
            "Quantify and visualise missing data.",
            "Compare Simple / KNN / Iterative imputation RMSE.",
            "Connects directly to MCAR/MAR ideas from lecture.",
        ],
    )

    numc = numeric_cols(current_df)
    st.markdown("### Missing values summary")
    if numc:
        miss_cnt = current_df[numc].isna().sum()
        miss_pct = current_df[numc].isna().mean() * 100
        miss_df = pd.DataFrame(
            {"missing_count": miss_cnt, "missing_pct": miss_pct}
        ).sort_values("missing_pct", ascending=False)
        st.dataframe(miss_df, use_container_width=True)

        st.markdown("#### Missingness bar chart")
        miss_nonzero = miss_df[miss_df["missing_count"] > 0]
        if not miss_nonzero.empty:
            fig = px.bar(
                miss_nonzero.reset_index().rename(columns={"index": "column"}),
                x="column",
                y="missing_pct",
                template=PLOTLY_TEMPLATE,
                title="Percent missing by column",
                height=350,
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No missing values in numeric columns.")

        st.markdown("#### Missingness heatmap (rows × columns)")
        miss_matrix = current_df[numc].isna().astype(int)
        if not miss_matrix.empty:
            fig_hm = px.imshow(
                miss_matrix.transpose(),
                aspect="auto",
                color_continuous_scale="Viridis",
                template=PLOTLY_TEMPLATE,
                labels=dict(color="Missing"),
                title="Missingness heatmap (1=missing)",
            )
            st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No numeric columns, so missingness lab is trivial.")

    # imputation comparison
    st.markdown("---")
    st.markdown("### Imputation comparison for one column")

    if numc:
        target_col = st.selectbox(
            "Numeric column with missing values",
            numc,
            index=min(1, len(numc) - 1),
        )
        base = current_df[target_col]

        df_mcar = current_df.copy()
        rng = np.random.default_rng(7)
        idx = df_mcar.index.to_numpy()
        mask_extra = rng.choice(idx, size=int(0.1 * len(idx)), replace=False)
        df_mcar.loc[mask_extra, target_col] = np.nan

        results = []
        for label, imp in [
            ("Simple (median)", SimpleImputer(strategy="median")),
            ("KNN (k=5)", KNNImputer(n_neighbors=5)),
            ("Iterative (MICE)", IterativeImputer(random_state=7, max_iter=10)),
        ]:
            X = df_mcar[numc]
            imputed = imp.fit_transform(X)
            Ximp = pd.DataFrame(imputed, columns=numc, index=X.index)

            common = base.dropna()
            if target_col in Ximp.columns:
                rmse = float(
                    np.sqrt(
                        np.nanmean(
                            (Ximp.loc[common.index, target_col] - common) ** 2
                        )
                    )
                )
            else:
                rmse = np.nan
            results.append({"imputer": label, "RMSE_vs_original": rmse})

        res_df = pd.DataFrame(results).sort_values("RMSE_vs_original")
        st.dataframe(res_df, use_container_width=True)
        fig_imp = px.bar(
            res_df,
            x="imputer",
            y="RMSE_vs_original",
            template=PLOTLY_TEMPLATE,
            title=f"Imputation RMSE for {target_col}",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------------------------------
# ENCODING HELPERS
# ------------------------------------------------------------------
def build_encoded_matrices(df: pd.DataFrame, target: str, imputer_name: str):
    dfy = df.dropna(subset=[target]).copy()
    if dfy.empty:
        return None

    drop_cols = [target, "cap_ah"]
    feature_cols = [c for c in dfy.columns if c not in drop_cols]

    num_features = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(dfy[c]) and c != "cycle"
    ]
    cat_features = [
        c for c in feature_cols if dfy[c].dtype == "category" or dfy[c].dtype == "object"
    ]
    text_feature = "usage_text" if "usage_text" in cat_features else None
    if text_feature and text_feature in cat_features:
        cat_features = [c for c in cat_features if c != text_feature]

    X_struct = dfy[feature_cols].copy()
    y = dfy[target].copy()
    if target == "bucket":
        y = y.astype(str)

    cells = dfy["cell_id"].astype(str).values
    uniq = np.unique(cells)
    rng = np.random.default_rng(7)
    rng.shuffle(uniq)
    n_train_cells = max(1, int(0.7 * len(uniq)))
    train_cells = set(uniq[:n_train_cells])
    train_mask = np.isin(cells, list(train_cells))

    X_train_struct = X_struct.iloc[train_mask].copy()
    X_test_struct = X_struct.iloc[~train_mask].copy()
    y_train = y.iloc[train_mask].copy()
    y_test = y.iloc[~train_mask].copy()

    # text
    if text_feature:
        tfidf = TfidfVectorizer(max_features=80)
        X_text_all = tfidf.fit_transform(X_struct[text_feature].fillna("").astype(str))
        X_train_text = X_text_all[train_mask]
        X_test_text = X_text_all[~train_mask]
    else:
        tfidf = None
        X_train_text = X_test_text = None

    if imputer_name.startswith("Simple"):
        num_imputer = SimpleImputer(strategy="median")
    elif imputer_name.startswith("KNN"):
        num_imputer = KNNImputer(n_neighbors=5)
    else:
        num_imputer = IterativeImputer(random_state=7, max_iter=15)

    num_transformer = Pipeline(
        steps=[
            ("imputer", num_imputer),
            ("scaler", StandardScaler()),
        ]
    )
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if num_features:
        transformers.append(("num", num_transformer, num_features))
    if cat_features:
        transformers.append(("cat", cat_transformer, cat_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    X_train_struct_enc = preprocessor.fit_transform(X_train_struct)
    X_test_struct_enc = preprocessor.transform(X_test_struct)

    try:
        enc_feature_names = preprocessor.get_feature_names_out()
    except Exception:
        enc_feature_names = [f"f_{i}" for i in range(X_train_struct_enc.shape[1])]

    if X_train_text is not None:
        X_tr = np.hstack([X_train_struct_enc, X_train_text.toarray()])
        X_te = np.hstack([X_test_struct_enc, X_test_text.toarray()])
        feature_names_full = list(enc_feature_names) + [
            f"text_{i}" for i in range(X_train_text.shape[1])
        ]
    else:
        X_tr = X_train_struct_enc
        X_te = X_test_struct_enc
        feature_names_full = list(enc_feature_names)

    encoded_train_df = pd.DataFrame(
        X_tr, columns=feature_names_full, index=X_train_struct.index
    )

    mapping_rows = []
    for name in enc_feature_names:
        if name.startswith("num__"):
            raw_name = name.split("__", 1)[1]
            mapping_rows.append(
                dict(
                    encoded_column=name,
                    source_column=raw_name,
                    encoding_type="numeric → impute+scale",
                    category_value="—",
                )
            )
        elif name.startswith("cat__"):
            base = name.split("__", 1)[1]
            if "_" in base:
                src, cat_val = base.split("_", 1)
            else:
                src, cat_val = base, ""
            mapping_rows.append(
                dict(
                    encoded_column=name,
                    source_column=src,
                    encoding_type="categorical → one-hot",
                    category_value=cat_val,
                )
            )
        else:
            mapping_rows.append(
                dict(
                    encoded_column=name,
                    source_column="(other)",
                    encoding_type="other",
                    category_value="",
                )
            )
    encoding_map_df = pd.DataFrame(mapping_rows)

    return dict(
        X_tr=X_tr,
        X_te=X_te,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        tfidf=tfidf,
        feature_cols=feature_cols,
        train_struct=X_train_struct,
        encoded_train_df=encoded_train_df,
        encoding_map_df=encoding_map_df,
        dfy=dfy,
    )


# ------------------------------------------------------------------
# 5. ENCODING & MODELS
# ------------------------------------------------------------------
with tabs[4]:
    explain(
        "Encoding & Models",
        [
            "Table BEFORE encoding (raw).",
            "Design matrix AFTER encoding (all numeric).",
            "Encoding map (raw column → encoded column).",
            "Then model comparison and simple RUL estimate.",
        ],
    )

    target = "soh" if task_type == "SOH regression" else "bucket"
    enc = build_encoded_matrices(current_df, target, impute_choice)

    if enc is None or enc["dfy"].shape[0] < MIN_LABELS_TRAIN:
        st.info(
            f"Not enough labelled rows for target '{target}'. Need at least {MIN_LABELS_TRAIN}."
        )
    else:
        dfy = enc["dfy"]
        X_tr = enc["X_tr"]
        X_te = enc["X_te"]
        y_train = enc["y_train"]
        y_test = enc["y_test"]
        train_struct = enc["train_struct"]
        encoded_train_df = enc["encoded_train_df"]
        encoding_map_df = enc["encoding_map_df"]

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi("Rows with label", len(dfy))
        with c2:
            kpi("Train rows", len(y_train))
        with c3:
            kpi("Test rows", len(y_test))

        st.markdown("### 🔤 Before encoding (raw features)")
        show_cols = ["dataset", "cell_id", "cycle", target]
        extra = [
            c
            for c in [
                "q_abs",
                "e_abs",
                "temp_mean",
                "temp_max",
                "v_mean",
                "v_std",
                "bucket",
            ]
            if c in dfy.columns
        ]
        show_cols += extra
        show_cols = [c for c in show_cols if c in dfy.columns]
        st.dataframe(train_struct[show_cols].head(10), use_container_width=True)
        st.caption(
            "Above: original features, including numeric (q_abs, e_abs, temperatures, voltages), "
            "categorical (dataset, cell_id, bucket) and text (usage_text, not shown here)."
        )

        st.markdown("### 🔁 After encoding (design matrix)")
        st.dataframe(encoded_train_df.head(10), use_container_width=True)
        st.caption(
            "After encoding: numeric → imputed + standardised, categorical → one-hot 0/1, "
            "text (usage_text) → TF-IDF `text_*` features."
        )

        st.markdown("### 🧬 Encoding map (raw → encoded)")
        st.dataframe(encoding_map_df, use_container_width=True)

        # ---- modelling ----
        st.markdown("---")
        st.subheader(f"Model comparison for target: **{target}**")

        models = {}
        if target == "soh":
            models["RandomForest"] = RandomForestRegressor(
                n_estimators=250,
                max_depth=None,
                random_state=7,
                n_jobs=-1,
            )
            models["GradientBoosting"] = GradientBoostingRegressor(random_state=7)
            if use_mlp:
                models["MLP"] = MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=400,
                    alpha=1e-3,
                    random_state=7,
                )
            if use_xgb and XGB_OK:
                models["XGBoost"] = xgb.XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=7,
                    n_jobs=-1,
                )
        else:
            models["RandomForest"] = RandomForestClassifier(
                n_estimators=250,
                max_depth=None,
                random_state=7,
                n_jobs=-1,
            )
            models["GradientBoosting"] = GradientBoostingClassifier(random_state=7)
            if use_mlp:
                models["MLP"] = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=400,
                    alpha=1e-3,
                    random_state=7,
                )

        rows = []
        preds_store = {}
        for name, model in models.items():
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            preds_store[name] = y_pred

            if target == "soh":
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rows.append({"model": name, "MAE": mae, "R2": r2})
            else:
                acc = accuracy_score(y_test, y_pred)
                rows.append({"model": name, "Accuracy": acc})

        res_df = pd.DataFrame(rows)
        st.dataframe(res_df, use_container_width=True)

        if target == "soh":
            fig_m = px.bar(
                res_df,
                x="model",
                y="MAE",
                color="R2",
                template=PLOTLY_TEMPLATE,
                title="SOH regression performance",
            )
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            fig_m = px.bar(
                res_df,
                x="model",
                y="Accuracy",
                template=PLOTLY_TEMPLATE,
                title="Bucket classification accuracy",
            )
            st.plotly_chart(fig_m, use_container_width=True)

        # Hyperparameter tuning
        st.markdown("### 🎛 Hyperparameter tuning (RandomizedSearchCV on RandomForest)")
        if target == "soh":
            base_rf = RandomForestRegressor(random_state=7, n_jobs=-1)
            scoring = "neg_mean_absolute_error"
        else:
            base_rf = RandomForestClassifier(random_state=7, n_jobs=-1)
            scoring = "accuracy"

        param_dist = {
            "n_estimators": [150, 250, 350],
            "max_depth": [None, 6, 10],
            "min_samples_leaf": [1, 2, 5],
        }

        search = RandomizedSearchCV(
            base_rf,
            param_distributions=param_dist,
            n_iter=6,
            cv=3,
            scoring=scoring,
            n_jobs=-1,
            random_state=7,
        )
        search.fit(X_tr, y_train)
        y_tuned = search.predict(X_te)

        if target == "soh":
            mae_tuned = mean_absolute_error(y_test, y_tuned)
            r2_tuned = r2_score(y_test, y_tuned)
            st.write(
                f"Best params: `{search.best_params_}` — MAE={mae_tuned:.4f}, R²={r2_tuned:.3f}"
            )
        else:
            acc_tuned = accuracy_score(y_test, y_tuned)
            st.write(f"Best params: `{search.best_params_}` — Accuracy={acc_tuned:.3f}")

        st.caption(
            "RandomizedSearchCV with n_jobs=-1 shows parallel hyperparameter search "
            "(high-performance computing)."
        )

        # Simple RUL estimate for regression target
        if target == "soh":
            st.markdown("---")
            st.subheader("🔮 Simple RUL estimate (Remaining Useful Life)")

            best_name = (
                res_df.sort_values("MAE").iloc[0]["model"]
                if not res_df.empty
                else "RandomForest"
            )
            best_model = models[best_name]

            rul_rows = []
            for cell, gcell in dfy.groupby("cell_id"):
                gcell = gcell.sort_values("cycle")
                Xc_struct = gcell[enc["feature_cols"]]
                Xc_enc = enc["preprocessor"].transform(Xc_struct)
                if enc["tfidf"] is not None and "usage_text" in Xc_struct.columns:
                    Xt_cell = enc["tfidf"].transform(
                        Xc_struct["usage_text"].fillna("").astype(str)
                    )
                    Xc = np.hstack([Xc_enc, Xt_cell.toarray()])
                else:
                    Xc = Xc_enc

                soh_hat = best_model.predict(Xc)
                cyc = gcell["cycle"].astype(int).values
                if len(cyc) < 4:
                    continue
                m, b = np.polyfit(cyc, soh_hat, 1)
                if m >= 0:
                    continue
                eol_cycle = int((EOL_THRESH - b) / m)
                current_cycle = int(cyc.max())
                rul = max(eol_cycle - current_cycle, 0)
                rul_rows.append(
                    dict(
                        cell_id=cell,
                        current_cycle=current_cycle,
                        predicted_eol_cycle=eol_cycle,
                        RUL_cycles=rul,
                    )
                )

            if rul_rows:
                st.dataframe(
                    pd.DataFrame(rul_rows).sort_values("RUL_cycles"),
                    use_container_width=True,
                )
            else:
                st.info(
                    "SOH trend not decreasing enough to estimate RUL reliably for these cells."
                )

# ------------------------------------------------------------------
# 6. TIME-SERIES & FORECAST
# ------------------------------------------------------------------
with tabs[5]:
    explain(
        "Time-series & forecast",
        [
            "Treat SOH as a time series and fit an AutoReg model.",
            "Shows forecasting and temporal reasoning for EV batteries.",
        ],
    )

    if not STATS_OK:
        st.info(
            "statsmodels is not installed, so the AutoReg forecast demo is disabled. "
            "Add `statsmodels` to requirements.txt to enable it."
        )
    else:
        dfy = current_df.dropna(subset=["soh"]).copy()
        if dfy.empty:
            st.info("No SOH labels available for forecasting.")
        else:
            cell_options = sorted(dfy["cell_id"].astype(str).unique())
            sel_cell = st.selectbox("Select cell for forecast", cell_options)
            g = dfy[dfy["cell_id"].astype(str) == sel_cell].sort_values("cycle")
            series = g["soh"].astype(float).values
            cycles = g["cycle"].values

            if len(series) < 20:
                st.info("Need at least 20 points to fit an AutoReg model.")
            else:
                N = min(80, len(series))
                s_train = series[-N:]
                c_train = cycles[-N:]
                model = AutoReg(s_train, lags=5, old_names=False).fit()
                steps = 20
                forecast = model.predict(start=len(s_train), end=len(s_train) + steps - 1)
                cyc_future = np.arange(c_train[-1] + 1, c_train[-1] + 1 + steps)

                fig = px.line(
                    x=c_train, y=s_train, template=PLOTLY_TEMPLATE,
                    labels={"x": "Cycle", "y": "SOH"},
                    title=f"AutoReg forecast for cell {sel_cell}",
                )
                fig.add_scatter(x=cyc_future, y=forecast, mode="lines+markers", name="Forecast")
                fig.add_hline(y=EOL_THRESH, line_dash="dot", line_color="red", annotation_text="EOL")
                st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 7. INSIGHTS & STORY
# ------------------------------------------------------------------
with tabs[6]:
    explain(
        "Insights & Story",
        [
            "High-level narrative for the report/presentation.",
            "Real-world recommendations for EV fleets based on results.",
        ],
    )

    st.markdown("### Who is the audience?")
    st.write(
        "- **Audience**: EV fleet managers / battery engineers who want to monitor "
        "battery health and plan maintenance or warranty policies."
    )

    st.markdown("### What does the app help them do?")
    st.write(
        """
        - Compare **Urban**, **Highway**, and **Mixed** duty cycles side-by-side.
        - See how **temperature**, **energy throughput**, and **current** relate to SOH degradation.
        - Understand where the data are **missing** and how different **imputation** schemes change the picture.
        - Train and compare multiple predictive models for **SOH** (and optionally health **buckets**).
        - Get a first-cut estimate of **Remaining Useful Life (RUL)** in cycles.
        - Explore a simple **time-series forecast** of future SOH.
        """
    )

    st.markdown("### Real-world recommendations (examples)")
    st.write(
        """
        - **Temperature control**: High `temp_max` cycles accelerate degradation and are associated with more
          missing SOH measurements. Fleet operators should monitor thermal events and reduce exposure to very
          high pack temperatures.
        - **Usage-dependent maintenance**: Urban profiles degrade faster than highway profiles, suggesting
          different maintenance intervals or warranty windows by usage class.
        - **Missing data strategy**: Comparing Simple, KNN, and Iterative imputation shows that more
          sophisticated methods can reduce bias when data are MAR (Missing At Random).
        - **Model choice & uncertainty**: Tree-based ensembles (RandomForest / GradientBoosting) provide strong
          baselines. MLP adds nonlinearity at the cost of interpretability. Using multiple models and comparing
          their errors gives a sense of uncertainty.
        """
    )

# ------------------------------------------------------------------
# 8. EXPORT
# ------------------------------------------------------------------
with tabs[7]:
    explain(
        "Export",
        [
            "Download a cleaned per-cycle dataset for further analysis.",
            "This CSV is what you would commit to GitHub as the main artifact.",
        ],
    )

    st.markdown("### Download cleaned dataset (current selection)")
    st.write(
        f"Rows: **{len(current_df)}**, columns: **{len(current_df.columns)}**. "
        f"Sources: **{', '.join(selected_sources)}**."
    )
    csv_bytes = current_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name="ev_battery_cleaned.csv",
        mime="text/csv",
    )

    st.caption(
        "Tip: put this CSV in `data/` in your GitHub repo and describe all columns in a data dictionary."
    )
