# app.py
# Robust EV Battery SOH & RUL:
# A Missing-Data–Aware, Multi-Dataset Analytics & Visualization Framework
#
# Run locally:
#   streamlit run app.py

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import plotly.graph_objects as go          # ← this one is missing
from plotly.subplots import make_subplots  # ← for the subplot histograms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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
    confusion_matrix,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

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

# -------------------------------------------------------------------
# STREAMLIT CONFIG & DARK THEME
# -------------------------------------------------------------------
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
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_SEQ = px.colors.qualitative.Set2
EOL_THRESH_DEFAULT = 0.80
MIN_LABELS_TRAIN_DEFAULT = 40


# -------------------------------------------------------------------
# SMALL HELPERS
# -------------------------------------------------------------------
def kpi(label, value, sub=""):
    """Nice KPI box with title, value, and subtitle."""
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
    """Expandable explanation at top of each tab."""
    with st.expander(f"ℹ️ What this tab shows — {title}", expanded=False):
        for b in bullets:
            st.write(f"- {b}")


def numeric_cols(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def corr_heatmap(df: pd.DataFrame, title: str, key: str):
    cols = [c for c in numeric_cols(df) if df[c].notna().sum() > 10]
    cols = cols[:10]
    if len(cols) < 2:
        st.info("Need at least two numeric columns with enough non-missing values.")
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
    st.caption(
        "Interpretation: dark red/blue cells indicate strong positive/negative correlation "
        "between features. This helps detect redundancy and multicollinearity."
    )


def pct_missing(df: pd.DataFrame) -> float:
    if df.size == 0:
        return 0.0
    return 100.0 * df.isna().mean().mean()


def plot_nn_architecture(layer_sizes, title="Neural network architecture"):
    """
    Simple architecture diagram: each layer = column of nodes.
    Example: [n_input, 128, 64, 32, n_output]
    """
    max_nodes_display = 10
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")

    n_layers = len(layer_sizes)
    x_spacing = 1.0 / (n_layers + 1)

    for i, size in enumerate(layer_sizes):
        x = (i + 1) * x_spacing
        display_size = min(size, max_nodes_display)
        y_positions = np.linspace(0.1, 0.9, display_size)

        ax.scatter(
            [x] * display_size,
            y_positions,
            s=100,
            color="#00e3ff" if 0 < i < n_layers - 1 else "#ffdd00",
            edgecolors="k",
            linewidths=0.5,
        )

        if i == 0:
            label = f"Input\n({size})"
        elif i == n_layers - 1:
            label = f"Output\n({size})"
        else:
            label = f"Hidden {i}\n({size})"

        ax.text(x, 0.97, label, ha="center", va="top", fontsize=8, color="white")

    ax.set_title(title, color="white")
    return fig


def limit_rows(df: pd.DataFrame, max_rows: int = 2500) -> pd.DataFrame:
    """Safety: downsample large tables for heavy plots or models."""
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=7)



#helper for rf tuning

@st.cache_data(show_spinner=False)
def run_rf_tuning(X_df: pd.DataFrame, y: np.ndarray):
    """
    Run a small GridSearchCV for RandomForestRegressor and return
    a full cv_results_ DataFrame with positive MAE.
    """
    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [50, 100, 200, 400],
        "max_depth": [None, 5, 10, 20],
    }

    grid = GridSearchCV(
        rf,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        return_train_score=False,
    )

    grid.fit(X_df, y)

    cvres = pd.DataFrame(grid.cv_results_)
    cvres["mae"] = -cvres["mean_test_score"]  # convert from negative MAE
    return cvres


# -------------------------------------------------------------------
# EV DATA (BASE SOURCES: Urban / Highway / Mixed)
# -------------------------------------------------------------------
@st.cache_data
def generate_ev_dataset(profile: str, n_cells=4, n_cycles=260, seed: int = 0) -> pd.DataFrame:
    """
    EV dataset for one usage profile:
    - 'Urban', 'Highway', or 'Mixed'
    - Per-cycle features, SOH, capacity, usage text
    - MCAR + MAR missingness
    """
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
        if soh >= EOL_THRESH_DEFAULT:
            return "Aging"
        return "EOL"

    df["bucket"] = df["soh"].apply(bucket)

    rng = np.random.default_rng(seed + 123)
    m_mcar_q = rng.random(len(df)) < 0.06
    df.loc[m_mcar_q, "q_abs"] = np.nan
    m_mcar_v = rng.random(len(df)) < 0.04
    df.loc[m_mcar_v, "v_std"] = np.nan

    high_temp = df["temp_max"] > (p["temp0"] + 12)
    mar_mask = high_temp & (rng.random(len(df)) < 0.35)
    df.loc[mar_mask, ["soh", "cap_ah"]] = np.nan

    dup_rows = df.sample(40, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


@st.cache_data
def get_base_sources():
    return {
        "Urban": generate_ev_dataset("Urban", seed=0),
        "Highway": generate_ev_dataset("Highway", seed=1),
        "Mixed": generate_ev_dataset("Mixed", seed=2),
    }


def build_metadata_tables(per_cycle_sources: dict[str, pd.DataFrame]):
    cells = []
    for dsname, df in per_cycle_sources.items():
        if "cell_id" not in df.columns:
            continue
        for cid in sorted(pd.Series(df["cell_id"]).astype(str).unique()):
            cells.append((dsname, cid))

    rng = np.random.default_rng(42)
    meta_rows = []
    for ds, cid in cells:
        manufacturer = rng.choice(["OEM_A", "OEM_B", "OEM_C"])
        cooling = rng.choice(["air", "liquid"])
        segment = rng.choice(["compact", "sedan", "SUV"])
        meta_rows.append(
            dict(
                dataset=ds,
                cell_id=cid,
                manufacturer=manufacturer,
                cooling=cooling,
                vehicle_segment=segment,
            )
        )
    cell_metadata = pd.DataFrame(meta_rows)

    env_rows = []
    for ds in per_cycle_sources.keys():
        if ds == "Urban":
            env_rows.append(dict(dataset=ds, region="hot city", climate_index="hot-humid"))
        elif ds == "Highway":
            env_rows.append(dict(dataset=ds, region="mild corridor", climate_index="temperate"))
        elif ds == "Mixed":
            env_rows.append(dict(dataset=ds, region="mixed region", climate_index="temperate"))
        else:
            env_rows.append(dict(dataset=ds, region="uploaded region", climate_index="unknown"))
    env_profile = pd.DataFrame(env_rows)

    return cell_metadata, env_profile


def feature_engineering(df: pd.DataFrame):
    d = df.copy()

    if {"temp_max", "temp_mean"}.issubset(d.columns):
        d["temp_spread"] = d["temp_max"] - d["temp_mean"]
    else:
        d["temp_spread"] = np.nan

    if {"temp_max", "current_rms"}.issubset(d.columns):
        d["stress_index"] = (d["temp_max"] - d["temp_max"].min()) / (
            d["temp_max"].max() - d["temp_max"].min() + 1e-6
        ) + (d["current_rms"] - d["current_rms"].min()) / (
            d["current_rms"].max() - d["current_rms"].min() + 1e-6
        )
    else:
        d["stress_index"] = np.nan

    if "cap_ah" in d.columns and "q_abs" in d.columns and "e_abs" in d.columns:
        d["q_norm"] = d["q_abs"] / (d["cap_ah"] + 1e-6)
        d["e_norm"] = d["e_abs"] / (d["cap_ah"] + 1e-6)
    else:
        d["q_norm"] = np.nan
        d["e_norm"] = np.nan

    if "cycle" in d.columns:
        d["cycle_bin"] = pd.cut(
            d["cycle"],
            bins=[-1, 50, 150, 1000],
            labels=["early", "mid", "late"],
        )
    else:
        d["cycle_bin"] = "unknown"

    return d


def clean_and_integrate(per_cycle_sources, cell_metadata, env_profile):
    cleaned_sources = {}
    for name, df in per_cycle_sources.items():
        d = df.copy()
        d = d.drop_duplicates()
        if "dataset" not in d.columns:
            d["dataset"] = name
        d["dataset"] = d["dataset"].astype("category")
        if "cell_id" not in d.columns:
            d["cell_id"] = f"{name}_CELL"
        d["cell_id"] = d["cell_id"].astype("category")
        if "cycle" in d.columns:
            d["cycle"] = pd.to_numeric(d["cycle"], errors="coerce").fillna(0).astype(int)
        if "bucket" in d.columns:
            d["bucket"] = d["bucket"].astype("category")
        cleaned_sources[name] = d

    combined = pd.concat(list(cleaned_sources.values()), ignore_index=True)
    combined = combined.merge(
        cell_metadata,
        on=["dataset", "cell_id"],
        how="left",
        validate="m:1",
    )
    combined = combined.merge(
        env_profile,
        on="dataset",
        how="left",
        validate="m:1",
    )
    combined = feature_engineering(combined)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    for col in [
        "manufacturer",
        "cooling",
        "vehicle_segment",
        "region",
        "climate_index",
        "cycle_bin",
    ]:
        if col in combined.columns:
            combined[col] = combined[col].astype("category")

    return cleaned_sources, combined


def build_encoded_matrices(df: pd.DataFrame, target: str, imputer_name: str):
    """
    Returns dict with:
      X_tr, X_te, y_train, y_test, preprocessor, tfidf,
      feature_cols, train_struct, encoded_train_df, encoding_map_df, dfy, num_features, cat_features, text_feature
    """
    dfy = df.dropna(subset=[target]).copy()
    if dfy.empty:
        return None

    dfy = dfy.loc[:, ~dfy.columns.duplicated()]

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

    stratify = y if target == "bucket" and y.nunique() > 1 else None
    X_train_struct, X_test_struct, y_train, y_test = train_test_split(
        X_struct,
        y,
        test_size=0.3,
        random_state=7,
        stratify=stratify,
    )

    # downsample training for heavy models
    if len(X_train_struct) > 1800:
        idx = X_train_struct.sample(1800, random_state=7).index
        X_train_struct = X_train_struct.loc[idx]
        y_train = y_train.loc[idx]

    if text_feature:
        tfidf = TfidfVectorizer(max_features=80)
        X_train_text = tfidf.fit_transform(
            X_train_struct[text_feature].fillna("").astype(str)
        )
        X_test_text = tfidf.transform(
            X_test_struct[text_feature].fillna("").astype(str)
        )
    else:
        tfidf = None
        X_train_text = X_test_text = None

    if imputer_name.startswith("Simple"):
        num_imputer = SimpleImputer(strategy="median")
    elif imputer_name.startswith("KNN"):
        num_imputer = KNNImputer(n_neighbors=5)
    else:
        num_imputer = IterativeImputer(random_state=7, max_iter=8)

    num_transformer = Pipeline(
        steps=[("imputer", num_imputer), ("scaler", StandardScaler())]
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
        num_features=num_features,
        cat_features=cat_features,
        text_feature=text_feature,
    )


# -------------------------------------------------------------------
# SIDEBAR: DATA SOURCES + CONTROLS
# -------------------------------------------------------------------
st.sidebar.title("Controls")

st.sidebar.subheader("Data sources")
st.sidebar.caption(
    "Uploaded data: **Urban**, **Highway**, **Mixed**.\n"
    "You can also upload **multiple CSV files** with your own EV per‑cycle data.\n"
    "Each file becomes a separate dataset (Upload_1, Upload_2, ...)."
)

uploaded_files = st.sidebar.file_uploader(
    "Upload extra EV per-cycle CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

base_sources = get_base_sources()
per_cycle_sources = base_sources.copy()

if uploaded_files:
    for i, f in enumerate(uploaded_files, start=1):
        try:
            df_up = pd.read_csv(f)
            ds_name = f"Upload_{i}"
            df_up = df_up.copy()
            df_up["dataset"] = ds_name
            if "cell_id" not in df_up.columns:
                df_up["cell_id"] = f"{ds_name}_CELL"
            if "cycle" not in df_up.columns:
                df_up["cycle"] = np.arange(len(df_up))
            per_cycle_sources[ds_name] = df_up
        except Exception as e:
            st.sidebar.warning(f"Could not read {f.name}: {e}")

cell_metadata, env_profile = build_metadata_tables(per_cycle_sources)
cleaned_sources, combined_all = clean_and_integrate(per_cycle_sources, cell_metadata, env_profile)

ds_names = list(per_cycle_sources.keys())
selected_sources = st.sidebar.multiselect(
    "Select dataset(s) to analyse",
    ds_names,
    default=ds_names,
)
if not selected_sources:
    selected_sources = ds_names

current_df = combined_all[combined_all["dataset"].isin(selected_sources)].copy()
current_df = current_df.loc[:, ~current_df.columns.duplicated()]

impute_choice = st.sidebar.selectbox(
    "Imputation method (for modelling)",
    ["Simple (median)", "KNN (k=5)", "Iterative (MICE)"],
    index=0,
)

task_type = st.sidebar.radio("Modelling task", ["SOH regression", "Bucket classification"])

st.sidebar.markdown("---")
EOL_THRESH = st.sidebar.slider("EOL threshold (SOH)", 0.6, 0.95, EOL_THRESH_DEFAULT, 0.01)
MIN_LABELS_TRAIN = st.sidebar.slider(
    "Min labelled rows required to train a model",
    20,
    200,
    MIN_LABELS_TRAIN_DEFAULT,
    5,
)

if "last_results" not in st.session_state:
    st.session_state["last_results"] = {}

# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------
tabs = st.tabs(
    [
        "📖 Introduction",
        "🏠 Summary",
        "📦 Data Overview",
        "📊 EDA & Viz Gallery",
        "🧩 Missingness Lab",
        "🔁 Encoding & Classical Models",
        "🧠 Deep Learning & Ensembles",
        "🔮 Predictions & Forecasting",
        "🌍 Insights & Rubric",
        "💾 Export",
    ]
)

# -------------------------------------------------------------------
# 0. INTRODUCTION TAB
# -------------------------------------------------------------------
with tabs[0]:
    st.title("🔋 Robust EV Battery SOH & RUL Dashboard")

    st.markdown(
        """
        This app is a full **end‑to‑end data science project** for **EV battery health**. This project presents a comprehensive framework 
        for analyzing and forecasting the health of electric vehicle (EV) batteries, focusing on State of Health (SOH) and Remaining Useful Life (RUL).
The framework is missing-data–aware, supporting multiple datasets with advanced preprocessing, visualization, and predictive modeling. 
It is designed as a user-friendly Streamlit web application for interactive exploration, comparison, and insights generation.

       
        You can:
        - Work with 3 datasets: **Urban**, **Highway**, **Mixed**.
        - You can also upload **multiple CSV files** → each becomes `Upload_1`, `Upload_2`, etc.
        - Analyse them **individually**, **side‑by‑side**, or **all combined**.

        

        We include:
        - **Missing data** (MCAR + MAR)
        - **Feature engineering** (thermal, energy, stress)
        - **Encoding** (numeric, categorical, text → TF‑IDF)
        - **Classical models** (Linear Regression, RandomForest, GradientBoosting)
        - **Advanced models** (deep neural network MLP, XGBoost)
        - **Time series forecasting** (SOH)
        - **Rich visualizations**: histograms, box/violin, scatter, 3D scatter,
          scatter matrix, PCA, correlation & missingness heatmaps, etc.

   

        To reduce energy use and protect the environment, many countries are shifting toward clean alternatives to fossil fuel vehicle power sources. 
        Lithium ion batteries have become widely used because they offer high energy density, long life, and no memory effect. 
        As these batteries operate over time, their capacity decreases and their internal resistance increases. 
        This process is known as battery aging and it affects both performance and safety, as well as the accuracy of important indicators such as the 
        State of Charge (SOC). SOC represents the amount of usable charge left in the battery at any moment.
        A battery is considered to have reached the end of its life when its capacity falls to about 80 percent of its original value or when its 
        internal resistance doubles. At this stage the battery can no longer meet performance needs and continued use may create safety risks. 
        For this reason it is important to estimate the State of Health (SOH) and the Remaining Useful Life (RUL). 
        SOH describes the overall aging condition of the battery compared to when it was new, while RUL indicates how much time or 
        how many cycles remain before the battery reaches the end of its life. RUL depends directly on SOH.
        Because SOH and RUL cannot be measured during operation, they must be estimated from battery behavior such as voltage, 
        temperature, charge and energy data. Recent data driven methods and feature extraction techniques are improving the accuracy of these estimates 
        and supporting safer and more efficient battery use.

        """
    )

    st.markdown("#### Quick tab guide")
    st.write(
        """
        1. **📖 Introduction** – story & tab descriptions.  
        2. **🏠 Summary** – KPIs + dataset mix + key plots.  
        3. **📦 Data Overview** – table, data types, stats, per‑dataset summary.  
        4. **📊 EDA & Viz Gallery** – all the classic EDA plots from lecture.  
        5. **🧩 Missingness Lab** – missing patterns + imputation comparison.  
        6. **🔁 Encoding & Classical Models** – before/after encoding + RF/GB/LR + RF tuning.  
        7. **🧠 Deep Learning & Ensembles** – neural net (MLP) + XGBoost (if installed).  
        8. **🔮 Predictions & Forecasting** – RUL and SOH time‑series forecast.  
        9. **🌍 Insights & Rubric** – real‑world conclusions & rubric mapping.  
        10. **💾 Export** – download cleaned & engineered data for GitHub.
        """
    )

# -------------------------------------------------------------------
# 1. SUMMARY TAB
# -------------------------------------------------------------------
with tabs[1]:
    explain(
        "Summary dashboard",
        [
            "High-level KPIs for the selected dataset(s).",
            "Dataset mix across Urban / Highway / Mixed / Uploads.",
            "SOH curves, energy throughput, health buckets, and overall missingness.",
        ],
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Data sources selected", len(selected_sources), ", ".join(selected_sources))
    with c2:
        kpi("Unique cells", int(current_df["cell_id"].nunique()), "cell_id")
    with c3:
        kpi("Rows (after cleaning & FE)", len(current_df))
    with c4:
        kpi("Avg % missing", f"{pct_missing(current_df):.1f}%", "across all columns")

    st.markdown("### Dataset mix (Urban + Highway + Mixed + Uploads)")
    ds_counts = (
        current_df["dataset"]
        .astype(str)
        .value_counts()
        .rename_axis("dataset")
        .reset_index(name="rows")
    )
    st.dataframe(ds_counts, use_container_width=True)
    fig_ds_mix = px.bar(
        ds_counts,
        x="dataset",
        y="rows",
        template=PLOTLY_TEMPLATE,
        color="dataset",
        color_discrete_sequence=COLOR_SEQ,
        title="Row count per dataset in current selection",
    )
    st.plotly_chart(fig_ds_mix, use_container_width=True)
    st.caption(
        "Interpretation: this shows how many rows come from each dataset "
        "(Urban/Highway/Mixed/Uploads). If one dominates, your models may be biased "
        "toward that usage pattern."
    )

    st.markdown("### SOH & energy overview")
    left, mid, right = st.columns([1.4, 1.1, 1.1])

    with left:
        df_plot = current_df.dropna(subset=["soh"]).copy()
        if not df_plot.empty:
            fig = px.line(
                df_plot,
                x="cycle",
                y="soh",
                color="cell_id",
                line_group="cell_id",
                facet_row="dataset",
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=COLOR_SEQ,
                height=350,
                title="SOH vs cycle by cell / dataset",
            )
            fig.add_hline(
                y=EOL_THRESH,
                line_dash="dot",
                line_color="red",
                annotation_text="EOL",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Interpretation: each line is a cell's SOH (State of Health) vs cycle. "
                "A downward trend indicates degradation. The dashed red line marks the "
                f"EOL threshold (SOH={EOL_THRESH:.2f})."
            )
        else:
            st.info("No SOH labels available for selected datasets.")

    with mid:
        if "e_abs" in current_df.columns:
            g = (
                current_df.groupby(["dataset", "cell_id"], as_index=False)["e_abs"]
                .sum()
                .sort_values("e_abs", ascending=False)
            )
            fig2 = px.bar(
                g,
                x="cell_id",
                y="e_abs",
                color="dataset",
                color_discrete_sequence=COLOR_SEQ,
                template=PLOTLY_TEMPLATE,
                title="Total energy throughput per cell",
                height=350,
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                "Interpretation: higher bars = cells that delivered more energy over "
                "their lifetime. These cells are more 'worked' and may degrade faster."
            )
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
                color="bucket",
                color_discrete_sequence=COLOR_SEQ,
                title="Health bucket distribution",
                height=350,
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.caption(
                "Interpretation: slices show what fraction of rows belong to each "
                "health bucket (Healthy / Monitor / Aging / EOL / Missing). "
                "An operator ideally wants most cells in the 'Healthy' segment."
            )
        else:
            st.info("Bucket labels not found.")

    st.markdown("### SOH vs stress_index (scatter)")
    if "soh" in current_df.columns and "stress_index" in current_df.columns:
        fig_sc = px.scatter(
            current_df,
            x="stress_index",
            y="soh",
            color="dataset",
            color_discrete_sequence=COLOR_SEQ,
            hover_data=["cycle", "cell_id"],
            template=PLOTLY_TEMPLATE,
            title="SOH vs stress_index",
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption(
            "Interpretation: points with high stress_index and low SOH are 'problem' "
            "cells. stress_index blends temperature and current to summarise how harsh "
            "the cycling has been."
        )

# -------------------------------------------------------------------
# 2. DATA OVERVIEW TAB
# ------------------# -------------------------------------------------------------------
with tabs[2]:
    explain(
        "EDA Gallery",
        [
            "Multiple visualisations (histograms, boxplots, scatter, correlation heatmaps, 3D plots, parallel coordinates).",
            "Shows distributions and relationships across datasets and features.",
        ],
    )

    # Use the CURRENT selection from the sidebar
    df = current_df.copy()
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c not in ["note"]]

    # Quick KPIs
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Rows", int(len(df)), "current selection")
    with c2:
        kpi("Cells", int(df["cell_id"].nunique()), "unique cell_id")
    with c3:
        kpi("Datasets", int(df["dataset"].nunique()), ", ".join(sorted(df["dataset"].unique())))

    # ------------------------------------------------------------------
    # Histograms for a few key numeric variables
    # ------------------------------------------------------------------
    st.markdown("### Histograms of key numeric variables")

    top_num = [c for c in ["soh", "soc", "temp_avg", "current_avg", "voltage_avg"] if c in num_cols][:4]
    if len(top_num) >= 1:
        n_plots = len(top_num)
        fig = make_subplots(rows=1, cols=n_plots, subplot_titles=top_num)
        for i, col in enumerate(top_num, start=1):
            fig.add_trace(
                go.Histogram(
                    x=df[col],
                    name=col,
                    marker_color=COLOR_SEQ[i - 1],
                    opacity=0.8,
                ),
                row=1,
                col=i,
            )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            showlegend=False,
            title="Histograms (current selection)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "These histograms show the distribution of key numeric features like SOH, SOC, and temperature."
        )

    # ------------------------------------------------------------------
    # Boxplots & scatter
    # ------------------------------------------------------------------
        # ------------------------------------------------------------------
    # Boxplots & scatter
    # ------------------------------------------------------------------
    st.markdown("### Boxplots & scatter")

    col_a, col_b = st.columns(2)

    # ---- Left: SOH by dataset (boxplot) ----
    with col_a:
        if "soh" in num_cols and "dataset" in df.columns:
            fig_box = px.box(
                df,
                x="dataset",
                y="soh",
                color="dataset",
                color_discrete_sequence=COLOR_SEQ,
                template=PLOTLY_TEMPLATE,
                title="SOH by dataset (boxplot)",
            )
            fig_box.update_layout(height=350, margin=dict(l=40, r=20, t=50, b=40))
            st.plotly_chart(fig_box, width="stretch")
            st.caption(
                "Boxplots show the distribution of **State of Health (SOH)** for each dataset "
                "(Urban / Highway / Mixed / uploaded). You can compare medians, spread, and outliers."
            )
        else:
            st.info("Need 'soh' and 'dataset' columns to plot SOH boxplots.")

    # ---- Right: Scatter (SOH vs another numeric feature) ----
    with col_b:
        # pick an X axis:
        #  1. prefer 'soc' if available
        #  2. otherwise any other numeric feature that isn't 'soh'
        x_candidates = []
        if "soc" in num_cols:
            x_candidates.append("soc")
        x_candidates += [c for c in num_cols if c not in ["soh", "soc", "cycle"]]

        if "soh" in num_cols and len(x_candidates) > 0:
            x_default = x_candidates[0]
            # optional: let the user change X-axis if you want more control
            x_axis = st.selectbox(
                "Scatter X‑axis (numeric)",
                options=x_candidates,
                index=0,
                key="overview_scatter_x",
                help="X‑axis feature for the scatter plot (e.g., SOC, temperature, current).",
            )

            fig_sc = px.scatter(
                df,
                x=x_axis,
                y="soh",
                color="dataset" if "dataset" in df.columns else None,
                color_discrete_sequence=COLOR_SEQ,
                template=PLOTLY_TEMPLATE,
                title=f"SOH vs {x_axis}",
                opacity=0.7,
            )
            fig_sc.update_layout(height=350, margin=dict(l=40, r=20, t=50, b=40))
            st.plotly_chart(fig_sc, width="stretch")
            st.caption(
                f"This scatter shows how **SOH** changes as **{x_axis}** changes. "
                "If you see a downward trend, higher values of this feature may be linked to faster degradation. "
                "Colours distinguish datasets."
            )
        elif "soh" in num_cols:
            st.info(
                "Only one numeric column (SOH) available, so a scatter plot isn't meaningful. "
                "Once you add more numeric features (e.g., SOC, currents, temperatures), a scatter will appear here."
            )
        else:
            st.info("No SOH column found — cannot plot SOH scatter.")

    # ------------------------------------------------------------------
    # Correlation heatmap
    # ------------------------------------------------------------------
    st.markdown("### Correlation heatmap (numeric features)")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            template=PLOTLY_TEMPLATE,
            title="Correlation heatmap (current selection)",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Correlation heatmap shows linear relationships between numeric variables.")
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")

    # ------------------------------------------------------------------
    # 3D scatter
    # ------------------------------------------------------------------
    st.markdown("### 3D Scatter (current_avg, temp_avg, SOH)")
    if {"current_avg", "temp_avg", "soh"}.issubset(df.columns):
        fig3d = px.scatter_3d(
            df,
            x="current_avg",
            y="temp_avg",
            z="soh",
            color="dataset",
            opacity=0.7,
            template=PLOTLY_TEMPLATE,
            title="3D Scatter: current vs temperature vs SOH",
        )
        st.plotly_chart(fig3d, use_container_width=True)
        st.caption(
            "3D scatter helps see how high current and high temperature jointly "
            "push SOH down in certain datasets."
        )

    # ------------------------------------------------------------------
    # Parallel coordinates plot (the plot in your screenshot)
    
        # ------------------------------------------------------------------
    # Parallel coordinates plot (multi‑feature view of cycles)
    # ------------------------------------------------------------------
    st.markdown("### Parallel coordinates view of cycles")

    # numeric candidates (already computed above as num_cols)
    numeric_candidates = [c for c in num_cols if c not in ["cell_id"]]

    # Prefer these if they exist, but fall back to any numeric cols
    preferred_order = ["cycle", "soh", "current_avg", "temp_avg", "soc", "voltage_avg"]
    par_dims = [c for c in preferred_order if c in numeric_candidates]

    # If we still have <3, pad with other numeric columns
    if len(par_dims) < 3:
        extras = [c for c in numeric_candidates if c not in par_dims]
        par_dims.extend(extras[: max(0, 3 - len(par_dims))])

    # Don’t let it explode: cap to first 6 dimensions
    par_dims = par_dims[:6]

    if len(par_dims) >= 3:
        # Build subset: selected numeric columns + dataset for context
        cols_for_par = par_dims + (["dataset"] if "dataset" in df.columns else [])
        df_par = df[cols_for_par].dropna()

        # Sample rows so the plot stays readable
        max_rows = st.slider(
            "Max number of cycles to show in parallel coordinates",
            min_value=200,
            max_value=5000,
            value=1000,
            step=200,
            help="Sampling keeps the plot readable when there are many cycles.",
        )
        if len(df_par) > max_rows:
            df_par = df_par.sample(max_rows, random_state=42)

        # Color by SOH if present, otherwise by the first dimension
        color_dim = "soh" if "soh" in par_dims else par_dims[0]

        # Pretty axis labels
        dim_labels = {c: c.replace("_", " ").title() for c in par_dims}

        fig_par = px.parallel_coordinates(
            df_par,
            dimensions=par_dims,
            color=color_dim,
            color_continuous_scale=px.colors.sequential.Blues,
            labels=dim_labels,
            template=PLOTLY_TEMPLATE,
        )
        # Make sure it fits and isn’t clipped
        fig_par.update_layout(
            title="Parallel coordinates: multi‑feature profile of cycles",
            height=500,                     # a bit taller for readability
            margin=dict(l=60, r=40, t=60, b=40),
            autosize=True,
        )

        st.plotly_chart(fig_par, width="stretch")
        st.caption(
            "Each vertical axis is a feature (e.g., cycle, SOH, current, temperature, SOC). "
            "Each polyline is one drive cycle passing through all axes."
        )

        # Extra explanation: how to read the plot
        st.markdown("#### How to read this plot")
        st.write(
            """
- Look for **bundles of lines**:
  - A tight bundle on the SOH axis near 1.0 means many healthy cycles.
  - Lines dipping lower on the SOH axis correspond to degraded cycles.
- Follow a single line across axes:
  - A low‑SOH line that also has **high current** and **high temperature** suggests stress‑induced degradation.
- Compare datasets:
  - If you selected multiple datasets in the sidebar, you’ll see **mixed patterns**;
    e.g., Urban cycles may show more lines with high current and temperature and lower SOH.
- The colour scale:
  - Darker blue = higher value of the colour dimension (usually SOH).
  - This helps you visually link “healthy” vs “aged” cycles to specific feature ranges.
"""
        )
    else:
        st.info(
            "This dataset has fewer than 3 numeric columns, so a parallel‑coordinates plot "
            "is not very informative."
        )


# -------------------------------------------------------------------
# 3. EDA & VIZ GALLERY TAB
# -------------------------------------------------------------------
with tabs[3]:
    explain(
        "EDA & Viz Gallery",
        [
            "Histogram, boxplot, violin, class imbalance, outlier detection.",
            "Scatter, 3D scatter, scatter matrix, correlation heatmap.",
            "PCA projection (2D) like in the SVD/PCA lectures.",
        ],
    )

    df_eda = limit_rows(current_df, max_rows=2000)
    numc = [c for c in numeric_cols(df_eda) if c != "cycle"]

    st.markdown("### Class imbalance: bucket distribution")
    if "bucket" in df_eda.columns:
        cat_counts = (
            df_eda["bucket"]
            .value_counts(dropna=False)
            .rename_axis("bucket")
            .reset_index(name="count")
        )
        cat_counts["pct"] = 100 * cat_counts["count"] / cat_counts["count"].sum()
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(cat_counts, use_container_width=True)
        with c2:
            fig = px.bar(
                cat_counts,
                x="bucket",
                y="count",
                color="bucket",
                color_discrete_sequence=COLOR_SEQ,
                template=PLOTLY_TEMPLATE,
                title="Bucket class counts",
                text="count",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Interpretation: this checks **class imbalance** in the bucket labels. "
            "Highly imbalanced classes may require re-weighting or resampling."
        )

    st.markdown("### Histograms by dataset")
    if numc:
        col_hist = st.selectbox(
            "Histogram feature",
            numc,
            index=numc.index("soh") if "soh" in numc else 0,
        )
        fig = px.histogram(
            df_eda,
            x=col_hist,
            color="dataset",
            nbins=30,
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=COLOR_SEQ,
            barmode="overlay",
            title=f"Histogram of {col_hist} by dataset",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Interpretation: overlaid histograms show how the distribution of this feature "
            "differs across usage profiles (Urban/Highway/Mixed)."
        )

    st.markdown("### Box, violin & outlier statistics")
    if numc:
        col_box = st.selectbox(
            "Feature for box/violin & outlier stats",
            numc,
            index=numc.index("stress_index") if "stress_index" in numc else 0,
            key="box_feat",
        )

        c1, c2 = st.columns(2)
        with c1:
            fig_box = px.box(
                df_eda,
                x="dataset",
                y=col_box,
                color="dataset",
                color_discrete_sequence=COLOR_SEQ,
                template=PLOTLY_TEMPLATE,
                title=f"Box plot of {col_box} by dataset",
            )
            st.plotly_chart(fig_box, use_container_width=True)
        with c2:
            fig_violin = px.violin(
                df_eda,
                x="dataset",
                y=col_box,
                color="dataset",
                color_discrete_sequence=COLOR_SEQ,
                box=True,
                points="all",
                template=PLOTLY_TEMPLATE,
                title=f"Violin plot of {col_box} by dataset",
            )
            st.plotly_chart(fig_violin, use_container_width=True)

        st.caption(
            "Interpretation: box/violin plots show median, spread, and potential outliers "
            "of a feature by dataset. Long tails or many points outside the whiskers "
            "indicate outliers."
        )

        outlier_stats = []
        s = df_eda[col_box].dropna()
        if len(s) > 0:
            Q1 = s.quantile(0.25)
            Q3 = s.quantile(0.75)
            IQR = Q3 - Q1
            low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = s[(s < low) | (s > high)]
            outlier_stats.append(
                {
                    "feature": col_box,
                    "outlier_count": len(outliers),
                    "pct_outliers": 100 * len(outliers) / len(s),
                }
            )
        st.dataframe(pd.DataFrame(outlier_stats), use_container_width=True)
        st.caption(
            "Interpretation: the outlier statistics use the IQR rule from lecture "
            "to quantify how many extreme points appear for this feature."
        )

    st.markdown("### Correlation heatmap (numeric)")
    corr_heatmap(df_eda, "Correlation heatmap (numeric features)", key="eda_corr")

    st.markdown("### 2D scatter plot")
    if len(numc) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            x_axis = st.selectbox("X-axis", numc, index=0, key="scatter_x")
        with c2:
            y_axis = st.selectbox("Y-axis", numc, index=1, key="scatter_y")
        color_by = st.selectbox(
            "Color points by",
            ["dataset", "cell_id", "bucket", "vehicle_segment"],
            index=0,
        )
        fig_sc = px.scatter(
            df_eda,
            x=x_axis,
            y=y_axis,
            color=color_by,
            color_discrete_sequence=COLOR_SEQ,
            hover_data=["cycle", "cell_id"],
            template=PLOTLY_TEMPLATE,
            height=500,
            title=f"{y_axis} vs {x_axis}",
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption(
            "Interpretation: a 2D scatter plot shows pairwise relationships (e.g., "
            "SOH vs stress_index). Color encoding helps compare datasets or buckets."
        )

    st.markdown("### 3D scatter plot")
    if len(numc) >= 3:
        c1, c2, c3 = st.columns(3)
        with c1:
            x3 = st.selectbox("X (3D)", numc, index=0, key="sc3_x")
        with c2:
            y3 = st.selectbox("Y (3D)", numc, index=1, key="sc3_y")
        with c3:
            z3 = st.selectbox("Z (3D)", numc, index=2, key="sc3_z")

        fig3d = px.scatter_3d(
            df_eda,
            x=x3,
            y=y3,
            z=z3,
            color="dataset",
            color_discrete_sequence=COLOR_SEQ,
            template=PLOTLY_TEMPLATE,
            title=f"3D scatter: {x3}, {y3}, {z3}",
        )
        st.plotly_chart(fig3d, use_container_width=True)
        st.caption(
            "Interpretation: 3D scatter plots reveal multi-dimensional patterns "
            "that are not visible in 2D (e.g., how temp_max, q_abs, and SOH interact)."
        )

    st.markdown("### Scatter matrix (small subset)")
    if len(numc) >= 3:
        subset_cols = numc[:4]
        fig_sm = px.scatter_matrix(
            df_eda,
            dimensions=subset_cols,
            color="dataset",
            color_discrete_sequence=COLOR_SEQ,
            template=PLOTLY_TEMPLATE,
            title=f"Scatter matrix ({', '.join(subset_cols)})",
        )
        st.plotly_chart(fig_sm, use_container_width=True)
        st.caption(
            "Interpretation: scatter matrix = all pairwise scatter plots + histograms "
            "on the diagonal, useful for initial IDA/EDA."
        )

    st.markdown("### PCA projection (2D)")
    if len(numc) >= 3:
        Xnum = df_eda[numc].copy()
        if Xnum.notna().sum().min() > 5:
            imp = SimpleImputer(strategy="median")
            Ximp = imp.fit_transform(Xnum)
            scaler = StandardScaler()
            Xscaled = scaler.fit_transform(Ximp)
            pca = PCA(n_components=2, random_state=7)
            Xp = pca.fit_transform(Xscaled)
            df_pca = pd.DataFrame(
                {
                    "PC1": Xp[:, 0],
                    "PC2": Xp[:, 1],
                    "dataset": df_eda["dataset"].astype(str).values,
                }
            )
            fig_pca = px.scatter(
                df_pca,
                x="PC1",
                y="PC2",
                color="dataset",
                color_discrete_sequence=COLOR_SEQ,
                template=PLOTLY_TEMPLATE,
                title="PCA projection (PC1 vs PC2)",
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            st.caption(
                f"Interpretation: PCA compresses high‑dimensional features into 2 axes. "
                f"PC1 and PC2 explain {pca.explained_variance_ratio_[0]:.2f} and "
                f"{pca.explained_variance_ratio_[1]:.2f} of the variance respectively."
            )

    st.markdown("### Text feature (usage_text) – most frequent tokens")
    if "usage_text" in df_eda.columns:
        from collections import Counter
        all_text = " ".join(df_eda["usage_text"].dropna().astype(str).tolist()).lower()
        tokens = [t.strip(",.! ") for t in all_text.split() if len(t) > 3]
        counts = Counter(tokens)
        top_tokens = counts.most_common(12)
        if top_tokens:
            tok_df = pd.DataFrame(top_tokens, columns=["token", "count"])
            fig_tok = px.bar(
                tok_df,
                x="token",
                y="count",
                color="token",
                color_discrete_sequence=COLOR_SEQ,
                template=PLOTLY_TEMPLATE,
                title="Top tokens in usage_text",
            )
            st.plotly_chart(fig_tok, use_container_width=True)
            st.caption(
                "Interpretation: this is a simple NLP flavour—shows which driving "
                "conditions are most common in the text descriptions."
            )
        else:
            st.info("Not enough text to show token frequencies.")


# -------------------------------------------------------------------
# 4. MISSINGNESS LAB TAB
# -------------------------------------------------------------------
with tabs[4]:
    explain(
        "Missingness Lab",
        [
            "Quantify missing values per column.",
            "Visualise missingness matrix.",
            "Compare Simple / KNN / Iterative (MICE) imputation RMSE on one column.",
        ],
    )

    numc = numeric_cols(current_df)
    st.markdown("### Missing values summary (numeric columns)")
    if numc:
        miss_cnt = current_df[numc].isna().sum()
        miss_pct = current_df[numc].isna().mean() * 100
        miss_df = pd.DataFrame(
            {"missing_count": miss_cnt, "missing_pct": miss_pct}
        ).sort_values("missing_pct", ascending=False)
        st.dataframe(miss_df, use_container_width=True)
        st.caption(
            "Interpretation: use this table to identify which numeric features have "
            "serious missingness issues."
        )

        st.markdown("#### Missingness bar chart")
        miss_nonzero = miss_df[miss_df["missing_count"] > 0]
        if not miss_nonzero.empty:
            fig = px.bar(
                miss_nonzero.reset_index().rename(columns={"index": "column"}),
                x="column",
                y="missing_pct",
                template=PLOTLY_TEMPLATE,
                color="missing_pct",
                color_continuous_scale="Reds",
                title="Percent missing by column",
                height=350,
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Interpretation: taller, redder bars = worse missingness. Those columns "
                "are critical to handle carefully."
            )
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
            st.caption(
                "Interpretation: vertical patterns show rows with many missing values; "
                "horizontal patterns show columns that are systematically missing."
            )
    else:
        st.info("No numeric columns to analyse missingness.")

    st.markdown("---")
    st.markdown("### Imputation comparison for one numeric column (MCAR)")

    if numc:
        target_col = st.selectbox(
            "Numeric column",
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
            ("Iterative (MICE)", IterativeImputer(random_state=7, max_iter=8)),
        ]:
            X = df_mcar[numc]
            imputed = imp.fit_transform(X)
            Ximp = pd.DataFrame(imputed, columns=numc, index=X.index)

            common = base.dropna()
            if target_col in Ximp.columns and not common.empty:
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
            color="imputer",
            color_discrete_sequence=COLOR_SEQ,
            title=f"Imputation RMSE for {target_col}",
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption(
            "Interpretation: lower RMSE = better imputation for this column under a "
            "simulated MCAR pattern. This mirrors lecture on comparing imputation methods."
        )

# -------------------------------------------------------------------
# 5. ENCODING & CLASSICAL MODELS TAB
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# 5. ENCODING & CLASSICAL MODELS TAB
# -------------------------------------------------------------------
with tabs[5]:
    explain(
        "Encoding & Classical Models",
        [
            "Show data BEFORE encoding (raw columns) for all selected datasets.",
            "Show encoded design matrix AFTER encoding.",
            "Show exactly which columns are encoded and how (numeric / categorical / TF-IDF).",
            "Train classical models: Linear Regression, RandomForest, GradientBoosting.",
            "Run RandomForest hyperparameter tuning and visualise its performance.",
        ],
    )

    target = "soh" if task_type == "SOH regression" else "bucket"
    enc = build_encoded_matrices(current_df, target, impute_choice)

    if enc is None or enc["dfy"].shape[0] < MIN_LABELS_TRAIN:
        st.info(
            f"Not enough labelled rows for target '{target}'. "
            f"Need at least {MIN_LABELS_TRAIN} rows after cleaning/encoding."
        )
    else:
        # unpack encoded bundle
        dfy = enc["dfy"].loc[:, ~enc["dfy"].columns.duplicated()]
        X_tr = enc["X_tr"]
        X_te = enc["X_te"]
        y_train = enc["y_train"]
        y_test = enc["y_test"]
        train_struct = enc["train_struct"]
        encoded_train_df = enc["encoded_train_df"]
        encoding_map_df = enc["encoding_map_df"]
        num_features = enc["num_features"]
        cat_features = enc["cat_features"]
        text_feature = enc["text_feature"]

        # high‑level counts
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi("Rows with label", len(dfy), "after cleaning & filtering")
        with c2:
            kpi("Train rows", len(y_train))
        with c3:
            kpi("Test rows", len(y_test))

        # -----------------------
        # What is being encoded?
        # -----------------------
        st.markdown("### Which features are being encoded?")
        st.write("**Numeric features (impute + scale):**", num_features or "None")
        st.write("**Categorical features (impute + one‑hot):**", cat_features or "None")
        st.write("**Text feature (TF‑IDF):**", text_feature or "None")
        st.caption(
            "Interpretation: this section shows how feature types are split: "
            "numeric → median imputation + standardization; "
            "categorical → imputation + one‑hot encoding; "
            "text → TF‑IDF vectorization."
        )

        # -----------------------
        # BEFORE encoding
        # -----------------------
        st.markdown("### BEFORE encoding – sample rows from ALL datasets")

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
                "manufacturer",
                "cooling",
                "vehicle_segment",
                "cycle_bin",
                "usage_text",
            ]
            if c in dfy.columns
        ]
        show_cols += extra

        # de‑duplicate while preserving order
        show_cols_unique = []
        for c in show_cols:
            if c in dfy.columns and c not in show_cols_unique:
                show_cols_unique.append(c)

        if "dataset" in dfy.columns:
            dfy_sample = (
                dfy[show_cols_unique]
                .sort_values("dataset")
                .groupby("dataset")
                .head(4)
            )
        else:
            dfy_sample = dfy[show_cols_unique].sample(
                min(12, len(dfy)), random_state=7
            )

        st.dataframe(dfy_sample, use_container_width=True)
        st.caption(
            "Interpretation: this is the **raw feature table** before encoding, "
            "showing rows from Urban, Highway, Mixed, and any uploaded datasets."
        )

        # -----------------------
        # AFTER encoding
        # -----------------------
        st.markdown("### AFTER encoding – design matrix")
        st.dataframe(encoded_train_df.head(10), use_container_width=True)
        st.caption(
            "Interpretation: this is the actual matrix that goes into the models. "
            "Each column is a numeric feature (scaled) or a one‑hot encoded category; "
            "all missing values have been imputed."
        )

        st.markdown("### Encoding map (raw → encoded)")
        st.dataframe(encoding_map_df, use_container_width=True)
        st.caption(
            "Interpretation: this map tells you which raw feature each encoded column "
            "came from and whether it's numeric scaling or one‑hot of a category value."
        )

        st.markdown("---")
        st.subheader(f"Classical model comparison – target: `{target}`")

        # -----------------------
        # Classical model training
        # -----------------------
        models = {}
        if target == "soh":
            models["LinearRegression"] = LinearRegression()
            models["RandomForestRegressor"] = RandomForestRegressor(
                n_estimators=200, random_state=7, n_jobs=-1
            )
            models["GradientBoostingRegressor"] = GradientBoostingRegressor(
                random_state=7
            )
        else:
            models["RandomForestClassifier"] = RandomForestClassifier(
                n_estimators=200, random_state=7, n_jobs=-1
            )
            models["GradientBoostingClassifier"] = GradientBoostingClassifier(
                random_state=7
            )

        rows = []
        best_name = None
        best_metric = None
        best_pred = None

        for name, model in models.items():
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)

            if target == "soh":  # regression metrics
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rows.append({"model": name, "MAE": mae, "R2": r2})
                # select best by lowest MAE
                if best_metric is None or mae < best_metric:
                    best_metric = mae
                    best_name = name
                    best_pred = y_pred
            else:  # classification metrics
                acc = accuracy_score(y_test, y_pred)
                rows.append({"model": name, "Accuracy": acc})
                # select best by highest Accuracy
                if best_metric is None or acc > best_metric:
                    best_metric = acc
                    best_name = name
                    best_pred = y_pred

        res_df = pd.DataFrame(rows)
        st.dataframe(res_df, use_container_width=True)
        st.caption(
            "Interpretation: this table compares classical models on the test set. "
            "For SOH regression we focus on MAE/R²; for bucket classification on Accuracy."
        )

        # bar plots + diagnostic plots
        if target == "soh":
            fig_m = px.bar(
                res_df,
                x="model",
                y="MAE",
                color="R2",
                template=PLOTLY_TEMPLATE,
                title="SOH regression (LinearReg + RF + GradientBoosting)",
            )
            st.plotly_chart(fig_m, use_container_width=True)
            st.caption(
                "Interpretation: shorter bars (lower MAE) indicate better models; "
                "R² shows how much variance in SOH is explained."
            )

            if best_pred is not None:
                st.markdown(f"#### Best classical model by MAE: `{best_name}`")
                fig_sc = px.scatter(
                    x=y_test,
                    y=best_pred,
                    labels={"x": "Actual SOH", "y": "Predicted SOH"},
                    template=PLOTLY_TEMPLATE,
                    title=f"{best_name}: predicted vs actual SOH",
                )
                fig_sc.add_shape(
                    type="line",
                    x0=float(min(y_test)),
                    y0=float(min(y_test)),
                    x1=float(max(y_test)),
                    y1=float(max(y_test)),
                    line=dict(color="red", dash="dot"),
                )
                st.plotly_chart(fig_sc, use_container_width=True)
                st.caption(
                    "Interpretation: points close to the diagonal line indicate "
                    "good calibration of SOH predictions."
                )

        else:  # classification visualisation
            fig_m = px.bar(
                res_df,
                x="model",
                y="Accuracy",
                color="model",
                color_discrete_sequence=COLOR_SEQ,
                template=PLOTLY_TEMPLATE,
                title="Bucket classification (RF + GradientBoosting)",
            )
            st.plotly_chart(fig_m, use_container_width=True)
            st.caption(
                "Interpretation: higher Accuracy bars show better classification of "
                "health buckets."
            )

            if best_pred is not None:
                cm = confusion_matrix(y_test, best_pred)
                labels = sorted(pd.Series(y_test).unique())
                fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    xticklabels=labels,
                    yticklabels=labels,
                    cmap="mako",
                    ax=ax_cm,
                )
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("True")
                ax_cm.set_title(f"{best_name} confusion matrix")
                st.pyplot(fig_cm, clear_figure=True)
                st.caption(
                    "Interpretation: the confusion matrix shows which bucket classes are "
                    "confused with which others."
                )

        # -------------------------------------------------
        # RandomForest hyperparameter tuning (kept in this tab)
        # -------------------------------------------------
        st.markdown("---")
        st.markdown("## 🔧 RandomForest hyperparameter tuning")

        # Use current_df directly to run a small RF + GridSearch
        if "soh" not in current_df.columns:
            st.info(
                "No 'soh' column found in the current dataset. "
                "RandomForest regression can't be tuned."
            )
        else:
            df_rf = current_df.dropna(subset=["soh"]).copy()
            num_cols_rf = df_rf.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols_rf = [
                c for c in num_cols_rf if c not in ["soh", "cycle", "time_s"]
            ]

            if len(df_rf) < 40 or len(feature_cols_rf) < 1:
                st.info(
                    "Need at least ~40 labelled rows and at least one numeric feature "
                    "to run a meaningful RF tuning."
                )
            else:
                X_rf = df_rf[feature_cols_rf]
                y_rf = df_rf["soh"].astype(float).values

                imp = SimpleImputer(strategy="median")
                X_imp_rf = imp.fit_transform(X_rf)
                scaler = StandardScaler()
                X_proc_rf = scaler.fit_transform(X_imp_rf)

                with st.spinner(
                    "Running RandomForest GridSearchCV (once per dataset selection)..."
                ):
                    cvres = run_rf_tuning(
                        pd.DataFrame(X_proc_rf, index=df_rf.index), y_rf
                    )

                best_row = cvres.loc[cvres["mae"].idxmin()]
                st.success(
                    f"**Best RF params** → "
                    f"`n_estimators={int(best_row['param_n_estimators'])}`, "
                    f"`max_depth={best_row['param_max_depth']}`, "
                    f"**CV MAE = {best_row['mae']:.4f}**"
                )

                st.markdown("**All tested combinations (sorted by MAE)**")
                st.dataframe(
                    cvres[["param_n_estimators", "param_max_depth", "mae"]]
                    .sort_values("mae")
                    .rename(
                        columns={
                            "param_n_estimators": "n_estimators",
                            "param_max_depth": "max_depth",
                            "mae": "CV MAE",
                        }
                    ),
                    use_container_width=True,
                )

                plot_df = cvres.copy()
                plot_df["param_n_estimators"] = plot_df[
                    "param_n_estimators"
                ].astype(int)
                plot_df["param_max_depth"] = plot_df["param_max_depth"].astype(str)

                fig_rf = px.line(
                    plot_df,
                    x="param_n_estimators",
                    y="mae",
                    color="param_max_depth",
                    markers=True,
                    template=PLOTLY_TEMPLATE,
                    labels={
                        "param_n_estimators": "Number of trees (n_estimators)",
                        "param_max_depth": "Max depth",
                        "mae": "CV MAE (lower is better)",
                    },
                    title="RandomForest tuning: MAE vs number of trees (by max_depth)",
                )
                fig_rf.update_layout(
                    height=400,
                    margin=dict(l=40, r=20, t=60, b=40),
                    legend_title_text="max_depth",
                )
                st.plotly_chart(fig_rf, use_container_width=True)

                st.caption(
                    "- Each **point** is one hyperparameter combination tested by GridSearchCV.\n"
                    "- The **x‑axis** is the number of trees in the forest.\n"
                    "- Each **line colour** is a different `max_depth` value.\n"
                    "- The **y‑axis** is cross‑validated MAE; **lower is better**.\n"
                    "This lets us visually compare the tuned RF against the baseline RF "
                    "in the classical model section above."
                )

# -------------------------------------------------------------------
# 6. DEEP LEARNING & ENSEMBLES TAB
# -------------------------------------------------------------------
with tabs[6]:
    explain(
        "Deep Learning & Ensembles",
        [
            "Deep learning = Multi-Layer Perceptron (MLP) neural network with 3 hidden layers.",
            "Ensemble = XGBoost (gradient-boosted trees) when available.",
            "Visualise architecture, loss curve, and predicted vs actual / confusion matrix.",
        ],
    )

    st.markdown(
        """
        ### Neural network model (MLP)

        - Input: encoded numeric + one-hot categorical + TF‑IDF text features  
        - Hidden layer 1: 128 neurons, ReLU  
        - Hidden layer 2: 64 neurons, ReLU  
        - Hidden layer 3: 32 neurons, ReLU  
        - Output:  
            - **SOH regression** → 1 neuron (continuous SOH)  
            - **Bucket classification** → one neuron per class
        """
    )

    target = "soh" if task_type == "SOH regression" else "bucket"
    enc_adv = build_encoded_matrices(current_df, target, impute_choice)

    if enc_adv is None or enc_adv["dfy"].shape[0] < MIN_LABELS_TRAIN:
        st.info(
            f"Not enough labelled rows for target '{target}' to train advanced models. "
            f"Need at least {MIN_LABELS_TRAIN}."
        )
    else:
        X_tr = enc_adv["X_tr"]
        X_te = enc_adv["X_te"]
        y_train = enc_adv["y_train"]
        y_test = enc_adv["y_test"]

        n_in = X_tr.shape[1]
        if target == "soh":
            n_out = 1
        else:
            n_out = int(pd.Series(y_train).nunique())

        st.subheader("Neural network architecture (schematic)")
        nn_fig = plot_nn_architecture(
            [n_in, 128, 64, 32, n_out],
            title=f"MLP architecture ({n_in} → 128 → 64 → 32 → {n_out})",
        )
        st.pyplot(nn_fig, clear_figure=True)
        st.caption(
            "Interpretation: this diagram shows the **structure** of the neural network: "
            "input layer, 3 hidden layers, and output layer."
        )

        advanced_models = {}

        if target == "soh":
            mlp = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                max_iter=250,
                alpha=1e-3,
                random_state=7,
            )
            advanced_models["MLPRegressor (3-layer NN)"] = mlp
        else:
            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                max_iter=250,
                alpha=1e-3,
                random_state=7,
            )
            advanced_models["MLPClassifier (3-layer NN)"] = mlp

        if XGB_OK:
            if target == "soh":
                xgb_model = xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=7,
                    n_jobs=-1,
                )
                advanced_models["XGBRegressor (XGBoost)"] = xgb_model
            else:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=7,
                    n_jobs=-1,
                    eval_metric="logloss",
                )
                advanced_models["XGBClassifier (XGBoost)"] = xgb_model
        else:
            st.warning(
                "xgboost library is not installed, so the XGBoost ensemble model is skipped. "
                "Add `xgboost` to requirements.txt to enable it."
            )

        rows = []
        nn_loss_curve = None
        y_pred_nn = None

        for name, model in advanced_models.items():
            try:
                model.fit(X_tr, y_train)
                y_pred = model.predict(X_te)

                if "MLP" in name:
                    if hasattr(model, "loss_curve_"):
                        nn_loss_curve = model.loss_curve_
                    y_pred_nn = y_pred

                if target == "soh":
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    rows.append({"model": name, "MAE": mae, "R2": r2})
                else:
                    acc = accuracy_score(y_test, y_pred)
                    rows.append({"model": name, "Accuracy": acc})

            except Exception as e:
                st.warning(f"Model {name} failed to train: {e}")

        if not rows:
            st.warning("No advanced models were successfully trained. See warnings above.")
        else:
            res_adv = pd.DataFrame(rows)
            st.markdown("### Advanced model results (Deep NN + XGBoost)")
            st.dataframe(res_adv, use_container_width=True)

            if target == "soh":
                fig_adv = px.bar(
                    res_adv,
                    x="model",
                    y="MAE",
                    color="model",
                    color_discrete_sequence=COLOR_SEQ,
                    template=PLOTLY_TEMPLATE,
                    title="SOH regression – deep learning & ensembles",
                )
                st.plotly_chart(fig_adv, use_container_width=True)
                st.caption(
                    "Interpretation: this compares the MLP neural net vs XGBoost (if available) "
                    "for SOH regression."
                )
            else:
                fig_adv = px.bar(
                    res_adv,
                    x="model",
                    y="Accuracy",
                    color="model",
                    color_discrete_sequence=COLOR_SEQ,
                    template=PLOTLY_TEMPLATE,
                    title="Bucket classification – deep learning & ensembles",
                )
                st.plotly_chart(fig_adv, use_container_width=True)
                st.caption(
                    "Interpretation: this compares the MLP neural net vs XGBoost (if available) "
                    "for bucket classification."
                )

        st.markdown("---")
        st.markdown("### Neural network diagnostics")

        if nn_loss_curve is not None:
            st.subheader("MLP training loss curve")
            fig_loss, ax = plt.subplots(figsize=(6, 3))
            ax.plot(nn_loss_curve, color="#00e3ff")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.set_title("MLP loss over iterations")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig_loss, clear_figure=True)
            st.caption(
                "Interpretation: a decreasing loss curve indicates the neural network "
                "is learning. A flat or increasing curve suggests underfitting or overfitting."
            )
        else:
            st.info("Loss curve not available for this MLP configuration.")

        if y_pred_nn is not None:
            if target == "soh":
                st.subheader("MLP predicted vs actual SOH (test set)")
                fig_sc = px.scatter(
                    x=y_test,
                    y=y_pred_nn,
                    labels={"x": "Actual SOH", "y": "Predicted SOH"},
                    template=PLOTLY_TEMPLATE,
                    title="MLPRegressor: predicted vs actual SOH",
                )
                fig_sc.add_shape(
                    type="line",
                    x0=float(min(y_test)),
                    y0=float(min(y_test)),
                    x1=float(max(y_test)),
                    y1=float(max(y_test)),
                    line=dict(color="red", dash="dot"),
                )
                st.plotly_chart(fig_sc, use_container_width=True)
                st.caption(
                    "Interpretation: this is the neural network equivalent of the regression "
                    "diagnostic scatter: good models hug the diagonal line."
                )
            else:
                st.subheader("MLP classification confusion matrix (test set)")
                cm = confusion_matrix(y_test, y_pred_nn)
                labels = sorted(pd.Series(y_test).unique())
                fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    xticklabels=labels,
                    yticklabels=labels,
                    cmap="mako",
                    ax=ax_cm,
                )
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("True")
                ax_cm.set_title("MLPClassifier confusion matrix")
                st.pyplot(fig_cm, clear_figure=True)
                st.caption(
                    "Interpretation: this shows how well the neural network classifies "
                    "health buckets."
                )

# -------------------------------------------------------------------
# 7. PREDICTIONS & FORECASTING TAB
# -------------------------------------------------------------------
with tabs[7]:
    explain(
        "Predictions & Forecasting",
        [
            "Final 'end results' tab: RUL (Remaining Useful Life) + time-series SOH forecast.",
            "Uses RandomForest regression plus optional AutoReg time-series for SOH vs cycle.",
        ],
    )

    if task_type != "SOH regression":
        st.info("Switch modelling task to 'SOH regression' in the sidebar to enable this tab.")
    else:
        enc_rul = build_encoded_matrices(current_df, "soh", impute_choice)
        if enc_rul is None or enc_rul["dfy"].shape[0] < MIN_LABELS_TRAIN:
            st.info("Not enough rows with SOH labels to estimate RUL.")
        else:
            dfy_r = enc_rul["dfy"]
            X_tr_r = enc_rul["X_tr"]
            X_te_r = enc_rul["X_te"]
            y_train_r = enc_rul["y_train"]
            y_test_r = enc_rul["y_test"]

            rf_rul = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=7,
                n_jobs=-1,
            )
            rf_rul.fit(X_tr_r, y_train_r)
            y_pred_r = rf_rul.predict(X_te_r)
            mae_r = mean_absolute_error(y_test_r, y_pred_r)
            r2_r = r2_score(y_test_r, y_pred_r)

            c1, c2 = st.columns(2)
            with c1:
                kpi("SOH RF MAE", mae_r, "baseline error for RUL")
            with c2:
                kpi("SOH RF R²", r2_r, "explained variance")

            st.markdown("### Simple RUL estimate (Remaining Useful Life in cycles)")
            rul_rows = []
            for cell, gcell in dfy_r.groupby("cell_id"):
                gcell = gcell.sort_values("cycle")
                Xc_struct = gcell[enc_rul["feature_cols"]]
                Xc_enc = enc_rul["preprocessor"].transform(Xc_struct)
                if enc_rul["tfidf"] is not None and "usage_text" in Xc_struct.columns:
                    Xt_cell = enc_rul["tfidf"].transform(
                        Xc_struct["usage_text"].fillna("").astype(str)
                    )
                    Xc = np.hstack([Xc_enc, Xt_cell.toarray()])
                else:
                    Xc = Xc_enc

                soh_hat = rf_rul.predict(Xc)
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
                rul_df = pd.DataFrame(rul_rows).sort_values("RUL_cycles")
                st.dataframe(rul_df, use_container_width=True)
                fig_rul = px.bar(
                    rul_df,
                    x="cell_id",
                    y="RUL_cycles",
                    template=PLOTLY_TEMPLATE,
                    color="cell_id",
                    color_discrete_sequence=COLOR_SEQ,
                    title="Estimated RUL (cycles) by cell",
                )
                st.plotly_chart(fig_rul, use_container_width=True)
                st.caption(
                    "Interpretation: RUL estimates how many cycles remain before a cell "
                    "hits the EOL threshold. Short bars = near end of life."
                )
            else:
                st.info("SOH trend not decreasing enough to estimate RUL reliably.")

        st.markdown("---")
        st.subheader("SOH time-series forecast (AutoReg)")

        if not STATS_OK:
            st.info(
                "statsmodels is not installed, so the AutoReg time-series forecast demo is disabled. "
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

                st.markdown("#### Historical SOH series")
                fig_hist = px.line(
                    g,
                    x="cycle",
                    y="soh",
                    color_discrete_sequence=["#00e3ff"],
                    template=PLOTLY_TEMPLATE,
                    title=f"Historical SOH for cell {sel_cell}",
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                st.caption(
                    "Interpretation: this shows the observed SOH trajectory of one cell "
                    "over its life."
                )

                if len(series) < 20:
                    st.info("Need at least 20 cycles to fit an AutoReg model.")
                else:
                    N = min(80, len(series))
                    s_train = series[-N:]
                    c_train = cycles[-N:]
                    model_ar = AutoReg(s_train, lags=5, old_names=False).fit()
                    steps = 20
                    forecast = model_ar.predict(
                        start=len(s_train), end=len(s_train) + steps - 1
                    )
                    cyc_future = np.arange(c_train[-1] + 1, c_train[-1] + 1 + steps)

                    fig = px.line(
                        x=c_train,
                        y=s_train,
                        template=PLOTLY_TEMPLATE,
                        labels={"x": "Cycle", "y": "SOH"},
                        color_discrete_sequence=["#00e3ff"],
                        title=f"AutoReg forecast for cell {sel_cell}",
                    )
                    fig.add_scatter(
                        x=cyc_future,
                        y=forecast,
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(color="#ffdd00"),
                    )
                    fig.add_hline(
                        y=EOL_THRESH,
                        line_dash="dot",
                        line_color="red",
                        annotation_text="EOL",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "Interpretation: this combines observed SOH (blue) and forecasted "
                        "future SOH (yellow). Crossing the EOL line indicates end-of-life."
                    )


   
  

# -------------------------------------------------------------------
# 8. INSIGHTS & RUBRIC TAB
# -------------------------------------------------------------------
with tabs[8]:
    explain(
        "Insights & Rubric mapping",
        [
            "Real-world insights for EV battery monitoring.",
            "Rubric coverage table for base + advanced requirements.",
        ],
    )

    st.markdown("### Real-world insights")
    st.write(
        """
        - **Thermal management**: high `temp_max` and large `temp_spread`
          correlate with faster SOH decay. EV fleet managers should monitor
          thermal events and redesign cooling where `stress_index` is high.
        - **Usage profiles differ**: Urban cycles degrade faster than Highway cycles,
          suggesting different maintenance / warranty strategies by usage profile.
        - **Missing data strategy**: naive imputation can bias capacity estimates
          under MAR missingness; our comparison of Simple/KNN/MICE helps justify
          a richer imputation strategy in production.
        - **Model diversity**: having classical models (LR, RF, GB), deep neural networks (MLP),
          and gradient-boosted trees (XGBoost) gives complementary views and robustness
          to model mis-specification.
        """
    )

    st.markdown("### stress_index by dataset (reliability triage)")
    if "stress_index" in current_df.columns:
        fig_st = px.box(
            current_df,
            x="dataset",
            y="stress_index",
            color="dataset",
            color_discrete_sequence=COLOR_SEQ,
            template=PLOTLY_TEMPLATE,
            title="stress_index distribution by dataset",
        )
        st.plotly_chart(fig_st, use_container_width=True)
        st.caption(
            "Interpretation: a dataset with consistently high stress_index values "
            "will typically show faster degradation."
        )

    st.markdown("### Rubric compliance summary")
    rubric_items = [
        (
            "Data Collection & Preparation",
            "3 datasets (Urban/Highway/Mixed) + multi-file CSV upload; "
            "advanced cleaning & integration; typed columns & features.",
        ),
        (
            "EDA & Visualisations",
            "Histograms, boxplots, violin plots, class imbalance, outlier stats, 2D & 3D scatter, "
            "scatter matrix, PCA projection, correlation & missingness heatmaps.",
        ),
        (
            "Data Processing & Feature Engineering",
            "Multiple imputers (Simple, KNN, Iterative MICE), scaling, engineered features "
            "(temp_spread, stress_index, q_norm, e_norm, cycle_bin), text TF-IDF.",
        ),
        (
            "Model Development & Evaluation",
            "Classical models (LinearRegression, RF, GB) + deep MLP + XGBoost; train/test split; metrics "
            "(MAE, R², Accuracy); RF RandomizedSearchCV for hyperparameter tuning.",
        ),
        (
            "Streamlit App",
            "Multi-tab app, dataset selector, upload of multiple datasets, task selector, "
            "imputation choice, interactive plots, expander documentation, dark theme.",
        ),
        (
            "GitHub & Documentation",
            "Export CSV in app; you add README, data dictionary, and Streamlit URL in the repo.",
        ),
        (
            "Advanced Modelling Techniques",
            "MLPRegressor/MLPClassifier as deep neural networks (3 hidden layers), XGBoost ensembles, "
            "hyperparameter tuning via RandomizedSearchCV.",
        ),
        (
            "Specialized Data Science",
            "Time-series AutoReg forecast of SOH vs cycle; text feature `usage_text` encoded via TF-IDF.",
        ),
        (
            "High-Performance Computing",
            "RandomizedSearchCV and RandomForest with n_jobs=-1, exploiting parallel CPUs.",
        ),
        (
            "Real-world Application & Impact",
            "Insights tab outlines how EV fleet operators can use these results for maintenance & design.",
        ),
        (
            "Exceptional Presentation & Visualisation",
            "Dark theme dashboard, multi‑coloured interactive Plotly figures, structured story across tabs.",
        ),
    ]
    rubric_df = pd.DataFrame(rubric_items, columns=["Rubric item", "How this app addresses it"])
    st.dataframe(rubric_df, use_container_width=True)
    st.caption(
        "Use this table directly in your report/presentation to show how you checked "
        "off each rubric requirement."
    )


# -------------------------------------------------------------------
# 9. EXPORT TAB
# -------------------------------------------------------------------
with tabs[9]:
    explain(
        "Export",
        [
            "Download cleaned per-cycle dataset (with engineered features).",
            "This CSV is the main artifact for your GitHub repo and README.",
        ],
    )

    st.markdown("### Download cleaned dataset (current selection)")
    st.write(
        f"Rows: **{len(current_df)}**, columns: **{len(current_df.columns)}**. "
        f"Sources included: **{', '.join(selected_sources)}**."
    )
    csv_bytes = current_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name="ev_battery_cleaned_and_engineered.csv",
        mime="text/csv",
    )

    st.caption(
        "Tip: put this CSV in `data/` in your GitHub repo and describe all columns in a data dictionary."
    )



























