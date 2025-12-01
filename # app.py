# app.py — Robust EV Battery SOH & RUL Dashboard
# Run with:  streamlit run app.py

from __future__ import annotations

import os, re, zipfile, warnings
from io import BytesIO
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestRegressor,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    average_precision_score,
)

# Optional SciPy for KDE in scatter-matrix
try:
    from scipy import stats

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG & THEME
# =============================================================================
st.set_page_config(
    page_title="EV Battery SOH & RUL",
    page_icon="🔋",
    layout="wide",
)

THEME = st.sidebar.selectbox(
    "Theme",
    ["Dark (bright text)", "Light"],
    index=0,
)
PLOTLY_TEMPLATE = "plotly_dark" if THEME.startswith("Dark") else "plotly_white"

if THEME.startswith("Dark"):
    plt.style.use("dark_background")
    BG = "#0e1117"
    FG = "#f0f2f6"
    ACCENT = "#22c55e"
else:
    plt.style.use("default")
    BG = "white"
    FG = "#111827"
    ACCENT = "#0ea5e9"

st.markdown(
    f"""
<style>
html, body, [data-testid="stAppViewContainer"], .block-container {{
  background: {BG}; color: {FG};
}}
h1, h2, h3, h4, .stMetric, .stMarkdown, .stText, .stCaption, .stDataFrame {{
  color: {FG};
}}
.kpi {{
  border: 1px solid #2f3b4a33;
  border-radius: 14px;
  padding: 10px 12px;
  height: 100%;
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.08));
}}
.kpi .label {{ font-size:.78rem; opacity:.85 }}
.kpi .value {{ font-size:1.35rem; font-weight:800; margin-top:2px; color:{ACCENT}; }}
.kpi .sub   {{ font-size:.72rem; opacity:.75 }}
.panel {{
  border: 1px solid #2f3b4a33;
  border-radius: 12px;
  padding: 10px 12px;
}}
hr {{ border-top: 1px solid #2f3b4a33; }}
</style>
""",
    unsafe_allow_html=True,
)


def kpi(label, value, sub=""):
    """Small KPI card with label, value, and subtitle."""
    if isinstance(value, (float, np.floating)):
        vtxt = f"{value:,.3f}"
    elif isinstance(value, (int, np.integer)):
        vtxt = f"{value:,}"
    else:
        vtxt = str(value)

    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>{label}</div>
          <div class='value'>{vtxt}</div>
          <div class='sub'>{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def donut(value, suffix="%", height=240):
    """Donut chart for a single percentage (used for missingness)."""
    v = float(np.clip(value if np.isfinite(value) else 0.0, 0, 100))
    fig = go.Figure(
        go.Pie(
            values=[v, 100 - v],
            labels=["missing", ""],
            hole=0.7,
            textinfo="none",
            sort=False,
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        annotations=[
            dict(
                text=f"{v:.1f}{suffix}",
                x=0.5,
                y=0.5,
                showarrow=False,
                font_size=18,
            )
        ],
    )
    return fig


def explain(title: str, bullets: List[str]):
    """Expandable explanation panel per tab (definitions + goals)."""
    with st.expander(f"ℹ️ What this tab shows — {title}", expanded=False):
        for b in bullets:
            st.write(f"- {b}")


# =============================================================================
# SIDEBAR: DATA SOURCES & SETTINGS
# =============================================================================
st.sidebar.header("Data Sources")

demo_choice = st.sidebar.selectbox(
    "Preloaded EV battery data",
    ["Battery Demo (3 synthetic fleets)", "None"],
    index=0,
)

feat_files = st.sidebar.file_uploader(
    "Optional: Upload feature CSV(s) or ZIP(s)",
    type=["csv", "zip"],
    accept_multiple_files=True,
)

raw_mode = st.sidebar.selectbox(
    "Optional: Upload raw time-series",
    ["None", "Raw CSV/ZIP"],
    index=0,
)
raw_files = None
if raw_mode == "Raw CSV/ZIP":
    raw_files = st.sidebar.file_uploader(
        "Raw CSV(s) or ZIP(s) with time/current/voltage",
        type=["csv", "zip"],
        accept_multiple_files=True,
    )

st.sidebar.header("Row & Memory")
keep_every = st.sidebar.number_input(
    "Downsample rows (keep every k‑th)",
    1,
    50,
    5,
    1,
)
max_rows_cell = st.sidebar.number_input(
    "Max rows per cell (raw → features)",
    50_000,
    2_000_000,
    300_000,
    50_000,
)

st.sidebar.header("Capacity / SOH & Health Buckets")
cap_mode = st.sidebar.selectbox(
    "Capacity integration",
    ["discharge-only", "min(charge,discharge)"],
    index=0,
)
baseline_N = st.sidebar.number_input("Baseline cycles (N)", 1, 20, 5, 1)
eol_threshold = st.sidebar.slider(
    "EOL threshold (SOH = End Of Life)",
    0.60,
    0.95,
    0.80,
    0.01,
)

t_healthy = st.sidebar.number_input("Healthy ≥", 0.50, 1.20, 0.90, 0.01)
t_monitor = st.sidebar.number_input("Monitor ≥", 0.50, 1.20, 0.85, 0.01)
t_eol = st.sidebar.number_input("EOL <", 0.10, 1.00, 0.80, 0.01)

st.sidebar.header("Imputation (missing-data)")
imp_choice = st.sidebar.selectbox(
    "Imputer",
    ["Median (Simple)", "KNN (k=5)", "Iterative (MICE)"],
    index=0,
)

min_labels_train = 20  # minimum labeled cycles for modeling


# =============================================================================
# GENERIC UTILITIES
# =============================================================================
def downcast_inplace(df: pd.DataFrame) -> None:
    """Downcast float64/int64 to save memory."""
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")


def numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def prep_numeric_matrix(
    X: pd.DataFrame, min_non_na: int = 1, drop_const: bool = True
) -> pd.DataFrame:
    """Select numeric columns with at least `min_non_na` non-missing values."""
    if X is None or X.empty:
        return pd.DataFrame(index=getattr(X, "index", None))
    Xn = X.select_dtypes(include=[np.number]).copy()
    keep_non_na = Xn.notna().sum(axis=0) >= int(min_non_na)
    Xn = Xn.loc[:, keep_non_na]
    if drop_const and Xn.shape[1] > 0:
        var = Xn.var(axis=0, skipna=True)
        Xn = Xn.loc[:, var > 0]
    return Xn


def impute_matrix(X: pd.DataFrame, choice: str):
    """Return (imputed numeric DataFrame, fitted imputer)."""
    Xn = prep_numeric_matrix(X, min_non_na=1, drop_const=True)
    if Xn is None or Xn.shape[1] == 0:
        return pd.DataFrame(index=getattr(X, "index", None)), None
    if choice.startswith("Median"):
        imp = SimpleImputer(strategy="median")
    elif choice.startswith("KNN"):
        imp = KNNImputer(n_neighbors=5, weights="uniform")
    else:
        imp = IterativeImputer(
            random_state=7, sample_posterior=False, max_iter=15, initial_strategy="median"
        )
    arr = imp.fit_transform(Xn)
    Xm = pd.DataFrame(arr, columns=Xn.columns, index=Xn.index)
    return Xm, imp


def pct_missing(df: pd.DataFrame) -> float:
    if df.size == 0:
        return 0.0
    return 100.0 * df.isna().mean().mean()


def corr_heatmap(df: pd.DataFrame, title: str, key: str):
    """Simple matplotlib correlation heatmap (avoids heavy interactive layout)."""
    cols = [c for c in numeric_cols(df) if df[c].notna().sum() >= 3][:18]
    if len(cols) < 2:
        st.info("Need ≥2 numeric columns with variation for correlation.")
        return
    C = df[cols].corr()
    fig = plt.figure(figsize=(0.55 * len(cols) + 3, 0.55 * len(cols) + 3))
    plt.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(shrink=0.8)
    plt.title(title)
    plt.xticks(range(len(cols)), cols, rotation=90, fontsize=9)
    plt.yticks(range(len(cols)), cols, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def bucketize_soh(s: pd.Series, th: float, tm: float, te: float) -> pd.Series:
    """
    Map continuous SOH to health buckets.

    SOH = State Of Health (fraction of nominal capacity).
    th  = Healthy threshold.
    tm  = Monitor threshold.
    te  = End-of-life threshold.
    """
    s = pd.to_numeric(s, errors="coerce")
    lab = pd.Series("Unknown", index=s.index, dtype="object")
    lab[(s >= th)] = "Healthy"
    lab[(s < th) & (s >= tm)] = "Monitor"
    lab[(s < tm) & (s >= te)] = "Aging"
    lab[(s < te)] = "EOL"
    lab[s.isna()] = "Missing"
    return lab


def bucket_shares(df: pd.DataFrame, th: float, tm: float, te: float) -> pd.DataFrame:
    """Return counts & shares of SOH health buckets."""
    if "soh" not in df.columns or df["soh"].notna().sum() == 0:
        return pd.DataFrame(columns=["bucket", "count", "share"])
    lab = bucketize_soh(df["soh"], th, tm, te)
    ct = (
        lab.value_counts(dropna=False)
        .rename_axis("bucket")
        .reset_index(name="count")
    )
    total = float(ct["count"].sum()) or 1.0
    ct["share"] = ct["count"] / total
    order = ["Healthy", "Monitor", "Aging", "EOL", "Missing", "Unknown"]
    ct["ord"] = ct["bucket"].apply(lambda b: order.index(b) if b in order else len(order))
    return ct.sort_values(["ord", "bucket"]).drop(columns=["ord"])


# =============================================================================
# SYNTHETIC EV BATTERY DEMO DATA (3 fleets)
# =============================================================================
def seed_everything(seed=7):
    np.random.seed(seed)


def ocv_soc_curve(soc):
    """Smooth S-shaped OCV vs SOC curve [V]."""
    return 3.0 + 1.2 * 1 / (1 + np.exp(-8 * (soc - 0.5))) + 0.05 * np.sin(
        6 * np.pi * soc
    )


def synth_battery_demo(
    n_cells=6,
    n_cycles=130,
    n_samples=400,
    mcar=0.06,
    mar=0.10,
    seed=7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create realistic but synthetic EV-like battery data.

    Returns:
        raw  : time-series (cell_id, cycle, time_s, current_a, voltage_v, temperature_c)
        feat : per-cycle features + SOH/capacity with MCAR+MAR missingness
    """
    seed_everything(seed)
    raw_rows = []
    feat_rows = []

    for c in range(n_cells):
        cell_id = f"CELL_{c+1:02d}"
        base_cap = np.random.uniform(2.6, 3.0)  # Ah
        r0 = np.random.normal(0.045, 0.008)  # Ohm
        deg_rate = np.clip(
            np.random.normal(0.0016, 0.0005),
            0.0009,
            0.0025,
        )  # per-cycle SOH drop
        therm_gain = np.random.uniform(0.015, 0.03)

        cap_per_cycle = []

        for cyc in range(n_cycles):
            T = np.random.uniform(900, 1800)  # seconds
            t = np.linspace(0, T, n_samples)

            # current profile: rest → discharge → rest → charge
            I = np.zeros_like(t)
            d1 = slice(int(0.05 * n_samples), int(0.55 * n_samples))
            I[d1] = -np.random.uniform(1.5, 3.0)  # discharge
            d2 = slice(int(0.65 * n_samples), int(0.9 * n_samples))
            I[d2] = +np.random.uniform(1.2, 2.2)  # charge
            I += np.random.normal(0, 0.05, size=n_samples)

            soh_true = max(
                0.6,
                1.0 - deg_rate * cyc + np.random.normal(0, 0.002),
            )
            cap_ah_true = base_cap * soh_true

            dt = t[1] - t[0]
            q_ah = np.cumsum(I * dt) / 3600.0
            soc = np.clip(0.6 - q_ah / cap_ah_true, 0.05, 0.98)

            ocv = ocv_soc_curve(soc)
            V = ocv - I * r0 + np.random.normal(0, 0.01, size=n_samples)

            Tamb = np.random.uniform(23, 29)
            Tm = Tamb + therm_gain * (I ** 2) * r0 * 20 + np.random.normal(
                0, 0.2, size=n_samples
            )

            thr = 0.3
            dmask = I < -thr
            cap_est = np.sum((-I[dmask]) * dt) / 3600.0

            raw_rows.append(
                np.c_[
                    np.full(n_samples, cell_id),
                    np.full(n_samples, cyc),
                    t,
                    I,
                    V,
                    Tm,
                ]
            )

            e_abs = np.sum(np.abs(I * V) * dt)
            q_abs = np.sum(np.abs(I) * dt) / 3600.0
            temp_mean = float(np.mean(Tm))
            temp_max = float(np.max(Tm))
            v_mean = float(np.mean(V))
            v_std = float(np.std(V))
            r_est = (
                np.median(np.diff(V) / np.diff(I + 1e-6))
                if (np.any(np.diff(I) != 0))
                else r0
            )

            feat_rows.append(
                {
                    "cell_id": cell_id,
                    "cycle": cyc,
                    "cap_ah": cap_est,
                    "q_abs": q_abs,
                    "e_abs": e_abs,
                    "temp_mean": temp_mean,
                    "temp_max": temp_max,
                    "v_mean": v_mean,
                    "v_std": v_std,
                    "r_est": float(r_est),
                }
            )
            cap_per_cycle.append(cap_est)

        cap_series = pd.Series(cap_per_cycle)
        base = float(cap_series.head(5).mean())
        soh_series = cap_series / base if base > 0 else np.nan

        for cyc in range(n_cycles):
            feat_rows[c * n_cycles + cyc]["soh"] = float(soh_series.iloc[cyc])

    raw = pd.DataFrame(
        np.vstack(raw_rows),
        columns=[
            "cell_id",
            "cycle",
            "time_s",
            "current_a",
            "voltage_v",
            "temperature_c",
        ],
    )
    raw["cell_id"] = raw["cell_id"].astype(str)
    for col in ["cycle", "time_s", "current_a", "voltage_v", "temperature_c"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    downcast_inplace(raw)

    feat = pd.DataFrame(feat_rows)
    downcast_inplace(feat)

    # -------------------------------------------------------------------------
    # Inject controlled missingness: MCAR + MAR on features table
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    numc = [c for c in feat.columns if c not in ["cell_id", "cycle", "soh"]]

    # MCAR (Missing Completely At Random)
    for c in numc:
        mask = rng.random(len(feat)) < mcar
        feat.loc[mask, c] = np.nan

    # MAR (Missing At Random): when temp_max high, hide e_abs and v_std
    if {"temp_max", "e_abs", "v_std"}.issubset(feat.columns):
        thr = feat["temp_max"].quantile(0.75)
        mar_mask = (feat["temp_max"] > thr) & (rng.random(len(feat)) < mar)
        feat.loc[mar_mask, ["e_abs", "v_std"]] = np.nan

    return raw, feat


def build_demo_fleets() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Three synthetic fleets with slightly different behaviors."""
    fleets = {}

    specs = [
        ("Fleet A", 7, 0.06, 0.10),
        ("Fleet B", 19, 0.10, 0.15),
        ("Fleet C", 31, 0.08, 0.12),
    ]

    for name, seed, mcar, mar in specs:
        raw, feat = synth_battery_demo(
            n_cells=6,
            n_cycles=130,
            n_samples=400,
            mcar=mcar,
            mar=mar,
            seed=seed,
        )
        raw["dataset"] = name
        feat["dataset"] = name
        fleets[name] = {"raw": raw, "feat": feat}

    return fleets


# =============================================================================
# READING USER DATA (OPTIONAL)
# =============================================================================
def read_csv_sample(buf: BytesIO, k_keep: int) -> pd.DataFrame:
    buf.seek(0)
    out = []
    for chunk in pd.read_csv(buf, chunksize=250_000, low_memory=True):
        if k_keep > 1:
            chunk = chunk.iloc[::k_keep].copy()
        downcast_inplace(chunk)
        out.append(chunk)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def to_feature_schema(df: pd.DataFrame, default_cell: str) -> pd.DataFrame:
    """
    Normalize arbitrary per-cycle feature table to expected schema:

    Required columns:
      - cell_id : battery ID
      - cycle   : cycle index (0,1,2,...)
      - soh     : State Of Health (optional; inferred from cap_ah if missing)
      - plus engineered features (q_abs, e_abs, temps, etc.)
    """
    out = df.copy()

    if "cell_id" not in out.columns:
        out["cell_id"] = default_cell
    out["cell_id"] = out["cell_id"].astype(str)

    if "cycle" not in out.columns:
        out = out.sort_values(["cell_id"]).reset_index(drop=True)
        out["cycle"] = out.groupby("cell_id").cumcount()
    out["cycle"] = pd.to_numeric(out["cycle"], errors="coerce")

    if "soh" not in out.columns:
        if "cap_ah" in out.columns:
            def _norm(sub):
                v = sub["cap_ah"][sub["cap_ah"] > 0]
                if v.empty:
                    return pd.Series([np.nan] * len(sub), index=sub.index)
                return sub["cap_ah"] / v.iloc[0]

            out["soh"] = out.groupby("cell_id", group_keys=False).apply(_norm)
        else:
            out["soh"] = np.nan

    for c in [
        "q_abs",
        "e_abs",
        "temp_mean",
        "temp_max",
        "v_mean",
        "v_std",
        "r_est",
    ]:
        if c not in out.columns:
            out[c] = np.nan

    if "dataset" not in out.columns:
        out["dataset"] = "Uploaded"

    keep = [
        "dataset",
        "cell_id",
        "cycle",
        "soh",
        "cap_ah",
        "q_abs",
        "e_abs",
        "temp_mean",
        "temp_max",
        "v_mean",
        "v_std",
        "r_est",
    ]
    return out[keep]


# -----------------------------------------------------------------------------
# Read uploaded feature files
# -----------------------------------------------------------------------------
feat_upload: Optional[pd.DataFrame] = None
feat_files = feat_files or []

if feat_files:
    feats = []
    for f in feat_files:
        try:
            if f.name.lower().endswith(".zip"):
                with zipfile.ZipFile(BytesIO(f.read())) as zf:
                    for zname in zf.namelist():
                        if zname.lower().endswith(".csv"):
                            df = read_csv_sample(BytesIO(zf.read(zname)), int(keep_every))
                            df = to_feature_schema(
                                df,
                                default_cell=os.path.splitext(os.path.basename(zname))[0],
                            )
                            feats.append(df)
            else:
                df = read_csv_sample(BytesIO(f.read()), int(keep_every))
                df = to_feature_schema(df, default_cell=os.path.splitext(f.name)[0])
                feats.append(df)
        except Exception as e:
            st.warning(f"Skipped {f.name}: {e}")
    if feats:
        feat_upload = pd.concat(feats, ignore_index=True)
        downcast_inplace(feat_upload)
        st.success(
            f"Loaded Feature source(s): shape={feat_upload.shape}, cells={feat_upload['cell_id'].nunique()}"
        )

# -----------------------------------------------------------------------------
# Read uploaded raw files
# -----------------------------------------------------------------------------
raw_upload: Optional[pd.DataFrame] = None

if raw_mode == "Raw CSV/ZIP" and raw_files:
    raws = []
    for f in raw_files:
        try:
            if f.name.lower().endswith(".zip"):
                with zipfile.ZipFile(BytesIO(f.read())) as zf:
                    for zname in zf.namelist():
                        if zname.lower().endswith(".csv"):
                            df = read_csv_sample(BytesIO(zf.read(zname)), int(keep_every))
                            raws.append(df)
            else:
                df = read_csv_sample(BytesIO(f.read()), int(keep_every))
                raws.append(df)
        except Exception as e:
            st.warning(f"Skipped {f.name}: {e}")
    if raws:
        raw_upload = pd.concat(raws, ignore_index=True)
        col_map = {}
        for c in raw_upload.columns:
            lc = c.lower()
            if re.search(r"\btime\b", lc):
                col_map[c] = "time_s"
            if re.search(r"\bvolt", lc):
                col_map[c] = "voltage_v"
            if re.search(r"\bcurr", lc):
                col_map[c] = "current_a"
            if re.search(r"temp", lc):
                col_map[c] = "temperature_c"
            if re.search(r"\bcycle\b", lc):
                col_map[c] = "cycle"
            if re.search(r"cell", lc) or lc == "id":
                col_map[c] = "cell_id"
        raw_upload = raw_upload.rename(columns=col_map)
        if "cell_id" not in raw_upload.columns:
            raw_upload["cell_id"] = "USER_001"
        if "cycle" not in raw_upload.columns:
            raw_upload["cycle"] = 0
        raw_upload["cell_id"] = raw_upload["cell_id"].astype(str)
        for c in ["cycle", "time_s", "current_a", "voltage_v", "temperature_c"]:
            if c in raw_upload.columns:
                raw_upload[c] = pd.to_numeric(raw_upload[c], errors="coerce")
        raw_upload["dataset"] = "Uploaded Raw"
        downcast_inplace(raw_upload)
        st.success(
            f"Loaded Raw source(s): shape={raw_upload.shape}, cells={raw_upload['cell_id'].nunique()}"
        )

# =============================================================================
# BUILD DATASET DICTIONARY (Fleets + uploaded)
# =============================================================================
datasets: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {}

# Synthetic fleets
if demo_choice.startswith("Battery Demo"):
    st.info(
        "Robust EV Battery SOH & RUL: A Missing‑Data–Aware Analytics and Visualization Framework"
    )
    demo_fleets = build_demo_fleets()
    datasets.update(demo_fleets)

# Uploaded data as an additional dataset
if (feat_upload is not None) or (raw_upload is not None):
    datasets["Uploaded"] = {"raw": raw_upload, "feat": feat_upload}

if not datasets:
    st.error("No data available. Turn on the demo or upload at least one dataset.")
    st.stop()

# Dataset selection (single / multiple / all)
st.sidebar.header("Dataset selection")
all_ds_names = list(datasets.keys())
selected_ds_names = st.sidebar.multiselect(
    "Choose dataset(s) / fleets",
    all_ds_names,
    default=all_ds_names,
)
if not selected_ds_names:
    st.warning("Select at least one dataset.")
    st.stop()

# Combine selected datasets into unified raw + feature tables
raw_list, feat_list = [], []
for name in selected_ds_names:
    ds = datasets[name]
    if ds.get("raw") is not None:
        tmp = ds["raw"].copy()
        if "dataset" not in tmp.columns:
            tmp["dataset"] = name
        raw_list.append(tmp)
    if ds.get("feat") is not None:
        tmp = ds["feat"].copy()
        if "dataset" not in tmp.columns:
            tmp["dataset"] = name
        feat_list.append(tmp)

raw: Optional[pd.DataFrame] = (
    pd.concat(raw_list, ignore_index=True) if raw_list else None
)
feat_pre: Optional[pd.DataFrame] = (
    pd.concat(feat_list, ignore_index=True) if feat_list else None
)

# Build per-cycle feature table if only raw present
if raw is None and feat_pre is None:
    st.error("No usable data. Try enabling the demo fleets.")
    st.stop()

if raw is not None:
    if "dataset" not in raw.columns:
        raw["dataset"] = "Demo"
    raw["cell_id"] = raw["cell_id"].astype(str)
    if "cycle" in raw.columns:
        raw["cycle"] = pd.to_numeric(raw["cycle"], errors="coerce").fillna(0).astype(int)
    downcast_inplace(raw)

if feat_pre is None and raw is not None:
    grp_cols = [c for c in ["dataset", "cell_id", "cycle"] if c in raw.columns]
    feat = raw.groupby(grp_cols, as_index=False).agg(
        cap_ah=("current_a", lambda x: np.nan),
        q_abs=(
            "current_a",
            lambda x: np.trapz(np.abs(x.dropna()), dx=1) / 3600.0
            if x.notna().sum() > 3
            else np.nan,
        ),
        e_abs=("voltage_v", lambda v: np.nan),
        temp_mean=("temperature_c", "mean"),
        temp_max=("temperature_c", "max"),
        v_mean=("voltage_v", "mean"),
        v_std=("voltage_v", "std"),
    )
    feat["soh"] = np.nan
else:
    feat = feat_pre.copy()

# Ensure standard columns exist
for c in [
    "dataset",
    "cell_id",
    "cycle",
    "soh",
    "cap_ah",
    "q_abs",
    "e_abs",
    "temp_mean",
    "temp_max",
    "v_mean",
    "v_std",
    "r_est",
]:
    if c not in feat.columns:
        if c == "dataset":
            feat[c] = "Demo"
        elif c == "cell_id":
            feat[c] = "CELL_001"
        elif c == "cycle":
            feat[c] = 0
        else:
            feat[c] = np.nan

feat["dataset"] = feat["dataset"].astype(str)
feat["cell_id"] = feat["cell_id"].astype(str)
feat["cycle"] = pd.to_numeric(feat["cycle"], errors="coerce").fillna(0).astype(int)
downcast_inplace(feat)

miss_all = pct_missing(feat)

# =============================================================================
# TABS
# =============================================================================
tabs = st.tabs(
    [
        "🏠 Summary",
        "Overview",
        "Data & Quality",
        "Feature Explorer",
        "Missingness Lab",
        "SOH & RUL",
        "Early-Life",
        "Anomaly & Thermal",
        "ΔSOH",
        "Robustness & OOD",
        "EDA Gallery",
        "Export",
    ]
)

# -----------------------------------------------------------------------------
# SUMMARY TAB
# -----------------------------------------------------------------------------
with tabs[0]:
    explain(
        "Summary (KPI + Story)",
        [
            "Main goal: quick narrative snapshot.",
            "Definitions: SOH = State Of Health, RUL = Remaining Useful Life, EOL = End Of Life.",
            "You see: KPIs, SOH vs cycle, energy throughput, missingness and health buckets.",
        ],
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Total Cells", int(feat["cell_id"].nunique()), "distinct cell_id")
    with c2:
        kpi("Labeled Cycles", int(feat["soh"].notna().sum()), "rows with SOH")
    with c3:
        kpi(
            "Average SOH",
            float(feat["soh"].mean(skipna=True))
            if feat["soh"].notna().any()
            else np.nan,
        )
    with c4:
        kpi("Avg % Missing", f"{miss_all:.1f}%", "across all columns")

    a, b, c = st.columns([1.3, 1.0, 1.0])

    # SOH vs cycle per cell (colored by dataset via facet)
    with a:
        st.markdown("**SOH vs Cycle (per cell, by dataset)**")
        g = feat.dropna(subset=["cycle"]).sort_values(
            ["dataset", "cell_id", "cycle"]
        )
        if g["soh"].notna().any():
            fig = px.line(
                g,
                x="cycle",
                y="soh",
                color="cell_id",
                facet_col="dataset",
                facet_col_wrap=2,
                template=PLOTLY_TEMPLATE,
                height=340,
            )
            fig.add_hline(y=eol_threshold, line_dash="dot", opacity=0.6)
            fig.update_traces(line=dict(width=2))
            fig.update_layout(margin=dict(l=6, r=6, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True, key="sum_soh_line_v3")
        else:
            st.info("No SOH labels available to plot.")

    # Energy throughput by cell
    with b:
        st.markdown("**Energy Throughput Σ(e_abs) by Cell**")
        if "e_abs" in feat.columns and feat["e_abs"].notna().any():
            grp = (
                feat.groupby(["dataset", "cell_id"], as_index=False)["e_abs"]
                .sum()
                .sort_values("e_abs")
            )
            fig2 = px.bar(
                grp,
                x="e_abs",
                y="cell_id",
                color="dataset",
                orientation="h",
                template=PLOTLY_TEMPLATE,
                height=320,
            )
            fig2.update_layout(
                margin=dict(l=6, r=6, t=30, b=0),
                xaxis_title="energy (arb.)",
                yaxis_title="cell_id",
            )
            st.plotly_chart(fig2, use_container_width=True, key="sum_energy_v3")
        else:
            st.info("No e_abs available.")

    # Missingness donuts
    with c:
        st.markdown("**Data Completeness**")
        num_cols_all = feat.select_dtypes(include=[np.number]).columns.tolist()
        miss_num = (
            float(100.0 * feat[num_cols_all].isna().mean().mean())
            if len(num_cols_all)
            else np.nan
        )
        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(
                donut(miss_num if np.isfinite(miss_num) else 0.0),
                use_container_width=True,
                key="donut_num_v3",
            )
        with d2:
            st.plotly_chart(
                donut(miss_all),
                use_container_width=True,
                key="donut_all_v3",
            )
        st.caption(
            "Left: average % missing across numeric columns. Right: across all columns."
        )

    st.write("---")
    st.markdown("**Health Buckets (based on SOH)**")
    share = bucket_shares(feat, t_healthy, t_monitor, t_eol)
    if not share.empty:
        figp = px.pie(
            share,
            values="count",
            names="bucket",
            hole=0.55,
            template=PLOTLY_TEMPLATE,
            height=300,
        )
        figp.update_traces(textinfo="percent+label")
        st.plotly_chart(figp, use_container_width=True, key="sum_buckets_v3")
    else:
        st.info("No SOH to compute buckets.")

    if "dataset" in feat.columns and feat["soh"].notna().any():
        st.markdown("**Mean SOH by dataset**")
        ds_summary = (
            feat.dropna(subset=["soh"])
            .groupby("dataset")["soh"]
            .mean()
            .reset_index(name="mean_soh")
        )
        fig_ds = px.bar(
            ds_summary,
            x="dataset",
            y="mean_soh",
            template=PLOTLY_TEMPLATE,
            height=280,
        )
        st.plotly_chart(fig_ds, use_container_width=True, key="sum_ds_mean_soh")


# -----------------------------------------------------------------------------
# OVERVIEW TAB
# -----------------------------------------------------------------------------
with tabs[1]:
    explain(
        "Overview (Data types & stats)",
        [
            "Goal: confirm that numeric / categorical fields and encodings look sensible.",
            "Shows: head(), type summary, % missing, unique counts, and full describe().",
            "Dataset column tells you which fleet or upload each row came from.",
        ],
    )

    st.write("**Head (first 40 rows)**")
    st.dataframe(feat.head(40), use_container_width=True)

    st.markdown("**Type Summary & Basic Stats**")
    c1, c2 = st.columns([1.0, 1.2])
    with c1:
        info = pd.DataFrame(
            {
                "column": feat.columns,
                "dtype": [str(feat[c].dtype) for c in feat.columns],
                "pct_missing": [
                    float(100.0 * feat[c].isna().mean()) for c in feat.columns
                ],
                "n_unique": [feat[c].nunique(dropna=True) for c in feat.columns],
            }
        )
        st.dataframe(info, use_container_width=True)
    with c2:
        st.dataframe(
            feat.describe(include="all").transpose(),
            use_container_width=True,
        )

# -----------------------------------------------------------------------------
# DATA & QUALITY (RAW TIME SERIES) TAB
# -----------------------------------------------------------------------------
with tabs[2]:
    explain(
        "Data & Quality (raw I/V/T)",
        [
            "Goal: inspect raw **time series**: I = current (A), V = voltage (V), T = temperature (°C).",
            "Shows oscilloscope‑style plots for a selected cell and cycle.",
            "Use this to sanity‑check experiment protocols and look for logging glitches.",
        ],
    )
    if raw is None or not {"time_s", "current_a", "voltage_v"}.issubset(raw.columns):
        st.info(
            "Raw signals not available. The synthetic demo fleets include raw I/V/T time series."
        )
    else:
        ucells = sorted(raw["cell_id"].astype(str).unique().tolist())
        cc1, cc2, cc3 = st.columns(3)
        pick_ds = cc1.selectbox(
            "dataset / fleet", sorted(raw["dataset"].unique()), key="dq_ds"
        )
        rc = raw[raw["dataset"] == pick_ds]
        cells_ds = sorted(rc["cell_id"].astype(str).unique().tolist())
        pick_cell = cc2.selectbox("cell_id", cells_ds, index=0, key="dq_cell")
        cycs = (
            rc.loc[rc["cell_id"] == pick_cell, "cycle"]
            .dropna()
            .astype(int)
            .unique()
        )
        cycs = sorted(cycs.tolist())
        if not cycs:
            st.info("No cycles for this cell.")
        else:
            pick_cyc = cc3.selectbox(
                "cycle (index)",
                cycs,
                index=min(10, len(cycs) - 1),
                key="dq_cycle",
            )
            g = rc[(rc["cell_id"] == pick_cell) & (rc["cycle"] == pick_cyc)].sort_values(
                "time_s"
            )
            if len(g) >= 5:
                fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
                axes[0].plot(g["time_s"], g["current_a"])
                axes[0].set_ylabel("Current (A)")
                axes[1].plot(g["time_s"], g["voltage_v"])
                axes[1].set_ylabel("Voltage (V)")
                if "temperature_c" in g.columns:
                    axes[2].plot(g["time_s"], g["temperature_c"])
                else:
                    axes[2].plot(g["time_s"], np.zeros(len(g)))
                axes[2].set_ylabel("Temp (°C)")
                axes[2].set_xlabel("Time (s)")
                for ax in axes:
                    ax.grid(True, alpha=0.3)
                st.pyplot(fig, clear_figure=True)

# -----------------------------------------------------------------------------
# FEATURE EXPLORER TAB
# -----------------------------------------------------------------------------
with tabs[3]:
    explain(
        "Feature Explorer",
        [
            "Goal: explore per‑cycle engineered features and their relationships.",
            "Includes table view, correlation heatmap, and several fixed scatter plots.",
        ],
    )

    st.write("**Per-cycle features (first 30)**")
    st.dataframe(feat.head(30), use_container_width=True)

    corr_heatmap(feat, "Feature Correlation", key="feat_corr_v3")

    # Fixed scatter plots (no user selections → faster)
    pairs = [("q_abs", "soh"), ("e_abs", "soh"), ("temp_max", "soh")]
    for i, (xcol, ycol) in enumerate(pairs, start=1):
        if xcol in feat.columns and ycol in feat.columns:
            st.markdown(f"**Scatter: {ycol} vs {xcol}**")
            fig = px.scatter(
                feat,
                x=xcol,
                y=ycol,
                color="dataset",
                hover_data=["cell_id", "cycle"],
                template=PLOTLY_TEMPLATE,
                opacity=0.85,
                height=320,
            )
            fig.update_traces(marker=dict(size=6))
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"feat_scatter_auto_{i}",
            )

# -----------------------------------------------------------------------------
# MISSINGNESS LAB TAB
# -----------------------------------------------------------------------------
with tabs[4]:
    explain(
        "Missingness Lab (MCAR/MAR + Imputation)",
        [
            "Goal: understand missing data patterns and test imputation approaches.",
            "MCAR = Missing Completely At Random, MAR = Missing At Random.",
            "Imputers: Simple median, k‑Nearest Neighbors (KNN), and Iterative (MICE).",
        ],
    )

    numc = [c for c in numeric_cols(feat) if feat[c].notna().sum() >= 5]
    if not numc:
        st.info("No numeric columns to analyze.")
    else:
        st.markdown("**% Missing by Column (numeric)**")
        pmiss = feat[numc].isna().mean().sort_values(ascending=False)
        st.dataframe(pmiss.to_frame("pct_missing"), use_container_width=True)
        figm = px.bar(
            pmiss.reset_index().rename(columns={"index": "column", 0: "pct_missing"}),
            x="column",
            y="pct_missing",
            template=PLOTLY_TEMPLATE,
            height=280,
        )
        figm.update_layout(xaxis_tickangle=45, margin=dict(l=6, r=6, t=6, b=6))
        st.plotly_chart(figm, use_container_width=True, key="miss_bar_v3")

        st.markdown("**Missingness Pattern (rows × columns)**")
        samp = feat[numc].head(300)
        miss_mat = samp.isna().astype(int)
        figmm = px.imshow(
            miss_mat.T,
            color_continuous_scale="Viridis",
            template=PLOTLY_TEMPLATE,
            aspect="auto",
            height=360,
            labels=dict(x="row", y="column", color="missing"),
        )
        st.plotly_chart(figmm, use_container_width=True, key="miss_mat_v2")

        # MCAR experiment on a fixed column
        target = "e_abs" if "e_abs" in numc else numc[0]
        rng = np.random.default_rng(7)
        f_mcar = feat.copy()
        idx = f_mcar.index.to_numpy()
        k = int(0.25 * len(idx))
        mask_idx = rng.choice(idx, size=k, replace=False)
        base_vec = f_mcar[target].copy()
        f_mcar.loc[mask_idx, target] = np.nan

        scores = []
        for label in ["Median (Simple)", "KNN (k=5)", "Iterative (MICE)"]:
            Xi, _ = impute_matrix(f_mcar[numc], label)
            if Xi.empty or target not in Xi.columns:
                rmse = np.nan
            else:
                rmse = float(np.sqrt(np.nanmean((Xi[target] - base_vec) ** 2)))
            scores.append({"imputer": label, "RMSE_vs_true": rmse})

        comp = pd.DataFrame(scores).sort_values("RMSE_vs_true")
        st.markdown(
            f"**Imputation comparison on MCAR target:** `{target}` (lower RMSE is better)"
        )
        st.dataframe(comp, use_container_width=True)
        figc = px.bar(
            comp,
            x="imputer",
            y="RMSE_vs_true",
            template=PLOTLY_TEMPLATE,
            height=260,
        )
        st.plotly_chart(figc, use_container_width=True, key="imp_compare_v3")
        st.caption(
            "Interpretation: if MCAR holds, simple imputation can work; "
            "Iterative/MICE often gives the smallest error."
        )

# -----------------------------------------------------------------------------
# SOH & RUL TAB
# -----------------------------------------------------------------------------
with tabs[5]:
    explain(
        "SOH & RUL modeling",
        [
            "SOH = State Of Health (capacity fraction).",
            "RUL = Remaining Useful Life (cycles until SOH ≤ EOL threshold).",
            "This tab trains multiple regressors (Linear, RandomForest, GradientBoosting, MLP neural net) using grouped cross‑validation across cells.",
        ],
    )

    n_labels = int(feat["soh"].notna().sum())
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("SOH labels", n_labels, "rows with SOH")
    with c2:
        kpi("Cells (labeled)", int(feat[feat["soh"].notna()]["cell_id"].nunique()))
    with c3:
        kpi("EOL threshold (SOH)", eol_threshold)

    if n_labels < max(10, min_labels_train // 2):
        st.info(
            "Not enough labeled SOH cycles to train regressors. The demo fleets should satisfy this; "
            "if you see this with your own data, ensure SOH is provided or derivable."
        )
    else:
        dfy = feat.dropna(subset=["soh"]).copy()
        Xcols = [c for c in dfy.columns if c not in ("dataset", "cell_id", "cycle", "soh")]
        Xraw = dfy[Xcols].copy()
        Ximp, _ = impute_matrix(Xraw, imp_choice)
        y = dfy["soh"].astype(float).values

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=120,
                n_jobs=-1,
                random_state=7,
            ),
            "GradientBoosting": GradientBoostingRegressor(random_state=7),
            "MLPRegressor (NN)": MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                learning_rate_init=0.001,
                max_iter=500,
                random_state=7,
            ),
        }

        results = []
        groups = dfy["cell_id"].astype(str).values
        if dfy["cell_id"].nunique() >= 2:
            gkf = GroupKFold(
                n_splits=min(5, max(2, dfy["cell_id"].nunique()))
            )
            splits = list(gkf.split(Ximp, y, groups=groups))
        else:
            # fallback CV via random splits
            splits = []
            for rs in range(3):
                tr, te = train_test_split(
                    np.arange(len(y)),
                    test_size=0.3,
                    random_state=rs,
                )
                splits.append((tr, te))

        scaler = StandardScaler()

        for name, mdl in models.items():
            maes, r2s = [], []
            for tr, te in splits:
                Xtr, Xte = Ximp.iloc[tr], Ximp.iloc[te]
                ytr, yte = y[tr], y[te]
                Xtr_s = scaler.fit_transform(Xtr)
                Xte_s = scaler.transform(Xte)
                mdl.fit(Xtr_s, ytr)
                yhat = mdl.predict(Xte_s)
                maes.append(mean_absolute_error(yte, yhat))
                r2s.append(r2_score(yte, yhat))
            results.append(
                {
                    "model": name,
                    "MAE_mean": np.mean(maes),
                    "MAE_std": np.std(maes),
                    "R2_mean": np.mean(r2s),
                    "R2_std": np.std(r2s),
                }
            )

        res_df = pd.DataFrame(results).sort_values("MAE_mean")
        st.markdown("**SOH regression model comparison (cross‑validation)**")
        st.dataframe(
            res_df.style.format(
                {
                    "MAE_mean": "{:.4f}",
                    "MAE_std": "{:.4f}",
                    "R2_mean": "{:.3f}",
                    "R2_std": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

        st.markdown(
            f"Best average MAE model on this run: **{res_df.iloc[0]['model']}**."
        )

        # Per-dataset performance for best model (GradientBoosting as a strong baseline)
        st.markdown("---")
        st.markdown("**Per‑dataset SOH performance (GradientBoosting baseline)**")
        gb = GradientBoostingRegressor(random_state=7)
        per_rows = []
        for ds_name, sub in dfy.groupby("dataset"):
            if len(sub) < 15:
                continue
            Xd = sub[Xcols].copy()
            Xd_imp, _ = impute_matrix(Xd, imp_choice)
            yd = sub["soh"].astype(float).values
            if len(yd) < 10:
                continue
            groups_ds = sub["cell_id"].astype(str).values
            if sub["cell_id"].nunique() >= 2:
                gkf_ds = GroupKFold(
                    n_splits=min(3, max(2, sub["cell_id"].nunique()))
                )
                maes_ds, r2s_ds = [], []
                for tr, te in gkf_ds.split(Xd_imp, yd, groups=groups_ds):
                    Xtr, Xte = Xd_imp.iloc[tr], Xd_imp.iloc[te]
                    ytr, yte = yd[tr], yd[te]
                    Xtr_s = scaler.fit_transform(Xtr)
                    Xte_s = scaler.transform(Xte)
                    gb.fit(Xtr_s, ytr)
                    yhat = gb.predict(Xte_s)
                    maes_ds.append(mean_absolute_error(yte, yhat))
                    r2s_ds.append(r2_score(yte, yhat))
                per_rows.append(
                    {
                        "dataset": ds_name,
                        "MAE_mean": np.mean(maes_ds),
                        "R2_mean": np.mean(r2s_ds),
                    }
                )

        if per_rows:
            st.dataframe(pd.DataFrame(per_rows), use_container_width=True)
        else:
            st.info(
                "Per‑dataset evaluation not shown (need ≥2 labeled cells per dataset)."
            )

        # ------------------------ RUL (cycles to EOL) ------------------------
        st.markdown("---")
        st.markdown("**RUL (Remaining Useful Life in cycles)**")

        rul_rows = []
        for (ds, cell), g in feat.groupby(["dataset", "cell_id"]):
            gs = g.sort_values("cycle")
            if gs["soh"].notna().sum() < 5:
                continue
            hit = gs.index[gs["soh"] <= eol_threshold]
            if len(hit) == 0:
                continue
            eol_cycle = int(gs.loc[hit[0], "cycle"])
            cur = gs[gs["cycle"] < eol_cycle]
            if len(cur) == 0:
                continue
            rr = cur.copy()
            rr["rul_cycles"] = eol_cycle - rr["cycle"].astype(int)
            rul_rows.append(rr)

        if not rul_rows:
            st.info(
                "No cells reach EOL in the current selection; lower the EOL threshold or include later cycles."
            )
        else:
            drul = pd.concat(rul_rows, ignore_index=True)
            Xr = drul[
                [c for c in drul.columns if c not in ("dataset", "cell_id", "cycle", "soh", "rul_cycles")]
            ]
            Xr_imp, _ = impute_matrix(Xr, imp_choice)
            yr = drul["rul_cycles"].astype(float).values
            if len(yr) < 10:
                st.info("Not enough RUL samples to train.")
            else:
                if drul["cell_id"].nunique() >= 2:
                    gkf = GroupKFold(
                        n_splits=min(5, max(2, drul["cell_id"].nunique()))
                    )
                    maes_r = []
                    for tr, te in gkf.split(
                        Xr_imp, yr, groups=drul["cell_id"].astype(str).values
                    ):
                        reg = RandomForestRegressor(
                            random_state=7, n_estimators=120, n_jobs=-1
                        ).fit(Xr_imp.iloc[tr], yr[tr])
                        yhat = reg.predict(Xr_imp.iloc[te])
                        maes_r.append(mean_absolute_error(yr[te], yhat))
                    st.metric("RUL MAE (cycles, CV)", f"{np.mean(maes_r):.2f}")
                else:
                    Xtr, Xte, ytr, yte = train_test_split(
                        Xr_imp, yr, test_size=0.3, random_state=7
                    )
                    reg = RandomForestRegressor(
                        random_state=7, n_estimators=120, n_jobs=-1
                    ).fit(Xtr, ytr)
                    yhat = reg.predict(Xte)
                    st.metric("RUL MAE (cycles, holdout)", f"{mean_absolute_error(yte, yhat):.2f}")

# -----------------------------------------------------------------------------
# EARLY-LIFE TAB
# -----------------------------------------------------------------------------
with tabs[6]:
    explain(
        "Early‑Life Screening",
        [
            "Goal: detect 'short‑life' cells using only the first N cycles.",
            "Short‑life = bottom ~30% by life length (cycles until SOH ≤ EOL) across cells.",
            "Classifier: GradientBoostingClassifier; metric: Average Precision (area under precision‑recall).",
        ],
    )

    def life_index(df: pd.DataFrame, eol: float) -> pd.Series:
        out = {}
        for (ds, cell), g in df.groupby(["dataset", "cell_id"]):
            s = g.sort_values("cycle")["soh"].astype(float)
            idx = np.argmax(s.values <= eol) if np.any(s.values <= eol) else len(s)
            out[f"{ds}:{cell}"] = int(idx)
        return pd.Series(out, name="life")

    lifemap = life_index(feat, eol_threshold)
    if len(lifemap) < 3:
        st.info("Need ≥3 dataset+cell combinations for grouped classification.")
    else:
        N = 15  # number of early cycles used
        features_used = [
            c for c in feat.columns if c not in ("dataset", "cell_id", "cycle", "soh")
        ]
        early_raw = feat[feat["cycle"] < N]
        early = (
            early_raw.groupby(["dataset", "cell_id"])[features_used]
            .mean(numeric_only=True)
            .reset_index()
        )
        idx_key = early["dataset"] + ":" + early["cell_id"]
        k = max(1, int(np.ceil(0.3 * len(lifemap))))
        short_ids = set(lifemap.sort_values().index[:k].tolist())
        y = idx_key.map(lambda cid: 1 if cid in short_ids else 0).astype(int)
        Ximp, _ = impute_matrix(
            early[features_used], imp_choice
        )  # numeric only

        if Ximp.shape[1] == 0 or np.unique(y.values).size < 2:
            st.info("Not enough diversity/columns for early-life classification.")
        else:
            groups = idx_key.values
            gkf = GroupKFold(n_splits=min(5, max(2, len(early))))
            aps = []
            for tr, te in gkf.split(Ximp.values, y.values, groups=groups):
                clf = GradientBoostingClassifier(random_state=7).fit(
                    Ximp.iloc[tr].values, y.values[tr]
                )
                proba = clf.predict_proba(Ximp.iloc[te].values)[:, 1]
                aps.append(average_precision_score(y.values[te], proba))
            st.metric("Average Precision (mean folds)", f"{np.mean(aps):.3f}")
            st.caption(
                "Interpretation: higher Average Precision means better ranking of short‑life cells "
                "based only on their first few cycles."
            )

# -----------------------------------------------------------------------------
# ANOMALY & THERMAL TAB
# -----------------------------------------------------------------------------
with tabs[7]:
    explain(
        "Anomaly & Thermal",
        [
            "Anomaly detection using PCA reconstruction residuals and IsolationForest.",
            "Thermal triage using Tmax (maximum temperature) and |dT/dt| (temperature ramp rate).",
            "OOD here means cycles that look unusual in feature space or in thermal behavior.",
        ],
    )

    feat_cols = [c for c in feat.columns if c not in ("dataset", "cell_id", "cycle", "soh")]
    Ximp, _ = impute_matrix(feat[feat_cols], imp_choice) if len(feat_cols) else (
        pd.DataFrame(),
        None,
    )

    if not Ximp.empty and Ximp.shape[1] >= 2 and Ximp.shape[0] >= 5:
        n_comp = max(1, min(8, Ximp.shape[1] - 1, Ximp.shape[0] - 1))
        pca = PCA(n_components=n_comp, random_state=7)
        Xp = pca.fit_transform(Ximp.values)
        Xh = pca.inverse_transform(Xp)
        resid = np.mean((Ximp.values - Xh) ** 2, axis=1)
        iso = IsolationForest(
            contamination=0.06, random_state=7, n_jobs=-1
        ).fit(Ximp.values)
        iso_score = -iso.score_samples(Ximp.values)

        ff = feat.copy()
        ff.loc[Ximp.index, "anom_pca"] = resid
        ff.loc[Ximp.index, "anom_iso"] = iso_score
        top = ff.loc[Ximp.index].nlargest(
            12, "anom_pca"
        )[["dataset", "cell_id", "cycle", "anom_pca", "anom_iso"]]
        st.write("**Top anomalies (by PCA residual)**")
        st.dataframe(top.reset_index(drop=True), use_container_width=True)
    else:
        st.info("Not enough rows/features for PCA/IsolationForest anomaly detection.")

    st.markdown("---")
    st.write("**Thermal screen** (fixed demo thresholds)")
    Tmax = 55.0
    dTlim = 0.25

    if raw is None or "time_s" not in raw.columns:
        st.info("Raw time-series not available for thermal plots.")
    else:
        uc = sorted(raw["cell_id"].astype(str).unique().tolist())
        cid = uc[0]
        g_all = raw[raw["cell_id"] == cid]
        cyc = int(np.median(g_all["cycle"]))
        g = g_all[g_all["cycle"] == cyc].sort_values("time_s")
        if len(g) >= 5:
            dt = np.median(np.diff(g["time_s"])) if g["time_s"].nunique() > 1 else 1.0
            if "temperature_c" in g.columns:
                dTdt = np.gradient(g["temperature_c"]) / max(dt, 1e-6)
            else:
                dTdt = np.zeros(len(g))
            fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            if "temperature_c" in g.columns:
                axes[0].plot(g["time_s"], g["temperature_c"])
            axes[0].hlines(
                Tmax,
                g["time_s"].min(),
                g["time_s"].max(),
                colors="r",
                linestyles="--",
            )
            axes[1].plot(g["time_s"], dTdt)
            axes[1].hlines(
                [dTlim, -dTlim],
                g["time_s"].min(),
                g["time_s"].max(),
                colors="r",
                linestyles="--",
            )
            axes[0].set_ylabel("Temp (°C)")
            axes[1].set_ylabel("dT/dt (°C/s)")
            axes[1].set_xlabel("Time (s)")
            for ax in axes:
                ax.grid(True, alpha=0.3)
            st.pyplot(fig, clear_figure=True)

# -----------------------------------------------------------------------------
# ΔSOH TAB
# -----------------------------------------------------------------------------
with tabs[8]:
    explain(
        "ΔSOH (next‑cycle change)",
        [
            "Goal: predict short‑term degradation ΔSOH(t+1) using a sliding history window.",
            "Model: RandomForestRegressor on small sequences of past SOH values.",
        ],
    )

    if feat["soh"].notna().sum() < 40:
        st.info("Need labeled SOH sequences to estimate ΔSOH.")
    else:

        def make_delta_dataset(s: np.ndarray, k_back=5):
            X, y = [], []
            for t in range(k_back, len(s) - 1):
                X.append(s[t - k_back : t])
                y.append(s[t + 1] - s[t])
            return np.array(X), np.array(y)

        k_back = 5
        Xd_list, yd_list = [], []
        for (_, cell), g in feat.groupby(["dataset", "cell_id"]):
            s = (
                g.sort_values("cycle")["soh"]
                .astype(float)
                .dropna()
                .values
            )
            if len(s) > k_back + 1:
                Xd, yd = make_delta_dataset(s, k_back=k_back)
                Xd_list.append(Xd)
                yd_list.append(yd)
        if not Xd_list:
            st.info("Not enough continuous SOH sequences for ΔSOH modeling.")
        else:
            Xd = np.vstack(Xd_list)
            yd = np.concatenate(yd_list)
            Xtr, Xte, ytr, yte = train_test_split(
                Xd, yd, test_size=0.25, random_state=7
            )
            dreg = RandomForestRegressor(
                random_state=7, n_estimators=150, n_jobs=-1
            ).fit(Xtr, ytr)
            yhat = dreg.predict(Xte)
            st.metric("ΔSOH MAE (absolute)", f"{mean_absolute_error(yte, yhat):.6f}")
            fig = plt.figure(figsize=(7, 3))
            n = min(250, len(yte))
            plt.plot(yte[:n], label="True")
            plt.plot(yhat[:n], label="Pred")
            plt.title("Next-cycle ΔSOH (sample)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(fig, clear_figure=True)

# -----------------------------------------------------------------------------
# ROBUSTNESS & OOD TAB
# -----------------------------------------------------------------------------
with tabs[9]:
    explain(
        "Robustness & OOD (Out‑Of‑Distribution)",
        [
            "Goal: identify distribution shift and suspicious cycles.",
            "We use Mahalanobis distance in feature space and IsolationForest OOD scores.",
        ],
    )

    dfy = feat.dropna(subset=["soh"]).copy()
    cols = [c for c in dfy.columns if c not in ("dataset", "cell_id", "cycle", "soh")]
    if dfy.empty or not cols:
        st.info("Need labeled features to compute OOD.")
    else:
        Ximp, _ = impute_matrix(dfy[cols], imp_choice)
        if Ximp.empty:
            st.info("No usable numeric features after imputation.")
        else:
            Xn = Ximp.values
            mu = Xn.mean(axis=0, keepdims=True)
            cov = np.cov(Xn, rowvar=False) + 1e-6 * np.eye(Xn.shape[1])
            try:
                inv = np.linalg.inv(cov)
                d2 = np.array([(x - mu) @ inv @ (x - mu).T for x in Xn]).ravel()
                fig = plt.figure(figsize=(7, 3))
                plt.hist(d2, bins=40, alpha=0.85)
                plt.title("Mahalanobis distance (labeled rows)")
                st.pyplot(fig, clear_figure=True)
            except np.linalg.LinAlgError:
                st.info("Covariance not invertible; skipping Mahalanobis distance plot.")

            iso = IsolationForest(
                contamination=0.06, random_state=7, n_jobs=-1
            ).fit(Xn)
            ood_iso = -iso.score_samples(Xn)
            out = dfy[["dataset", "cell_id", "cycle"]].copy()
            out["ood_iso"] = ood_iso
            st.write("**IsolationForest OOD scores (higher = more suspicious)**")
            st.dataframe(out.head(40), use_container_width=True)

# -----------------------------------------------------------------------------
# EDA GALLERY TAB
# -----------------------------------------------------------------------------
with tabs[10]:
    explain(
        "EDA Gallery",
        [
            "Goal: one‑stop, no‑click gallery that demonstrates all IDA/EDA skills.",
            "Includes class imbalance, missingness, outliers, correlation, distributions, 2D/3D scatter, and a scatter‑matrix with KDE diagonals.",
        ],
    )

    # A. Class imbalance — by health bucket and by cell_id
    st.subheader("⚖️ Class Imbalance")
    share = bucket_shares(feat, t_healthy, t_monitor, t_eol)
    cA, cB = st.columns(2)
    if not share.empty:
        with cA:
            figp = px.pie(
                share,
                values="count",
                names="bucket",
                hole=0.45,
                template=PLOTLY_TEMPLATE,
                height=320,
            )
            st.plotly_chart(figp, use_container_width=True, key="eda_pie_buckets3")
        with cB:
            figb = px.bar(
                share,
                x="bucket",
                y="count",
                text="count",
                template=PLOTLY_TEMPLATE,
                height=320,
            )
            st.plotly_chart(figb, use_container_width=True, key="eda_bar_buckets3")
    else:
        st.info("No SOH to compute bucket distributions.")

    counts_cell = (
        feat.groupby(["dataset", "cell_id"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    fig_cell = px.bar(
        counts_cell,
        x="cell_id",
        y="count",
        color="dataset",
        template=PLOTLY_TEMPLATE,
        height=300,
    )
    st.plotly_chart(fig_cell, use_container_width=True, key="eda_bar_cell_count2")

    st.markdown("---")

    # B. Missing values — summary, bar, heatmap
    st.subheader("🔍 Missing Values")
    numc_all = numeric_cols(feat)
    missing = feat[numc_all].isna().sum()
    missing_pct = (missing / len(feat) * 100).round(2)
    miss_df = pd.DataFrame(
        {"column": missing.index, "missing": missing.values, "pct": missing_pct.values}
    ).sort_values("pct", ascending=False)
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.dataframe(miss_df, use_container_width=True)
    with c2:
        mf = miss_df[miss_df["missing"] > 0]
        if len(mf):
            fig = px.bar(
                mf,
                x="column",
                y="missing",
                template=PLOTLY_TEMPLATE,
                height=300,
                text="missing",
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True, key="eda_miss_bar_full2")
        else:
            st.info("No missing values found!")

    miss_mat = feat[numc_all].head(300).isna().astype(int)
    figmm2 = px.imshow(
        miss_mat.T,
        color_continuous_scale="Viridis",
        template=PLOTLY_TEMPLATE,
        aspect="auto",
        height=360,
        labels=dict(x="row", y="column", color="missing"),
    )
    st.plotly_chart(figmm2, use_container_width=True, key="eda_miss_heat2")

    st.markdown("---")

    # C. Outliers — box plots + IQR table
    st.subheader("📈 Outlier Detection (IQR)")
    num_cols = [
        c
        for c in ["cap_ah", "q_abs", "e_abs", "temp_max", "v_std", "r_est"]
        if c in feat.columns
    ]
    if num_cols:
        rows = int(np.ceil(len(num_cols) / 3))
        fig = make_subplots(
            rows=rows, cols=3, subplot_titles=num_cols, horizontal_spacing=0.08
        )
        for i, col in enumerate(num_cols):
            row = i // 3 + 1
            colpos = i % 3 + 1
            fig.add_trace(
                go.Box(y=feat[col], name=col, boxpoints="outliers"),
                row=row,
                col=colpos,
            )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, showlegend=False, height=280 * rows
        )
        st.plotly_chart(fig, use_container_width=True, key="eda_box_grid2")

    outlier_data = []
    for col in num_cols:
        s = pd.to_numeric(feat[col], errors="coerce").dropna()
        if len(s) < 5:
            outlier_data.append({"Feature": col, "Outlier Count": 0, "Percentage": 0.0})
            continue
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outs = s[(s < lb) | (s > ub)]
        outlier_data.append(
            {
                "Feature": col,
                "Outlier Count": int(len(outs)),
                "Percentage": round(100 * len(outs) / len(s), 2),
            }
        )
    st.dataframe(pd.DataFrame(outlier_data), use_container_width=True)

    st.markdown("---")

    # D. Correlation — table, heatmap, strong pairs
    st.subheader("🔗 Correlation Analysis")
    numeric_df = feat.select_dtypes(include=[np.number]).copy()
    corr_matrix = numeric_df.corr().round(3)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(corr_matrix, use_container_width=True)
    with c2:
        figcorr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            template=PLOTLY_TEMPLATE,
            height=500,
            labels=dict(color="corr"),
        )
        st.plotly_chart(figcorr, use_container_width=True, key="eda_corr_big2")

    strong = []
    cols_corr = corr_matrix.columns.tolist()
    for i in range(len(cols_corr)):
        for j in range(i + 1, len(cols_corr)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.5 and np.isfinite(r):
                strong.append(
                    {
                        "Feature 1": cols_corr[i],
                        "Feature 2": cols_corr[j],
                        "Correlation": r,
                    }
                )
    if strong:
        st.dataframe(
            pd.DataFrame(strong).sort_values(
                "Correlation", key=lambda s: s.abs(), ascending=False
            ),
            use_container_width=True,
        )
    else:
        st.info("No strong correlations found (|r| > 0.5).")

    st.markdown("---")

    # E. Distributions — histograms + violin
    st.subheader("📊 Distributions")
    cand = [
        c
        for c in ["cap_ah", "q_abs", "e_abs", "temp_max", "v_std", "r_est", "soh"]
        if c in feat.columns
    ]
    if cand:
        rows = int(np.ceil(len(cand) / 3))
        fig = make_subplots(
            rows=rows, cols=3, subplot_titles=cand, horizontal_spacing=0.08
        )
        for i, col in enumerate(cand):
            row = i // 3 + 1
            colpos = i % 3 + 1
            s = pd.to_numeric(feat[col], errors="coerce").dropna()
            fig.add_trace(
                go.Histogram(x=s, opacity=0.85),
                row=row,
                col=colpos,
            )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, showlegend=False, height=280 * rows
        )
        st.plotly_chart(fig, use_container_width=True, key="eda_hist_grid2")

    if "soh" in feat.columns:
        top_cells = (
            feat["cell_id"]
            .astype(str)
            .value_counts()
            .head(10)
            .index
            .tolist()
        )
        vdf = feat[feat["cell_id"].astype(str).isin(top_cells)]
        figv = px.violin(
            vdf,
            y="soh",
            x=vdf["cell_id"].astype(str),
            color="dataset",
            box=True,
            points="all",
            template=PLOTLY_TEMPLATE,
            height=380,
        )
        st.plotly_chart(figv, use_container_width=True, key="eda_violin_soh_cell2")

    st.markdown("---")

    # F. Scatter — 2D & 3D
    st.subheader("🎯 Scatter (2D & 3D)")
    pairs_auto = [("q_abs", "soh"), ("e_abs", "soh"), ("temp_max", "soh")]
    for i, (xcol, ycol) in enumerate(pairs_auto, start=1):
        if xcol in feat.columns and ycol in feat.columns:
            fig = px.scatter(
                feat,
                x=xcol,
                y=ycol,
                color="dataset",
                hover_data=["cell_id", "cycle"],
                template=PLOTLY_TEMPLATE,
                opacity=0.75,
                height=320,
            )
            fig.update_traces(marker=dict(size=6))
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"eda_scatter2d_{i}",
            )

    three = [
        c
        for c in ["q_abs", "e_abs", "temp_max", "v_std", "r_est"]
        if c in feat.columns
    ][:3]
    if len(three) == 3:
        fig3 = px.scatter_3d(
            feat,
            x=three[0],
            y=three[1],
            z=three[2],
            color="dataset",
            opacity=0.7,
            template=PLOTLY_TEMPLATE,
            height=550,
        )
        st.plotly_chart(fig3, use_container_width=True, key="eda_scatter3d2")

    st.markdown("---")

    # G. Scatter matrix with KDE diagonals (top-4 numerics)
    st.subheader("🧮 Scatter Matrix (with KDE diagonals)")
    top_num = [
        c
        for c in ["cap_ah", "q_abs", "e_abs", "temp_max", "v_std", "r_est", "soh"]
        if c in feat.columns
    ]
    top_num = top_num[:4] if len(top_num) >= 4 else top_num
    if len(top_num) >= 2:
        n_vars = len(top_num)
        fig = make_subplots(
            rows=n_vars,
            cols=n_vars,
            vertical_spacing=0.02,
            horizontal_spacing=0.02,
        )
        lab = bucketize_soh(feat["soh"], t_healthy, t_monitor, t_eol)
        cats = lab.fillna("Unknown").astype(str)
        cat_vals = cats.unique().tolist()
        pal = px.colors.qualitative.Plotly + px.colors.qualitative.D3
        color_map = {cat: pal[i % len(pal)] for i, cat in enumerate(cat_vals)}

        for i, cy in enumerate(top_num):
            for j, cx in enumerate(top_num):
                row = i + 1
                col = j + 1
                if i == j:
                    for cat in cat_vals:
                        data = pd.to_numeric(
                            feat.loc[cats == cat, cx], errors="coerce"
                        ).dropna()
                        if len(data) < 3:
                            continue
                        xs = np.linspace(data.min(), data.max(), 200)
                        if SCIPY_OK:
                            kde = stats.gaussian_kde(data)
                            ys = kde(xs)
                            fig.add_trace(
                                go.Scatter(
                                    x=xs,
                                    y=ys,
                                    mode="lines",
                                    name=str(cat),
                                    line=dict(
                                        color=color_map[cat],
                                        width=2,
                                    ),
                                    showlegend=(i == 0 and j == 0),
                                ),
                                row=row,
                                col=col,
                            )
                        else:
                            fig.add_trace(
                                go.Histogram(
                                    x=data,
                                    name=str(cat),
                                    marker_color=color_map[cat],
                                    showlegend=(i == 0 and j == 0),
                                    opacity=0.5,
                                ),
                                row=row,
                                col=col,
                            )
                else:
                    for cat in cat_vals:
                        data = feat[cats == cat]
                        fig.add_trace(
                            go.Scatter(
                                x=data[cx],
                                y=data[cy],
                                mode="markers",
                                name=str(cat),
                                marker=dict(
                                    size=4,
                                    opacity=0.5,
                                    color=color_map[cat],
                                ),
                                showlegend=False,
                            ),
                            row=row,
                            col=col,
                        )
                fig.update_xaxes(title_text=cx if i == n_vars - 1 else "", row=row, col=col)
                fig.update_yaxes(title_text=cy if j == 0 else "", row=row, col=col)

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=240 * n_vars,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True, key="eda_scatter_matrix2")
    else:
        st.info("Need ≥2 numeric columns for scatter matrix.")

# -----------------------------------------------------------------------------
# EXPORT TAB
# -----------------------------------------------------------------------------
with tabs[11]:
    explain(
        "Export",
        [
            "Goal: provide a clean per‑cycle feature table for GitHub and reproducibility.",
            "This CSV is what your README and notebook should refer to.",
        ],
    )
    feat_cols = [
        c
        for c in feat.columns
        if c not in ("dataset", "cell_id", "cycle", "soh")
    ]
    extra = [c for c in ["soh", "cap_ah"] if c in feat.columns]
    out_cols = ["dataset", "cell_id", "cycle"] + feat_cols + extra
    st.write("**Columns included in export:**", ", ".join(out_cols))
    st.download_button(
        "⬇️ Download per‑cycle features (CSV)",
        data=feat[out_cols].to_csv(index=False),
        file_name="ev_battery_features_export.csv",
        mime="text/csv",
    )
