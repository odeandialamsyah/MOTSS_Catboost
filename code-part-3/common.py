"""Shared preprocessing that mirrors MOTSS_PSO_Y1.ipynb (cells 8-52)."""
import os, numpy as np, pandas as pd, warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
warnings.filterwarnings("ignore")

# Dataset path inside the repository (dataset/ folder next to code-part-3/)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "dataset_pipeline_penelitian_utama.xlsx")
os.makedirs("figures", exist_ok=True)

TARGET = "potential_biogas_day"
DERIVED = ["total_manure_potential_year_cattle_ton", "total_manure_potential_day_cattle_kg",
           "total_manure_potential_year_chicken_ton", "total_manure_potential_day_chicken_kg"]

# PSO-optimal hyper-parameters reported in the notebook output (cell 55)
PSO_PARAMS = {
    "LR": {}, "Ridge": {"alpha": 10.0}, "Lasso": {"alpha": 0.11155173890309131},
    "ElasticNet": {"alpha": 0.11417021046864367, "l1_ratio": 0.9664065573555832},
    "KNN": {"n_neighbors": 5, "weights": "distance"},
    "CART": {"max_depth": 6, "min_samples_split": 17, "min_samples_leaf": 2},
    "RF": {"n_estimators": 300, "max_depth": 6, "min_samples_split": 2},
    "GBM": {"n_estimators": 169, "learning_rate": 0.1762506889315551, "max_depth": 4},
    "XGBoost": {"n_estimators": 132, "learning_rate": 0.19566773880062174, "max_depth": 4, "subsample": 0.8029329338557394},
    "LightGBM": {"n_estimators": 50, "learning_rate": 0.2, "num_leaves": 255, "max_depth": 15},
    "CatBoost": {"depth": 4, "learning_rate": 0.08251316857563709, "n_estimators": 199},
}

def base_models():
    return {
        "LR": LinearRegression(), "Ridge": Ridge(), "Lasso": Lasso(), "ElasticNet": ElasticNet(max_iter=10000),
        "KNN": KNeighborsRegressor(), "CART": DecisionTreeRegressor(random_state=42),
        "RF": RandomForestRegressor(random_state=42, n_jobs=2), "GBM": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=2),
        "LightGBM": LGBMRegressor(random_state=42, verbose=-1, n_jobs=2),
        "CatBoost": CatBoostRegressor(verbose=False, random_state=42, thread_count=2),
    }

def make_model(name, tuned=True):
    m = base_models()[name]
    if tuned: m.set_params(**PSO_PARAMS[name])
    return m

def iqr_limits(s):
    q1, q3 = s.quantile(.25), s.quantile(.75); iqr = q3 - q1
    if iqr == 0: return None, None
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

def load(path=DATA, cap_target=True, fit_mask=None):
    """Return df (lower-case columns) after notebook preprocessing.
    fit_mask: boolean index of rows used to estimate capping limits and scaler (None = all rows, as in notebook)."""
    raw = pd.read_excel(path)
    df = raw.copy()
    for c in ["district", "subdistrict"]: df[c] = df[c].astype(str).str.strip()
    year = df["year"].values
    district = df["district"].values
    df = pd.get_dummies(df, columns=["district"], prefix="district", dtype=int)
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() >= 10]
    if not cap_target: num = [c for c in num if c != "Potential_biogas_day"]
    ref = df if fit_mask is None else df[fit_mask]
    for c in num:
        lo, up = iqr_limits(ref[c])
        if lo is None: continue
        df[c] = df[c].astype(float).clip(lo, up)
    df = df.drop(columns=["year"])
    df.columns = df.columns.str.replace(" ", "_").str.replace("[^A-Za-z0-9_]+", "", regex=True).str.lower()
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
    sc = MinMaxScaler().fit(df.loc[fit_mask, feats] if fit_mask is not None else df[feats])
    df[feats] = sc.transform(df[feats])
    X = df[feats].copy(); y = df[TARGET].astype(float).copy()
    return X, y, year, district, raw
