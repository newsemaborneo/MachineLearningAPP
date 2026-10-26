"""Logika machine learning murni (tidak bergantung pada Streamlit).

Seluruh langkah (fitur turunan -> preprocessing -> seleksi fitur -> model) dibungkus
dalam satu sklearn ``Pipeline``. Data baru di halaman Prediksi cukup diberikan dalam
bentuk mentah (kolom asli), tanpa perlu meniru encoding atau scaling secara manual.
"""
from __future__ import annotations

import pickle
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error,
                             mean_absolute_percentage_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV, RepeatedKFold,
                                     RepeatedStratifiedKFold, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, RobustScaler, StandardScaler, TargetEncoder)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    import xgboost as xgb
except ImportError:  # XGBoost opsional
    xgb = None

CLASSIFICATION = "Klasifikasi"
REGRESSION = "Regresi"
CLUSTERING = "Clustering"
TASKS = (CLASSIFICATION, REGRESSION, CLUSTERING)

NUMERIC, CATEGORICAL, DATETIME = "Numerik", "Kategorikal", "Tanggal"
BASELINE = "Baseline (Dummy)"

IMPUTE_OPTIONS = {"Mean": "mean", "Median": "median", "Nilai terbanyak": "most_frequent"}
SCALING_OPTIONS = {"Tanpa scaling": None, "Standard (z-score)": StandardScaler,
                   "Min-Max (0–1)": MinMaxScaler, "Robust (tahan outlier)": RobustScaler}
ENCODING_OPTIONS = ("One-Hot", "Ordinal", "Target encoding")
DERIVED_OPS = {"Rasio (A ÷ B)": "/", "Selisih (A − B)": "-", "Jumlah (A + B)": "+", "Perkalian (A × B)": "*"}

# Metrik per tugas: nama tampilan -> (scorer sklearn untuk CV, semakin besar semakin baik?)
METRICS = {
    CLASSIFICATION: {
        "F1 (macro)": ("f1_macro", True),
        "Akurasi": ("accuracy", True),
        "Balanced accuracy": ("balanced_accuracy", True),
        "ROC-AUC": ("roc_auc_ovr", True),
    },
    REGRESSION: {
        "R²": ("r2", True),
        "RMSE": ("neg_root_mean_squared_error", False),
        "MAE": ("neg_mean_absolute_error", False),
        "MAPE": ("neg_mean_absolute_percentage_error", False),
    },
    CLUSTERING: {"Silhouette": (None, True)},
}


def higher_is_better(task: str, metric: str) -> bool:
    return METRICS[task][metric][1]


# --------------------------------------------------------------------------- #
# Algoritma & ruang hyperparameter (untuk RandomizedSearchCV)
# --------------------------------------------------------------------------- #
CLASSIFIERS = {
    "Logistic Regression": lambda rs: LogisticRegression(max_iter=3000, random_state=rs),
    "Decision Tree": lambda rs: DecisionTreeClassifier(random_state=rs),
    "Random Forest": lambda rs: RandomForestClassifier(random_state=rs),
    "Gradient Boosting": lambda rs: GradientBoostingClassifier(random_state=rs),
    "HistGradientBoosting": lambda rs: HistGradientBoostingClassifier(random_state=rs),
    "SVM": lambda rs: SVC(probability=True, random_state=rs),
    "KNN": lambda rs: KNeighborsClassifier(),
    "Naive Bayes": lambda rs: GaussianNB(),
}
REGRESSORS = {
    "Linear Regression": lambda rs: LinearRegression(),
    "Ridge": lambda rs: Ridge(random_state=rs),
    "Lasso": lambda rs: Lasso(random_state=rs, max_iter=10000),
    "Decision Tree": lambda rs: DecisionTreeRegressor(random_state=rs),
    "Random Forest": lambda rs: RandomForestRegressor(random_state=rs),
    "Gradient Boosting": lambda rs: GradientBoostingRegressor(random_state=rs),
    "HistGradientBoosting": lambda rs: HistGradientBoostingRegressor(random_state=rs),
    "SVR": lambda rs: SVR(),
    "KNN": lambda rs: KNeighborsRegressor(),
}
if xgb is not None:
    CLASSIFIERS["XGBoost"] = lambda rs: xgb.XGBClassifier(random_state=rs, n_jobs=1, verbosity=0)
    REGRESSORS["XGBoost"] = lambda rs: xgb.XGBRegressor(random_state=rs, n_jobs=1, verbosity=0)

CLUSTERERS = {
    "K-Means": lambda rs, p: KMeans(n_clusters=p["n_clusters"], random_state=rs, n_init="auto"),
    "Agglomerative": lambda rs, p: AgglomerativeClustering(n_clusters=p["n_clusters"]),
    "DBSCAN": lambda rs, p: DBSCAN(eps=p["eps"], min_samples=p["min_samples"]),
}
ALGORITHMS = {CLASSIFICATION: CLASSIFIERS, REGRESSION: REGRESSORS, CLUSTERING: CLUSTERERS}
DEFAULT_ALGORITHMS = {
    CLASSIFICATION: ["Logistic Regression", "Random Forest", "HistGradientBoosting"],
    REGRESSION: ["Ridge", "Random Forest", "HistGradientBoosting"],
    CLUSTERING: ["K-Means"],
}

_C = list(np.logspace(-3, 3, 13))
PARAM_SPACES = {
    "Logistic Regression": {"C": _C},
    "Decision Tree": {"max_depth": [None, 3, 5, 8, 12], "min_samples_leaf": [1, 2, 5, 10]},
    "Random Forest": {"n_estimators": [100, 200, 400], "max_depth": [None, 5, 10, 20],
                      "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.5, 1.0]},
    "Gradient Boosting": {"n_estimators": [100, 200, 300], "learning_rate": [0.03, 0.1, 0.2],
                          "max_depth": [2, 3, 5]},
    "HistGradientBoosting": {"learning_rate": [0.03, 0.1, 0.2], "max_leaf_nodes": [7, 15, 31],
                             "min_samples_leaf": [2, 5, 10, 20], "l2_regularization": [0.0, 0.1, 1.0]},
    "SVM": {"C": list(np.logspace(-2, 2, 9)), "gamma": ["scale", "auto"]},
    "SVR": {"C": list(np.logspace(-2, 2, 9)), "epsilon": [0.01, 0.1, 0.5]},
    "KNN": {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]},
    "Naive Bayes": {"var_smoothing": list(np.logspace(-11, -7, 5))},
    "Ridge": {"alpha": list(np.logspace(-3, 2, 11))},
    "Lasso": {"alpha": list(np.logspace(-3, 2, 11))},
    "Linear Regression": {},
    "XGBoost": {"n_estimators": [100, 300], "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.1, 0.3], "subsample": [0.7, 1.0]},
}


# --------------------------------------------------------------------------- #
# Analisis kolom
# --------------------------------------------------------------------------- #
def _looks_like_date(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(200)
    if sample.empty or not sample.str.contains(r"\d{1,4}[-/.]\d{1,2}").mean() >= 0.9:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.9


def column_kind(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME
    if pd.api.types.is_numeric_dtype(series):
        return NUMERIC
    return DATETIME if _looks_like_date(series) else CATEGORICAL


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        n_unique = s.nunique(dropna=True)
        kind = column_kind(s)
        note = ""
        if n_unique <= 1:
            note = "Konstan — tidak informatif"
        elif kind == CATEGORICAL and n_unique == len(s.dropna()):
            note = "Semua nilai unik — kemungkinan ID"
        elif kind == CATEGORICAL and n_unique > 50:
            note = "Kardinalitas tinggi"
        rows.append({
            "Kolom": col,
            "Jenis": kind,
            "Tipe pandas": str(s.dtype),
            "Unik": n_unique,
            "Missing": int(s.isna().sum()),
            "Missing %": round(100 * s.isna().mean(), 1),
            "Catatan": note,
        })
    return pd.DataFrame(rows)


def suggest_task(series: pd.Series) -> str:
    if column_kind(series) != NUMERIC or pd.api.types.is_bool_dtype(series):
        return CLASSIFICATION
    if pd.api.types.is_integer_dtype(series) and series.nunique() <= 10:
        return CLASSIFICATION
    return REGRESSION


# --------------------------------------------------------------------------- #
# Transformer kustom
# --------------------------------------------------------------------------- #
class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Tambahkan fitur turunan dari dua kolom numerik, mis. Goals For ÷ Games Played."""

    def __init__(self, specs=()):
        self.specs = specs

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        for spec in self.specs:
            a = pd.to_numeric(X[spec["a"]], errors="coerce")
            b = pd.to_numeric(X[spec["b"]], errors="coerce")
            if spec["op"] == "/":
                X[spec["name"]] = a / b.replace(0, np.nan)
            elif spec["op"] == "-":
                X[spec["name"]] = a - b
            elif spec["op"] == "+":
                X[spec["name"]] = a + b
            else:
                X[spec["name"]] = a * b
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(self.feature_names_in_ if input_features is None else input_features)
        return np.asarray(base + [s["name"] for s in self.specs], dtype=object)


class DateParts(BaseEstimator, TransformerMixin):
    """Pecah kolom tanggal menjadi tahun, bulan, hari, dan hari-dalam-minggu."""

    PARTS = ("year", "month", "day", "dayofweek")

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(pd.DataFrame(X).columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = {}
        for col in X.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt = pd.to_datetime(X[col], errors="coerce", format="mixed")
            for part in self.PARTS:
                out[f"{col}_{part}"] = getattr(dt.dt, part)
        return pd.DataFrame(out, index=X.index).astype(float)

    def get_feature_names_out(self, input_features=None):
        cols = self.feature_names_in_ if input_features is None else input_features
        return np.array([f"{c}_{p}" for c in cols for p in self.PARTS], dtype=object)


class AsObject(BaseEstimator, TransformerMixin):
    """Samakan tipe kolom kategorikal (str/category/bool) menjadi object."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(pd.DataFrame(X).columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.astype(object).where(X.notna(), np.nan)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else np.asarray(input_features, dtype=object)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Potong nilai ekstrem ke batas Q1 − 1.5·IQR dan Q3 + 1.5·IQR (dipelajari dari data latih)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        self.lower_, self.upper_ = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _log1p_clipped(X):
    return np.log1p(np.clip(np.asarray(X, dtype=float), 0, None))


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def input_columns(config: dict) -> list[str]:
    """Kolom mentah yang dibutuhkan pipeline (fitur asli + sumber fitur turunan)."""
    derived = {s["name"] for s in config.get("derived", [])}
    cols = [c for c in config["features"] if c not in derived]
    for spec in config.get("derived", []):
        cols += [c for c in (spec["a"], spec["b"]) if c not in cols]
    return cols


def apply_derived(df: pd.DataFrame, specs) -> pd.DataFrame:
    return DerivedFeatures(specs).fit_transform(df) if specs else df


def build_preprocessor(config: dict) -> ColumnTransformer:
    kinds, features = config["kinds"], config["features"]
    log_cols = [c for c in config.get("log_features", []) if c in features]
    numeric = [c for c in features if kinds[c] == NUMERIC and c not in log_cols]
    categorical = [c for c in features if kinds[c] == CATEGORICAL]
    dates = [c for c in features if kinds[c] == DATETIME]
    scaler_cls = SCALING_OPTIONS[config["scaling"]]

    def numeric_steps(impute, log=False):
        steps = [("impute", SimpleImputer(strategy=impute))]
        if log:
            steps.append(("log", FunctionTransformer(_log1p_clipped, feature_names_out="one-to-one")))
        if config.get("clip_outliers"):
            steps.append(("clip", IQRClipper()))
        if scaler_cls is not None:
            steps.append(("scale", scaler_cls()))
        return steps

    rare = config.get("rare_min") or None
    if config["encoding"] == "Target encoding":
        encoder = TargetEncoder(random_state=0)
    elif config["encoding"] == "Ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=rare)
    else:
        encoder = OneHotEncoder(handle_unknown="infrequent_if_exist" if rare else "ignore",
                                min_frequency=rare, sparse_output=False)

    impute = IMPUTE_OPTIONS[config["impute"]]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(numeric_steps(impute)), numeric))
    if log_cols:
        transformers.append(("log", Pipeline(numeric_steps(impute, log=True)), log_cols))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("object", AsObject()),
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", encoder),
        ]), categorical))
    if dates:
        transformers.append(("date", Pipeline([("parts", DateParts()), *numeric_steps("median")]), dates))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(config: dict, estimator) -> Pipeline:
    steps = []
    if config.get("derived"):
        steps.append(("derive", DerivedFeatures(config["derived"])))
    steps.append(("prep", build_preprocessor(config)))
    if config.get("select_k"):
        score_func = f_classif if config["task"] == CLASSIFICATION else f_regression
        steps.append(("select", SelectKBest(score_func, k=config["select_k"])))
    if config["task"] == REGRESSION and config.get("log_target"):
        estimator = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
    steps.append(("model", estimator))
    return Pipeline(steps)


def prepare_xy(df: pd.DataFrame, config: dict):
    """Ambil X (kolom mentah), y (ter-encode untuk klasifikasi), dan daftar nama kelas."""
    data = df.dropna(subset=[config["target"]])
    X = data[input_columns(config)]
    y_raw = data[config["target"]]
    if config["task"] == CLASSIFICATION:
        values = y_raw if pd.api.types.is_numeric_dtype(y_raw) else y_raw.astype(str)
        encoder = LabelEncoder().fit(values)
        return X, encoder.transform(values), encoder.classes_.tolist()
    return X, y_raw.astype(float).to_numpy(), None


def preview_transform(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Hasil preprocessing pada seluruh data, untuk ditampilkan di halaman Persiapan."""
    steps = build_pipeline(config, None)[:-1]
    if config["task"] == CLUSTERING:
        X, y = df[input_columns(config)], None
    else:
        X, y, _ = prepare_xy(df, config)
    out = steps.fit_transform(X, y)
    return pd.DataFrame(out, columns=steps.get_feature_names_out(), index=X.index)


# --------------------------------------------------------------------------- #
# Kebocoran data
# --------------------------------------------------------------------------- #
def leakage_report(df: pd.DataFrame, config: dict) -> dict:
    """Cari fitur yang (hampir) menentukan target secara langsung.

    Mengembalikan tabel per fitur numerik dan, jika ada, persamaan linear yang
    menyusun target dari fitur (tanda kebocoran paling kuat).
    """
    from mlapp import eda

    data = apply_derived(df.dropna(subset=[config["target"]]), config.get("derived"))
    numeric = [c for c in config["features"] if config["kinds"].get(c) == NUMERIC]
    target = data[config["target"]]
    rows = []
    for col in numeric:
        x = pd.to_numeric(data[col], errors="coerce")
        if config["task"] == REGRESSION:
            strength = abs(x.corr(target.astype(float)))
            measure = "|r| Pearson"
        else:
            strength = eda.correlation_ratio(target.astype(str), x)
            measure = "η (correlation ratio)"
        rows.append({"Fitur": col, "Ukuran": measure, "Kekuatan": strength,
                     "Status": "Curiga bocor" if strength is not None and strength >= 0.95 else "OK"})
    table = pd.DataFrame(rows).sort_values("Kekuatan", ascending=False) if rows else pd.DataFrame()

    relation = None
    if config["task"] == REGRESSION and numeric:
        relation = eda.linear_relation(data, config["target"], numeric)
    return {"table": table, "relation": relation}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainOptions:
    algorithms: list
    primary: str
    test_size: float = 0.2
    cv_folds: int | None = 5
    cv_repeats: int = 3
    random_state: int = 42
    tune: bool = False
    n_iter: int = 15
    balanced: bool = False
    baseline: bool = True


@dataclass
class Run:
    algorithm: str
    task: str
    primary: str
    pipeline: Pipeline | None = None
    metrics: dict = field(default_factory=dict)      # skor data uji
    cv: dict = field(default_factory=dict)           # metrik -> (mean, std) dari repeated CV
    train_score: float | None = None
    best_params: dict | None = None
    train_time: float = 0.0
    error: str | None = None
    is_baseline: bool = False
    X_train: pd.DataFrame | None = None
    y_train: np.ndarray | None = None
    # Data uji (supervised) atau seluruh data (clustering), untuk evaluasi.
    X_eval: pd.DataFrame | None = None
    y_eval: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    y_proba: np.ndarray | None = None
    classes: list | None = None
    embedding: np.ndarray | None = None
    cache: dict = field(default_factory=dict)        # hasil analisis mahal (SHAP, importance, …)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def validation(self) -> str:
        return "CV" if self.primary in self.cv else "Uji"

    @property
    def primary_value(self):
        if self.primary in self.cv:
            return self.cv[self.primary][0]
        return self.metrics.get(self.primary)

    @property
    def score(self) -> float:
        """Skor untuk ranking (selalu: semakin besar semakin baik)."""
        value = self.primary_value
        if not self.ok or value is None or np.isnan(value):
            return -np.inf
        return value if higher_is_better(self.task, self.primary) else -value

    @property
    def can_predict(self) -> bool:
        return self.ok and hasattr(self.pipeline, "predict")


def make_cv(task, y, folds, repeats, random_state):
    if task == CLASSIFICATION:
        min_count = int(np.bincount(y).min())
        if min_count >= 2:
            k = min(folds, min_count)
            if repeats > 1:
                return RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=random_state)
            return StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    k = max(2, min(folds, len(y) // 2))
    if repeats > 1:
        return RepeatedKFold(n_splits=k, n_repeats=repeats, random_state=random_state)
    return KFold(n_splits=k, shuffle=True, random_state=random_state)


def _scoring(task, n_classes):
    scoring = {}
    for name, (scorer, _) in METRICS[task].items():
        if scorer == "roc_auc_ovr" and n_classes == 2:
            scorer = "roc_auc"
        scoring[name] = scorer
    return scoring


def test_metrics(task, y_true, y_pred, y_proba=None):
    if task == CLASSIFICATION:
        kw = {"average": "macro", "zero_division": 0}
        out = {
            "Akurasi": accuracy_score(y_true, y_pred),
            "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
            "Presisi (macro)": precision_score(y_true, y_pred, **kw),
            "Recall (macro)": recall_score(y_true, y_pred, **kw),
            "F1 (macro)": f1_score(y_true, y_pred, **kw),
            "ROC-AUC": None,
        }
        if y_proba is not None:
            try:
                out["ROC-AUC"] = (roc_auc_score(y_true, y_proba[:, 1]) if y_proba.shape[1] == 2
                                  else roc_auc_score(y_true, y_proba, multi_class="ovr",
                                                     labels=range(y_proba.shape[1])))
            except ValueError:  # data uji tidak memuat semua kelas
                pass
        return out
    return {
        "R²": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": None if np.any(y_true == 0) else mean_absolute_percentage_error(y_true, y_pred),
    }


def _estimator(task, name, opts):
    if name == BASELINE:
        return DummyClassifier(strategy="prior") if task == CLASSIFICATION else DummyRegressor()
    est = ALGORITHMS[task][name](opts.random_state)
    if opts.balanced and "class_weight" in est.get_params():
        est.set_params(class_weight="balanced")
    return est


def _param_space(config, name):
    prefix = "model__regressor__" if config["task"] == REGRESSION and config.get("log_target") else "model__"
    return {prefix + k: v for k, v in PARAM_SPACES.get(name, {}).items()}


def train_supervised(df, config, opts: TrainOptions, on_progress=None):
    task = config["task"]
    X, y, classes = prepare_xy(df, config)
    stratify = y if task == CLASSIFICATION and np.bincount(y).min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.test_size, random_state=opts.random_state, stratify=stratify)
    scoring = _scoring(task, len(classes) if classes else 0)

    names = ([BASELINE] if opts.baseline else []) + list(opts.algorithms)
    runs = []
    for i, name in enumerate(names):
        if on_progress:
            on_progress(i, len(names), name)
        run = Run(algorithm=name, task=task, primary=opts.primary, classes=classes,
                  is_baseline=name == BASELINE)
        try:
            pipe = build_pipeline(config, _estimator(task, name, opts))
            space = _param_space(config, name)
            start = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if opts.tune and space and not run.is_baseline:
                    search = RandomizedSearchCV(
                        pipe, space, n_iter=opts.n_iter, scoring=scoring[opts.primary],
                        cv=make_cv(task, y_train, opts.cv_folds or 3, 1, opts.random_state),
                        random_state=opts.random_state, n_jobs=-1, error_score=np.nan)
                    search.fit(X_train, y_train)
                    pipe = search.best_estimator_
                    run.best_params = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
                else:
                    pipe.fit(X_train, y_train)
                run.train_time = time.perf_counter() - start

                run.y_pred = pipe.predict(X_test)
                if hasattr(pipe, "predict_proba"):
                    run.y_proba = pipe.predict_proba(X_test)
                run.metrics = test_metrics(task, y_test, run.y_pred, run.y_proba)
                train_proba = pipe.predict_proba(X_train) if run.y_proba is not None else None
                run.train_score = test_metrics(task, y_train, pipe.predict(X_train), train_proba)[opts.primary]

                if opts.cv_folds:
                    cv = make_cv(task, y_train, opts.cv_folds, opts.cv_repeats, opts.random_state)
                    res = cross_validate(clone(pipe), X_train, y_train, cv=cv, scoring=scoring,
                                         error_score=np.nan, n_jobs=-1)
                    for metric, (_, hib) in METRICS[task].items():
                        scores = res[f"test_{metric}"] * (1 if hib else -1)
                        if not np.all(np.isnan(scores)):
                            run.cv[metric] = (float(np.nanmean(scores)), float(np.nanstd(scores)))
            run.pipeline, run.X_eval, run.y_eval = pipe, X_test, y_test
            run.X_train, run.y_train = X_train, y_train
        except Exception as exc:  # satu algoritma gagal tidak menghentikan yang lain
            run.error = str(exc)
        runs.append(run)
    return runs


def train_clustering(df, config, algorithms, params, random_state, on_progress=None):
    X = df[input_columns(config)]
    runs = []
    for i, name in enumerate(algorithms):
        if on_progress:
            on_progress(i, len(algorithms), name)
        run = Run(algorithm=name, task=CLUSTERING, primary="Silhouette")
        try:
            pipe = build_pipeline(config, CLUSTERERS[name](random_state, params))
            start = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = pipe.fit_predict(X)
            run.train_time = time.perf_counter() - start
            X_t = pipe[:-1].transform(X)
            run.metrics = {
                "Jumlah cluster": len(set(labels) - {-1}),
                "Silhouette": (silhouette_score(X_t, labels)
                               if 2 <= len(set(labels)) < len(X) else None),
                "Noise (DBSCAN)": int((labels == -1).sum()),
            }
            run.embedding = _embed_2d(X_t, random_state)
            run.pipeline, run.X_eval, run.y_pred = pipe, X, labels
        except Exception as exc:
            run.error = str(exc)
        runs.append(run)
    return runs


def _embed_2d(X_t, random_state):
    if X_t.shape[1] >= 2:
        return PCA(n_components=2, random_state=random_state).fit_transform(X_t)
    return np.column_stack([X_t[:, 0], np.zeros(len(X_t))])


def cluster_k_sweep(df, config, ks, random_state=42) -> pd.DataFrame:
    """Inertia (elbow) dan silhouette K-Means untuk beberapa nilai k."""
    X_t = build_pipeline(config, None)[:-1].fit_transform(df[input_columns(config)])
    rows = []
    for k in ks:
        if k >= len(X_t):
            break
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit(X_t)
        rows.append({"k": k, "Inertia": km.inertia_, "Silhouette": silhouette_score(X_t, km.labels_)})
    return pd.DataFrame(rows)


def k_distance(df, config, k) -> np.ndarray:
    """Jarak ke tetangga ke-k (diurutkan) untuk memilih eps DBSCAN; cari 'siku'-nya."""
    X_t = build_pipeline(config, None)[:-1].fit_transform(df[input_columns(config)])
    dist, _ = NearestNeighbors(n_neighbors=min(k + 1, len(X_t))).fit(X_t).kneighbors(X_t)
    return np.sort(dist[:, -1])


def leaderboard(runs: list[Run]) -> pd.DataFrame:
    task, primary = runs[0].task, runs[0].primary
    baseline = next((r for r in runs if r.is_baseline and r.ok), None)
    rows = []
    for i, run in enumerate(runs):
        row = {"#": i, "Algoritma": run.algorithm}
        if run.ok:
            if run.cv:
                mean, std = run.cv.get(primary, (None, None))
                row[f"CV {primary}"] = mean
                row["± std"] = std
            row.update({k: v for k, v in run.metrics.items()})
            if task != CLUSTERING:
                row[f"Train {primary}"] = run.train_score
            if baseline is not None and not run.is_baseline:
                row["> Baseline"] = "Ya" if run.score > baseline.score else "Tidak"
            row["Waktu (detik)"] = run.train_time
            if run.best_params:
                row["Parameter terbaik"] = ", ".join(f"{k}={_fmt(v)}" for k, v in run.best_params.items())
            row["Status"] = "OK"
        else:
            row["Status"] = f"Gagal: {run.error}"
        rows.append(row)
    order = sorted(range(len(runs)), key=lambda i: runs[i].score, reverse=True)
    return pd.DataFrame(rows).set_index("#").loc[order]


def _fmt(value):
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def best_run_index(runs: list[Run]) -> int | None:
    candidates = [i for i, r in enumerate(runs) if r.ok and not r.is_baseline]
    if not candidates:
        candidates = [i for i, r in enumerate(runs) if r.ok]
    return max(candidates, key=lambda i: runs[i].score) if candidates else None


def session_summary(runs, config, opts_label: str) -> dict:
    best = runs[best_run_index(runs)] if best_run_index(runs) is not None else None
    return {
        "Waktu": pd.Timestamp.now().strftime("%H:%M:%S"),
        "Tugas": config["task"],
        "Target": config.get("target") or "—",
        "Fitur": len(config["features"]),
        "Pengaturan": opts_label,
        "Model terbaik": best.algorithm if best else "—",
        "Metrik": best.primary if best else "—",
        "Skor": best.primary_value if best else None,
        "Validasi": best.validation if best else "—",
        "Jumlah model": sum(r.ok for r in runs),
    }


# --------------------------------------------------------------------------- #
# Evaluasi & prediksi
# --------------------------------------------------------------------------- #
def feature_importance(run: Run, random_state: int = 42) -> pd.DataFrame:
    """Permutation importance terhadap kolom asli (bukan kolom hasil encoding)."""
    if "importance" not in run.cache:
        scorer = _scoring(run.task, len(run.classes) if run.classes else 0)[run.primary]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = permutation_importance(run.pipeline, run.X_eval, run.y_eval, scoring=scorer,
                                            n_repeats=10, random_state=random_state, n_jobs=-1)
        run.cache["importance"] = (pd.DataFrame({
            "Fitur": run.X_eval.columns,
            "Importance": result.importances_mean,
            "Std": result.importances_std,
        }).sort_values("Importance", ascending=False).reset_index(drop=True))
    return run.cache["importance"]


def predict_frame(run: Run, X: pd.DataFrame) -> pd.DataFrame:
    pred = run.pipeline.predict(X)
    out = pd.DataFrame(index=X.index)
    if run.task == CLASSIFICATION:
        out["Prediksi"] = np.asarray(run.classes, dtype=object)[pred]
        if hasattr(run.pipeline, "predict_proba"):
            proba = run.pipeline.predict_proba(X)
            for j, cls in enumerate(run.classes):
                out[f"Prob. {cls}"] = proba[:, j]
    elif run.task == CLUSTERING:
        out["Cluster"] = pred
    else:
        out["Prediksi"] = pred
    return out


def export_model(run: Run, config: dict, df: pd.DataFrame) -> bytes:
    """Bundel pipeline + metadata. Muat kembali dengan pickle (butuh paket ``mlapp``)."""
    return pickle.dumps({
        "pipeline": run.pipeline,
        "algorithm": run.algorithm,
        "task": run.task,
        "inputs": input_columns(config),
        "features": config["features"],
        "derived": config.get("derived", []),
        "target": config.get("target"),
        "classes": run.classes,
        "metrics_test": run.metrics,
        "metrics_cv": run.cv,
        "best_params": run.best_params,
        "sklearn_version": sklearn.__version__,
        "data_rows": len(df),
        "data_hash": int(pd.util.hash_pandas_object(df, index=False).sum()),
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
    })
