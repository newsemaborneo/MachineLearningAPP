"""Interpretasi & diagnosis model (tanpa Streamlit).

Semua fungsi menerima ``ml.Run`` dan menyimpan hasil mahal di ``run.cache``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import partial_dependence
from sklearn.metrics import brier_score_loss, precision_recall_curve
from sklearn.model_selection import learning_curve as sk_learning_curve

from mlapp import ml


# --------------------------------------------------------------------------- #
# Learning curve
# --------------------------------------------------------------------------- #
def learning_curve(run: ml.Run, folds=5, random_state=42) -> pd.DataFrame:
    """Skor latih vs validasi untuk ukuran data latih yang bertambah."""
    key = "learning_curve"
    if key not in run.cache:
        scoring = ml._scoring(run.task, len(run.classes) if run.classes else 0)[run.primary]
        cv = ml.make_cv(run.task, run.y_train, folds, 1, random_state)
        hib = ml.higher_is_better(run.task, run.primary)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sizes, train, valid = sk_learning_curve(
                clone(run.pipeline), run.X_train, run.y_train, cv=cv, scoring=scoring,
                train_sizes=np.linspace(0.3, 1.0, 6), n_jobs=-1, error_score=np.nan, shuffle=True,
                random_state=random_state)
        sign = 1 if hib else -1
        run.cache[key] = pd.DataFrame({
            "Jumlah data latih": sizes,
            "Latih": sign * np.nanmean(train, axis=1), "Latih std": np.nanstd(train, axis=1),
            "Validasi": sign * np.nanmean(valid, axis=1), "Validasi std": np.nanstd(valid, axis=1),
        })
    return run.cache[key]


def diagnose_learning_curve(curve: pd.DataFrame, higher_is_better: bool) -> str:
    last = curve.iloc[-1]
    gap = (last["Latih"] - last["Validasi"]) * (1 if higher_is_better else -1)
    trend = (curve["Validasi"].iloc[-1] - curve["Validasi"].iloc[-2]) * (1 if higher_is_better else -1)
    scale = max(abs(last["Latih"]), 1e-9)
    if gap / scale > 0.15:
        msg = "**Varians tinggi (overfitting):** skor latih jauh di atas validasi. "
        msg += ("Kurva validasi masih naik — menambah data kemungkinan membantu."
                if trend > 0 else "Coba model lebih sederhana atau regularisasi lebih kuat.")
        return msg
    if higher_is_better and last["Validasi"] < 0.6 * max(curve["Latih"].max(), 1e-9):
        return "**Bias tinggi (underfitting):** skor latih dan validasi sama-sama rendah. Coba model lebih kompleks atau fitur yang lebih informatif."
    return "**Seimbang:** selisih skor latih dan validasi kecil."


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def _transformed(run: ml.Run, X: pd.DataFrame):
    pre = run.pipeline[:-1]
    Xt = pre.transform(X)
    return np.asarray(Xt, dtype=float), list(pre.get_feature_names_out())


def shap_values(run: ml.Run, class_index: int | None = None, max_samples=200, random_state=42):
    """Nilai SHAP pada data uji, dalam ruang fitur hasil preprocessing.

    Mengembalikan (values[n, f], base_value, data[n, f], feature_names, satuan).
    """
    import shap

    key = ("shap", class_index)
    if key in run.cache:
        return run.cache[key]

    X_eval = run.X_eval.sample(min(len(run.X_eval), max_samples), random_state=random_state)
    X_bg = run.X_train.sample(min(len(run.X_train), 100), random_state=random_state)
    Xt, names = _transformed(run, X_eval)
    Bt, _ = _transformed(run, X_bg)
    model = run.pipeline[-1]
    unit = "prediksi"
    if isinstance(model, TransformedTargetRegressor):
        model, unit = model.regressor_, "log(1 + target)"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            explainer = shap.Explainer(model, Bt, feature_names=names)
            if isinstance(explainer, shap.TreeExplainer):
                exp = explainer(Xt, check_additivity=False)
            else:
                exp = explainer(Xt)
            if run.task == ml.CLASSIFICATION:
                unit = "log-odds" if isinstance(explainer, shap.LinearExplainer) else "probabilitas / margin"
        except Exception:
            if run.task == ml.CLASSIFICATION:
                idx = 1 if class_index is None else class_index
                fn = lambda data: model.predict_proba(data)[:, idx]  # noqa: E731
                unit = "probabilitas"
            else:
                fn = model.predict
            explainer = shap.Explainer(fn, Bt, feature_names=names, algorithm="permutation")
            exp = explainer(Xt[:60], max_evals=max(2 * Xt.shape[1] + 1, 100), silent=True)

    values, base = np.asarray(exp.values), np.asarray(exp.base_values)
    if values.ndim == 3:  # (n, fitur, kelas)
        idx = 1 if class_index is None and values.shape[2] == 2 else (class_index or 0)
        values, base = values[:, :, idx], base[:, idx] if base.ndim == 2 else base
    base_value = float(np.mean(base))
    result = (values, base_value, Xt[:len(values)], names, unit, X_eval.index[:len(values)])
    run.cache[key] = result
    return result


def shap_global(values, names) -> pd.DataFrame:
    return (pd.DataFrame({"Fitur": names, "Mean |SHAP|": np.abs(values).mean(axis=0)})
            .sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True))


def shap_beeswarm_frame(values, data, names, top) -> pd.DataFrame:
    rows = []
    for name in top:
        j = names.index(name)
        col = data[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        norm = (col - lo) / (hi - lo) if hi > lo else np.full_like(col, 0.5)
        rows.append(pd.DataFrame({"Fitur": name, "SHAP": values[:, j], "Nilai fitur (relatif)": norm,
                                  "Nilai fitur": col}))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Partial dependence
# --------------------------------------------------------------------------- #
def pdp(run: ml.Run, feature: str, ice_lines: int = 40, random_state=42):
    """PDP + ICE untuk satu kolom mentah. Mengembalikan (grid, rata-rata[k, g], ice[k, n, g])."""
    key = ("pdp", feature)
    if key not in run.cache:
        X = run.X_train.copy()
        if pd.api.types.is_integer_dtype(X[feature]):  # sklearn menolak PDP untuk kolom integer
            X[feature] = X[feature].astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = partial_dependence(run.pipeline, X, [feature], kind="both", grid_resolution=30)
        ice = res["individual"]
        if ice.shape[1] > ice_lines:
            pick = np.random.default_rng(random_state).choice(ice.shape[1], ice_lines, replace=False)
            ice = ice[:, pick]
        run.cache[key] = (res["grid_values"][0], res["average"], ice)
    return run.cache[key]


# --------------------------------------------------------------------------- #
# Klasifikasi: kalibrasi & threshold
# --------------------------------------------------------------------------- #
def calibration(run: ml.Run, positive: int = 1):
    y = (np.asarray(run.y_eval) == positive).astype(int)
    p = run.y_proba[:, positive]
    bins = int(np.clip(len(y) // 10, 3, 10))
    frac, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"Rata-rata prediksi": mean_pred, "Proporsi aktual": frac}), brier_score_loss(y, p)


def threshold_table(run: ml.Run) -> pd.DataFrame:
    """Presisi, recall, dan F1 kelas positif untuk setiap threshold (klasifikasi biner)."""
    precision, recall, thresholds = precision_recall_curve(run.y_eval, run.y_proba[:, 1])
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall + 1e-12), 0)
    return pd.DataFrame({"Threshold": thresholds, "Presisi": precision[:-1], "Recall": recall[:-1],
                         "F1": f1[:-1]})


# --------------------------------------------------------------------------- #
# Analisis error
# --------------------------------------------------------------------------- #
def error_table(run: ml.Run) -> pd.DataFrame:
    df = run.X_eval.copy()
    if run.task == ml.REGRESSION:
        df["Aktual"] = run.y_eval
        df["Prediksi"] = run.y_pred
        df["Error"] = run.y_pred - run.y_eval
        df["|Error|"] = np.abs(df["Error"])
        return df.sort_values("|Error|", ascending=False)
    classes = np.asarray(run.classes, dtype=object)
    df["Aktual"] = classes[run.y_eval]
    df["Prediksi"] = classes[run.y_pred]
    df["Benar?"] = np.where(run.y_eval == run.y_pred, "Ya", "Tidak")
    if run.y_proba is not None:
        df["Prob. kelas aktual"] = run.y_proba[np.arange(len(run.y_eval)), run.y_eval]
        df["Keyakinan prediksi"] = run.y_proba.max(axis=1)
        return df.sort_values(["Benar?", "Keyakinan prediksi"], ascending=[False, False])
    return df.sort_values("Benar?", ascending=False)


def error_by_segment(run: ml.Run, column: str) -> pd.DataFrame:
    """Kinerja per kelompok nilai satu kolom (kategori, atau kuartil untuk numerik)."""
    seg = run.X_eval[column]
    if ml.column_kind(seg) == ml.NUMERIC and seg.nunique() > 6:
        seg = pd.qcut(seg, q=4, duplicates="drop").astype(str)
    else:
        seg = seg.astype(str)
    frame = pd.DataFrame({"Segmen": seg.values, "y": run.y_eval, "p": run.y_pred})
    if run.task == ml.REGRESSION:
        out = frame.groupby("Segmen").apply(
            lambda g: pd.Series({"n": len(g), "MAE": np.mean(np.abs(g.p - g.y))}), include_groups=False)
    else:
        out = frame.groupby("Segmen").apply(
            lambda g: pd.Series({"n": len(g), "Akurasi": np.mean(g.p == g.y)}), include_groups=False)
    return out.reset_index()
