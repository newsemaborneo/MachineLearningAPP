"""Statistik untuk analisis data eksploratif lanjutan (tanpa Streamlit)."""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from mlapp.ml import CATEGORICAL, NUMERIC, column_kind


def split_columns(df: pd.DataFrame):
    kinds = {c: column_kind(df[c]) for c in df.columns}
    numeric = [c for c, k in kinds.items() if k == NUMERIC]
    categorical = [c for c, k in kinds.items() if k == CATEGORICAL]
    return numeric, categorical


# --------------------------------------------------------------------------- #
# Kualitas data
# --------------------------------------------------------------------------- #
def iqr_bounds(s: pd.Series):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


def outlier_mask(s: pd.Series) -> pd.Series:
    lo, hi = iqr_bounds(s)
    return (s < lo) | (s > hi)


def numeric_quality(df: pd.DataFrame, cols) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna().astype(float)
        if s.empty:
            continue
        skew = float(stats.skew(s)) if len(s) > 2 else np.nan
        rows.append({
            "Kolom": c,
            "Mean": s.mean(), "Median": s.median(), "Std": s.std(),
            "Min": s.min(), "Max": s.max(),
            "Skewness": skew,
            "Kurtosis": float(stats.kurtosis(s)) if len(s) > 3 else np.nan,
            "Outlier (IQR)": int(outlier_mask(s).sum()),
            "Outlier %": 100 * outlier_mask(s).mean(),
            "Nol %": 100 * (s == 0).mean(),
            "Bentuk": ("Sangat miring" if abs(skew) > 1 else "Agak miring" if abs(skew) > 0.5
                       else "Simetris") if not np.isnan(skew) else "—",
        })
    return pd.DataFrame(rows)


def categorical_quality(df: pd.DataFrame, cols) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        freq = s.value_counts(normalize=True)
        numeric_like = pd.to_numeric(s, errors="coerce").notna().mean()
        rows.append({
            "Kolom": c,
            "Unik": s.nunique(),
            "Nilai teratas": freq.index[0],
            "Frekuensi teratas %": 100 * freq.iloc[0],
            "Kategori langka (<1%)": int((freq < 0.01).sum()),
            "Tipe campuran": "Ya" if 0.05 < numeric_like < 0.95 else "Tidak",
        })
    return pd.DataFrame(rows)


def insights(df: pd.DataFrame, target: str | None = None) -> list[tuple[str, str]]:
    """Temuan otomatis: list (tingkat, pesan), tingkat ∈ {"warning", "info", "success"}."""
    out = []
    numeric, categorical = split_columns(df)
    n = len(df)
    if n < 100:
        out.append(("warning", f"Data hanya {n} baris. Hasil model akan sangat bervariasi; "
                               "gunakan cross-validation dan jangan terlalu percaya satu skor uji."))
    dup = int(df.duplicated().sum())
    if dup:
        out.append(("warning", f"{dup} baris duplikat ditemukan."))
    for c in df.columns:
        miss = df[c].isna().mean()
        if miss > 0.3:
            out.append(("warning", f"**{c}** kosong {miss:.0%}. Pertimbangkan membuang kolom ini."))
        elif miss > 0:
            out.append(("info", f"**{c}** kosong {miss:.1%} — perlu imputasi."))
        if df[c].nunique(dropna=True) <= 1:
            out.append(("warning", f"**{c}** bernilai konstan — tidak informatif."))
    for c in categorical:
        if df[c].nunique() == df[c].notna().sum() and n > 1:
            out.append(("info", f"**{c}** unik di setiap baris — kemungkinan ID, jangan dipakai sebagai fitur."))
    for r in numeric_quality(df, numeric).to_dict("records"):
        if abs(r["Skewness"]) > 1:
            out.append(("info", f"**{r['Kolom']}** sangat miring (skew {r['Skewness']:.2f}). "
                                "Transformasi log bisa membantu model linear."))
        if r["Outlier %"] > 5:
            out.append(("info", f"**{r['Kolom']}** memiliki {r['Outlier %']:.0f}% outlier (IQR)."))
    if len(numeric) >= 2:
        corr = df[numeric].corr().abs()
        pairs = [(a, b, corr.loc[a, b]) for a, b in itertools.combinations(numeric, 2) if corr.loc[a, b] > 0.9]
        for a, b, v in sorted(pairs, key=lambda p: -p[2])[:5]:
            out.append(("warning", f"**{a}** dan **{b}** berkorelasi sangat tinggi (|r| = {v:.2f}) — "
                                   "salah satunya mungkin redundan."))
        for rel in deterministic_relations(df, numeric)[:5]:
            out.append(("warning", f"Relasi pasti terdeteksi: `{rel['Persamaan']}` (R² = {rel['R²']:.4f})."))
    if target and target in df:
        y = df[target]
        if column_kind(y) == NUMERIC and y.nunique() > 10:
            if abs(stats.skew(y.dropna())) > 1:
                out.append(("info", f"Target **{target}** miring — coba opsi *log-transform target* saat Persiapan."))
        else:
            ratio = y.value_counts(normalize=True)
            if len(ratio) > 1 and ratio.iloc[0] / ratio.iloc[-1] > 3:
                out.append(("warning", f"Target **{target}** tidak seimbang (kelas terbesar {ratio.iloc[0]:.0%}, "
                                       f"terkecil {ratio.iloc[-1]:.0%}). Gunakan *class weight* dan metrik F1/balanced accuracy."))
    if not out:
        out.append(("success", "Tidak ada masalah kualitas data yang menonjol."))
    return out


# --------------------------------------------------------------------------- #
# Korelasi & asosiasi
# --------------------------------------------------------------------------- #
def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramér's V dengan koreksi bias (0 = tidak berhubungan, 1 = hubungan sempurna)."""
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    rc, kc = r - (r - 1) ** 2 / max(n - 1, 1), k - (k - 1) ** 2 / max(n - 1, 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float | None:
    """η: seberapa besar variasi nilai numerik dijelaskan oleh kategori (0–1)."""
    data = pd.DataFrame({"c": categories, "v": pd.to_numeric(values, errors="coerce")}).dropna()
    if data.empty or data["v"].var() == 0:
        return None
    grand = data["v"].mean()
    groups = data.groupby("c")["v"]
    between = (groups.count() * (groups.mean() - grand) ** 2).sum()
    total = ((data["v"] - grand) ** 2).sum()
    return float(np.sqrt(between / total))


def association_matrix(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Matriks asosiasi campuran 0–1: |r| (num–num), Cramér's V (kat–kat), η (kat–num)."""
    kinds = {c: column_kind(df[c]) for c in cols}
    # Kategori yang hampir unik per baris (mis. ID) selalu tampak "berasosiasi sempurna".
    id_like = {c for c in cols if kinds[c] != NUMERIC and df[c].nunique() > 0.5 * df[c].notna().sum()}
    mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for a, b in itertools.combinations(cols, 2):
        ka, kb = kinds[a], kinds[b]
        if a in id_like or b in id_like:
            mat.loc[a, b] = mat.loc[b, a] = np.nan
            continue
        if ka == NUMERIC and kb == NUMERIC:
            v = abs(df[a].corr(df[b]))
        elif ka != NUMERIC and kb != NUMERIC:
            v = cramers_v(df[a].astype(str), df[b].astype(str))
        else:
            cat, num = (a, b) if ka != NUMERIC else (b, a)
            v = correlation_ratio(df[cat].astype(str), df[num])
        mat.loc[a, b] = mat.loc[b, a] = v if v is not None else np.nan
    return mat


def top_pairs(corr: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    rows = [{"Kolom A": a, "Kolom B": b, "Nilai": corr.loc[a, b]}
            for a, b in itertools.combinations(corr.columns, 2) if pd.notna(corr.loc[a, b])]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.reindex(out["Nilai"].abs().sort_values(ascending=False).index).head(n).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
def _encode_for_mi(df, cols):
    X = pd.DataFrame(index=df.index)
    discrete = []
    for c in cols:
        if column_kind(df[c]) == NUMERIC:
            X[c] = df[c].astype(float)
            discrete.append(False)
        else:
            X[c] = df[c].astype(str).astype("category").cat.codes
            discrete.append(True)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=cols, index=df.index)
    return X, np.array(discrete)


def mutual_information(df: pd.DataFrame, target: str, regression: bool, random_state=42) -> pd.DataFrame:
    data = df.dropna(subset=[target])
    cols = [c for c in data.columns if c != target and 1 < data[c].nunique(dropna=True)
            and not (column_kind(data[c]) != NUMERIC and data[c].nunique() >= data[c].notna().sum())]
    if not cols:
        return pd.DataFrame(columns=["Fitur", "Mutual information"])
    X, discrete = _encode_for_mi(data, cols)
    if regression:
        mi = mutual_info_regression(X, data[target].astype(float), discrete_features=discrete,
                                    random_state=random_state)
    else:
        mi = mutual_info_classif(X, data[target].astype(str), discrete_features=discrete,
                                 random_state=random_state)
    return (pd.DataFrame({"Fitur": cols, "Mutual information": mi})
            .sort_values("Mutual information", ascending=False).reset_index(drop=True))


def missing_vs_target(df: pd.DataFrame, target: str, regression: bool) -> pd.DataFrame:
    """Bandingkan target pada baris yang kolomnya kosong vs terisi (indikasi MAR/MNAR)."""
    rows = []
    y = df[target]
    for c in df.columns:
        if c == target or not df[c].isna().any():
            continue
        miss = df[c].isna()
        if regression:
            a, b = y[miss].astype(float).mean(), y[~miss].astype(float).mean()
            label = "Rata-rata target"
        else:
            top = y.value_counts().index[0]
            a, b = (y[miss] == top).mean(), (y[~miss] == top).mean()
            label = f"Proporsi '{top}'"
        rows.append({"Kolom": c, "Baris kosong": int(miss.sum()), "Ukuran": label,
                     "Saat kosong": a, "Saat terisi": b, "Selisih": a - b})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Multikolinearitas & relasi deterministik
# --------------------------------------------------------------------------- #
def _clean_numeric(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce").dropna()


def vif(df: pd.DataFrame, cols) -> pd.DataFrame:
    data = _clean_numeric(df, cols)
    data = data.loc[:, data.std() > 0]
    rows = []
    for c in data.columns:
        others = [o for o in data.columns if o != c]
        if not others:
            continue
        r2 = LinearRegression().fit(data[others], data[c]).score(data[others], data[c])
        value = np.inf if r2 >= 0.999999 else 1 / (1 - r2)
        rows.append({"Kolom": c, "R² dari kolom lain": r2, "VIF": value,
                     "Tingkat": "Sempurna (redundan)" if np.isinf(value) else
                                "Tinggi" if value > 10 else "Sedang" if value > 5 else "Rendah"})
    return pd.DataFrame(rows).sort_values("R² dari kolom lain", ascending=False).reset_index(drop=True)


def linear_relation(df: pd.DataFrame, target: str, candidates, tol=0.9999, max_terms=4):
    """Cari kombinasi linear terkecil kolom lain yang menyusun `target` (forward selection)."""
    cols = [c for c in candidates if c != target]
    data = _clean_numeric(df, cols + [target])
    if len(data) < 5 or data[target].std() == 0:
        return None
    y, chosen, best_r2 = data[target], [], -np.inf
    for _ in range(min(max_terms, len(cols))):
        scores = {}
        for c in cols:
            if c in chosen:
                continue
            X = data[chosen + [c]]
            scores[c] = LinearRegression().fit(X, y).score(X, y)
        if not scores:
            break
        c, r2 = max(scores.items(), key=lambda kv: kv[1])
        if r2 <= best_r2 + 1e-9:
            break
        chosen.append(c)
        best_r2 = r2
        if r2 >= tol:
            # Buang suku yang koefisiennya praktis nol (sisa greedy selection).
            model = LinearRegression().fit(data[chosen], y)
            scale = data[chosen].std().to_numpy() * np.abs(model.coef_)
            chosen = [c for c, w in zip(chosen, scale) if w > 1e-6 * y.std()]
            model = LinearRegression().fit(data[chosen], y)
            return {"target": target, "terms": dict(zip(chosen, model.coef_)),
                    "intercept": model.intercept_, "r2": r2, "equation": _equation(target, chosen, model)}
    return None


def _equation(target, cols, model):
    terms = [(coef, c if np.isclose(abs(coef), 1, atol=0.01) else f"{abs(coef):.3g}×{c}")
             for c, coef in zip(cols, model.coef_)]
    if not np.isclose(model.intercept_, 0, atol=1e-6):
        terms.append((model.intercept_, f"{abs(model.intercept_):.3g}"))
    expr = ("−" if terms[0][0] < 0 else "") + terms[0][1]
    for coef, text in terms[1:]:
        expr += f" {'−' if coef < 0 else '+'} {text}"
    return f"{target} ≈ {expr}"


def deterministic_relations(df: pd.DataFrame, cols) -> list[dict]:
    """Kolom numerik yang dapat dihitung tepat dari kolom lain (R² ≥ 0.9999)."""
    found, seen = [], set()
    for c in cols:
        rel = linear_relation(df, c, cols, max_terms=3)
        if rel:
            members = frozenset([c, *rel["terms"]])
            if members in seen:  # relasi yang sama, hanya ditulis ulang untuk kolom lain
                continue
            seen.add(members)
            found.append({"Kolom": c, "Persamaan": rel["equation"], "R²": rel["r2"]})
    return found


# --------------------------------------------------------------------------- #
# Reduksi dimensi
# --------------------------------------------------------------------------- #
def _scaled(df, cols):
    X = SimpleImputer(strategy="median").fit_transform(df[cols].astype(float))
    return StandardScaler().fit_transform(X)


def pca_summary(df: pd.DataFrame, cols, random_state=42):
    X = _scaled(df, cols)
    pca = PCA(random_state=random_state).fit(X)
    var = pd.DataFrame({
        "Komponen": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
        "Varians %": 100 * pca.explained_variance_ratio_,
        "Kumulatif %": 100 * np.cumsum(pca.explained_variance_ratio_),
    })
    loadings = pd.DataFrame(pca.components_[:2].T, index=cols, columns=["PC1", "PC2"])
    return var, pca.transform(X)[:, :2], loadings


def tsne_embedding(df: pd.DataFrame, cols, perplexity=30, random_state=42):
    X = _scaled(df, cols)
    perplexity = max(2, min(perplexity, (len(X) - 1) // 3))
    return TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca").fit_transform(X)


# --------------------------------------------------------------------------- #
# Uji statistik
# --------------------------------------------------------------------------- #
def _p_text(p):
    return "signifikan (p < 0.05)" if p < 0.05 else "tidak signifikan (p ≥ 0.05)"


def group_test(df: pd.DataFrame, num: str, cat: str) -> dict:
    """Apakah rata-rata `num` berbeda antar kelompok `cat`? (t-test/ANOVA + versi non-parametrik)."""
    data = df[[num, cat]].dropna()
    groups = [g[num].astype(float).values for _, g in data.groupby(data[cat].astype(str)) if len(g) >= 2]
    if len(groups) < 2:
        return {"error": "Butuh minimal 2 kelompok dengan ≥ 2 data."}
    summary = (data.groupby(data[cat].astype(str))[num].agg(["count", "mean", "std", "median"])
               .rename(columns={"count": "n", "mean": "Mean", "std": "Std", "median": "Median"}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(groups) == 2:
            a, b = groups
            t, p = stats.ttest_ind(a, b, equal_var=False)
            pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
            d = (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan
            u, p_np = stats.mannwhitneyu(a, b)
            rows = [("Welch t-test", t, p, "Cohen's d", d), ("Mann–Whitney U", u, p_np, "—", np.nan)]
        else:
            f, p = stats.f_oneway(*groups)
            allv = np.concatenate(groups)
            ss_between = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
            eta2 = ss_between / ((allv - allv.mean()) ** 2).sum()
            h, p_np = stats.kruskal(*groups)
            rows = [("ANOVA satu arah", f, p, "η²", eta2), ("Kruskal–Wallis", h, p_np, "—", np.nan)]
    table = pd.DataFrame(rows, columns=["Uji", "Statistik", "p-value", "Effect size", "Nilai effect"])
    table["Kesimpulan"] = table["p-value"].map(_p_text)
    return {"summary": summary, "tests": table}


def chi_square(df: pd.DataFrame, a: str, b: str) -> dict:
    data = df[[a, b]].dropna().astype(str)
    table = pd.crosstab(data[a], data[b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"error": "Kedua kolom butuh minimal 2 kategori."}
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {"table": table, "chi2": chi2, "p": p, "dof": dof, "cramers_v": cramers_v(data[a], data[b]),
            "low_expected": float((expected < 5).mean()), "conclusion": _p_text(p)}


def normality(df: pd.DataFrame, cols) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna().astype(float)
        if len(s) < 3 or s.std() == 0:
            continue
        stat, p = stats.shapiro(s.sample(min(len(s), 5000), random_state=0))
        rows.append({"Kolom": c, "Shapiro-Wilk W": stat, "p-value": p,
                     "Normal?": "Ya (p ≥ 0.05)" if p >= 0.05 else "Tidak (p < 0.05)"})
    return pd.DataFrame(rows)


def qq_points(s: pd.Series):
    (theoretical, ordered), (slope, intercept, _) = stats.probplot(s.dropna().astype(float), dist="norm")
    return theoretical, ordered, slope, intercept
