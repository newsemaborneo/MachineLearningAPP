import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mlapp import charts, eda, ml, ui

NO_TARGET = "— Tanpa target —"
SECTIONS = {
    "Ringkasan": ":material/summarize:",
    "Kualitas": ":material/fact_check:",
    "Missing": ":material/grid_off:",
    "Distribusi": ":material/bar_chart:",
    "Outlier": ":material/scatter_plot:",
    "Korelasi": ":material/grid_on:",
    "Target": ":material/target:",
    "Multikolinearitas": ":material/hub:",
    "Dimensi": ":material/blur_on:",
    "Uji statistik": ":material/functions:",
}

# Komputasi berat di-cache per dataset.
_quality_num = st.cache_data(eda.numeric_quality, show_spinner=False)
_quality_cat = st.cache_data(eda.categorical_quality, show_spinner=False)
_insights = st.cache_data(eda.insights, show_spinner="Menganalisis data…")
_assoc = st.cache_data(eda.association_matrix, show_spinner="Menghitung asosiasi…")
_mi = st.cache_data(eda.mutual_information, show_spinner="Menghitung mutual information…")
_vif = st.cache_data(eda.vif, show_spinner=False)
_relations = st.cache_data(eda.deterministic_relations, show_spinner="Mencari relasi antar kolom…")
_pca = st.cache_data(eda.pca_summary, show_spinner=False)
_tsne = st.cache_data(eda.tsne_embedding, show_spinner="Menjalankan t-SNE…")
_normality = st.cache_data(eda.normality, show_spinner=False)


def _is_regression_target(df, target):
    return ml.column_kind(df[target]) == ml.NUMERIC and df[target].nunique() > 10


def _need_numeric(cols, n=1):
    if len(cols) < n:
        st.info(f"Butuh minimal {n} kolom numerik.", icon=":material/info:")
        return False
    return True


# --------------------------------------------------------------------------- #
def _overview(df, numeric, categorical, target):
    ui.metric_row({
        "Baris": len(df), "Kolom": df.shape[1], "Numerik": len(numeric), "Kategorikal": len(categorical),
        "Sel kosong %": round(100 * df.isna().mean().mean(), 1), "Duplikat": int(df.duplicated().sum()),
    }, fmt="{:.1f}")
    st.markdown("##### Temuan otomatis")
    icons = {"warning": ":material/warning:", "info": ":material/lightbulb:", "success": ":material/check_circle:"}
    for level, message in _insights(df, target):
        getattr(st, level)(message, icon=icons[level])
    if numeric:
        st.markdown("##### Statistik numerik")
        st.dataframe(df[numeric].describe().T, width="stretch")


def _quality(df, numeric, categorical):
    if numeric:
        st.markdown("##### Kolom numerik")
        st.dataframe(_quality_num(df, numeric), width="stretch", hide_index=True, column_config={
            "Outlier %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
            "Nol %": st.column_config.NumberColumn(format="%.1f%%"),
            **{c: st.column_config.NumberColumn(format="%.3g") for c in
               ["Mean", "Median", "Std", "Min", "Max", "Skewness", "Kurtosis"]},
        })
        st.caption("Skewness > |1| = sangat miring · Kurtosis > 3 = ekor tebal (banyak nilai ekstrem) · "
                   "Outlier dihitung dengan aturan IQR (di luar Q1 − 1.5·IQR … Q3 + 1.5·IQR).")
    if categorical:
        st.markdown("##### Kolom kategorikal")
        st.dataframe(_quality_cat(df, categorical), width="stretch", hide_index=True, column_config={
            "Frekuensi teratas %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
        st.caption("*Tipe campuran* = sebagian nilai berupa angka dan sebagian teks — biasanya tanda data kotor.")


def _missing(df, target):
    missing = df.isna().mean().mul(100).loc[lambda s: s > 0].sort_values()
    if missing.empty:
        st.success("Tidak ada missing value.", icon=":material/check_circle:")
        return
    left, right = st.columns(2)
    with left:
        fig = px.bar(missing, orientation="h", labels={"value": "Kosong (%)", "index": ""},
                     title="Persentase kosong per kolom")
        fig.update_layout(showlegend=False)
        ui.chart(fig)
    with right:
        sample = df.sample(min(len(df), 300), random_state=0).sort_index()
        mat = sample[missing.index[::-1]].isna().astype(int)
        fig = px.imshow(mat.T, aspect="auto", color_continuous_scale=["#EEF2FF", "#DC2626"],
                        labels={"x": "Baris", "y": "", "color": "Kosong"}, title="Pola missing (merah = kosong)")
        fig.update_coloraxes(showscale=False)
        ui.chart(fig)
    st.caption("Pola yang membentuk garis horizontal serentak berarti beberapa kolom sering kosong bersamaan.")
    if target:
        table = eda.missing_vs_target(df, target, _is_regression_target(df, target))
        if not table.empty:
            st.markdown("##### Apakah kosongnya berkaitan dengan target?")
            st.dataframe(table, width="stretch", hide_index=True)
            st.caption("Selisih besar → data kosong *tidak acak* (MAR/MNAR). Pertimbangkan menambah fitur "
                       "penanda 'kolom X kosong' alih-alih hanya mengimputasi.")


def _distribution(df, numeric, categorical):
    col = st.selectbox("Kolom", numeric + categorical, key="eda_dist_col")
    if col in numeric:
        s = df[col].dropna().astype(float)
        log = st.toggle("Tampilkan log(1 + x)", key="eda_dist_log", disabled=bool((s < 0).any()),
                        help="Mengecilkan ekor panjang agar pola terlihat. Hanya untuk nilai ≥ 0.")
        values = np.log1p(s) if log else s
        label = f"log(1 + {col})" if log else col
        left, right = st.columns([3, 2])
        with left:
            ui.chart(px.histogram(x=values, nbins=40, marginal="box", labels={"x": label},
                                  title=f"Distribusi {label}"))
        with right:
            theo, ordered, slope, intercept = eda.qq_points(pd.Series(values))
            fig = px.scatter(x=theo, y=ordered, labels={"x": "Kuantil normal teoretis", "y": "Kuantil data"},
                             title="Q–Q plot terhadap distribusi normal")
            fig.add_trace(go.Scatter(x=theo, y=slope * theo + intercept, mode="lines", name="Normal",
                                     line=dict(color="#0F172A", dash="dash")))
            fig.update_layout(showlegend=False)
            ui.chart(fig)
        norm = eda.normality(pd.DataFrame({label: values}), [label])
        stats_row = {"Skewness": float(pd.Series(values).skew()), "Kurtosis": float(pd.Series(values).kurt())}
        if not norm.empty:
            stats_row["Shapiro p-value"] = float(norm["p-value"].iloc[0])
        ui.metric_row(stats_row, fmt="{:.3f}")
        st.caption("Titik Q–Q yang menjauh dari garis = distribusi tidak normal (ekor tebal atau miring).")
    else:
        counts = df[col].astype(str).value_counts()
        left, right = st.columns([3, 2])
        with left:
            top = counts.head(25)
            fig = px.bar(x=top.values, y=top.index, orientation="h", labels={"x": "Jumlah", "y": ""},
                         title=f"Frekuensi {col} (25 teratas)")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=max(380, 22 * len(top)))
            ui.chart(fig)
        with right:
            share = (counts / counts.sum() * 100).reset_index()
            share.columns = [col, "Persen"]
            share["Kumulatif %"] = share["Persen"].cumsum()
            st.dataframe(share, width="stretch", hide_index=True, height=380,
                         column_config={"Persen": st.column_config.NumberColumn(format="%.1f%%"),
                                        "Kumulatif %": st.column_config.NumberColumn(format="%.1f%%")})


def _outliers(df, numeric):
    if not _need_numeric(numeric):
        return
    z = (df[numeric] - df[numeric].mean()) / df[numeric].std().replace(0, np.nan)
    long = z.melt(var_name="Kolom", value_name="z-score").dropna()
    fig = px.box(long, x="Kolom", y="z-score", points="outliers", title="Semua kolom numerik (distandarkan)")
    fig.add_hrect(y0=-3, y1=3, fillcolor="#EEF2FF", opacity=0.4, line_width=0)
    ui.chart(fig)
    st.caption("Area biru = ±3 standar deviasi. Titik di luar kotak adalah outlier menurut aturan IQR.")
    col = st.selectbox("Lihat baris outlier untuk kolom", numeric, key="eda_outlier_col")
    lo, hi = eda.iqr_bounds(df[col].dropna())
    rows = df[eda.outlier_mask(df[col])]
    st.markdown(f"Batas normal **{col}**: {lo:.4g} … {hi:.4g} · **{len(rows)}** baris di luar batas.")
    if not rows.empty:
        st.dataframe(rows, width="stretch")


def _correlation(df, numeric):
    method = st.segmented_control("Metode", ["Pearson", "Spearman", "Kendall", "Asosiasi campuran"],
                                  default="Pearson", key="eda_corr_method") or "Pearson"
    if method == "Asosiasi campuran":
        cols = [c for c in df.columns if df[c].nunique() > 1][:30]
        mat = _assoc(df, cols)
        fig = px.imshow(mat, text_auto=".2f", zmin=0, zmax=1, aspect="auto",
                        color_continuous_scale=ui.SEQUENTIAL_SCALE, title="Kekuatan asosiasi (0–1) semua tipe kolom")
        st.caption("|r| untuk numerik–numerik, Cramér's V untuk kategorikal–kategorikal, η untuk "
                   "kategorikal–numerik. Kolom mirip ID dikosongkan karena selalu tampak berasosiasi sempurna.")
    else:
        if not _need_numeric(numeric, 2):
            return
        mat = df[numeric].corr(method=method.lower())
        fig = px.imshow(mat, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                        color_continuous_scale=ui.DIVERGING_SCALE, title=f"Korelasi {method}")
        st.caption({"Pearson": "Pearson mengukur hubungan **linear**.",
                    "Spearman": "Spearman mengukur hubungan **monoton** (berbasis peringkat), tahan outlier.",
                    "Kendall": "Kendall: berbasis peringkat, lebih konservatif untuk data kecil."}[method])
    fig.update_layout(height=max(420, 34 * len(mat)))
    ui.chart(fig)
    st.markdown("##### Pasangan terkuat")
    st.dataframe(eda.top_pairs(mat), width="stretch", hide_index=True,
                 column_config={"Nilai": st.column_config.NumberColumn(format="%.3f")})


def _target(df, target):
    if not target:
        st.info("Pilih **kolom target** di atas untuk menganalisis hubungan fitur dengan target.",
                icon=":material/target:")
        return
    regression = _is_regression_target(df, target)
    y = df[target]
    left, right = st.columns(2)
    with left:
        if regression:
            ui.chart(px.histogram(df, x=target, nbins=40, marginal="box", title=f"Distribusi target {target}"))
        else:
            counts = y.astype(str).value_counts()
            ui.chart(px.bar(x=counts.index, y=counts.values, labels={"x": target, "y": "Jumlah"},
                            title=f"Distribusi kelas {target}", text_auto=True))
            ratio = counts.iloc[0] / counts.iloc[-1] if len(counts) > 1 else 1
            st.caption(f"Rasio kelas terbesar : terkecil = **{ratio:.1f} : 1**" +
                       (" — tidak seimbang, gunakan *class weight*." if ratio > 3 else " — cukup seimbang."))
    with right:
        mi = _mi(df, target, regression)
        fig = px.bar(mi.head(20).iloc[::-1], x="Mutual information", y="Fitur", orientation="h",
                     title="Fitur paling informatif (mutual information)")
        fig.update_layout(height=max(380, 24 * min(len(mi), 20)))
        ui.chart(fig)
        st.caption("Mutual information menangkap hubungan **non-linear** juga. 0 = tidak memberi informasi.")

    st.markdown("##### Fitur vs target")
    feature = st.selectbox("Fitur", [c for c in df.columns if c != target], key="eda_target_feature")
    is_num = ml.column_kind(df[feature]) == ml.NUMERIC and df[feature].nunique() > 10
    if regression and is_num:
        fig = px.scatter(df, x=feature, y=target, opacity=0.7, title=f"{target} vs {feature}")
        charts.add_trendline(fig, df, feature, target)
    elif regression:
        fig = px.box(df, x=df[feature].astype(str), y=target, points="all",
                     labels={"x": feature}, title=f"{target} per {feature}")
    elif is_num:
        fig = px.violin(df, x=y.astype(str), y=feature, box=True, points="all",
                        labels={"x": target}, title=f"{feature} per kelas {target}")
    else:
        ct = pd.crosstab(df[feature].astype(str), y.astype(str), normalize="index").mul(100)
        long = ct.reset_index().melt(id_vars=feature, var_name=target, value_name="Persen")
        fig = px.bar(long, x=feature, y="Persen", color=target, barmode="stack",
                     title=f"Komposisi {target} per {feature} (%)")
    ui.chart(fig)


def _multicollinearity(df, numeric):
    if not _need_numeric(numeric, 2):
        return
    table = _vif(df, numeric)
    left, right = st.columns([3, 2])
    with left:
        fig = px.bar(table.iloc[::-1], x="R² dari kolom lain", y="Kolom", orientation="h", color="Tingkat",
                     color_discrete_map={"Sempurna (redundan)": "#DC2626", "Tinggi": "#F59E0B",
                                         "Sedang": "#818CF8", "Rendah": "#0D9488"},
                     title="Seberapa bisa ditebak dari kolom lain", range_x=[0, 1.02])
        ui.chart(fig)
    with right:
        st.dataframe(table.assign(VIF=table["VIF"].map(lambda v: "∞" if np.isinf(v) else f"{v:.2f}")),
                     width="stretch", hide_index=True,
                     column_config={"R² dari kolom lain": st.column_config.NumberColumn(format="%.4f")})
    st.caption("VIF > 10 = multikolinearitas tinggi; koefisien model linear menjadi tidak stabil.")
    st.markdown("##### Relasi pasti antar kolom")
    relations = _relations(df, numeric)
    if relations:
        st.warning("Kolom berikut bisa dihitung **tepat** dari kolom lain. Jangan memakai keduanya sebagai "
                   "fitur, dan jangan memakai kolom penyusun target sebagai fitur (kebocoran data).",
                   icon=":material/warning:")
        st.dataframe(pd.DataFrame(relations), width="stretch", hide_index=True)
    else:
        st.success("Tidak ada kolom yang merupakan kombinasi linear pasti dari kolom lain.",
                   icon=":material/check_circle:")


def _dimension(df, numeric, target):
    cols = [c for c in numeric if df[c].nunique() > 1 and c != target]
    if not _need_numeric(cols, 2):
        return
    var, coords, loadings = _pca(df, cols)
    if target:
        color = df[target] if _is_regression_target(df, target) else df[target].astype(str)
    else:
        color = None
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_bar(x=var["Komponen"], y=var["Varians %"], name="Per komponen", marker_color=ui.PRIMARY)
        fig.add_scatter(x=var["Komponen"], y=var["Kumulatif %"], name="Kumulatif", mode="lines+markers",
                        line=dict(color=ui.DONE))
        fig.update_layout(title="Varians yang dijelaskan PCA (scree plot)", yaxis_title="%")
        ui.chart(fig)
        n90 = min(int((var["Kumulatif %"] < 90).sum()) + 1, len(var))
        st.caption(f"**{n90}** dari {len(cols)} komponen sudah menjelaskan ≥ 90% variasi data.")
    with right:
        ui.chart(px.scatter(x=coords[:, 0], y=coords[:, 1], color=color, opacity=0.75,
                            labels={"x": "PC1", "y": "PC2", "color": target or ""}, title="Proyeksi PCA 2D"))
    with st.expander("Loadings (kontribusi kolom ke PC1 & PC2)"):
        st.dataframe(loadings.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1).format("{:.3f}"),
                     width="stretch")
    if st.toggle("Tampilkan t-SNE (non-linear, lebih lambat)", key="eda_tsne"):
        emb = _tsne(df, cols)
        ui.chart(px.scatter(x=emb[:, 0], y=emb[:, 1], color=color, opacity=0.75,
                            labels={"x": "t-SNE 1", "y": "t-SNE 2", "color": target or ""}, title="Proyeksi t-SNE"))
        st.caption("t-SNE menjaga kedekatan lokal: kelompok yang terpisah jelas menandakan struktur cluster.")


def _tests(df, numeric):
    groupable = [c for c in df.columns if 2 <= df[c].nunique() <= 20]
    mode = st.segmented_control("Uji", ["Perbandingan kelompok", "Chi-square", "Normalitas"],
                                default="Perbandingan kelompok", key="eda_test_mode") or "Perbandingan kelompok"
    if mode == "Perbandingan kelompok":
        if not numeric or not groupable:
            st.info("Butuh satu kolom numerik dan satu kolom pengelompok (2–20 nilai unik).", icon=":material/info:")
            return
        c1, c2 = st.columns(2)
        num = c1.selectbox("Variabel numerik", numeric, key="eda_test_num")
        cat = c2.selectbox("Dikelompokkan menurut", [g for g in groupable if g != num], key="eda_test_cat")
        if cat is None:
            return
        result = eda.group_test(df, num, cat)
        if "error" in result:
            st.warning(result["error"], icon=":material/warning:")
            return
        left, right = st.columns([2, 3])
        with left:
            st.dataframe(result["summary"], width="stretch")
        with right:
            ui.chart(px.box(df, x=df[cat].astype(str), y=num, points="all", labels={"x": cat},
                            title=f"{num} per {cat}"))
        st.dataframe(result["tests"], width="stretch", hide_index=True,
                     column_config={"p-value": st.column_config.NumberColumn(format="%.4g"),
                                    "Statistik": st.column_config.NumberColumn(format="%.4g"),
                                    "Nilai effect": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Uji parametrik mengasumsikan data mendekati normal; versi non-parametrik "
                   "(Mann–Whitney / Kruskal–Wallis) lebih aman untuk data kecil atau miring. "
                   "Effect size: Cohen's d ≈ 0.2 kecil, 0.5 sedang, 0.8 besar · η² ≈ 0.01 kecil, 0.06 sedang, 0.14 besar.")
    elif mode == "Chi-square":
        if len(groupable) < 2:
            st.info("Butuh dua kolom kategorikal (2–20 nilai unik).", icon=":material/info:")
            return
        c1, c2 = st.columns(2)
        a = c1.selectbox("Kolom A", groupable, key="eda_chi_a")
        b = c2.selectbox("Kolom B", [g for g in groupable if g != a], key="eda_chi_b")
        result = eda.chi_square(df, a, b)
        if "error" in result:
            st.warning(result["error"], icon=":material/warning:")
            return
        ui.metric_row({"χ²": result["chi2"], "p-value": result["p"], "Derajat bebas": int(result["dof"]),
                       "Cramér's V": result["cramers_v"]}, fmt="{:.4g}")
        st.markdown(f"Hubungan **{a}** dan **{b}**: {result['conclusion']}.")
        if result["low_expected"] > 0.2:
            st.warning(f"{result['low_expected']:.0%} sel memiliki frekuensi harapan < 5; hasil chi-square "
                       "kurang andal. Gabungkan kategori kecil atau tambah data.", icon=":material/warning:")
        ui.chart(px.imshow(result["table"], text_auto=True, aspect="auto",
                           color_continuous_scale=ui.SEQUENTIAL_SCALE, title="Tabel kontingensi"))
    else:
        if not _need_numeric(numeric):
            return
        st.dataframe(_normality(df, numeric), width="stretch", hide_index=True,
                     column_config={"p-value": st.column_config.NumberColumn(format="%.4g")})
        st.caption("Shapiro–Wilk: p < 0.05 berarti distribusi berbeda nyata dari normal. Untuk data besar, "
                   "penyimpangan kecil pun menjadi signifikan — lihat juga Q–Q plot di bagian Distribusi.")


def render():
    ui.step_header("explore", "Analisis data eksploratif lanjutan: kualitas data, outlier, korelasi, "
                              "hubungan dengan target, multikolinearitas, reduksi dimensi, dan uji statistik.")
    ui.require(st.session_state.df is not None, "Muat dataset terlebih dahulu.", "data")

    df = st.session_state.df
    numeric, categorical = eda.split_columns(df)
    config = st.session_state.config or {}

    with st.container(border=True, key="card-explore-controls"):
        options = [NO_TARGET] + df.columns.tolist()
        target = st.selectbox("Kolom target (opsional)", options,
                              index=options.index(config["target"]) if config.get("target") in options else 0,
                              key="eda_target", help="Mengaktifkan analisis target dan pewarnaan grafik.")
        target = None if target == NO_TARGET else target
        section = st.segmented_control("Analisis", list(SECTIONS), default="Ringkasan", key="eda_section",
                                       format_func=lambda s: f"{SECTIONS[s]} {s}") or "Ringkasan"

    with st.container(border=True, key=f"card-eda-{section.lower().replace(' ', '-')}"):
        st.subheader(section, anchor=False)
        {
            "Ringkasan": lambda: _overview(df, numeric, categorical, target),
            "Kualitas": lambda: _quality(df, numeric, categorical),
            "Missing": lambda: _missing(df, target),
            "Distribusi": lambda: _distribution(df, numeric, categorical),
            "Outlier": lambda: _outliers(df, numeric),
            "Korelasi": lambda: _correlation(df, numeric),
            "Target": lambda: _target(df, target),
            "Multikolinearitas": lambda: _multicollinearity(df, numeric),
            "Dimensi": lambda: _dimension(df, numeric, target),
            "Uji statistik": lambda: _tests(df, numeric),
        }[section]()

    ui.next_step_button("visualize")
