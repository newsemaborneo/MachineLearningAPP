import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mlapp import ml, state, ui


def _config_summary(config):
    target = config["target"] or "—"
    extras = []
    if config.get("derived"):
        extras.append(f"{len(config['derived'])} fitur turunan")
    if config.get("log_features"):
        extras.append(f"log pada {len(config['log_features'])} kolom")
    if config.get("clip_outliers"):
        extras.append("potong outlier")
    if config.get("log_target"):
        extras.append("log target")
    if config.get("select_k"):
        extras.append(f"SelectKBest k={config['select_k']}")
    st.markdown(
        f"**Tugas:** {config['task']} · **Target:** {target} · **Fitur:** {len(config['features'])}  \n"
        f"Imputasi {config['impute']} · {config['scaling']} · Encoding {config['encoding']}"
        + (" · " + " · ".join(extras) if extras else "")
    )
    st.page_link(ui.PAGES["prepare"], label="Ubah konfigurasi", icon=":material/edit:")


def _supervised_settings(task, n_rows):
    algorithms = st.multiselect("Algoritma", list(ml.ALGORITHMS[task]), default=ml.DEFAULT_ALGORITHMS[task])
    c1, c2 = st.columns(2)
    metrics = list(ml.METRICS[task])
    primary = c1.selectbox("Metrik utama (untuk ranking & tuning)", metrics,
                           help="Klasifikasi tidak seimbang: pakai F1 atau balanced accuracy, bukan akurasi.")
    baseline = c2.toggle("Sertakan model baseline (Dummy)", value=True,
                         help="Model yang selalu menebak kelas terbanyak / rata-rata. Model yang tidak "
                              "mengalahkannya tidak berguna.")

    st.markdown("**Validasi**")
    c1, c2, c3, c4 = st.columns(4)
    test_size = c1.slider("Porsi data uji", 0.1, 0.5, 0.2, 0.05, format="%.2f")
    use_cv = c2.toggle("Repeated k-fold CV", value=True,
                       help="Ranking memakai rata-rata CV pada data latih — jauh lebih stabil daripada satu split.")
    folds = c3.number_input("Jumlah fold", 2, 10, 5, disabled=not use_cv)
    repeats = c4.number_input("Pengulangan", 1, 10, 3 if n_rows < 2000 else 1, disabled=not use_cv)
    if n_rows * test_size < 30:
        st.caption(f":material/warning: Data uji hanya ±{int(n_rows * test_size)} baris — skor uji akan "
                   "sangat bervariasi. Andalkan skor CV.")

    st.markdown("**Optimasi**")
    c1, c2, c3, c4 = st.columns(4)
    tune = c1.toggle("Tuning hyperparameter", help="RandomizedSearchCV pada data latih dengan CV internal.")
    n_iter = c2.number_input("Iterasi tuning", 5, 100, 15, disabled=not tune)
    balanced = c3.toggle("Class weight seimbang", disabled=task != ml.CLASSIFICATION,
                         help="Memberi bobot lebih pada kelas minoritas (untuk algoritma yang mendukung).")
    seed = c4.number_input("Random state", 0, 9999, 42)
    opts = ml.TrainOptions(algorithms=algorithms, primary=primary, test_size=test_size,
                           cv_folds=int(folds) if use_cv else None, cv_repeats=int(repeats),
                           random_state=int(seed), tune=tune, n_iter=int(n_iter), balanced=balanced,
                           baseline=baseline)
    label = f"{'CV ' + str(folds) + '×' + str(repeats) if use_cv else 'holdout'}" + (" · tuning" if tune else "")
    return algorithms, opts, label


def _clustering_helper(df, config):
    with st.expander("Bantu pilih jumlah cluster / epsilon", icon=":material/help:"):
        left, right = st.columns(2)
        with left:
            if st.button("Hitung elbow & silhouette (k = 2…10)", key="k_sweep_button"):
                st.session_state.k_sweep = (config, ml.cluster_k_sweep(df, config, range(2, 11)))
            saved = st.session_state.get("k_sweep")
            sweep = saved[1] if saved and saved[0] == config else None
            if sweep is not None and not sweep.empty:
                fig = go.Figure()
                fig.add_scatter(x=sweep["k"], y=sweep["Inertia"], name="Inertia (elbow)", mode="lines+markers")
                fig.add_scatter(x=sweep["k"], y=sweep["Silhouette"], name="Silhouette", yaxis="y2",
                                mode="lines+markers", line=dict(color=ui.DONE))
                fig.update_layout(title="Memilih k", xaxis_title="k",
                                  yaxis=dict(title="Inertia"), yaxis2=dict(title="Silhouette", overlaying="y", side="right"))
                ui.chart(fig)
                best = int(sweep.loc[sweep["Silhouette"].idxmax(), "k"])
                st.caption(f"Silhouette tertinggi pada **k = {best}**. Pada elbow, pilih titik di mana penurunan melandai.")
        with right:
            k = st.number_input("min_samples (untuk k-distance)", 2, 50, 5, key="kdist_k")
            dist = ml.k_distance(df, config, int(k))
            fig = px.line(x=np.arange(len(dist)), y=dist, labels={"x": "Titik (diurutkan)", "y": f"Jarak ke tetangga ke-{k}"},
                          title="Plot k-distance untuk DBSCAN")
            ui.chart(fig)
            st.caption("Pilih **epsilon** di titik 'siku', tempat kurva mulai naik tajam.")


def _clustering_settings(df, config):
    algorithms = st.multiselect("Algoritma", list(ml.CLUSTERERS), default=ml.DEFAULT_ALGORITHMS[ml.CLUSTERING])
    c1, c2, c3, c4 = st.columns(4)
    params = {
        "n_clusters": int(c1.number_input("Jumlah cluster", 2, 20, 3, help="K-Means & Agglomerative")),
        "eps": c2.number_input("Epsilon", 0.05, 20.0, 0.5, 0.05, help="DBSCAN"),
        "min_samples": int(c3.number_input("Min samples", 1, 50, 5, help="DBSCAN")),
    }
    seed = int(c4.number_input("Random state", 0, 9999, 42))
    _clustering_helper(df, config)
    return algorithms, params, seed


def _run(df, config, algorithms, settings, label):
    progress = st.progress(0.0)

    def on_progress(i, total, name):
        progress.progress(i / total, text=f"Melatih {name} ({i + 1}/{total})…")

    if config["task"] == ml.CLUSTERING:
        params, seed = settings
        runs = ml.train_clustering(df, config, algorithms, params, seed, on_progress=on_progress)
    else:
        runs = ml.train_supervised(df, config, settings, on_progress=on_progress)
    progress.empty()
    state.set_runs(runs, ml.best_run_index(runs))
    st.session_state.history.append(ml.session_summary(runs, config, label))


def _comparison_chart(runs):
    ok = [r for r in runs if r.ok and r.primary_value is not None]
    if not ok:
        return
    primary = runs[0].primary
    data = pd.DataFrame({
        "Algoritma": [r.algorithm for r in ok],
        "Skor": [r.primary_value for r in ok],
        "Std": [r.cv.get(primary, (None, 0))[1] for r in ok],
        "Jenis": ["Baseline" if r.is_baseline else "Model" for r in ok],
    })
    data = data.sort_values("Skor", ascending=not ml.higher_is_better(runs[0].task, primary))
    source = "CV (mean ± std)" if ok[0].cv else "data uji"
    fig = px.bar(data, x="Skor", y="Algoritma", orientation="h", error_x="Std", color="Jenis",
                 color_discrete_map={"Model": ui.PRIMARY, "Baseline": "#94A3B8"},
                 title=f"{primary} per algoritma — {source}")
    fig.update_layout(height=max(260, 46 * len(data)), yaxis={"categoryorder": "array",
                                                             "categoryarray": data["Algoritma"].tolist()[::-1]})
    ui.chart(fig)


def _results(runs):
    primary = runs[0].primary
    table = ml.leaderboard(runs)
    numeric_cols = [c for c in table.columns if c not in ("Algoritma", "Status", "> Baseline", "Parameter terbaik")]
    st.dataframe(table, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in numeric_cols})
    hib = "semakin tinggi semakin baik" if ml.higher_is_better(runs[0].task, primary) else "semakin rendah semakin baik"
    basis = "rata-rata CV pada data latih" if runs[0].cv else "data uji"
    st.caption(f"Diurutkan berdasarkan **{primary}** ({basis}; {hib}). Kolom tanpa awalan = skor pada data uji.")
    _comparison_chart(runs)

    ok = [i for i, r in enumerate(runs) if r.ok]
    if not ok:
        st.error("Semua algoritma gagal dilatih. Periksa konfigurasi di langkah Persiapan.", icon=":material/error:")
        return False
    best = ml.best_run_index(runs)
    baseline = next((r for r in runs if r.is_baseline and r.ok), None)
    if baseline is not None and runs[best].score <= baseline.score:
        st.error("Tidak ada model yang mengalahkan baseline. Fitur kemungkinan tidak informatif terhadap target.",
                 icon=":material/report:")
    choice = st.selectbox(
        "Model aktif (dipakai di Evaluasi & Prediksi)", ok,
        index=ok.index(st.session_state.active_run) if st.session_state.active_run in ok else 0,
        format_func=lambda i: runs[i].algorithm + ("  ★ terbaik" if i == best else ""),
    )
    st.session_state.active_run = choice
    return True


def _history():
    history = st.session_state.history
    if len(history) < 1:
        return
    with ui.section("Riwayat eksperimen", "Ringkasan setiap sesi training pada dataset ini, "
                                          "untuk membandingkan efek perubahan konfigurasi."):
        table = pd.DataFrame(history)
        table.index = range(1, len(table) + 1)
        st.dataframe(table, width="stretch", column_config={"Skor": st.column_config.NumberColumn(format="%.4f")})
        if len(table) > 1 and table["Metrik"].nunique() == 1:
            fig = px.line(table.reset_index(), x="index", y="Skor", markers=True, hover_data=["Model terbaik", "Pengaturan"],
                          labels={"index": "Sesi"}, title=f"{table['Metrik'].iloc[0]} terbaik per sesi")
            fig.update_xaxes(dtick=1)
            ui.chart(fig)


def render():
    ui.step_header("train", "Latih beberapa algoritma dengan konfigurasi yang sama, validasi dengan "
                            "cross-validation, dan pilih model terbaik.")
    ui.require(st.session_state.config is not None,
               "Simpan konfigurasi target, fitur, dan preprocessing.", "prepare")

    df, config = st.session_state.df, st.session_state.config

    with ui.section("Konfigurasi data"):
        _config_summary(config)

    with ui.section("Pengaturan training"):
        if config["task"] == ml.CLUSTERING:
            algorithms, params, seed = _clustering_settings(df, config)
            settings, label = (params, seed), f"k={params['n_clusters']}"
        else:
            algorithms, settings, label = _supervised_settings(config["task"], len(df))
        if st.button(f"Latih {len(algorithms)} model", type="primary", icon=":material/play_arrow:",
                     disabled=not algorithms):
            _run(df, config, algorithms, settings, label)
            st.rerun()

    runs = st.session_state.runs
    if not runs:
        st.info("Belum ada hasil. Pilih algoritma lalu klik **Latih**.", icon=":material/info:")
        _history()
        return

    with ui.section("Hasil · perbandingan model"):
        has_model = _results(runs)
    _history()

    if has_model:
        ui.next_step_button("evaluate")
