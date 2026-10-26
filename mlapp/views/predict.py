import pandas as pd
import plotly.express as px
import streamlit as st

from mlapp import ml, state, ui


def _input_widget(col, series, kind):
    clean = series.dropna()
    if kind == ml.NUMERIC:
        if pd.api.types.is_integer_dtype(series):
            return st.number_input(col, value=int(clean.median()) if len(clean) else 0, step=1)
        return st.number_input(col, value=float(clean.median()) if len(clean) else 0.0, format="%.4f")
    if kind == ml.DATETIME:
        return st.date_input(col, value=pd.to_datetime(clean, errors="coerce", format="mixed").max())
    options = clean.astype(str).value_counts().index.tolist()[:200]
    return st.selectbox(col, options)


def _show_result(run, result):
    with st.container(key="result"):
        _result_body(run, result)


def _result_body(run, result):
    if run.task == ml.CLUSTERING:
        st.metric("Masuk ke cluster", int(result["Cluster"].iloc[0]), border=True)
        return
    value = result["Prediksi"].iloc[0]
    st.metric("Hasil prediksi", f"{value:,.4f}" if run.task == ml.REGRESSION else str(value), border=True)
    proba_cols = [c for c in result.columns if c.startswith("Prob. ")]
    if proba_cols:
        proba = result[proba_cols].iloc[0].rename(lambda c: c.removeprefix("Prob. "))
        fig = px.bar(x=proba.index, y=proba.values, labels={"x": "Kelas", "y": "Probabilitas"},
                     title="Probabilitas per kelas", range_y=[0, 1])
        ui.chart(fig)


def _manual(run, config, df):
    inputs = ml.input_columns(config)
    kinds = {c: ml.column_kind(df[c]) for c in inputs}
    with st.form("manual_prediction"):
        values = {}
        cols = st.columns(3)
        for i, col in enumerate(inputs):
            with cols[i % 3]:
                values[col] = _input_widget(col, df[col], kinds[col])
        submitted = st.form_submit_button("Prediksi", type="primary", icon=":material/bolt:")
    st.caption("Nilai awal = median (numerik) atau nilai terbanyak (kategorikal) dari data latih.")
    if submitted:
        X = pd.DataFrame([values]).astype({c: df[c].dtype for c in inputs if kinds[c] == ml.NUMERIC})
        _show_result(run, ml.predict_frame(run, X))


def _batch(run, config):
    uploaded = st.file_uploader("CSV dengan kolom fitur yang sama seperti data latih", type=["csv"],
                                key="batch_file")
    st.caption("Kolom yang dibutuhkan: " + ", ".join(f"`{c}`" for c in ml.input_columns(config)))
    if uploaded is None:
        return
    try:
        data = pd.read_csv(uploaded, sep=None, engine="python")
    except Exception as exc:
        st.error(f"File tidak bisa dibaca: {exc}", icon=":material/error:")
        return
    missing = [c for c in ml.input_columns(config) if c not in data.columns]
    if missing:
        st.error("Kolom berikut tidak ada di file: " + ", ".join(f"`{c}`" for c in missing),
                 icon=":material/error:")
        return
    result = pd.concat([data, ml.predict_frame(run, data[ml.input_columns(config)])], axis=1)
    st.success(f"{len(result):,} baris berhasil diprediksi.", icon=":material/check_circle:")
    st.dataframe(result, width="stretch")
    st.download_button("Unduh hasil (CSV)", result.to_csv(index=False).encode(), "prediksi.csv",
                       "text/csv", icon=":material/download:")


def render():
    ui.step_header("predict", "Gunakan model aktif untuk data baru. Isi nilai dalam bentuk asli; "
                              "preprocessing dijalankan otomatis oleh pipeline.")
    ui.require(state.active_run() is not None, "Latih minimal satu model.", "train")

    run, config, df = state.active_run(), st.session_state.config, st.session_state.df

    with ui.section(f"Model aktif · {run.algorithm}"):
        score = run.primary_value
        basis = "CV" if run.validation == "CV" else "data uji"
        st.markdown(f"{run.task} · {run.primary} ({basis}) = **{score:.4f}**" if score is not None else run.task)
        left, right = st.columns(2)
        left.page_link(ui.PAGES["train"], label="Ganti model aktif", icon=":material/swap_horiz:")
        right.download_button(
            "Unduh model (.pkl)", ml.export_model(run, config, df),
            f"model_{run.algorithm.lower().replace(' ', '_')}.pkl", "application/octet-stream",
            icon=":material/download:", width="stretch",
            help="Berisi pipeline lengkap (preprocessing + model) dan metadata.",
        )

    if not run.can_predict:
        st.info(f"{run.algorithm} tidak bisa menempatkan data baru ke cluster. "
                "Gunakan K-Means untuk prediksi.", icon=":material/info:")
        return

    manual, batch = st.tabs(["Input manual", "Batch dari CSV"])
    with manual:
        _manual(run, config, df)
    with batch:
        _batch(run, config)
