import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

from mlapp import explain, ml, state, ui

SECTIONS = {
    "Kinerja": ":material/speed:",
    "Diagnostik": ":material/troubleshoot:",
    "Interpretasi": ":material/psychology:",
    "Kalibrasi & threshold": ":material/tune:",
}


def _cv_text(run):
    if run.primary not in run.cv:
        return None
    mean, std = run.cv[run.primary]
    return f"{mean:.4f} ± {std:.4f}"


def _overfit_hint(run):
    train, test = run.train_score, run.metrics.get(run.primary)
    if train is None or test is None:
        return
    hib = ml.higher_is_better(run.task, run.primary)
    gap = (train - test) if hib else (test - train)
    relative = gap / max(abs(train if hib else test), 1e-9)
    if relative > 0.1:
        st.warning(f"{run.primary} data latih ({train:.3f}) jauh lebih baik daripada data uji ({test:.3f}). "
                   "Model kemungkinan **overfitting** — lihat learning curve di bagian Diagnostik.",
                   icon=":material/warning:")
    else:
        st.caption(f":material/check: {run.primary} latih {train:.3f} vs uji {test:.3f} — tidak ada tanda overfitting.")


# --------------------------------------------------------------------------- #
# Kinerja
# --------------------------------------------------------------------------- #
def _classification(run):
    labels = [str(c) for c in run.classes]
    ui.metric_row({k: run.metrics.get(k) for k in ["Akurasi", "Balanced accuracy", "F1 (macro)", "ROC-AUC"]})
    if _cv_text(run):
        st.markdown(f"**CV {run.primary}:** {_cv_text(run)} *(rata-rata ± std pada fold data latih)*")
    _overfit_hint(run)

    left, right = st.columns(2)
    with left:
        normalize = st.toggle("Normalisasi per kelas aktual (%)", key="cm_norm")
        cm = confusion_matrix(run.y_eval, run.y_pred, labels=range(len(labels))).astype(float)
        if normalize:
            cm = 100 * cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig = px.imshow(cm, x=labels, y=labels, text_auto=".1f" if normalize else ".0f",
                        color_continuous_scale=ui.SEQUENTIAL_SCALE,
                        labels={"x": "Prediksi", "y": "Aktual", "color": "%" if normalize else "Jumlah"},
                        title="Confusion matrix")
        ui.chart(fig)
    with right:
        if run.y_proba is not None:
            fig = go.Figure()
            pairs = [(1, labels[1])] if len(labels) == 2 else list(enumerate(labels))
            for idx, name in pairs:
                y_bin = (np.asarray(run.y_eval) == idx).astype(int)
                if y_bin.min() == y_bin.max():
                    continue
                fpr, tpr, _ = roc_curve(y_bin, run.y_proba[:, idx])
                fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc(fpr, tpr):.3f})")
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
            fig.update_layout(title="Kurva ROC" + (" (one-vs-rest)" if len(labels) > 2 else ""),
                              xaxis_title="False positive rate", yaxis_title="True positive rate")
            ui.chart(fig)
        else:
            st.info("Model ini tidak menghasilkan probabilitas, kurva ROC tidak tersedia.", icon=":material/info:")
    report = classification_report(run.y_eval, run.y_pred, labels=range(len(labels)),
                                   target_names=labels, output_dict=True, zero_division=0)
    with st.expander("Laporan per kelas (presisi, recall, F1)"):
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")


def _regression(run):
    ui.metric_row({k: run.metrics.get(k) for k in ["R²", "RMSE", "MAE", "MAPE"]})
    if _cv_text(run):
        st.markdown(f"**CV {run.primary}:** {_cv_text(run)} *(rata-rata ± std pada fold data latih)*")
    _overfit_hint(run)
    residuals = run.y_eval - run.y_pred
    c1, c2, c3 = st.columns(3)
    with c1:
        lo, hi = float(np.min(run.y_eval)), float(np.max(run.y_eval))
        fig = px.scatter(x=run.y_eval, y=run.y_pred, opacity=0.7, labels={"x": "Aktual", "y": "Prediksi"},
                         title="Aktual vs prediksi")
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal",
                                 line=dict(dash="dash", color="gray")))
        fig.update_layout(showlegend=False)
        ui.chart(fig)
    with c2:
        fig = px.scatter(x=run.y_pred, y=residuals, opacity=0.7, labels={"x": "Prediksi", "y": "Residual"},
                         title="Residual vs prediksi")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_yaxes(tickformat=".3~g")
        ui.chart(fig)
    with c3:
        ui.chart(px.histogram(x=residuals, nbins=30, labels={"x": "Residual"}, title="Distribusi residual"))
    st.caption("Residual yang menyebar acak di sekitar 0 tanpa pola (corong/lengkung) menandakan model sudah "
               "menangkap struktur data. Pola corong = varians tidak konstan → coba log-transform target.")


def _clustering(run):
    ui.metric_row({k: run.metrics.get(k) for k in ["Jumlah cluster", "Silhouette", "Noise (DBSCAN)"]})
    st.caption("Silhouette berkisar −1 sampai 1; semakin tinggi berarti cluster semakin terpisah.")
    labels = pd.Series(run.y_pred).map(lambda c: "Noise" if c == -1 else f"Cluster {c}")
    left, right = st.columns([1, 2])
    with left:
        counts = labels.value_counts().sort_index()
        ui.chart(px.bar(x=counts.index, y=counts.values, labels={"x": "", "y": "Jumlah data"}, title="Ukuran cluster"))
    with right:
        ui.chart(px.scatter(x=run.embedding[:, 0], y=run.embedding[:, 1], color=labels,
                            labels={"x": "Komponen 1", "y": "Komponen 2", "color": ""},
                            title="Proyeksi cluster (PCA 2D)"))
    numeric = run.X_eval.select_dtypes("number")
    if not numeric.empty:
        st.markdown("##### Profil cluster")
        profile = numeric.groupby(labels.values).mean()
        z = (profile - numeric.mean()) / numeric.std().replace(0, np.nan)
        fig = px.imshow(z, text_auto=".2f", aspect="auto", color_continuous_scale=ui.DIVERGING_SCALE,
                        zmin=-2, zmax=2, labels={"color": "z"}, title="Ciri khas tiap cluster (rata-rata distandarkan)")
        ui.chart(fig)
        st.caption("Biru = di atas rata-rata keseluruhan, merah = di bawah. Gunakan untuk menamai cluster.")
        with st.expander("Nilai rata-rata asli"):
            st.dataframe(profile, width="stretch")


# --------------------------------------------------------------------------- #
# Diagnostik
# --------------------------------------------------------------------------- #
def _diagnostics(run):
    st.markdown("##### Learning curve")
    st.caption("Skor latih vs validasi saat data latih ditambah. Membantu memutuskan: butuh data lebih banyak, "
               "model lebih sederhana, atau model lebih kompleks?")
    if "learning_curve" in run.cache or st.button("Hitung learning curve", icon=":material/show_chart:"):
        try:
            with st.spinner("Melatih ulang model pada beberapa ukuran data…"):
                curve = explain.learning_curve(run)
        except Exception as exc:
            st.warning(f"Learning curve tidak bisa dihitung: {exc}", icon=":material/warning:")
        else:
            fig = go.Figure()
            x = curve["Jumlah data latih"].tolist()
            for name, color in (("Latih", ui.PRIMARY), ("Validasi", ui.DONE)):
                upper = (curve[name] + curve[f"{name} std"]).tolist()
                lower = (curve[name] - curve[f"{name} std"]).tolist()
                fig.add_scatter(x=x + x[::-1], y=upper + lower[::-1], fill="toself", fillcolor=color,
                                opacity=0.12, line=dict(width=0), showlegend=False, hoverinfo="skip")
                fig.add_scatter(x=x, y=curve[name], mode="lines+markers", name=name,
                                line=dict(color=color, width=3))
            fig.update_layout(title=f"Learning curve — {run.primary}", xaxis_title="Jumlah data latih",
                              yaxis_title=run.primary)
            ui.chart(fig)
            st.markdown(explain.diagnose_learning_curve(curve, ml.higher_is_better(run.task, run.primary)))

    st.divider()
    st.markdown("##### Analisis error")
    errors = explain.error_table(run)
    if run.task == ml.REGRESSION:
        st.caption("Baris data uji dengan error absolut terbesar.")
        st.dataframe(errors.head(20), width="stretch")
    else:
        wrong = errors[errors["Benar?"] == "Tidak"]
        st.caption(f"{len(wrong)} dari {len(errors)} baris uji salah diprediksi, diurutkan dari yang paling "
                   "*yakin tapi salah* — biasanya petunjuk label keliru atau fitur yang kurang.")
        st.dataframe(wrong.head(20), width="stretch")
    segment_col = st.selectbox("Kinerja per segmen kolom", run.X_eval.columns.tolist(), key="err_segment")
    seg = explain.error_by_segment(run, segment_col)
    metric = "MAE" if run.task == ml.REGRESSION else "Akurasi"
    ui.chart(px.bar(seg, x="Segmen", y=metric, text_auto=".3f", hover_data=["n"],
                    title=f"{metric} per segmen {segment_col}"))


# --------------------------------------------------------------------------- #
# Interpretasi
# --------------------------------------------------------------------------- #
def _permutation(run):
    st.markdown("##### Permutation importance")
    st.caption("Seberapa besar skor model turun jika nilai satu kolom diacak (per kolom asli, pada data uji).")
    if len(run.X_eval) < 30:
        st.caption(f":material/warning: Data uji hanya {len(run.X_eval)} baris — importance bisa tidak stabil.")
    if "importance" not in run.cache and not st.button("Hitung permutation importance", icon=":material/calculate:"):
        return
    with st.spinner("Menghitung…"):
        imp = ml.feature_importance(run).head(15)
    fig = px.bar(imp.iloc[::-1], x="Importance", y="Fitur", error_x="Std", orientation="h",
                 title=f"Penurunan {run.primary} saat kolom diacak")
    fig.update_layout(height=max(320, 28 * len(imp)))
    ui.chart(fig)


def _shap(run):
    st.markdown("##### SHAP")
    st.caption("Kontribusi setiap fitur (setelah preprocessing) terhadap prediksi: global dan per baris.")
    class_index = None
    if run.task == ml.CLASSIFICATION and len(run.classes) > 2:
        class_index = st.selectbox("Jelaskan probabilitas kelas", range(len(run.classes)),
                                   format_func=lambda i: str(run.classes[i]), key="shap_class")
    if ("shap", class_index) not in run.cache and not st.button("Hitung SHAP", icon=":material/psychology:"):
        return
    try:
        with st.spinner("Menghitung nilai SHAP…"):
            values, base, data, names, unit, index = explain.shap_values(run, class_index)
    except Exception as exc:
        st.warning(f"SHAP tidak bisa dihitung untuk model ini: {exc}", icon=":material/warning:")
        return
    glob = explain.shap_global(values, names).head(15)
    order = glob["Fitur"].head(10).tolist()
    left, right = st.columns(2)
    with left:
        fig = px.bar(glob.iloc[::-1], x="Mean |SHAP|", y="Fitur", orientation="h", title="Pengaruh rata-rata")
        fig.update_layout(height=max(380, 26 * len(glob)))
        ui.chart(fig)
    with right:
        bees = explain.shap_beeswarm_frame(values, data, names, order)
        fig = px.scatter(bees, x="SHAP", y="Fitur", color="Nilai fitur (relatif)", hover_data=["Nilai fitur"],
                         color_continuous_scale=["#0EA5E9", "#E2E8F0", "#E11D48"],
                         category_orders={"Fitur": order}, title="Arah pengaruh (beeswarm)")
        fig.update_traces(marker=dict(size=7, opacity=0.7))
        fig.update_layout(height=max(380, 26 * len(glob)), coloraxis_colorbar=dict(title="Nilai"))
        fig.add_vline(x=0, line_color="gray")
        ui.chart(fig)
    st.caption(f"Satuan SHAP: **{unit}**. Beeswarm: setiap titik satu baris. Kanan = mendorong prediksi naik; warna merah = nilai fitur tinggi, "
               "biru = rendah.")

    st.markdown("**Penjelasan satu prediksi**")
    i = st.slider("Baris data uji", 0, len(values) - 1, 0, key="shap_row") if len(values) > 1 else 0
    contrib = pd.Series(values[i], index=names)
    top = contrib.abs().sort_values(ascending=False).index[:12]
    steps = list(contrib[top].items())
    if len(contrib) > len(top):
        steps.append(("Fitur lainnya", contrib.drop(top).sum()))
    fig = go.Figure(go.Waterfall(
        orientation="h", y=[n for n, _ in steps] + ["Prediksi"], x=[v for _, v in steps] + [0],
        measure=["relative"] * len(steps) + ["total"], base=base,
        increasing=dict(marker=dict(color=ui.PRIMARY)), decreasing=dict(marker=dict(color="#DC2626")),
        totals=dict(marker=dict(color=ui.DONE))))
    fig.update_layout(title=f"Dari nilai dasar {base:.3g} ke {base + contrib.sum():.3g} ({unit})",
                      height=max(380, 30 * (len(steps) + 1)), yaxis=dict(autorange="reversed"))
    ui.chart(fig)
    with st.expander("Nilai asli baris ini"):
        st.dataframe(run.X_eval.loc[[index[i]]], width="stretch")


def _pdp(run):
    st.markdown("##### Partial dependence (PDP) & ICE")
    st.caption("Bagaimana prediksi rata-rata berubah saat satu fitur divariasikan, fitur lain tetap.")
    numeric = [c for c in run.X_train.columns if ml.column_kind(run.X_train[c]) == ml.NUMERIC]
    if not numeric:
        st.info("Tidak ada fitur numerik mentah.", icon=":material/info:")
        return
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    feature = c1.selectbox("Fitur", numeric, key="pdp_feature")
    show_ice = c2.toggle("Garis ICE", value=True, help="Satu garis per baris data: memperlihatkan efek yang berbeda-beda.")
    try:
        grid, avg, ice = explain.pdp(run, feature)
    except Exception as exc:
        st.warning(f"PDP tidak bisa dihitung: {exc}", icon=":material/warning:")
        return
    fig = go.Figure()
    for k in range(avg.shape[0]):
        if run.task == ml.CLASSIFICATION:
            name = str(run.classes[k] if avg.shape[0] > 1 else run.classes[1])
        else:
            name = "Prediksi"
        if show_ice:
            for line in ice[k]:
                fig.add_scatter(x=grid, y=line, mode="lines", line=dict(width=1, color="#CBD5E1"),
                                showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=grid, y=avg[k], mode="lines", name=f"PDP {name}", line=dict(width=4))
    fig.update_layout(title=f"Pengaruh {feature}", xaxis_title=feature,
                      yaxis_title="Probabilitas" if run.task == ml.CLASSIFICATION else "Prediksi")
    ui.chart(fig)


# --------------------------------------------------------------------------- #
# Kalibrasi & threshold
# --------------------------------------------------------------------------- #
def _calibration(run):
    if run.task != ml.CLASSIFICATION or run.y_proba is None or len(run.classes) != 2:
        st.info("Bagian ini untuk klasifikasi **biner** dengan model yang menghasilkan probabilitas.",
                icon=":material/info:")
        return
    positive = str(run.classes[1])
    table = explain.threshold_table(run)
    best = table.loc[table["F1"].idxmax()]
    left, right = st.columns(2)
    with left:
        curve, brier = explain.calibration(run)
        fig = px.line(curve, x="Rata-rata prediksi", y="Proporsi aktual", markers=True,
                      title=f"Reliability diagram (Brier = {brier:.3f})")
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
        fig.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1])
        ui.chart(fig)
        st.caption(f"Garis dekat diagonal = probabilitas '{positive}' bisa dipercaya apa adanya. "
                   "Brier score: semakin kecil semakin baik.")
    with right:
        long = table.melt(id_vars="Threshold", var_name="Metrik", value_name="Nilai")
        fig = px.line(long, x="Threshold", y="Nilai", color="Metrik",
                      title=f"Presisi/recall kelas '{positive}' per threshold")
        fig.add_vline(x=best["Threshold"], line_dash="dot", line_color=ui.DONE)
        ui.chart(fig)
    threshold = st.slider("Threshold keputusan", 0.0, 1.0, 0.5, 0.01, key="threshold",
                          help=f"Prediksi '{positive}' jika probabilitasnya ≥ threshold. "
                               f"F1 terbaik pada {best['Threshold']:.2f}.")
    pred = (run.y_proba[:, 1] >= threshold).astype(int)
    m = ml.test_metrics(ml.CLASSIFICATION, run.y_eval, pred)
    ui.metric_row({k: m[k] for k in ["Akurasi", "Presisi (macro)", "Recall (macro)", "F1 (macro)"]})
    labels = [str(c) for c in run.classes]
    ui.chart(px.imshow(confusion_matrix(run.y_eval, pred, labels=[0, 1]), x=labels, y=labels, text_auto=True,
                       color_continuous_scale=ui.SEQUENTIAL_SCALE,
                       labels={"x": "Prediksi", "y": "Aktual", "color": "Jumlah"},
                       title=f"Confusion matrix pada threshold {threshold:.2f}"))
    st.caption("Turunkan threshold bila melewatkan kasus positif mahal (recall penting); naikkan bila alarm palsu mahal.")


def render():
    ui.step_header("evaluate", "Periksa kinerja model aktif pada data yang tidak dilihat saat training, "
                               "diagnosis bias/varians, dan pahami alasan di balik prediksinya.")
    ui.require(state.active_run() is not None, "Latih minimal satu model.", "train")

    runs = st.session_state.runs
    ok = [i for i, r in enumerate(runs) if r.ok]
    with st.container(border=True, key="card-eval-controls"):
        choice = st.selectbox("Model", ok, index=ok.index(st.session_state.active_run),
                              format_func=lambda i: runs[i].algorithm + (" (baseline)" if runs[i].is_baseline else ""))
        st.session_state.active_run = choice
        run = runs[choice]
        sections = ["Kinerja"] if run.task == ml.CLUSTERING else list(SECTIONS)
        section = st.segmented_control("Bagian", sections, default="Kinerja", key=f"eval_section_{run.task}",
                                       format_func=lambda s: f"{SECTIONS[s]} {s}") or "Kinerja"
        if section not in sections:
            section = "Kinerja"

    with st.container(border=True, key=f"card-eval-{section.split()[0].lower()}"):
        st.subheader(section, anchor=False)
        if section == "Kinerja":
            {ml.CLASSIFICATION: _classification, ml.REGRESSION: _regression,
             ml.CLUSTERING: _clustering}[run.task](run)
        elif section == "Diagnostik":
            _diagnostics(run)
        elif section == "Interpretasi":
            _permutation(run)
            st.divider()
            _shap(run)
            st.divider()
            _pdp(run)
        else:
            _calibration(run)

    ui.next_step_button("predict")
