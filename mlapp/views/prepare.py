import re

import pandas as pd
import streamlit as st

from mlapp import ml, state, ui

NO_TARGET = "— Tanpa target (clustering) —"


@st.cache_data(show_spinner="Menerapkan preprocessing…")
def _preview(df, config):
    return ml.preview_transform(df, config)


@st.cache_data(show_spinner="Memeriksa kebocoran data…")
def _leakage(df, config):
    return ml.leakage_report(df, config)


def _index(options, value, fallback=0):
    return options.index(value) if value in options else fallback


def _features_key(target):
    return f"prep_features::{target}"


def _choose_target(df, saved):
    columns = df.columns.tolist()
    options = columns + [NO_TARGET]
    saved_target = (saved.get("target") or NO_TARGET) if saved else columns[-1]
    target = st.selectbox("Kolom target (yang ingin diprediksi)", options,
                          index=_index(options, saved_target, len(columns) - 1))
    if target == NO_TARGET:
        st.caption("Tanpa target, model akan mengelompokkan data (clustering).")
        return None, ml.CLUSTERING

    suggested = ml.suggest_task(df[target])
    numeric_target = ml.column_kind(df[target]) == ml.NUMERIC
    tasks = [ml.CLASSIFICATION, ml.REGRESSION] if numeric_target else [ml.CLASSIFICATION]
    default = saved["task"] if saved and saved.get("target") == target else suggested
    task = st.segmented_control("Jenis tugas", tasks, default=default if default in tasks else suggested) or suggested
    n_unique = df[target].nunique()
    st.caption(f"Target **{target}** punya {n_unique} nilai unik → disarankan **{suggested}**."
               + ("" if numeric_target else " Regresi tidak tersedia karena target bukan angka."))
    if task == ml.CLASSIFICATION and n_unique > 30:
        st.warning(f"{n_unique} kelas untuk klasifikasi cukup banyak. Apakah ini seharusnya regresi?",
                   icon=":material/warning:")
    if df[target].isna().any():
        st.caption(f":material/info: {int(df[target].isna().sum())} baris dengan target kosong "
                   "akan diabaikan saat training.")
    return target, task


def _choose_features(df, target, saved, profile, derived_names):
    candidates = [c for c in df.columns if c != target] + derived_names
    key = _features_key(target)
    if key not in st.session_state:
        if saved and saved.get("target") == target:
            default = [c for c in saved["features"] if c in candidates]
        else:
            flagged = set(profile.loc[profile["Catatan"] != "", "Kolom"])
            default = [c for c in candidates if c not in flagged]
        st.session_state[key] = default
    # Fitur turunan baru otomatis ikut dipilih; yang dihapus otomatis hilang.
    current = [c for c in st.session_state[key] if c in candidates]
    added = [n for n in derived_names if n not in st.session_state.get("prep_known_derived", [])]
    st.session_state[key] = current + [n for n in added if n not in current]
    st.session_state["prep_known_derived"] = list(derived_names)

    features = st.multiselect("Fitur (input model)", candidates, key=key)
    excluded = profile[profile["Kolom"].isin(set(candidates) - set(features)) & (profile["Catatan"] != "")]
    if not excluded.empty:
        st.caption("Tidak dipilih: " + ", ".join(
            f"**{r.Kolom}** ({r.Catatan.lower()})" for r in excluded.itertuples()))
    return features


def _derived_editor(df, target):
    """Kelola fitur turunan (draft; ikut tersimpan saat konfigurasi disimpan)."""
    drafts = st.session_state.draft_derived
    numeric = [c for c in df.columns if c != target and ml.column_kind(df[c]) == ml.NUMERIC]
    if drafts:
        for i, spec in enumerate(drafts):
            symbol = {"/": "÷", "-": "−", "+": "+", "*": "×"}[spec["op"]]
            left, right = st.columns([5, 1], vertical_alignment="center")
            left.markdown(f"`{spec['name']}` = **{spec['a']}** {symbol} **{spec['b']}**")
            if right.button("Hapus", key=f"del_derived_{i}", icon=":material/delete:"):
                drafts.pop(i)
                st.rerun()
    else:
        st.caption("Belum ada fitur turunan. Contoh: rasio total terhadap jumlah kejadian (nilai per unit), "
                   "selisih dua besaran, atau interaksi dua variabel.")
    if len(numeric) < 2:
        st.caption("Butuh minimal dua kolom numerik.")
        return
    with st.form("add_derived", border=False):
        c1, c2, c3 = st.columns([2, 1.3, 2])
        a = c1.selectbox("Kolom A", numeric)
        op = c2.selectbox("Operasi", list(ml.DERIVED_OPS))
        b = c3.selectbox("Kolom B", numeric, index=min(1, len(numeric) - 1))
        name = st.text_input("Nama fitur baru (opsional)", placeholder="otomatis")
        if st.form_submit_button("Tambah fitur turunan", icon=":material/add:"):
            symbol = ml.DERIVED_OPS[op]
            word = {"/": "per", "-": "minus", "+": "plus", "*": "kali"}[symbol]
            name = name.strip() or re.sub(r"\s+", "_", f"{a} {word} {b}")
            if name in df.columns or any(d["name"] == name for d in drafts):
                st.error(f"Nama `{name}` sudah dipakai.", icon=":material/error:")
            else:
                drafts.append({"name": name, "op": symbol, "a": a, "b": b})
                st.rerun()


def _choose_preprocessing(saved, task, numeric_features, target_series):
    saved = saved or {}
    c1, c2, c3 = st.columns(3)
    impute = c1.radio("Isi missing value numerik", list(ml.IMPUTE_OPTIONS),
                      index=_index(list(ml.IMPUTE_OPTIONS), saved.get("impute")))
    scaling = c2.radio("Scaling numerik", list(ml.SCALING_OPTIONS),
                       index=_index(list(ml.SCALING_OPTIONS), saved.get("scaling"), 1),
                       help="Robust memakai median & IQR sehingga tidak terpengaruh outlier.")
    encodings = list(ml.ENCODING_OPTIONS) if task != ml.CLUSTERING else ["One-Hot", "Ordinal"]
    encoding = c3.radio("Encoding kategorikal", encodings, index=_index(encodings, saved.get("encoding")),
                        help="One-Hot: satu kolom per kategori. Ordinal: satu kolom nomor kategori. "
                             "Target encoding: rata-rata target per kategori (cocok untuk kategori yang sangat "
                             "banyak; dihitung dengan cross-fitting untuk mencegah kebocoran).")

    st.markdown("**Penanganan lanjutan**")
    c1, c2 = st.columns(2)
    log_features = c1.multiselect("Transformasi log(1 + x) untuk kolom miring", numeric_features,
                                  default=[c for c in saved.get("log_features", []) if c in numeric_features],
                                  help="Lihat bagian Kualitas di Eksplorasi untuk kolom dengan skewness > 1.")
    rare_pct = c2.slider("Gabungkan kategori langka (< x% data)", 0, 10, int(100 * (saved.get("rare_min") or 0)),
                         disabled=encoding == "Target encoding", format="%d%%",
                         help="Kategori yang sangat jarang digabung menjadi satu kelompok 'infrequent'.")
    c1, c2 = st.columns(2)
    clip = c1.toggle("Potong outlier (batas IQR)", value=saved.get("clip_outliers", False),
                     help="Nilai di luar Q1 − 1.5·IQR … Q3 + 1.5·IQR dipotong ke batasnya (dipelajari dari data latih).")
    log_target = False
    if task == ml.REGRESSION:
        values = pd.to_numeric(target_series, errors="coerce").dropna()
        can_log = not values.empty and values.min() > -1
        log_target = c2.toggle("Log-transform target", value=saved.get("log_target", False) and can_log,
                               disabled=not can_log,
                               help="Model belajar pada log(1 + target), prediksi dikembalikan ke skala asli. "
                                    "Berguna jika target sangat miring.")
    return {"impute": impute, "scaling": scaling, "encoding": encoding, "log_features": log_features,
            "rare_min": rare_pct / 100, "clip_outliers": clip, "log_target": log_target}


def _drop_features(key, cols):
    st.session_state[key] = [c for c in st.session_state[key] if c not in cols]


def _leakage_section(df, config):
    report = _leakage(df, config)
    table, relation = report["table"], report["relation"]
    suspects = [] if table.empty else table.loc[table["Status"] != "OK", "Fitur"].tolist()
    if relation:
        st.error(f"**Target dapat dihitung tepat dari fitur:** `{relation['equation']}` "
                 f"(R² = {relation['r2']:.4f}). Model akan tampak sempurna, tetapi tidak berguna untuk data "
                 "baru di mana nilai-nilai ini belum diketahui.", icon=":material/report:")
        suspects = list(dict.fromkeys(suspects + list(relation["terms"])))
    elif suspects:
        st.warning("Beberapa fitur hampir menentukan target sendirian. Pastikan nilainya tersedia "
                   "*sebelum* target diketahui.", icon=":material/warning:")
    else:
        st.success("Tidak ada tanda kebocoran data pada fitur numerik.", icon=":material/verified:")
    if not table.empty:
        st.dataframe(table, width="stretch", hide_index=True, column_config={
            "Kekuatan": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)})
    if suspects:
        st.button(f"Buang {len(suspects)} fitur yang dicurigai: {', '.join(suspects)}",
                  icon=":material/remove_circle:", on_click=_drop_features,
                  args=(_features_key(config["target"]), suspects))


def render():
    ui.step_header("prepare", "Tentukan apa yang ingin diprediksi dan bagaimana data disiapkan. "
                              "Data mentah tidak diubah; semua langkah ini disimpan sebagai resep "
                              "yang dijalankan bersama model.")
    ui.require(st.session_state.df is not None, "Muat dataset terlebih dahulu.", "data")

    df = st.session_state.df
    saved = st.session_state.config
    profile = ml.column_profile(df)
    if "draft_derived" not in st.session_state:
        st.session_state.draft_derived = [dict(d) for d in (saved or {}).get("derived", [])]

    with ui.section("A. Target & jenis tugas"):
        target, task = _choose_target(df, saved)

    with ui.section("B. Fitur turunan (opsional)",
                    "Buat fitur baru dari dua kolom numerik. Dihitung otomatis di dalam pipeline, "
                    "termasuk saat prediksi data baru."):
        _derived_editor(df, target)
    derived = [d for d in st.session_state.draft_derived if target not in (d["a"], d["b"])]
    derived_names = [d["name"] for d in derived]

    with ui.section("C. Fitur", "Kolom ID atau konstan sebaiknya tidak dipakai sebagai fitur."):
        features = _choose_features(df, target, saved, profile, derived_names)
    if not features:
        st.error("Pilih minimal satu fitur.", icon=":material/error:")
        st.stop()

    full = ml.apply_derived(df, derived)
    kinds = {c: ml.column_kind(full[c]) for c in features}
    numeric_features = [c for c in features if kinds[c] == ml.NUMERIC]
    with ui.section("D. Preprocessing"):
        prep = _choose_preprocessing(saved, task, numeric_features, df[target] if target else None)

    config = {
        "target": target,
        "task": task,
        "features": features,
        "derived": [d for d in derived if d["name"] in features],
        "kinds": kinds,
        **prep,
        "select_k": None,
    }
    try:
        transformed = _preview(df, config)
    except Exception as exc:
        st.error(f"Preprocessing gagal dengan konfigurasi ini: {exc}", icon=":material/error:")
        st.stop()

    if task != ml.CLUSTERING:
        with ui.section("E. Seleksi fitur (opsional)",
                        "Hanya pertahankan k kolom hasil preprocessing dengan skor statistik "
                        "tertinggi (SelectKBest)."):
            n_out = transformed.shape[1]
            saved_k = (saved or {}).get("select_k")
            if st.toggle("Aktifkan seleksi fitur", value=bool(saved_k), disabled=n_out < 2):
                config["select_k"] = st.slider("Jumlah fitur (k)", 1, n_out - 1,
                                               min(saved_k or max(1, n_out // 2), n_out - 1))
        with ui.section("F. Cek kebocoran data",
                        "Fitur yang (hampir) menentukan target membuat skor terlihat sempurna "
                        "tetapi model gagal pada data baru."):
            _leakage_section(df, config)

    is_saved = saved == config
    with st.container(border=True, key="card-save"):
        left, right = st.columns([3, 2], vertical_alignment="center")
        if is_saved:
            left.markdown(":green[:material/check_circle:] **Konfigurasi tersimpan**  \n"
                          ":gray[Dipakai untuk training di langkah berikutnya.]")
        elif saved:
            left.markdown(":orange[:material/edit:] **Ada perubahan yang belum disimpan**  \n"
                          ":gray[Menyimpan akan menghapus hasil training lama.]")
        else:
            left.markdown(":blue[:material/info:] **Belum disimpan**  \n"
                          ":gray[Simpan konfigurasi untuk membuka langkah Training.]")
        if right.button("Tersimpan" if is_saved else "Simpan konfigurasi", type="primary",
                        icon=":material/save:", disabled=is_saved, width="stretch"):
            state.set_config(config)
            st.rerun()

    with st.expander(f"Pratinjau hasil preprocessing · {transformed.shape[1]} kolom"):
        st.dataframe(transformed.head(20), width="stretch")
        st.caption(f"Kolom mentah yang dibutuhkan saat prediksi: {', '.join(ml.input_columns(config))}")

    if is_saved:
        ui.next_step_button("train")
