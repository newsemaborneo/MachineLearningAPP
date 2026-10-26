"""Session state dan status alur kerja.

Semua data yang dibagi antar halaman disimpan di sini, dengan urutan
ketergantungan yang tegas:

    data  ->  config (persiapan)  ->  runs (hasil training)

Mengubah satu tahap otomatis mengosongkan tahap sesudahnya, sehingga
model tidak pernah dievaluasi/dipakai dengan data atau konfigurasi yang
sudah tidak berlaku.
"""
from dataclasses import dataclass

import streamlit as st

DONE, READY, LOCKED = "done", "ready", "locked"


@dataclass(frozen=True)
class Step:
    key: str
    number: int
    title: str
    icon: str
    summary: str
    optional: bool = False


STEPS = [
    Step("data", 1, "Data", ":material/upload_file:",
         "Unggah file CSV dan periksa profil kolomnya."),
    Step("explore", 2, "Eksplorasi", ":material/query_stats:",
         "Kualitas data, outlier, korelasi, uji statistik, dan analisis target.", optional=True),
    Step("visualize", 3, "Visualisasi", ":material/insert_chart:",
         "Bangun grafik apa pun dari data: 15+ jenis grafik interaktif.", optional=True),
    Step("prepare", 4, "Persiapan", ":material/tune:",
         "Pilih target, fitur, fitur turunan, dan cek kebocoran data."),
    Step("train", 5, "Training", ":material/model_training:",
         "Latih, tuning, dan bandingkan algoritma dengan cross-validation."),
    Step("evaluate", 6, "Evaluasi", ":material/analytics:",
         "Metrik, SHAP, PDP, learning curve, kalibrasi, dan analisis error."),
    Step("predict", 7, "Prediksi", ":material/online_prediction:",
         "Prediksi data baru dan unduh model."),
]
STEP_BY_KEY = {s.key: s for s in STEPS}

_DEFAULTS = {
    "df": None,            # DataFrame mentah hasil unggahan
    "data_name": None,     # nama sumber data (untuk ditampilkan)
    "data_id": None,       # identitas file, agar unggahan yang sama tidak me-reset alur
    "config": None,        # dict hasil langkah Persiapan
    "runs": [],            # list[ml.Run] hasil training terakhir
    "active_run": None,    # indeks run yang dipakai Evaluasi & Prediksi
    "history": [],         # riwayat eksperimen (ringkasan tiap sesi training)
    "visited": set(),      # langkah opsional yang sudah dibuka
}


def init():
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (list, set)) else value


def reset_all():
    for key in _DEFAULTS:
        st.session_state.pop(key, None)
    init()


def set_data(df, name, data_id):
    st.session_state.df = df
    st.session_state.data_name = name
    st.session_state.data_id = data_id
    st.session_state.visited = set()
    st.session_state.history = []
    st.session_state.pop("draft_derived", None)
    set_config(None)


def set_config(config):
    st.session_state.config = config
    set_runs([])


def set_runs(runs, active=None):
    st.session_state.runs = runs
    st.session_state.active_run = active


def mark_visited(step_key):
    st.session_state.visited.add(step_key)


def active_run():
    runs, idx = st.session_state.runs, st.session_state.active_run
    if idx is None or idx >= len(runs):
        return None
    return runs[idx]


def step_status():
    """Status tiap langkah: DONE, READY (bisa dikerjakan), atau LOCKED."""
    has_data = st.session_state.df is not None
    has_config = st.session_state.config is not None
    has_runs = any(r.ok for r in st.session_state.runs)
    visited = st.session_state.visited

    def status(done, ready):
        return DONE if done else READY if ready else LOCKED

    return {
        "data": status(has_data, True),
        "explore": status(has_data and "explore" in visited, has_data),
        "visualize": status(has_data and "visualize" in visited, has_data),
        "prepare": status(has_config, has_data),
        "train": status(has_runs, has_config),
        "evaluate": status(has_runs and "evaluate" in visited, has_runs),
        "predict": status(has_runs and "predict" in visited, has_runs),
    }


def next_step():
    """Langkah wajib pertama yang belum selesai (atau None jika semua selesai)."""
    statuses = step_status()
    for step in STEPS:
        if not step.optional and statuses[step.key] == READY:
            return step
    return None
