from io import BytesIO

import pandas as pd
import streamlit as st

from mlapp import ml, state, ui

SEPARATORS = {"Otomatis": None, "Koma (,)": ",", "Titik koma (;)": ";", "Tab": "\t"}


@st.cache_data(show_spinner=False)
def _read_csv(content: bytes, sep: str | None) -> pd.DataFrame:
    return pd.read_csv(BytesIO(content), sep=sep, engine="python" if sep is None else "c")


def _load_source():
    col_file, col_sep = st.columns([3, 1])
    uploaded = col_file.file_uploader("File CSV (UTF-8, baris pertama = header)", type=["csv"])
    sep_label = col_sep.selectbox("Pemisah kolom", list(SEPARATORS))
    if uploaded is None:
        return
    data_id = f"{uploaded.file_id}:{sep_label}"
    if data_id == st.session_state.data_id:
        return
    try:
        df = _read_csv(uploaded.getvalue(), SEPARATORS[sep_label])
    except Exception as exc:
        st.error(f"File tidak bisa dibaca sebagai CSV: {exc}", icon=":material/error:")
        return
    if df.shape[1] < 2:
        st.warning("Hanya 1 kolom terbaca. Coba ganti **Pemisah kolom**.", icon=":material/warning:")
    state.set_data(df, uploaded.name, data_id)
    st.rerun()


def render():
    ui.step_header("data", "Mulai dengan memuat dataset. Memuat data baru akan mengosongkan "
                           "konfigurasi dan model dari dataset sebelumnya.")

    with ui.section("Sumber data"):
        _load_source()

    df = st.session_state.df
    if df is None:
        st.info("Belum ada data. Unggah file CSV untuk melanjutkan.",
                icon=":material/info:")
        return

    profile = ml.column_profile(df)
    with ui.section(f"Ringkasan · {st.session_state.data_name}"):
        ui.metric_row({
            "Baris": df.shape[0],
            "Kolom": df.shape[1],
            "Numerik": int((profile["Jenis"] == ml.NUMERIC).sum()),
            "Kategorikal": int((profile["Jenis"] == ml.CATEGORICAL).sum()),
            "Missing": int(df.isna().sum().sum()),
            "Baris duplikat": int(df.duplicated().sum()),
        })
        st.dataframe(df.head(20), width="stretch")

    with ui.section("Profil kolom", "Jenis kolom ini menentukan preprocessing di langkah Persiapan."):
        st.dataframe(
            profile, width="stretch", hide_index=True,
            column_config={"Missing %": st.column_config.ProgressColumn(
                "Missing %", format="%.1f%%", min_value=0, max_value=100)},
        )
        flagged = profile[profile["Catatan"] != ""]
        for _, row in flagged.iterrows():
            st.caption(f":material/flag: **{row['Kolom']}** — {row['Catatan']}. "
                       "Pertimbangkan untuk tidak memakainya sebagai fitur.")

    ui.next_step_button("explore")
