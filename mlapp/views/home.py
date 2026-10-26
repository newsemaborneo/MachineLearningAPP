import streamlit as st

from mlapp import state, ui


def render():
    ui.hero()

    nxt = state.next_step()
    with st.container(border=True, key="card-cta"):
        left, right = st.columns([3, 2], vertical_alignment="center")
        if nxt is not None:
            left.markdown(f"**Langkah berikutnya · {nxt.number}. {nxt.title}**  \n:gray[{nxt.summary}]")
            if right.button(f"Mulai langkah {nxt.number}", type="primary", icon=":material/play_arrow:",
                            width="stretch"):
                st.switch_page(ui.PAGES[nxt.key])
        else:
            left.markdown("**Semua langkah wajib selesai**  \n:gray[Model siap dipakai untuk prediksi.]")
            if right.button("Buka Prediksi", type="primary", icon=":material/online_prediction:",
                            width="stretch"):
                st.switch_page(ui.PAGES["predict"])

    st.space("small")
    st.subheader("Alur kerja", anchor=False)
    statuses = state.step_status()
    for row in (state.STEPS[:4], state.STEPS[4:]):
        for col, step in zip(st.columns(4), row):
            with col:
                ui.step_card(step, statuses[step.key])

    with st.expander("Bagaimana data mengalir antar langkah?", icon=":material/account_tree:"):
        st.markdown(
            """
            | Tahap | Disimpan | Dipakai oleh |
            |---|---|---|
            | **1 · Data** | DataFrame mentah | Eksplorasi, Persiapan |
            | **3 · Persiapan** | Target, fitur, jenis tugas, strategi preprocessing | Training |
            | **4 · Training** | Pipeline *(preprocessing + model)* per algoritma | Evaluasi, Prediksi |

            Preprocessing tidak mengubah data mentah. Semua langkah (imputasi, encoding,
            scaling, seleksi fitur) dibungkus bersama model dalam satu `sklearn.Pipeline`,
            sehingga data baru di halaman Prediksi cukup diisi dalam bentuk aslinya.
            """
        )
