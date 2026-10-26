"""ML Workflow Studio — titik masuk aplikasi.

Jalankan dengan:  streamlit run app.py
"""
import streamlit as st

from mlapp import state, ui
from mlapp.views import data, evaluate, explore, home, predict, prepare, train, visualize

st.set_page_config(page_title="ML Workflow Studio", page_icon=":material/hub:", layout="wide")
state.init()

VIEWS = {
    "data": data.render,
    "explore": explore.render,
    "visualize": visualize.render,
    "prepare": prepare.render,
    "train": train.render,
    "evaluate": evaluate.render,
    "predict": predict.render,
}

home_page = st.Page(home.render, title="Beranda", icon=":material/home:", default=True)
ui.PAGES.update({
    step.key: st.Page(VIEWS[step.key], title=f"{step.number}. {step.title}", icon=step.icon,
                      url_path=step.key)
    for step in state.STEPS
})

nav = st.navigation([home_page, *ui.PAGES.values()], position="hidden")
ui.inject_css()

# Tandai langkah yang sedang dibuka sebelum sidebar digambar, agar progresnya langsung terbarui.
if nav.url_path in ui.PAGES:
    state.mark_visited(nav.url_path)

with st.sidebar:
    ui.sidebar(home_page)
    st.space("small")
    if st.button("Mulai ulang", icon=":material/restart_alt:", width="stretch",
                 disabled=st.session_state.df is None, help="Hapus data, konfigurasi, dan model."):
        state.reset_all()
        st.switch_page(home_page)

nav.run()
