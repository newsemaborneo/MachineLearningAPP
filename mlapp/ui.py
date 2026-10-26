"""Sistem desain & komponen UI bersama.

Setiap halaman memakai struktur yang sama:

    stepper  ->  judul langkah  ->  (cek prasyarat)  ->  kartu-kartu isi  ->  tombol langkah berikutnya

Warna di bawah adalah salinan token di .streamlit/config.toml; ubah keduanya bersamaan.
"""
import re
from html import escape

import plotly.express as px
import streamlit as st

from mlapp import state

# Diisi oleh app.py: key langkah -> st.Page
PAGES: dict = {}

INK, MUTED, LINE = "#0F172A", "#64748B", "#E2E8F0"
PRIMARY, PRIMARY_SOFT = "#4F46E5", "#EEF2FF"
DONE, DONE_SOFT = "#0D9488", "#CCFBF1"

# Skala warna grafik yang selaras dengan tema.
SEQUENTIAL_SCALE = ["#F8FAFC", "#C7D2FE", "#818CF8", "#4F46E5", "#312E81"]
DIVERGING_SCALE = ["#DC2626", "#FCA5A5", "#F8FAFC", "#A5B4FC", "#4F46E5"]

STATUS_BADGE = {
    state.DONE: (":material/check_circle:", "green", "Selesai"),
    state.READY: (":material/radio_button_unchecked:", "blue", "Siap"),
    state.LOCKED: (":material/lock:", "gray", "Terkunci"),
}

px.defaults.template = "plotly_white"
px.defaults.height = 380

CSS = f"""
<style>
:root {{
  --ink: {INK}; --muted: {MUTED}; --line: {LINE};
  --primary: {PRIMARY}; --primary-soft: {PRIMARY_SOFT};
  --done: {DONE}; --done-soft: {DONE_SOFT};
  --card: #FFFFFF; --shadow: 0 1px 2px rgba(15,23,42,.04), 0 6px 20px rgba(15,23,42,.05);
}}
[data-testid="stMainBlockContainer"] {{ max-width: 1180px; padding-top: 3.75rem; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1 {{ letter-spacing: -0.02em; }}

/* Kartu (st.container dengan key "card-…") */
[class*="st-key-card-"] {{
  background: var(--card); border: 1px solid var(--line) !important; border-radius: 16px;
  box-shadow: var(--shadow); padding: 1.35rem 1.5rem 1.5rem !important;
}}
[class*="st-key-card-"] h3 {{ font-size: 1.15rem; padding-top: 0; }}
[data-testid="stMetric"] {{ background: var(--card); box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
[data-testid="stMetricValue"] {{ font-weight: 700; color: var(--ink); }}
[data-testid="stForm"] {{ background: var(--card); border-radius: 16px; box-shadow: var(--shadow); }}
.st-key-result [data-testid="stMetric"] {{ background: var(--primary-soft); border-color: #C7D2FE; }}
.st-key-result [data-testid="stMetricValue"] {{ color: var(--primary); }}
[data-testid="stExpander"] details {{ background: var(--card); border-radius: 12px; }}
[data-testid="stBaseButton-primary"] {{ box-shadow: 0 4px 14px rgba(79,70,229,.28); font-weight: 600; }}

/* Stepper horizontal di atas setiap halaman langkah */
.stepper {{ display: flex; align-items: flex-start; margin: 0 0 1.75rem; }}
.stepper .st {{ display: flex; flex-direction: column; align-items: center; gap: .4rem; min-width: 72px; }}
.stepper .dot {{
  width: 32px; height: 32px; border-radius: 50%; display: grid; place-items: center;
  font: 600 .85rem/1 Inter, sans-serif; border: 2px solid var(--line); background: #fff; color: var(--muted);
}}
.stepper .lbl {{ font-size: .78rem; color: var(--muted); white-space: nowrap; }}
.stepper .bar {{ flex: 1; height: 2px; background: var(--line); margin-top: 15px; border-radius: 2px; }}
.stepper .done .dot {{ background: var(--done); border-color: var(--done); color: #fff; }}
.stepper .ready .dot {{ border-color: #A5B4FC; color: var(--primary); }}
.stepper .current .dot {{ background: var(--primary); border-color: var(--primary); color: #fff;
  box-shadow: 0 0 0 5px var(--primary-soft); }}
.stepper .current .lbl {{ color: var(--ink); font-weight: 600; }}
.stepper .bar.done {{ background: var(--done); }}
@media (max-width: 720px) {{ .stepper .lbl {{ display: none; }} .stepper .st {{ min-width: 36px; }} }}

.eyebrow {{ font-size: .75rem; font-weight: 600; letter-spacing: .08em; color: var(--primary); }}
.eyebrow .opt {{ color: var(--muted); }}

/* Banner beranda */
.hero {{
  background: radial-gradient(120% 140% at 100% 0%, #14B8A6 0%, rgba(20,184,166,0) 45%),
              linear-gradient(135deg, #1E1B4B 0%, #3730A3 50%, #4F46E5 100%);
  color: #fff; border-radius: 20px; padding: 2.4rem 2.6rem; margin-bottom: 1.25rem;
  box-shadow: 0 12px 32px rgba(49,46,129,.25);
}}
.hero .tag {{ display: inline-block; font-size: .75rem; font-weight: 600; letter-spacing: .06em;
  background: rgba(255,255,255,.14); padding: .3rem .7rem; border-radius: 999px; }}
.hero h1 {{ color: #fff; font-size: 2.4rem; margin: .8rem 0 .5rem; padding: 0; }}
.hero p {{ color: #E0E7FF; font-size: 1.02rem; max-width: 640px; margin: 0; }}
.hero .meter {{ margin-top: 1.4rem; max-width: 420px; }}
.hero .meter .track {{ height: 8px; background: rgba(255,255,255,.18); border-radius: 99px; overflow: hidden; }}
.hero .meter .fill {{ height: 100%; background: #5EEAD4; border-radius: 99px; }}
.hero .meter small {{ display: block; color: #C7D2FE; margin-top: .45rem; }}

/* Kartu langkah di beranda */
.step-head {{ display: flex; align-items: center; gap: .75rem; }}
.step-head .num {{ width: 36px; height: 36px; border-radius: 10px; display: grid; place-items: center;
  font-weight: 700; background: var(--primary-soft); color: var(--primary); }}
.step-head.done .num {{ background: var(--done-soft); color: var(--done); }}
.step-head.locked .num {{ background: #F1F5F9; color: #94A3B8; }}
.step-head b {{ color: var(--ink); font-size: 1rem; }}
.step-head .chip {{ margin-left: auto; font-size: .72rem; font-weight: 600; padding: .2rem .55rem;
  border-radius: 999px; background: var(--primary-soft); color: var(--primary); }}
.step-head.done .chip {{ background: var(--done-soft); color: var(--done); }}
.step-head.locked .chip {{ background: #F1F5F9; color: #94A3B8; }}
.step-desc {{ color: var(--muted); font-size: .9rem; margin: .5rem 0 0; min-height: 2.6em; }}

/* Sidebar */
[data-testid="stSidebarContent"] {{ padding-top: .5rem; }}
.brand {{ display: flex; align-items: center; gap: .6rem; margin: 0 0 1.1rem; }}
.brand .logo {{ width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center;
  background: linear-gradient(135deg, #6366F1, #14B8A6); color: #fff; font-weight: 800; }}
.brand b {{ color: #F8FAFC; font-size: 1rem; }}
.brand small {{ display: block; color: #64748B; font-size: .72rem; }}
.sb-label {{ font-size: .7rem; font-weight: 600; letter-spacing: .08em; color: #64748B; margin: .4rem 0 .35rem; }}
.sb-meter {{ display: flex; align-items: center; gap: .6rem; margin-bottom: .6rem; }}
.sb-meter .track {{ flex: 1; height: 6px; background: #1E293B; border-radius: 99px; overflow: hidden; }}
.sb-meter .fill {{ height: 100%; background: linear-gradient(90deg, #6366F1, #2DD4BF); }}
.sb-meter span {{ font-size: .75rem; color: #94A3B8; }}
[class*="st-key-nav-done"] [data-testid="stIconMaterial"] {{ color: #2DD4BF; }}
[class*="st-key-nav-ready"] [data-testid="stIconMaterial"] {{ color: #A5B4FC; }}
[class*="st-key-nav-locked"] {{ opacity: .6; }}
.ds-card {{ background: #1E293B; border-radius: 12px; padding: .8rem .9rem; font-size: .85rem; color: #CBD5E1; }}
.ds-card b {{ color: #F8FAFC; }}
.ds-card .muted {{ color: #64748B; }}
.ds-card .row {{ display: flex; justify-content: space-between; margin-top: .3rem; }}
</style>
"""


def inject_css():
    st.html(CSS)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# --------------------------------------------------------------------------- #
# Struktur halaman
# --------------------------------------------------------------------------- #
def stepper(current: str):
    statuses = state.step_status()
    parts = []
    for i, step in enumerate(state.STEPS):
        cls = "current" if step.key == current else statuses[step.key]
        mark = "✓" if statuses[step.key] == state.DONE and step.key != current else step.number
        if i:
            prev_done = statuses[state.STEPS[i - 1].key] == state.DONE
            parts.append(f'<div class="bar {"done" if prev_done else ""}"></div>')
        parts.append(f'<div class="st {cls}"><span class="dot">{mark}</span>'
                     f'<span class="lbl">{step.title}</span></div>')
    st.html(f'<div class="stepper">{"".join(parts)}</div>')


def step_header(step_key: str, description: str):
    step = state.STEP_BY_KEY[step_key]
    stepper(step_key)
    optional = ' <span class="opt">· OPSIONAL</span>' if step.optional else ""
    st.html(f'<div class="eyebrow">LANGKAH {step.number} DARI {len(state.STEPS)}{optional}</div>')
    st.title(step.title, anchor=False)
    st.markdown(f":gray[{description}]")


def section(title: str, help_text: str | None = None):
    """Kartu putih untuk mengelompokkan satu bagian di dalam halaman."""
    box = st.container(border=True, key=f"card-{_slug(title)}")
    box.subheader(title, anchor=False)
    if help_text:
        box.caption(help_text)
    return box


def require(ok: bool, message: str, step_key: str):
    """Hentikan halaman dengan pesan yang jelas jika prasyarat belum terpenuhi."""
    if ok:
        return
    target = state.STEP_BY_KEY[step_key]
    with st.container(border=True, key="card-locked"):
        st.markdown(f"#### :gray[:material/lock:] Selesaikan langkah {target.number} dulu")
        st.markdown(f":gray[{message}]")
        if st.button(f"Buka langkah {target.number} · {target.title}", type="primary",
                     icon=":material/arrow_forward:"):
            st.switch_page(PAGES[step_key])
    st.stop()


def next_step_button(step_key: str, label: str | None = None):
    target = state.STEP_BY_KEY[step_key]
    st.space("medium")
    with st.container(border=True, key="card-next"):
        left, right = st.columns([3, 2], vertical_alignment="center")
        left.markdown(f"**Berikutnya · {target.title}**  \n:gray[{target.summary}]")
        if right.button(label or f"Lanjut ke langkah {target.number}", type="primary",
                        icon=":material/arrow_forward:", width="stretch"):
            st.switch_page(PAGES[step_key])


def metric_row(items: dict, fmt: str = "{:.4f}"):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items.items()):
        if value is None:
            shown = "—"
        elif isinstance(value, float):
            shown = fmt.format(value)
        else:
            shown = f"{value:,}" if isinstance(value, int) else str(value)
        col.metric(label, shown, border=True)


def chart(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), title_font=dict(size=15),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      modebar=dict(bgcolor="rgba(0,0,0,0)", color="#94A3B8", activecolor=PRIMARY))
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# Beranda
# --------------------------------------------------------------------------- #
def progress_counts():
    statuses = state.step_status()
    required = [s for s in state.STEPS if not s.optional]
    return sum(statuses[s.key] == state.DONE for s in required), len(required)


def hero():
    done, total = progress_counts()
    st.html(f"""
    <div class="hero">
      <span class="tag">SCIKIT-LEARN · STREAMLIT</span>
      <h1>ML Workflow Studio</h1>
      <p>Bangun model machine learning dari file CSV dalam enam langkah berurutan,
         mulai dari data mentah sampai model siap pakai.</p>
      <div class="meter"><div class="track"><div class="fill" style="width:{100 * done / total:.0f}%"></div></div>
        <small>{done} dari {total} langkah wajib selesai</small></div>
    </div>""")


def step_card(step, status):
    _, _, label = STATUS_BADGE[status]
    num = "✓" if status == state.DONE else step.number
    with st.container(border=True, key=f"card-step-{step.key}"):
        st.html(f'<div class="step-head {status}"><span class="num">{num}</span>'
                f'<b>{step.title}</b><span class="chip">{label}</span></div>'
                f'<p class="step-desc">{"Opsional · " if step.optional else ""}{step.summary}</p>')
        st.page_link(PAGES[step.key], label="Buka langkah", icon=":material/arrow_forward:",
                     disabled=status == state.LOCKED)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar(home_page):
    st.html('<div class="brand"><span class="logo">ML</span>'
            '<div><b>ML Workflow Studio</b><small>Dari CSV ke model</small></div></div>')

    done, total = progress_counts()
    st.html(f'<div class="sb-label">PROGRES</div><div class="sb-meter"><div class="track">'
            f'<div class="fill" style="width:{100 * done / total:.0f}%"></div></div>'
            f'<span>{done}/{total}</span></div>')

    st.page_link(home_page, label="Beranda", icon=":material/home:")
    st.html('<div class="sb-label">ALUR KERJA</div>')
    statuses = state.step_status()
    for step in state.STEPS:
        status = statuses[step.key]
        icon = STATUS_BADGE[status][0]
        suffix = " · opsional" if step.optional else ""
        with st.container(key=f"nav-{status}-{step.key}"):
            st.page_link(PAGES[step.key], label=f"{step.number}. {step.title}{suffix}", icon=icon,
                         disabled=status == state.LOCKED)

    st.html('<div class="sb-label" style="margin-top:1rem">DATASET AKTIF</div>')
    _dataset_card()


def _dataset_card():
    df = st.session_state.df
    if df is None:
        st.html('<div class="ds-card"><span class="muted">Belum ada data dimuat.</span></div>')
        return
    rows = [f'<b>{escape(st.session_state.data_name)}</b>',
            f'<div class="row"><span class="muted">Ukuran</span><span>{df.shape[0]:,} × {df.shape[1]}</span></div>']
    config = st.session_state.config
    if config:
        rows.append(f'<div class="row"><span class="muted">Tugas</span><span>{config["task"]}</span></div>')
        rows.append(f'<div class="row"><span class="muted">Target</span>'
                    f'<span>{escape(str(config.get("target") or "—"))}</span></div>')
    run = state.active_run()
    if run:
        rows.append(f'<div class="row"><span class="muted">Model</span><span>{run.algorithm}</span></div>')
    st.html(f'<div class="ds-card">{"".join(rows)}</div>')
