import streamlit as st

from mlapp import charts, eda, ml, ui

NONE = "—"


def _pick(label, options, key, optional=False, default=0):
    choices = ([NONE] if optional else []) + list(options)
    if not choices:
        st.caption(f"{label}: tidak ada kolom yang cocok.")
        return None
    value = st.selectbox(label, choices, index=min(default, len(choices) - 1), key=key)
    return None if value == NONE else value


def _controls(kind, df, numeric, categorical):
    spec = charts.CHARTS[kind]
    numeric_fields = charts.NUMERIC_FIELDS.get(kind, [])
    all_cols = df.columns.tolist()
    low_card = [c for c in all_cols if df[c].nunique() <= 50]
    p = {}

    def options(field):
        if field in numeric_fields:
            return numeric
        if field in ("color", "facet", "names"):
            return low_card if field != "color" else all_cols
        return all_cols

    labels = {"x": "Sumbu X", "y": "Sumbu Y", "z": "Sumbu Z / nilai", "color": "Warna", "size": "Ukuran",
              "facet": "Pecah per (facet)", "names": "Kategori", "values": "Nilai"}
    for i, field in enumerate(spec["required"] + spec["optional"]):
        optional = field in spec["optional"]
        if field == "dims":
            p["dims"] = st.multiselect("Kolom numerik", numeric, default=numeric[:4], key=f"viz_{kind}_dims")
        elif field == "path":
            p["path"] = st.multiselect("Hierarki kategori (urut dari luar)", low_card,
                                       default=[c for c in categorical if c in low_card][:2], key=f"viz_{kind}_path")
        else:
            p[field] = _pick(labels[field], options(field), f"viz_{kind}_{field}", optional=optional,
                             default=0 if optional else i)

    with st.expander("Opsi lanjutan", icon=":material/tune:"):
        if kind in ("Line", "Area", "Bar", "Heatmap pivot"):
            p["agg"] = st.selectbox("Agregasi", list(charts.AGGREGATIONS), index=2, key=f"viz_{kind}_agg")
        if kind in ("Histogram", "Density 2D"):
            p["nbins"] = st.slider("Jumlah bin", 5, 100, 30, key=f"viz_{kind}_bins")
        if kind == "Histogram":
            marginal = st.selectbox("Grafik tepi", [NONE, "box", "violin", "rug"], key="viz_hist_marginal")
            p["marginal"] = None if marginal == NONE else marginal
            norm = st.selectbox("Satuan", ["Jumlah", "Persen", "Kepadatan"], key="viz_hist_norm")
            p["histnorm"] = {"Jumlah": None, "Persen": "percent", "Kepadatan": "probability density"}[norm]
        if kind == "Box plot":
            p["points"] = st.selectbox("Titik", ["outliers", "all", False],
                                       format_func=lambda v: {"outliers": "Outlier saja", "all": "Semua",
                                                              False: "Tidak ada"}[v], key="viz_box_points")
            p["notched"] = st.toggle("Notch (interval median)", key="viz_box_notch")
        if kind == "Scatter / Bubble":
            p["trend"] = st.toggle("Garis tren linear", value=True, key="viz_trend")
            p["opacity"] = st.slider("Transparansi", 0.1, 1.0, 0.8, key="viz_opacity")
        if kind == "Bar":
            p["horizontal"] = st.toggle("Horizontal", key="viz_bar_h")
            p["barmode"] = st.selectbox("Mode", ["group", "stack", "relative"], key="viz_bar_mode")
            p["top"] = st.slider("Tampilkan N kategori teratas", 3, 50, 20, key="viz_bar_top")
        if kind == "Pie / Donut":
            p["hole"] = st.slider("Lubang donut", 0.0, 0.7, 0.45, key="viz_pie_hole")
            p["top"] = st.slider("Kategori terbanyak (sisanya 'Lainnya')", 3, 20, 10, key="viz_pie_top")
        if kind not in ("Pie / Donut", "Treemap", "Sunburst", "Scatter matrix", "Parallel coordinates", "Heatmap pivot"):
            c1, c2 = st.columns(2)
            p["log_x"] = c1.toggle("Log sumbu X", key=f"viz_{kind}_logx")
            p["log_y"] = c2.toggle("Log sumbu Y", key=f"viz_{kind}_logy")
        p["title"] = st.text_input("Judul grafik", value="", placeholder=kind, key=f"viz_{kind}_title")
    return p


def _missing_required(kind, p):
    for field in charts.CHARTS[kind]["required"]:
        value = p.get(field)
        if not value or (field == "dims" and len(value) < 2):
            return field
    return None


def _builder(df, numeric, categorical):
    left, right = st.columns([1, 2.2], gap="medium")
    with left, st.container(border=True, key="card-viz-controls"):
        st.markdown("**Jenis grafik**")
        groups = {}
        for name, spec in charts.CHARTS.items():
            groups.setdefault(spec["group"], []).append(name)
        group = st.segmented_control("Kelompok", list(groups), default="Hubungan", key="viz_group",
                                     label_visibility="collapsed") or "Hubungan"
        kind = st.selectbox("Grafik", groups[group], key=f"viz_kind_{group}", label_visibility="collapsed")
        st.caption(charts.CHARTS[kind]["help"])
        st.divider()
        params = _controls(kind, df, numeric, categorical)

    with right, st.container(border=True, key="card-viz-chart"):
        missing = _missing_required(kind, params)
        if missing:
            st.info("Lengkapi pilihan kolom di panel kiri untuk menampilkan grafik.", icon=":material/info:")
            return
        try:
            fig = charts.make(kind, df, params)
        except Exception as exc:
            st.error(f"Grafik tidak bisa dibuat dengan kombinasi kolom ini: {exc}", icon=":material/error:")
            return
        if fig.layout.height is None:
            fig.update_layout(height=560)
        ui.chart(fig)
        st.download_button("Unduh grafik (HTML interaktif)", fig.to_html(include_plotlyjs="cdn"),
                           file_name=f"{kind.split(' ')[0].lower()}.html", mime="text/html",
                           icon=":material/download:")


def _dashboard(df, numeric, categorical):
    limit = st.slider("Maksimum grafik per jenis kolom", 3, 24, 9, key="viz_dash_limit")
    figs = charts.auto_dashboard(df, numeric, categorical, limit)
    for i in range(0, len(figs), 3):
        for j, (col, fig) in enumerate(zip(st.columns(3), figs[i:i + 3])):
            with col, st.container(border=True, key=f"card-dash-{i + j}"):
                ui.chart(fig)


def render():
    ui.step_header("visualize", "Bangun grafik interaktif dari kolom mana pun, atau lihat dashboard "
                                "otomatis untuk seluruh kolom.")
    ui.require(st.session_state.df is not None, "Muat dataset terlebih dahulu.", "data")

    df = st.session_state.df
    config = st.session_state.config or {}
    if config.get("derived"):
        if st.toggle("Sertakan fitur turunan dari langkah Persiapan", value=True, key="viz_derived"):
            df = ml.apply_derived(df, config["derived"])
    numeric, categorical = eda.split_columns(df)

    mode = st.segmented_control("Mode", ["Pembuat grafik", "Dashboard otomatis"], default="Pembuat grafik",
                                key="viz_mode", label_visibility="collapsed") or "Pembuat grafik"
    if len(df) > 20000:
        df = df.sample(20000, random_state=0)
        st.caption("Menampilkan sampel acak 20.000 baris agar grafik tetap responsif.")
    if mode == "Pembuat grafik":
        _builder(df, numeric, categorical)
    else:
        _dashboard(df, numeric, categorical)

    ui.next_step_button("prepare")
