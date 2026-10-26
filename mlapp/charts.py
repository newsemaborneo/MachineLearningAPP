"""Pembuat grafik generik untuk halaman Visualisasi (tanpa Streamlit).

``CHARTS`` mendeskripsikan setiap jenis grafik: grup, field yang dibutuhkan, dan opsional.
``make(kind, df, p)`` membangun figure Plotly dari parameter ``p``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

AGGREGATIONS = {"Jumlah baris": "count", "Total": "sum", "Rata-rata": "mean", "Median": "median",
                "Minimum": "min", "Maksimum": "max"}

# field: x, y, z, color, size, facet, names, values, path, dims
CHARTS = {
    "Histogram": {"group": "Distribusi", "required": ["x"], "optional": ["color", "facet"],
                  "help": "Sebaran nilai satu kolom."},
    "Box plot": {"group": "Distribusi", "required": ["y"], "optional": ["x", "color"],
                 "help": "Median, kuartil, dan outlier; bandingkan antar kelompok lewat sumbu X."},
    "Violin": {"group": "Distribusi", "required": ["y"], "optional": ["x", "color"],
               "help": "Seperti box plot, ditambah bentuk kepadatan distribusi."},
    "Strip (jitter)": {"group": "Distribusi", "required": ["y"], "optional": ["x", "color"],
                       "help": "Setiap titik data ditampilkan; cocok untuk data kecil."},
    "ECDF": {"group": "Distribusi", "required": ["x"], "optional": ["color"],
             "help": "Proporsi kumulatif data ≤ nilai tertentu."},
    "Density 2D": {"group": "Distribusi", "required": ["x", "y"], "optional": [],
                   "help": "Kepadatan titik pada dua variabel numerik."},
    "Scatter / Bubble": {"group": "Hubungan", "required": ["x", "y"], "optional": ["color", "size", "facet"],
                         "help": "Hubungan dua variabel; isi 'Ukuran' untuk bubble chart."},
    "Line": {"group": "Hubungan", "required": ["x", "y"], "optional": ["color"],
             "help": "Tren nilai Y terhadap X (diagregasi per nilai X)."},
    "Area": {"group": "Hubungan", "required": ["x", "y"], "optional": ["color"],
             "help": "Seperti line chart, area di bawahnya diwarnai dan ditumpuk per warna."},
    "Scatter matrix": {"group": "Hubungan", "required": ["dims"], "optional": ["color"],
                       "help": "Semua pasangan scatter dari beberapa kolom numerik (pairplot)."},
    "Parallel coordinates": {"group": "Hubungan", "required": ["dims"], "optional": ["color"],
                             "help": "Pola banyak variabel numerik sekaligus per baris."},
    "Scatter 3D": {"group": "Hubungan", "required": ["x", "y", "z"], "optional": ["color", "size"],
                   "help": "Tiga variabel numerik dalam ruang 3D (bisa diputar)."},
    "Bar": {"group": "Komposisi", "required": ["x"], "optional": ["y", "color"],
            "help": "Agregasi Y per kategori X; tanpa Y = jumlah baris."},
    "Pie / Donut": {"group": "Komposisi", "required": ["names"], "optional": ["values"],
                    "help": "Porsi tiap kategori terhadap total."},
    "Treemap": {"group": "Komposisi", "required": ["path"], "optional": ["values", "color"],
                "help": "Komposisi hierarkis beberapa kolom kategorikal."},
    "Sunburst": {"group": "Komposisi", "required": ["path"], "optional": ["values"],
                 "help": "Komposisi hierarkis berbentuk cincin."},
    "Heatmap pivot": {"group": "Matriks", "required": ["x", "y"], "optional": ["z"],
                      "help": "Agregasi Z untuk setiap kombinasi kategori X × Y."},
}

NUMERIC_FIELDS = {"Box plot": ["y"], "Violin": ["y"], "Strip (jitter)": ["y"], "ECDF": ["x"],
                  "Density 2D": ["x", "y"], "Scatter / Bubble": ["y", "size"], "Line": ["y"], "Area": ["y"],
                  "Scatter 3D": ["x", "y", "z", "size"], "Bar": ["y"], "Pie / Donut": ["values"],
                  "Treemap": ["values"], "Sunburst": ["values"], "Heatmap pivot": ["z"]}


def _agg(df, keys, value, how):
    if value is None or how == "count":
        return df.groupby(keys, dropna=False).size().reset_index(name="Jumlah"), "Jumlah"
    out = df.groupby(keys, dropna=False)[value].agg(how).reset_index()
    return out, value


def add_trendline(fig, df, x, y):
    data = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 2 or data[x].nunique() < 2:
        return
    slope, intercept = np.polyfit(data[x], data[y], 1)
    xs = np.linspace(data[x].min(), data[x].max(), 50)
    r = data[x].corr(data[y])
    fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name=f"Tren linear (r = {r:.2f})",
                             line=dict(color="#0F172A", dash="dash", width=2)))


def make(kind: str, df: pd.DataFrame, p: dict) -> go.Figure:
    x, y, z, color = p.get("x"), p.get("y"), p.get("z"), p.get("color")
    facet, size = p.get("facet"), p.get("size")
    agg = AGGREGATIONS.get(p.get("agg", "Rata-rata"), "mean")
    title = p.get("title") or kind
    wrap = 3 if facet else None

    if kind == "Histogram":
        fig = px.histogram(df, x=x, color=color, facet_col=facet, facet_col_wrap=wrap, nbins=p.get("nbins"),
                           marginal=p.get("marginal"), histnorm=p.get("histnorm"), barmode="overlay",
                           opacity=0.75 if color else 1)
    elif kind == "Box plot":
        fig = px.box(df, x=x, y=y, color=color, points=p.get("points", "outliers"), notched=p.get("notched", False))
    elif kind == "Violin":
        fig = px.violin(df, x=x, y=y, color=color, box=True, points=p.get("points", False))
    elif kind == "Strip (jitter)":
        fig = px.strip(df, x=x, y=y, color=color)
    elif kind == "ECDF":
        fig = px.ecdf(df, x=x, color=color)
    elif kind == "Density 2D":
        fig = px.density_heatmap(df, x=x, y=y, nbinsx=p.get("nbins"), nbinsy=p.get("nbins"),
                                 marginal_x="histogram", marginal_y="histogram")
    elif kind == "Scatter / Bubble":
        data = df.dropna(subset=[size]) if size else df
        fig = px.scatter(data, x=x, y=y, color=color, size=size, facet_col=facet, facet_col_wrap=wrap,
                         opacity=p.get("opacity", 0.8), hover_data=data.columns[:8])
        if p.get("trend"):
            add_trendline(fig, data, x, y)
    elif kind in ("Line", "Area"):
        keys = [x] + ([color] if color and color != x else [])
        data, value = _agg(df, keys, y, agg)
        data = data.sort_values(x)
        fn = px.line if kind == "Line" else px.area
        fig = fn(data, x=x, y=value, color=color, markers=kind == "Line")
    elif kind == "Scatter matrix":
        fig = px.scatter_matrix(df, dimensions=p["dims"], color=color, opacity=0.7)
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        fig.update_layout(height=max(500, 150 * len(p["dims"])))
    elif kind == "Parallel coordinates":
        data = df.copy()
        color_col = color
        if color and not pd.api.types.is_numeric_dtype(data[color]):
            data["_kode"] = data[color].astype("category").cat.codes
            color_col = "_kode"
        fig = px.parallel_coordinates(data, dimensions=p["dims"], color=color_col,
                                      color_continuous_scale=px.colors.sequential.Plasma)
    elif kind == "Scatter 3D":
        data = df.dropna(subset=[size]) if size else df
        fig = px.scatter_3d(data, x=x, y=y, z=z, color=color, size=size, opacity=0.8)
        fig.update_layout(height=600)
    elif kind == "Bar":
        keys = [x] + ([color] if color and color != x else [])
        data, value = _agg(df, keys, y, agg)
        data[x] = data[x].astype(str)
        if p.get("sort", True):
            order = data.groupby(x)[value].sum().sort_values(ascending=False)
            data = data[data[x].isin(order.index[:p.get("top", 30)])]
            category_orders = {x: list(order.index[:p.get("top", 30)])}
        else:
            category_orders = None
        horizontal = p.get("horizontal")
        fig = px.bar(data, x=value if horizontal else x, y=x if horizontal else value, color=color,
                     orientation="h" if horizontal else "v", barmode=p.get("barmode", "group"),
                     category_orders=category_orders, text_auto=".3s")
    elif kind == "Pie / Donut":
        data, value = _agg(df, [p["names"]], p.get("values"), "sum" if p.get("values") else "count")
        top = data.nlargest(p.get("top", 12), value)
        rest = data[value].sum() - top[value].sum()
        if rest > 0:
            top = pd.concat([top, pd.DataFrame({p["names"]: ["Lainnya"], value: [rest]})])
        fig = px.pie(top, names=p["names"], values=value, hole=p.get("hole", 0.45))
        fig.update_traces(textposition="inside", textinfo="percent+label")
    elif kind in ("Treemap", "Sunburst"):
        data = df.copy()
        for col in p["path"]:
            data[col] = data[col].astype(object).where(data[col].notna(), "(kosong)").astype(str)
        values = p.get("values")
        if values:
            data = data[data[values] > 0]
        fn = px.treemap if kind == "Treemap" else px.sunburst
        fig = fn(data, path=p["path"], values=values, color=color if kind == "Treemap" else None)
        fig.update_layout(height=560)
    elif kind == "Heatmap pivot":
        data, value = _agg(df, [y, x], z, agg if z else "count")
        pivot = data.pivot(index=y, columns=x, values=value)
        fig = px.imshow(pivot, text_auto=".3s", aspect="auto", color_continuous_scale="Blues",
                        labels={"color": value})
    else:
        raise ValueError(f"Jenis grafik tidak dikenal: {kind}")

    if p.get("log_x"):
        fig.update_xaxes(type="log")
    if p.get("log_y"):
        fig.update_yaxes(type="log")
    fig.update_layout(title=title)
    return fig


def auto_dashboard(df: pd.DataFrame, numeric, categorical, limit=12):
    """Grafik ringkas otomatis: histogram tiap numerik dan bar tiap kategorikal."""
    figs = []
    for col in numeric[:limit]:
        fig = px.histogram(df, x=col, nbins=30, title=col, marginal="box")
        fig.update_layout(height=280, showlegend=False)
        figs.append(fig)
    for col in categorical[:limit]:
        counts = df[col].astype(str).value_counts().head(15)
        fig = px.bar(x=counts.values, y=counts.index, orientation="h", title=col,
                     labels={"x": "Jumlah", "y": ""})
        fig.update_layout(height=280, yaxis={"categoryorder": "total ascending"})
        figs.append(fig)
    return figs
