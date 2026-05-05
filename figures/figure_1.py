import os

import plotly.graph_objects as go

data = [
    {"name": "BLT",             "vocab": 4.5,    "compression": 0.0127, "gini": 0.212},
    {"name": "BLT",             "vocab": 6,      "compression": 0.0145, "gini": 0.219},
    {"name": "BLT",             "vocab": 8,      "compression": 0.0161, "gini": 0.227},
    {"name": "MYTE",            "vocab": 45056,  "compression": 0.0085, "gini": 0.085},
    {"name": "MYTE",            "vocab": 90112,  "compression": 0.0089, "gini": 0.086},
    {"name": "MYTE",            "vocab": 135168, "compression": 0.0089, "gini": 0.095},
    {"name": "Byte-level BPE",  "vocab": 45056,  "compression": 0.0257, "gini": 0.243},
    {"name": "Byte-level BPE",  "vocab": 90112,  "compression": 0.0293, "gini": 0.220},
    {"name": "Byte-level BPE",  "vocab": 135168, "compression": 0.0314, "gini": 0.203},
    {"name": "Parity-aware BPE","vocab": 45056,  "compression": 0.0250, "gini": 0.021},
    {"name": "Parity-aware BPE","vocab": 90112,  "compression": 0.0272, "gini": 0.028},
    {"name": "Parity-aware BPE","vocab": 135168, "compression": 0.0280, "gini": 0.029},
]

families = ["BLT", "MYTE", "Byte-level BPE", "Parity-aware BPE"]
colors_map = {"BLT": "#636EFA", "MYTE": "#EF553B", "Byte-level BPE": "#00CC96", "Parity-aware BPE": "#AB63FA"}
symbols_map = {"BLT": "diamond", "MYTE": "circle", "Byte-level BPE": "square", "Parity-aware BPE": "triangle-up"}

textpos_map = {
    "BLT":             ["top center", "top center", "top center"],
    "MYTE":            ["middle left", "bottom right", "top center"],
    "Byte-level BPE":  ["top center", "top center", "top center"],
    "Parity-aware BPE":["top center", "top center", "middle right"],
}

def pareto_front(points):
    sorted_pts = sorted(enumerate(points), key=lambda x: -x[1][0])
    front_indices, min_gini = [], float('inf')
    for idx, (comp, gini) in sorted_pts:
        if gini < min_gini:
            min_gini = gini
            front_indices.append(idx)
    return sorted(front_indices, key=lambda i: points[i][0])

all_points = [(d["compression"], d["gini"]) for d in data]
pf_indices = pareto_front(all_points)
pf_x = [all_points[i][0] for i in pf_indices]
pf_y = [all_points[i][1] for i in pf_indices]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=pf_x, y=pf_y,
    mode='lines',
    line=dict(color='rgba(140,140,140,0.7)', dash='dot', width=2),
    name='Pareto Front',
    showlegend=True
))

for family in families:
    fd = [d for d in data if d["name"] == family]
    xs = [d["compression"] for d in fd]
    ys = [d["gini"] for d in fd]
    text_labels = [f"{d['vocab']//1000}K" if d["vocab"] > 10000 else d['vocab'] for d in fd]
    hover = [
        f"{d['name']} ({d['vocab']:,})<br>Compression: {d['compression']:.4f}<br>Gini: {d['gini']:.3f}"
        if d["vocab"] else
        f"BLT<br>Compression: {d['compression']:.4f}<br>Gini: {d['gini']:.3f}"
        for d in fd
    ]
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='markers+text',
        name=family,
        marker=dict(
            symbol=symbols_map[family],
            size=16,
            color=colors_map[family],
            line=dict(width=1.5, color='white')
        ),
        text=text_labels,
        textposition=textpos_map[family],
        textfont=dict(size=14),
        hovertext=hover,
        hoverinfo="text",
    ))

# Arrow pointing bottom-right (high compression, low gini = better)
# Anchor annotation at upper-left area, arrow points toward bottom-right
fig.add_annotation(
    x=0.031, y=0.030,       # arrowhead tip: bottom-right (high comp, high gini area to lead toward ideal)
    ax=0.022, ay=0.120,     # arrow tail: upper-left of the tip
    xref="x", yref="y", axref="x", ayref="y",
    text="Ideal direction",
    showarrow=True,
    arrowhead=2, arrowsize=1.2, arrowcolor="gray",
    font=dict(size=11, color="black"),
    bgcolor="rgba(255,255,255,0.9)",
)

fig.update_layout(
    font=dict(color="black"),
    title={
        "text": "<b>Efficiency versus Equity Trade-off</b><br><span style='font-size:16px;font-weight:normal;'>Higher compression + lower Gini = better. Ideal direction: bottom-right.</span>",
        "x": 0.5, "xanchor": "center",
    },
    legend=dict(
        orientation='v', yanchor='middle', y=0.5,
        xanchor='left', x=1.02,
        bgcolor='rgba(255,255,255,0.85)', bordercolor='lightgrey', borderwidth=1
    ),
    xaxis=dict(
        title_text="Compression Rate (higher = better)",
        showgrid=True, gridcolor='rgba(200,200,200,0.4)',
        tickformat=".3f",
        title_font_size=16,
    ),
    yaxis=dict(
        title_text="Gini Coefficient (lower = better)",
        type='linear',
        showgrid=True, gridcolor='rgba(200,200,200,0.4)',
        tickformat=".2f",
        title_font_size=16,
    ),
    width=900,
    height=600,
    plot_bgcolor='#edf2f7',
)
fig.update_traces(cliponaxis=False)

OUT_DIR = "plot-pareto"
os.makedirs(OUT_DIR, exist_ok=True)
fig.write_image(f"{OUT_DIR}/z0.png", scale=3)
print("done")