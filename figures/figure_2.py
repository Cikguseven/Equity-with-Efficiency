import plotly.graph_objects as go

languages = ["en", "id", "th", "vi", "zh", "ms", "ta", "tl", "my", "km", "lo"]
vocab_sizes = ["45k", "90k", "135k"]

z_data = [
    [38326, 44857, 44990, 44973, 44982, 44976, 44994, 44970, 44994, 44995, 44980],
    [32873, 41687, 41773, 41756, 41767, 41723, 41775, 41775, 41776, 41776, 41776],
    [31685, 40676, 40742, 40744, 40744, 40744, 40747, 40747, 40747, 40748, 40748],
]

z_transposed = [[z_data[j][i] for j in range(len(vocab_sizes))] for i in range(len(languages))]

fig = go.Figure(go.Heatmap(
    z=z_transposed,
    x=vocab_sizes,
    y=languages,
    colorscale="Viridis",
    colorbar=dict(title="Tokens"),
    reversescale=True,
    xgap=2,
    ygap=2,
))

fig.update_layout(
    title={
        "text": (
            "<b>Token Count by Vocab. Size & Language</b><br>"
            "<span style='font-size:15px;font-weight:normal;'>"
            "Parity-aware BPE | FLORES+ dataset</span>"
        ),
        "x": 0.5,
        "xanchor": "center",
    },
    margin=dict(l=60, r=65, t=75, b=55),
    width=500,
    height=600,
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
)
fig.update_xaxes(title_text="Vocab. Size", side="bottom")
fig.update_yaxes(title_text="Language", tickfont=dict(size=12))

fig.write_image("z2.png", scale=2)
print("Done")