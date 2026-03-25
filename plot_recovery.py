from pathlib import Path

import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig_dir = Path("figures/")
fig_dir.mkdir(exist_ok=True)

out_dir = Path("results/")
files = out_dir.glob("recovery_*.joblib")
fig = make_subplots(
    rows=5,
    cols=5,
    vertical_spacing=0.02,
    horizontal_spacing=0.02,
)
for file in files:
    i, j = file.stem.removeprefix("recovery_").split("_")
    i, j = int(i), int(j)
    data = joblib.load(file)
    lr = np.ravel(data["samples"]["lr"])
    beta_rt = np.ravel(data["samples"]["beta_rt"])
    true_lr = data["params"]["lr"]
    true_beta_rt = data["params"]["beta_rt"]
    fig.add_histogram2d(
        x=lr,
        y=beta_rt,
        row=i + 1,
        col=j + 1,
        showscale=False,
        xbins={"start": 0, "size": 1},
        bingroup=1,
    )
    mean_lr = data["summary"]["lr"]["mean"]
    mean_beta_rt = data["summary"]["beta_rt"]["mean"]
    fig.add_scatter(
        x=[mean_lr],
        y=[mean_beta_rt],
        row=i + 1,
        col=j + 1,
        error_x=dict(
            type="data",
            symmetric=False,
            array=[data["summary"]["lr"]["95.0%"] - mean_lr],
            arrayminus=[mean_lr - data["summary"]["lr"]["5.0%"]],
            color="white",
            thickness=3,
            width=3,
        ),
        error_y=dict(
            type="data",
            symmetric=False,
            array=[data["summary"]["beta_rt"]["95.0%"] - mean_beta_rt],
            arrayminus=[mean_beta_rt - data["summary"]["beta_rt"]["5.0%"]],
            color="white",
            thickness=3,
            width=3,
        ),
        marker=dict(
            symbol=None, color="white", size=12, line=dict(color="black", width=1)
        ),
        mode="markers",
        showlegend=False,
    )
    fig.add_scatter(
        x=[true_lr],
        y=[true_beta_rt],
        row=i + 1,
        col=j + 1,
        marker=dict(
            size=12, symbol="x", color="black", line=dict(color="white", width=1)
        ),
        mode="markers",
        showlegend=False,
    )
fig = fig.update_xaxes(matches="x")
fig = fig.update_yaxes(matches="y")
fig = fig.update_layout(
    template="plotly_white", margin=dict(l=5, r=5, b=5, t=5), width=1000, height=800
)
fig.show()


out_dir = Path("results/")
files = out_dir.glob("*recovery*.joblib")
fig = make_subplots(
    rows=2,
    cols=3,
    vertical_spacing=0.1,
    horizontal_spacing=0.01,
    subplot_titles=[f"$\\gamma={beta_rt:.2f}$" for beta_rt in np.linspace(0.1, 2.0, 5)]
    + ["No Reaction Times"],
)
records = []
colors = px.colors.qualitative.Bold
for file in files:
    if file.stem.startswith("lr-only-recovery_"):
        data = joblib.load(file)
        summary = data["summary"]
        true_lr = data["params"]["lr"]
        lr = np.ravel(data["samples"]["lr"])
        j = int(file.stem.removeprefix("lr-only-recovery_"))
        # fig.add_box(
        #     x0=true_lr,
        #     y=lr,
        #     row=2,
        #     col=3,
        #     showlegend=False,
        # )
        mean_lr = summary["lr"]["median"]
        fig.add_scatter(
            x=[true_lr],
            y=[mean_lr],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[summary["lr"]["95.0%"] - mean_lr],
                arrayminus=[mean_lr - summary["lr"]["5.0%"]],
                thickness=3,
                width=3,
            ),
            mode="markers",
            marker=dict(color=colors[j]),
            showlegend=False,
            row=2,
            col=3,
        )
        if j == 0:
            fig.add_scatter(
                x=[0, 2],
                y=[0, 2],
                row=2,
                col=3,
                line=dict(color="black", width=2, dash="dash"),
                showlegend=False,
                name="Ideal",
                mode="lines",
            )
        continue
    i, j = file.stem.removeprefix("recovery_").split("_")
    i, j = int(i), int(j)
    data = joblib.load(file)
    summary = data["summary"]
    true_lr = data["params"]["lr"]
    lr = np.ravel(data["samples"]["lr"])
    row = (i // 3) + 1
    col = (i % 3) + 1
    mean_lr = summary["lr"]["median"]
    fig.add_scatter(
        x=[true_lr],
        y=[mean_lr],
        error_y=dict(
            type="data",
            symmetric=False,
            array=[summary["lr"]["95.0%"] - mean_lr],
            arrayminus=[mean_lr - summary["lr"]["5.0%"]],
            thickness=3,
            width=3,
        ),
        mode="markers",
        marker=dict(color=colors[j]),
        showlegend=False,
        row=row,
        col=col,
    )
    # fig.add_box(
    #    x0=true_lr,
    #    y=lr,
    #    row=row,
    #    col=col,
    #    showlegend=False,
    # )
    if j == 0:
        fig.add_scatter(
            x=[0, 2],
            y=[0, 2],
            row=row,
            col=col,
            line=dict(color="black", width=2, dash="dash"),
            showlegend=False,
            name="Ideal",
            mode="lines",
        )
fig = fig.update_layout(
    template="plotly_white", margin=dict(l=5, r=5, b=5, t=20), width=1100, height=500
)
fig.show()

fig.write_image(fig_dir.joinpath("recovery.png"), scale=2)
