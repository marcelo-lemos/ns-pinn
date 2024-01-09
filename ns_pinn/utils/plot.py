import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


def make_html_plot(target_velocity, target_pressure, pred_velocity, pred_pressure, file):
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.12, subplot_titles=(
        "Velocity - Ground Truth", "Velocity - Predicted", "Pressure - Ground Truth", "Pressure - Predicted"))

    for step in range(len(target_velocity)):
        fig.add_trace(go.Heatmap(
            visible=False, z=target_velocity[step], coloraxis="coloraxis1"), row=1, col=1)
        fig.add_trace(go.Heatmap(
            visible=False, z=pred_velocity[step], coloraxis="coloraxis1"), row=1, col=2)
        fig.add_trace(go.Heatmap(
            visible=False, z=target_pressure[step], coloraxis="coloraxis2"), row=2, col=1)
        fig.add_trace(go.Heatmap(
            visible=False, z=pred_pressure[step], coloraxis="coloraxis2"), row=2, col=2)

    fig.update_layout(title="Navier-Stokes PINN - Frame 0")
    fig.data[0].visible = True
    fig.data[1].visible = True
    fig.data[2].visible = True
    fig.data[3].visible = True

    steps = []
    for i in range(len(fig.data)//4):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Navier-Stokes PINN - Frame {i}"}],
            label=f"{i}"
        )
        step["args"][0]["visible"][(4*i):(4*i)+3] = [True, True, True, True]
        steps.append(step)

    sliders = [dict(
        currentvalue={"prefix": "Frame: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        height=600,
        width=580,
        coloraxis1={
            "colorscale": "jet",
            'cmin': 0,
            'cmax': 1,
            'colorbar': {'y': 0.775, 'len': 0.5}
        },
        coloraxis2={
            "colorscale": "jet",
            'cmin': -450,
            'cmax': 450,
            'colorbar': {'y': 0.225, 'len': 0.5}
        }, title_x=0.5)

    fig.write_html(file, include_plotlyjs='cdn', full_html=True)


class MakePlotCallback(Callback):
    def __init__(self, data_file, predictions_file, html_file) -> None:
        super().__init__()
        self.data_file = data_file
        self.predictions_file = predictions_file
        self.html_file = html_file

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        target_df = pl.read_csv(self.data_file, has_header=False).to_numpy()
        pred_df = pl.read_csv(self.predictions_file, has_header=False).to_numpy()
        
        target_velocity = np.sqrt(
            np.square(target_df[:, 3].reshape(-1, 128, 128)) +
            np.square(target_df[:, 4].reshape(-1, 128, 128)))
        pred_velocity = np.sqrt(
            np.square(pred_df[:, 0].reshape(-1, 128, 128)) +
            np.square(pred_df[:, 1].reshape(-1, 128, 128)))
        target_pressure = target_df[:, 5].reshape(-1, 128, 128)
        pred_pressure = pred_df[:, 2].reshape(-1, 128, 128)

        make_html_plot(
            target_velocity,
            target_pressure,
            pred_velocity,
            pred_pressure,
            self.html_file
        )
