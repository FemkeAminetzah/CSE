import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html


def calc_curve(
    a1: float,
    a2: float,
    w1: float,
    w2: float,
    nsteps: int = 100,
    xmin=-5,
    xmax=5,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenating two sigmoids at x=0"""
    left = a1 / (1 + np.exp(-w1 * np.linspace(xmin, 0, nsteps)))
    right = a2 / (1 + np.exp(-w2 * np.linspace(0, xmax, nsteps)))

    # to connect the right side to the left, we need to match height at 0
    hzero_l = a1 / 2
    hzero_r = a2 / 2
    right -= hzero_r - hzero_l
    right = right[
        1:
    ]  # the first point would be the value at 0 which we choose to take from left

    return np.linspace(xmin, xmax, (nsteps * 2 - 1)), np.concatenate(
        [left, right]
    )


app = Dash(__name__)

app.layout = html.Div(
    [
        html.Div(id="graph_div", children=[]),
        html.Div(
            id="slider1_div",
            children=[
                html.Div("a1:"),
                # value is the default value
                dcc.Slider(id="slider_a1", min=0, max=10, step=0.1, value=1),
                html.Div("a2:"),
                dcc.Slider(id="slider_a2", min=0, max=10, step=0.1, value=1),
                html.Div("w1:"),
                dcc.Slider(id="slider_w1", min=0, max=10, step=0.1, value=1),
                html.Div("w2:"),
                dcc.Slider(id="slider_w2", min=0, max=10, step=0.1, value=1),
            ],
        ),
    ]
)


@callback(
    Output("graph_div", "children"),
    [
        Input("slider_a1", "value"),
        Input("slider_a2", "value"),
        Input("slider_w1", "value"),
        Input("slider_w2", "value"),
    ],
)
def update_graph(a1: float, a2: float, w1: float, w2: float):
    """
    Calculate according to new parameters. Note: We recreate the whole
    graph as it is small anyways, no need to optimize for performance.
    """

    x, y = calc_curve(a1, a2, w1, w2)
    fig = go.Figure(data=[go.Scatter(x=x, y=y, line_color="#35f")])

    # Some layout
    fig.update_yaxes(title="Sigmoid", range=[0, 3])
    fig.update_xaxes(title="x")

    graph = dcc.Graph(figure=fig, id="graph")

    return graph


if __name__ == "__main__":
    app.run(debug=True)