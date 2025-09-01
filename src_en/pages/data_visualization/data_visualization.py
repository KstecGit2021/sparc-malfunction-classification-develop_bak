import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import taipy.gui.builder as tgb

from state_class import State


graph_selector: list[str] = ["Scatter", "Histogram"]
graph_selected: str = graph_selector[0]

scatter: go.Figure = None
histogram: go.Figure = None


def create_scatter(
    dataset: pd.DataFrame, x_selected: str, y_selected: str, force_ratio=False
):
    dataset["PASS/FAIL"] = dataset["PASS/FAIL"].astype(int)
    dataset["PASS/FAIL"] = dataset["PASS/FAIL"].map({0: "Pass", 1: "Fail"})
    if force_ratio:
        fig = px.scatter(
            dataset,
            x=x_selected,
            y=y_selected,
            color="PASS/FAIL",
            hover_data=["RADIUS", "SENSOROFFSETHOT-COLD", "PASS/FAIL"],
            title="Die Position and Failure",
        )
        fig.update_layout(
            width=700,
            height=700,
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
        )
    else:
        fig = px.scatter(
            dataset,
            x=x_selected,
            y=y_selected,
            color="PASS/FAIL",
            marginal_x="histogram",
            marginal_y="violin",
            hover_data=["RADIUS", "SENSOROFFSETHOT-COLD", "PASS/FAIL"],
            title="2D Distribution of Fail",
        )
    return fig


def create_histogram(dataset: pd.DataFrame, x_selected: str):
    dataset["PASS/FAIL"] = dataset["PASS/FAIL"].astype(int)
    dataset["PASS/FAIL"] = dataset["PASS/FAIL"].map({0: "Pass", 1: "Fail"})
    return px.histogram(
        dataset,
        x=x_selected,
        color="PASS/FAIL",
        marginal="box",
        hover_data=["RADIUS", "SENSOROFFSETHOT-COLD", "PASS/FAIL"],
        title="Fail distribution",
        barmode="overlay",
    )


def on_change_data_visualization(state: State):
    state.scatter = create_scatter(
        state.train_dataset.copy(), state.x_selected, state.y_selected
    )
    state.histogram = create_histogram(state.train_dataset.copy(), state.x_selected)


# data_visualization = Markdown("pages/data_visualization/data_visualization.md")


with tgb.Page() as data_visualization:
    tgb.text("# *Malfunction Classification* - Data **Visualization**", mode="md")

    with tgb.part("container card"):
        tgb.text(
            """
**Malfunction** is when a semiconductor fails to operate as intended or stops working altogether.

Ensuring reliability is a **top priority** because preventing failures is often more cost-effective than dealing with the aftermath of a breakdown.

By **identifying semiconductors** that are likely to fail early, using input data to spot warning signs and applying preventive strategies, manufacturers can lower malfunction rates and enhance product performance.          
""",
            mode="md",
        )

    tgb.text(
        """Visualize the historical dataset and the distribution of churn with histogram and scatter plots.""",
        mode="md",
    )

    tgb.toggle("{graph_selected}", lov="{graph_selector}")

    tgb.html("hr")

    with tgb.expandable("Position analysis", expanded=False):
        with tgb.layout("1 1"):
            tgb.chart(
                figure=lambda preprocessed_dataset: create_scatter(
                    preprocessed_dataset, "X", "Y", force_ratio=True
                ),
                height="700px",
            )

    with tgb.part(render=lambda graph_selected: graph_selected == "Scatter"):
        tgb.text("### Scatter Plot", mode="md")
        tgb.text("Choose two columns to see the 2D Fail distribution (Pass or Fail).")
        with tgb.layout("1 1 1"):
            tgb.selector(
                "{x_selected}",
                lov="{select_x}",
                dropdown=True,
                label="X-axis",
                filter=True,
            )
            tgb.selector(
                "{y_selected}",
                lov="{select_y}",
                dropdown=True,
                label="Y-axis",
                filter=True,
            )
        tgb.chart(figure="{scatter}", height="700px")

    with tgb.part(render=lambda graph_selected: graph_selected == "Histogram"):
        tgb.text("### Histogram", mode="md")
        tgb.text("Choose one column to see the Fail distribution (Pass or Fail).")
        tgb.html("br")
        with tgb.layout("1 1 1"):
            tgb.selector(
                "{x_selected}",
                lov="{select_x}",
                dropdown=True,
                label="X-axis",
                filter=True,
            )
        tgb.chart(figure="{histogram}", height="700px")
