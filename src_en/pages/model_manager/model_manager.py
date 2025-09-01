import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
import shap
import tempfile

import taipy.gui.builder as tgb
from state_class import State
import matplotlib.pyplot as plt

graph_selector_scenario: list[str] = ["Metrics", "Features", "Scatter", "Histogram"]
graph_selected_scenario: str = graph_selector_scenario[0]

pie_color_dict_2: dict[str, list[str]] = {"piecolorway": ["#00D08A", "#FE913C"]}
pie_color_dict_4: dict[str, list[str]] = {
    "piecolorway": ["#00D08A", "#81F1A0", "#F3C178", "#FE913C"]
}

margin_features: dict[str, dict[str, int]] = {"margin": {"l": 150}}

metrics: pd.DataFrame = None
score_table: pd.DataFrame = None
values: pd.DataFrame = None

scatter_for_prediction: go.Figure = None
histogram_for_prediction: go.Figure = None

accuracy_pie: pd.DataFrame = None
distrib_class: pd.DataFrame = None
pie_confusion_matrix: pd.DataFrame = None
features_table: pd.DataFrame = None

forecast_series: pd.Series = None
shap_values, X = None, None


def create_scatter_for_prediction(
    test_dataset: pd.DataFrame,
    true_false_positive_negative_series: pd.Series,
    x_selected: str,
    y_selected: str,
):
    test_dataset["True/False/Positive/Negative"] = true_false_positive_negative_series

    return px.scatter(
        test_dataset,
        x=x_selected,
        y=y_selected,
        color="True/False/Positive/Negative",
        marginal_x="histogram",
        marginal_y="violin",
        hover_data=["RADIUS", "SENSOROFFSETHOT-COLD", "PASS/FAIL"],
        title="2D Distribution of Fail for Predictions",
    )


def create_histogram_for_prediction(
    test_dataset: pd.DataFrame,
    true_false_positive_negative_series: pd.Series,
    x_selected: str,
):
    test_dataset["True/False/Positive/Negative"] = true_false_positive_negative_series

    return px.histogram(
        test_dataset,
        x=x_selected,
        color="True/False/Positive/Negative",
        marginal="box",
        hover_data=["RADIUS", "SENSOROFFSETHOT-COLD", "PASS/FAIL"],
        title="Fail distribution for Predictions",
        barmode="overlay",
    )


def change_distribution_chart_prediction(state: State):
    state.scatter_for_prediction = create_scatter_for_prediction(
        state.test_dataset.copy(),
        state.values["True/False/Positive/Negative"],
        state.x_selected,
        state.y_selected,
    )
    state.histogram_for_prediction = create_histogram_for_prediction(
        state.test_dataset.copy(),
        state.values["True/False/Positive/Negative"],
        state.x_selected,
    )


def on_change_model_manager(state: State):
    model_type = state.algorithm_selected.lower().replace(" ", "_")
    state.values = state.scenario.data_nodes[f"results_{model_type}"].read()
    state.threshold = state.scenario.data_nodes[f"threshold_{model_type}"].read() * 1000
    state.forecast_series = state.values["Forecast"]

    state.metrics = state.scenario.data_nodes[f"metrics_{model_type}"].read()
    state.shap_values, state.X = state.scenario.data_nodes[
        f"shap_values_{model_type}"
    ].read()

    state.roc_dataset = state.scenario.data_nodes[f"roc_data_{model_type}"].read()
    state.features_table = state.scenario.data_nodes[
        f"feature_importance_{model_type}"
    ].read()

    state.accuracy_pie = pd.DataFrame(
        {
            "values": [
                state.metrics["number_of_good_predictions"],
                state.metrics["number_of_false_predictions"],
            ],
            "labels": ["Correct predictions", "False predictions"],
        }
    )

    state.distrib_class = pd.DataFrame(
        {
            "values": [
                len(state.values[state.values["Historical"] == 0]),
                len(state.values[state.values["Historical"] == 1]),
            ],
            "labels": ["Stayed", "PASS/FAIL"],
        }
    )

    state.score_table = pd.DataFrame(
        {
            "Score": ["Predicted Pass", "Predicted Fail"],
            "Pass": [
                state.metrics["dict_ftpn"]["tn"],
                state.metrics["dict_ftpn"]["fp"],
            ],
            "Fail": [
                state.metrics["dict_ftpn"]["fn"],
                state.metrics["dict_ftpn"]["tp"],
            ],
        }
    )

    state.pie_confusion_matrix = pd.DataFrame(
        {
            "values": [
                state.metrics["dict_ftpn"]["tp"],
                state.metrics["dict_ftpn"]["tn"],
                state.metrics["dict_ftpn"]["fp"],
                state.metrics["dict_ftpn"]["fn"],
            ],
            "labels": [
                "True Positive",
                "True Negative",
                "False Positive",
                "False Negative",
            ],
        }
    )

    change_distribution_chart_prediction(state)


# model_manager = Markdown("pages/model_manager/model_manager.md")

with tgb.Page() as model_manager:
    tgb.text(
        "# *Malfunction Classification* - **Model** Manager",
        mode="md",
    )

    tgb.text(
        """
Two models have been created to predict semiconductor malfunction: a Baseline model and a Machine Learning model. 
Both models use the same dataset but may differ in overall performance and in which features they consider most important.

You can check:

- Their overall metrics with ROC curves, confusion matrices, and scores,
- Their feature importance to know what influences their decisions,
- Their predictions on the test set and where the wrong predictions occur.
""",
        mode="md",
    )

    with tgb.layout("3 2 2 2", class_name="align-columns-center"):
        tgb.toggle("{graph_selected_scenario}", lov="{graph_selector_scenario}")

        tgb.selector(
            "{algorithm_selected}",
            lov="{algorithm_selector}",
            dropdown=True,
            label="Algorithm",
            filter=True,
        )

        tgb.button("Show ROC", on_action=lambda s: s.assign("show_roc", True))

        tgb.text(
            "**Number of predictions:** {metrics.number_of_predictions}", mode="md"
        )

    tgb.html("hr")


    with tgb.part(
        "Metrics",
        render=lambda graph_selected_scenario: graph_selected_scenario == "Metrics",
    ):
        tgb.text(
            """
Three metrics are displayed for each model: **Accuracy**, **AUC**, and **F1**‐score.
A perfect model should ideally have a score of 1.
""",
            mode="md",
        )

        tgb.text("### Metrics", mode="md")

        with tgb.layout("1 1 1", columns_mobile="1"):
            with tgb.part("text-center"):
                tgb.indicator(
                    "{metrics.accuracy}", value="{metrics.accuracy}", min=0, max=1
                )
                tgb.text("**Model Accuracy**", mode="md")

                tgb.chart(
                    "{accuracy_pie}",
                    title="Accuracy of Predictions Model",
                    values="values",
                    labels="labels",
                    type="pie",
                    layout="{pie_color_dict_2}",
                )

            with tgb.part("text-center"):
                tgb.indicator(
                    "{metrics.auc_score}", value="{metrics.auc_score}", min=0, max=1
                )
                tgb.text("**Model AUC**", mode="md")

                tgb.chart(
                    "{pie_confusion_matrix}",
                    title="Confusion Matrix",
                    values="values",
                    labels="labels",
                    type="pie",
                    layout="{pie_color_dict_4}",
                )

            with tgb.part("text-center"):
                tgb.indicator(
                    "{metrics.f1_score}", value="{metrics.f1_score}", min=0, max=1
                )
                tgb.text("**Model F1-score**", mode="md")

                tgb.chart(
                    "{distrib_class}",
                    title="Distribution between Working and Malfunction",
                    values="values",
                    labels="labels",
                    type="pie",
                    layout="{pie_color_dict_2}",
                )

    with tgb.part(
        "Features",
        render=lambda graph_selected_scenario: graph_selected_scenario == "Features",
    ):
        tgb.text("### Features", mode="md")
        tgb.text(
            """
Importance scores help you understand how the models reach their decisions.
They also provide insight into which factors are most critical in predicting malfunctions.
""",
            mode="md",
        )

        tgb.html("br")

        with tgb.layout("1 1", columns_mobile="1"):
            tgb.chart(
                "{features_table}",
                type="bar",
                y="Features",
                x="Importance",
                orientation="h",
                layout="{margin_features}",
                title="Features Importance",
                height="600px",
            )

            tgb.chart(
                "{features_table}",
                type="treemap",
                labels="Features",
                values="Importance_abs",
                layout="{margin_features}",
                title="Features Importance (Absolute)",
                height="600px",
            )


    with tgb.part(
        "Histogram",
        render=lambda graph_selected_scenario: graph_selected_scenario == "Histogram",
    ):
        tgb.text("### Histogram", mode="md")
        tgb.text(
            "Choose a column to compare correct vs. incorrect predictions.", mode="md"
        )
        tgb.html("br")

        tgb.selector(
            "{x_selected}",
            lov="{select_x}",
            dropdown=True,
            label="Select x",
            filter=True,
        )

        tgb.chart(figure="{histogram_for_prediction}", height="700px")


    with tgb.part(
        "Scatter",
        render=lambda graph_selected_scenario: graph_selected_scenario == "Scatter",
    ):
        tgb.text("### Scatter", mode="md")
        tgb.text(
            "Choose two columns to visualize the 2D distribution of correct vs. incorrect predictions."
        )
        tgb.html("br")

        with tgb.layout("1 2"):
            tgb.selector(
                "{x_selected}",
                lov="{select_x}",
                dropdown=True,
                label="Select x",
                filter=True,
            )
            tgb.selector(
                "{y_selected}",
                lov="{select_y}",
                dropdown=True,
                label="Select y",
                filter=True,
            )

        tgb.chart(figure="{scatter_for_prediction}", height="700px")
