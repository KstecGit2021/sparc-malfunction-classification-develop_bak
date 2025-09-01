import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

# from taipy.gui import Markdown
import taipy.gui.builder as tgb

from state_class import State


comparison_graph: go.Figure = None
all_metrics_df: pd.DataFrame = None


def compare_all_models(scenario, model_types: str):
    """This function creates the objects for the three plots of the compare models page"""

    def _get_metrics(scenario, model_type):
        return scenario.data_nodes[f"metrics_{model_type}"].read()

    metrics_list = [_get_metrics(scenario, model_type) for model_type in model_types]

    accuracies = [metrics["accuracy"] for metrics in metrics_list]
    f1_scores = [metrics["f1_score"] for metrics in metrics_list]
    scores_auc = [metrics["auc_score"] for metrics in metrics_list]
    recall = [metrics["recall"] for metrics in metrics_list]
    precision = [metrics["precision"] for metrics in metrics_list]
    names = [model_type.upper() for model_type in model_types]

    all_metrics_df = pd.DataFrame(
        {
            "Model": names,
            "Accuracy": accuracies,
            "F1 Score": f1_scores,
            "AUC Score": scores_auc,
            "Recall": recall,
            "Precision": precision,
        }
    )

    # 1. Melt the DataFrame to long format
    melted_df = pd.melt(
        all_metrics_df,
        id_vars="Model",
        value_vars=["Accuracy", "F1 Score", "AUC Score", "Precision", "Recall"],
        var_name="Metric",
        value_name="Value",
    )

    # 2. Create a grouped bar chart
    fig = px.bar(
        melted_df,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        title="Comparison of Metrics by Model",
        text="Value",
    )

    # Optional: Adjust layout and formatting
    fig.update_layout(
        xaxis_title="Model", yaxis_title="Metric Value", legend_title="Metric"
    )
    fig.update_traces(textposition="outside")
    return fig, all_metrics_df


def on_init_compare_models(state: State):
    state.comparison_graph, state.all_metrics_df = compare_all_models(
        state.scenario,
        ["baseline", "logistic_regression", "random_forest", "xgboost"],  # "tree",
    )


with tgb.Page() as compare_models:
    # Title and horizontal rule
    tgb.text("# *Malfunction Classification* - **Model** comparison", mode="md")
    tgb.html("hr")

    # Intro text
    tgb.text(
        "Check which model is best for predicting malfunctions. All metrics are compared for both models.",
        mode="md",
    )

    # Some spacing
    tgb.html("br")

    tgb.chart(figure="{comparison_graph}")
