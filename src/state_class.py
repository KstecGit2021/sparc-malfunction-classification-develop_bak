import pathlib
from taipy.gui import State
from taipy import Scenario
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class State(State):
    # Main
    # Read datasets
    train_dataset: pd.DataFrame
    test_dataset: pd.DataFrame
    roc_dataset: pd.DataFrame
    scenario: Scenario

    # Root
    # dialog
    show_roc: bool

    # Common variables to several pages
    algorithm_selector: list[str]
    algorithm_selected: str

    select_x: list[str]
    x_selected: str
    select_y: list[str]
    y_selected: str

    # Data Vizualization
    graph_selector: list[str]
    graph_selected: str

    scatter: go.Figure
    histogram: go.Figure

    # Compare models
    accuracy_graph: go.Figure
    f1_score_graph: go.Figure
    auc_score_graph: go.Figure
    all_metrics_df: pd.DataFrame

    # Model Manager
    graph_selector_scenario: list[str]
    graph_selected_scenario: str

    pie_color_dict_2: dict[str, list[str]]
    pie_color_dict_4: dict[str, list[str]]

    margin_features: dict[str, dict[str, int]]

    metrics: pd.DataFrame
    score_table: pd.DataFrame
    values: pd.DataFrame

    scatter_for_prediction: go.Figure
    histogram_for_prediction: go.Figure

    accuracy_pie: pd.DataFrame
    distrib_class: pd.DataFrame
    pie_confusion_matrix: pd.DataFrame
    features_table: pd.DataFrame

    forecast_series: pd.Series

    # Databases
    algorithm_selected: str
    # This path is used to create a temporary CSV file download the table
    tempdir: pathlib.Path
    PATH_TO_TABLE: str

    # Selector to select the table to show
    table_selector: list[str]
    table_selected: str
