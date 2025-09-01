from typing import Any
import taipy as tp
from taipy.gui import Gui
from config.config import scenario_cfg
from pages.root import *
from state_class import State


def on_change(state: State, var_name: str, var_value: Any):
    """Handle variable changes in the GUI."""
    if var_name in ["x_selected", "y_selected"] and state.forecast_series is not None:
        on_change_data_visualization(state)
        change_distribution_chart_prediction(state)
    elif var_name == "algorithm_selected":
        state.algorithm_selected = state.algorithm_selected
        on_change_model_manager(state)
    if var_name in [
        "algorithm_selected",
        "algorithm_selected",
        "table_selected",
    ]:
        handle_temp_csv_path(state)


def on_init(state: State):
    """Handle initialization of the GUI."""
    state.select_x = test_dataset.drop("PASS/FAIL", axis=1).columns.tolist()
    state.x_selected = "RADIUS"
    state.select_y = state.select_x
    state.y_selected = "SensorOffsetHot-Cold".upper()

    on_change_model_manager(state)
    on_init_compare_models(state)
    on_change_data_visualization(state)


# Define pages
pages = {
    "/": root,
    "Data-Visualization": data_visualization,
    "Model-Manager": model_manager,
    "Compare-Models": compare_models,
    "Databases": databases,
    "ROC-Curve": roc_curve_page,
}

if __name__ == "__main__":
    tp.Orchestrator().run()

    if len(tp.get_scenarios()) == 0:
        scenario = tp.create_scenario(scenario_cfg)
        # No need of wait=True here because we are synchronous;
        # This is just best practice
        tp.submit(scenario, wait=True)
    else:
        scenario = tp.get_scenarios()[0]

    # Read datasets
    preprocessed_dataset = scenario.preprocessed_dataset.read()
    train_dataset = scenario.train_dataset.read()
    test_dataset = scenario.test_dataset.read()
    roc_dataset = scenario.roc_data_random_forest.read()
    # roc_dataset = scenario.roc_data_logistic_regression.read()

    # Process test dataset columns
    test_dataset.columns = [str(column).upper() for column in test_dataset.columns]
    train_dataset.columns = [str(column).upper() for column in train_dataset.columns]
    preprocessed_dataset.columns = [
        str(column).upper() for column in preprocessed_dataset.columns
    ]

    gui = Gui(pages=pages)
    gui.run(title="Fail classification", dark_mode=True, port=8494)
