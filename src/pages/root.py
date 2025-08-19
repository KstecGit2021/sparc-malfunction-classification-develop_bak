from taipy.gui import Icon, navigate
import taipy.gui.builder as tgb
from pages.compare_models.compare_models import *
from pages.data_visualization.data_visualization import *
from pages.databases.databases import *
from pages.model_manager.model_manager import *
from state_class import State

# dialog
show_roc: bool = False

# Common variables to several pages
algorithm_selector: list[str] = [
    "Baseline",
    # "Tree",
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
]
algorithm_selected: str = "Baseline"

select_x: list[str] = []
x_selected: str = None
select_y: list[str] = []
y_selected: str = None

threshold = 800

menu_lov = [
    ("Data Visualization", Icon("images/histogram_menu.svg", "Data Visualization")),
    ("Model Manager", Icon("images/model.svg", "Model Manager")),
    ("Compare Models", Icon("images/compare.svg", "Compare Models")),
    ("Databases", Icon("images/Datanode.svg", "Databases")),
]


def menu_fct(state: State, var_name: str, var_value: dict):
    """Function that is called when there is a change in the menu control."""
    page = var_value["args"][0].replace(" ", "-")
    navigate(state, page)


def close_dialog(state: State):
    state.show_roc = False


with tgb.Page() as roc_curve_page:
    tgb.chart(
        "{roc_dataset}",
        x="False positive rate",
        y="True positive rate",
        label="True positive rate",
        height="500px",
        width="90%",
        type="scatter",
    )


def change_threshold(state: State):
    model_type = state.algorithm_selected.lower().replace(" ", "_")

    state.scenario.data_nodes[f"threshold_{model_type}"].write(state.threshold / 1000)
    state.scenario.sequences[f"change_threshold_{model_type}"].submit(wait=True)

    state.on_change_model_manager(state)
    state.on_init_compare_models(state)
    state.on_change_data_visualization(state)


with tgb.Page() as root:
    tgb.toggle(theme=True)
    tgb.menu(label="Menu", lov=menu_lov, on_action=menu_fct)

    tgb.dialog(
        open="{show_roc}",
        page="ROC-Curve",
        title="ROC Curve",
        width="100%",
        on_action=close_dialog,
    )

    # tgb.number(
    #     "{threshold}",
    # )
    # tgb.button("Change threshold", on_action=change_threshold)
