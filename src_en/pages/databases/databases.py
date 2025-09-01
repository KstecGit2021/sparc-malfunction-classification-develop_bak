import pathlib

# from taipy.gui import Markdown
import taipy.gui.builder as tgb
from state_class import State

# This path is used to create a temporary CSV file download the table
tempdir: pathlib.Path = pathlib.Path(".tmp")
tempdir.mkdir(exist_ok=True)
PATH_TO_TABLE: str = str(tempdir / "table.csv")

# Selector to select the table to show
table_selector: list[str] = [
    "Training Dataset",
    "Test Dataset",
    "Forecast Dataset",
    "Confusion Matrix",
]
table_selected: str = "Confusion Matrix"
algorithm_selected = "Baseline"


def handle_temp_csv_path(state: State):
    """This function checks if the temporary csv file exists. If it does, it is deleted. Then, the temporary csv file
    is created for the right table

    Args:
        state: object containing all the variables used in the GUI
    """
    if state.table_selected == "Test Dataset":
        state.test_dataset.to_csv(PATH_TO_TABLE, sep=";")
    if state.table_selected == "Confusion Matrix":
        state.score_table.to_csv(PATH_TO_TABLE, sep=";")
    if state.table_selected == "Training Dataset":
        state.train_dataset.to_csv(PATH_TO_TABLE, sep=";")
    if state.table_selected == "Forecast Dataset":
        state.values.to_csv(PATH_TO_TABLE, sep=";")


# Aggregation of the strings to create the complete page
# databases = Markdown("pages/databases/databases.md")

with tgb.Page() as databases:
    # Title
    tgb.text("# *Malfunction Classification* - Data**bases**", mode="md")
    tgb.html("hr")

    # Description
    tgb.text(
        """
The database page shows all the data used in this application. 
From the training dataset to the predictions, you can view the underlying data and download it.
""",
        mode="md",
    )

    # A 3-column layout for selectors and file download
    with tgb.layout("2 2 1"):
        tgb.selector(
            "{algorithm_selected}",
            lov="{algorithm_selector}",
            dropdown=True,
            label="Algorithm",
        )

        tgb.selector(
            "{table_selected}",
            lov="{table_selector}",
            dropdown=True,
            label="Table",
        )

        tgb.file_download(
            "{PATH_TO_TABLE}",
            name="table.csv",
            label="Download table",
        )

    # PART: Confusion Matrix
    with tgb.part(
        "Confusion",
        render=lambda table_selected: table_selected == "Confusion Matrix",
    ):
        tgb.table(
            "{score_table}",
            width="fit-content",
            show_all=True,
            class_name="ml-auto mr-auto",
            filter=True,
        )

    # PART: Training Dataset
    # with tgb.part(
    #     "Training",
    #     render=lambda table_selected: table_selected == "Training Dataset",
    # ):
    #     tgb.table("{train_dataset}")

    # PART: Forecast Dataset
    with tgb.part(
        "Forecast",
        render=lambda table_selected: table_selected == "Forecast Dataset",
    ):
        tgb.table(
            "{values}",
            width="fit-content",
            row_class_name="{lambda s, i, r: 'red_color' if r['Historical'] != r['Forecast'] else 'green_color'}",
            class_name="ml-auto mr-auto",
            filter=True,
        )

    with tgb.part(
        render=lambda table_selected, algorithm_selected: table_selected
        == "Training Dataset"
        and algorithm_selected == "ML"
    ):
        tgb.table(
            lambda scenario: (
                scenario.train_dataset_random_forest.read() if scenario else None
            ),
            row_class_name=lambda s, i, r: (
                "red_color" if r["Pass/Fail"] != r["Forecast"] else "green_color"
            ),
            class_name="ml-auto mr-auto",
            filter=True,
        )

    with tgb.part(
        render=lambda table_selected, algorithm_selected: table_selected
        == "Training Dataset"
        and algorithm_selected == "Baseline"
    ):
        tgb.table(
            lambda scenario: (
                scenario.train_dataset_baseline.read() if scenario else None
            ),
            row_class_name=lambda s, i, r: (
                "red_color" if r["Pass/Fail"] != r["Forecast"] else "green_color"
            ),
            class_name="ml-auto mr-auto",
            filter=True,
        )

    with tgb.part(
        render=lambda table_selected, algorithm_selected: table_selected
        == "Training Dataset"
        and algorithm_selected == "Logistic Regression"
    ):
        tgb.table(
            lambda scenario: (
                scenario.train_dataset_logistic_regression.read() if scenario else None
            ),
            row_class_name=lambda s, i, r: (
                "red_color" if r["Pass/Fail"] != r["Forecast"] else "green_color"
            ),
            class_name="ml-auto mr-auto",
            filter=True,
        )

    with tgb.part(
        render=lambda table_selected, algorithm_selected: table_selected
        == "Training Dataset"
        and algorithm_selected == "XGBoost"
    ):
        tgb.table(
            lambda scenario: (
                scenario.train_dataset_xgboost.read() if scenario else None
            ),
            row_class_name=lambda s, i, r: (
                "red_color" if r["Pass/Fail"] != r["Forecast"] else "green_color"
            ),
            class_name="ml-auto mr-auto",
            filter=True,
        )

    with tgb.part(
        "test_dataset",
        render=lambda table_selected: table_selected == "Test Dataset",
    ):
        tgb.table("{test_dataset}", filter=True)
