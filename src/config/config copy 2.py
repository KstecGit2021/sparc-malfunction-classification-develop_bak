from algos.algos import *
from taipy import Config, Scope


Config.configure_job_executions(mode="standalone", max_nb_of_workers=2)

##############################################################################################################################
# Creation of the datanodes
##############################################################################################################################
# How to connect to the database
# path_to_csv = "data/REl data and Cp data joined dec20_gg.csv"
path_to_pickle = "data/initial_dataset.p"


# path for csv and file_path for pickle
initial_dataset_cfg = Config.configure_data_node(
    # id="initial_dataset", path=path_to_csv, storage_type="csv", has_header=True
    id="initial_dataset", path=path_to_pickle, storage_type="pickle", has_header=True
)

preprocessed_dataset_cfg = Config.configure_data_node(id="preprocessed_dataset")
train_dataset_cfg = Config.configure_data_node(id="train_dataset")
test_dataset_cfg = Config.configure_data_node(id="test_dataset")

feature_selector_default = \
{
    "feature_selector_name": "CorrelationsClassifier",
    "f2_scorer": {
        "name": "fbeta_score",
        "beta": 4,
        "pos_label": 1
    },
    "variance_threshold_filter": {
        "threshold": 0
    },
    "correlation_filter": {
        "threshold": 0.02
        # "threshold": 0.08
    }
}

feature_selector_default_cfg = Config.configure_data_node(id="feature_selector_default", storage_type="json", default_data=feature_selector_default)
feature_selector_cfg = Config.configure_data_node(id="feature_selector", storage_type="json", default_data=feature_selector_default)
feature_selection_info_cfg = Config.configure_data_node(id="feature_selection_info", storage_type="json")

# 훈련/테스트 데이터 분할 및 샘플링을 적용 파라미터.
# split_parameter (dict, optional): 데이터 분할 및 샘플링 관련 파라미터.
#     - 'test_size' (float, optional): 테스트 데이터셋 비율. Defaults to 0.2.
#     - 'random_state' (int, optional): 난수 시드. Defaults to 42.
#     - 'var_threshold' (float, optional): 분산 필터 임계값. Defaults to 0.0.
#     - 'corr_threshold' (float, optional): 상관 관계 필터 임계값. Defaults to 0.98.
#     - 'sampling_ratio' (float, optional): 소수 클래스(y=1) 비율.
#         - 1.0: 1:1 비율로 샘플링 (오버/언더 자동 적용).
#         - 1.0 초과: 오버샘플링. 1 : n_samples_majority * sampling_ratio[2, 4, ...]
#         - 1.0 미만: 언더샘플링. 0 : int(n_samples_minority / sampling_ratio[0.5, 0.25, ...])
#         - None: 샘플링 미적용.
split_parameter_default = \
    {
        'test_size': 0.2,
        'random_state': 42,
        # create_train_test_data에서 직접 적용되는 필터
        'apply_filter_split': True,
        'var_threshold_split': 0.0,
        'corr_threshold_split': 0.98,
        
        'sampling_ratio': None,
        
        'apply_feature_generation': True,
        'sum_features': True,  # 쌍별 합 피처 생성 여부
        'diff_features': True,  # 쌍별 차 피처 생성 여부
        'poly_features': True,  # 다항식 피처 생성 여부
        'poly_degree': 2,

        # feature_generator 내부에서 적용되는 필터
        'apply_filter_gen': True,
        'var_threshold_gen': 0.0,
        'corr_threshold_gen': 0.1,
    }
    # {
    #     'test_size': 0.2,
    #     'random_state': 42,
    #     'var_threshold': 0.0,
    #     # 'corr_threshold': 0.98,
    #     'corr_threshold': 0.1,
    #     'sampling_ratio': None,
    #     # 'apply_feature_generation': False, # 피처 생성 적용 여부
    #     'apply_feature_generation': True, # 피처 생성 적용 여부
    #     'sum_features': True,            # 쌍별 합 피처 생성 여부
    #     'diff_features': False,           # 쌍별 차 피처 생성 여부
    #     'poly_features': False,           # 다항식 피처 생성 여부
    #     'poly_degree': 2                 # 다항식 피처 차수
    # }
    # {
    #     'test_size': 0.2,
    #     'random_state': 42,
    #     'var_threshold': 0.0,
    #     'corr_threshold': 0.98,
    #     'sampling_ratio': None,
    #     # 'sampling_ratio': 4,
    #     'apply_feature_generation': False  # 기본값으로 False 설정
    #     # 'apply_feature_generation': True  # 기본값으로 False 설정
    # }

    
split_parameter_cfg = Config.configure_data_node(
    id="split_parameter", storage_type="json",  default_data=split_parameter_default
)
split_parameter_info_cfg = Config.configure_data_node(id="split_parameter_info", storage_type="json")

train_parameters_list_default = \
{'baseline': {'train_model_baseline': {'function_name': 'train_model_baseline',
                                       'importance_data': {'Features': ['SensorOffsetHot-Cold',
                                                                        'band '
                                                                        'gap '
                                                                        'dpat_ok '
                                                                        'for '
                                                                        'band '
                                                                        'gap',
                                                                        'Radius'],
                                                           'Importance': [56.6,
                                                                          4.65,
                                                                          96.9]}}},
 'logistic_regression': {'train_model_logistic_regression': {'class_weight_multiplier': 'len(y) '
                                                                                        '- '
                                                                                        'n_pos',
                                                             'function_name': 'train_model_logistic_regression',
                                                            #  'max_iter': 1000,
                                                             'max_iter': 10,
                                                             'solver': 'lbfgs'},
                         'train_parameters_optuna': {'f2_rare_scorer': {'beta': 2,
                                                                        'name': 'fbeta_score',
                                                                        'pos_label': 1},
                                                     'function_name': 'train_model_logistic_regression_optuna',
                                                     'n_trials': 50,
                                                     'param_ranges': {'C': [0.0001,
                                                                            20],
                                                                      'class_weight_multiplier': [1,
                                                                                                  20],
                                                                    #   'max_iter': 1000,
                                                                      'max_iter': 10,
                                                                      'solver': ['liblinear',
                                                                                 'saga']}}},
 'random_forest': {'train_model_rf_cv': {'class_weight_multiplier': 'len(y) - '
                                                                    'sum(y)',
                                         'cv': 3,
                                         'f2_rare_scorer': {'beta': 2,
                                                            'name': 'fbeta_score',
                                                            'pos_label': 1},
                                         'function_name': 'train_model_rf_cv',
                                         'param_grid': {'max_depth': [2,
                                                                      5,
                                                                      None],
                                                        'min_samples_split': [2,
                                                                              5],
                                                        'n_estimators': [20,
                                                                         50,
                                                                         100]},
                                         'verbose': 1},
                   'train_model_rf_optuna': {'cv': 5,
                                             'function_name': 'train_model_rf_optuna',
                                             'n_trials': 50,
                                             'param_ranges': {'max_depth': {'high': 30,
                                                                            'low': 10},
                                                              'max_features': {'choices': ['sqrt',
                                                                                           0.5,
                                                                                           0.8]},
                                                              'min_samples_leaf': {'high': 10,
                                                                                   'low': 1},
                                                              'min_samples_split': {'high': 20,
                                                                                    'low': 2},
                                                              'n_estimators': {'high': 300,
                                                                               'low': 100}}}},
 'xgboost': {'train_model_xgboost_cv': {'cv': 3,
                                        'f2_rare_scorer': {'beta': 2,
                                                           'name': 'fbeta_score',
                                                           'pos_label': 1},
                                        'function_name': 'train_model_xgboost_cv',
                                        'param_grid': {'learning_rate': [0.01,
                                                                         0.1,
                                                                         0.2],
                                                       'max_depth': [2, 5],
                                                       'n_estimators': [30,
                                                                        50,
                                                                        100]},
                                        'scale_pos_weight_multiplier': 2,
                                        'verbose': 1},
             'train_model_xgboost_optuna': {'cv': 5,
                                            'function_name': 'train_model_xgboost_optuna',
                                            'n_trials': 50,
                                            'param_ranges': {'colsample_bytree': {'high': 1.0,
                                                                                  'low': 0.7},
                                                             'gamma': {'high': 0.5,
                                                                       'low': 0.1},
                                                             'learning_rate': {'high': 0.2,
                                                                               'low': 0.01},
                                                             'max_depth': {'high': 20,
                                                                           'low': 5},
                                                             'n_estimators': {'high': 300,
                                                                              'low': 100},
                                                             'reg_alpha': {'high': 0.1,
                                                                           'low': 1e-06},
                                                             'reg_lambda': {'high': 0.1,
                                                                            'low': 1e-06},
                                                             'subsample': {'high': 1.0,
                                                                           'low': 0.7}},
                                            'ratio_multiplier_range': {'high': 1.2,
                                                                       'low': 0.8}}}}

train_parameters_list_default_cfg = Config.configure_data_node(
    id="train_parameters_list_default", storage_type="json",  default_data=train_parameters_list_default
)

# Define models and their corresponding functions
models = {
    "baseline": train_model_baseline,
    # "logistic_regression": train_model_logistic_regression,
    "logistic_regression": train_model_logistic_regression_optuna,
    # "random_forest": train_model_rf_cv,
    "random_forest": train_model_rf_optuna,
    # "xgboost": train_model_xgboost_cv,
    "xgboost": train_model_xgboost_optuna,
    # "tree": train_model_decision_tree,
}


# Create data nodes for each model
data_nodes = {}
for model in models:
    function_name = models[model].__name__
    # train_parameters_default = get_train_parameters_default(function_name)
    data_nodes[model] = {
        "train_dataset_proba": Config.configure_data_node(
            id=f"train_dataset_proba_{model}"
        ),
        "train_dataset_metrics": Config.configure_data_node(
            id=f"train_dataset_metrics_{model}"
        ),
        "train_parameters": Config.configure_data_node(
            id=f"train_parameters_{model}", storage_type="json", default_data=train_parameters_list_default.get(model, {}).get(function_name, {})
        ),
        "trained_model": Config.configure_data_node(id=f"trained_model_{model}"),
        "train_parameters_info": Config.configure_data_node(
            id=f"train_parameters_info_{model}", storage_type="json"
        ),
        "threshold": Config.configure_data_node(
            id=f"threshold_{model}", storage_type="json"
        ),
        "forecast_dataset": Config.configure_data_node(id=f"forecast_dataset_{model}"),
        "shap_values": Config.configure_data_node(id=f"shap_values_{model}"),
        "roc_data": Config.configure_data_node(id=f"roc_data_{model}"),
        "auc_score": Config.configure_data_node(id=f"auc_score_{model}"),
        "metrics": Config.configure_data_node(id=f"metrics_{model}"),
        "feature_importance": Config.configure_data_node(
            id=f"feature_importance_{model}"
        ),
        "results": Config.configure_data_node(id=f"results_{model}"),
    }

##############################################################################################################################
# Creation of the tasks
##############################################################################################################################

# initial_dataset --> preprocess dataset --> preprocessed_dataset
task_preprocess_dataset_cfg = Config.configure_task(
    id="preprocess_dataset",
    input=[initial_dataset_cfg],
    function=preprocess_dataset,
    output=preprocessed_dataset_cfg,
    skippable=True,
)

# preprocessed_dataset --> create train data --> train_dataset, test_dataset
task_create_train_test_cfg = Config.configure_task(
    id="create_train_and_test_data",
    input=[preprocessed_dataset_cfg, split_parameter_cfg],
    function=create_train_test_data,
    output=[train_dataset_cfg, test_dataset_cfg, split_parameter_info_cfg],
    skippable=True,
)

# preprocessed_dataset --> create train data --> train_dataset, test_dataset --> select_feature
task_select_feature_cfg = Config.configure_task(
    id="select_feature",
    input=[test_dataset_cfg, train_dataset_cfg, feature_selector_cfg],
    function=select_feature,
    output=feature_selection_info_cfg,
    skippable=True,
)

# Create tasks for each model
tasks = {}
for model, function in models.items():
    tasks[model] = {
        "train_model": Config.configure_task(
            id=f"train_model_{model}",
            input=[
                train_dataset_cfg, 
                feature_selection_info_cfg,
                data_nodes[model]["train_parameters"],
            ],
            function=function,
            output=[
                data_nodes[model]["trained_model"],
                data_nodes[model]["feature_importance"],
                data_nodes[model]["train_parameters_info"],
            ],
            skippable=True,
        ),
        "find_best_threshold": Config.configure_task(
            id=f"find_best_threshold_{model}",
            input=[
                data_nodes[model]["trained_model"],
                train_dataset_cfg,
                feature_selection_info_cfg,
            ],
            function=find_best_threshold,
            output=[
                data_nodes[model]["train_dataset_proba"],
                data_nodes[model]["threshold"],
            ],
            skippable=True,
        ),
        "create_metrics_on_train": Config.configure_task(
            id=f"create_metrics_on_train_{model}",
            input=[
                data_nodes[model]["train_dataset_proba"],
                data_nodes[model]["threshold"],
            ],
            function=create_metrics_on_train,
            output=[data_nodes[model]["train_dataset_metrics"]],
            skippable=True,
        ),
        "forecast": Config.configure_task(
            id=f"predict_the_test_data_{model}",
            input=[test_dataset_cfg, 
                   data_nodes[model]["trained_model"],
                   feature_selection_info_cfg,
            ],
            function=forecast,
            output=[
                data_nodes[model]["forecast_dataset"],
                data_nodes[model]["shap_values"],
            ],
            skippable=True,
        ),
        "roc": Config.configure_task(
            id=f"task_roc_{model}",
            input=[data_nodes[model]["forecast_dataset"], test_dataset_cfg],
            function=roc_from_scratch,
            output=[data_nodes[model]["roc_data"], data_nodes[model]["auc_score"]],
            skippable=True,
        ),
        "create_metrics": Config.configure_task(
            id=f"task_create_metrics_{model}",
            input=[
                data_nodes[model]["forecast_dataset"],
                test_dataset_cfg,
                data_nodes[model]["auc_score"],
                data_nodes[model]["threshold"],
            ],
            function=create_metrics,
            output=data_nodes[model]["metrics"],
            skippable=True,
        ),
        "create_results": Config.configure_task(
            id=f"task_create_results_{model}",
            input=[
                data_nodes[model]["forecast_dataset"],
                test_dataset_cfg,
                data_nodes[model]["threshold"],
            ],
            function=create_results,
            output=data_nodes[model]["results"],
            skippable=True,
        ),
    }

##############################################################################################################################
# Creation of the scenario
##############################################################################################################################

scenario_cfg = Config.configure_scenario(
    id="churn_classification",
    additional_data_node_configs = [
        train_parameters_list_default_cfg,
        feature_selector_default_cfg,
    ],
    task_configs=[
        task_preprocess_dataset_cfg,
        task_create_train_test_cfg,
        task_select_feature_cfg,
    ]
    + [task for model_tasks in tasks.values() for task in model_tasks.values()],
    sequences={
        f"change_threshold_{model}": [
            tasks[model]["create_results"],
            tasks[model]["create_metrics"],
            tasks[model]["create_metrics_on_train"],
        ]
        for model in tasks.keys()
    },
)

Config.export("config/config.toml")
