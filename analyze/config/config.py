from algos.algos import *
from taipy import Config, Scope


Config.configure_job_executions(mode="standalone", max_nb_of_workers=2)

##############################################################################################################################
# Creation of the datanodes
##############################################################################################################################
# How to connect to the database
path_to_pickle = "data/initial_dataset.p"

initial_dataset_cfg = Config.configure_data_node(
    id="initial_dataset", path=path_to_pickle, storage_type="pickle", has_header=True
)

preprocessed_dataset_cfg = Config.configure_data_node(id="preprocessed_dataset")
train_dataset_cfg = Config.configure_data_node(id="train_dataset")
test_dataset_cfg = Config.configure_data_node(id="test_dataset")

# Applying a Filter Method (FeatureFilter)
feature_selector_params_FeatureFilter_default = {
    "feature_selector_name": "FeatureFilter",
    "filter_methods": {
        "apply_variance_filter": False,
        "var_threshold": 0.01,
        "apply_target_linear_corr_filter": False,
        "target_linear_corr_threshold": 0.05,
        "apply_target_xicor_filter": False,
        "target_xicor_threshold": 0.1,
        "apply_feature_linear_corr_filter": False,
        "feature_linear_corr_threshold": 0.95,
        "apply_feature_xicor_filter": False,
        "feature_xicor_threshold": 0.9
    }
}
feature_selector_params_FeatureFilter_default_cfg = Config.configure_data_node(id="feature_selector_params_FeatureFilter_default", storage_type="json", default_data=feature_selector_params_FeatureFilter_default)


feature_selector_params_estimator_LogisticRegression_default = \
{
    "name": "LogisticRegression"
}
feature_selector_params_estimator_RandomForestClassifier_default = \
{
    "name": "RandomForestClassifier",
    "params": {
        # "n_estimators": 150, # Number of trees
        "n_estimators": 250, # Number of trees
        # "max_depth": 10 # Maximum depth of trees
        "max_depth": 12 # Maximum depth of trees
    }
}
feature_selector_params_estimator_LGBMClassifier_default = \
{
    "name": "LGBMClassifier",
    "params": {
        "n_estimators": 200, # Number of boosting stages
        "learning_rate": 0.05 # Learning rate
    }
}



# RFE (Recursive Feature Elimination)
feature_selector_params_rfe_default = \
{
    "feature_selector_name": "RFE",
    "params": {
        "n_features_to_select": 15, # Number of features to select
        "step": 1, # Number of features to remove at each step
        "estimator": feature_selector_params_estimator_LogisticRegression_default
        # "estimator": feature_selector_params_estimator_RandomForestClassifier_default
    },
    "filter_methods": "apply_RecursiveFeatureElimination_filter"
}
feature_selector_params_rfe_default_cfg = Config.configure_data_node(id="feature_selector_params_rfe_default", storage_type="json", default_data=feature_selector_params_rfe_default)

# SFS (Sequential Feature Selector)
feature_selector_params_sfs_default = \
{
    "feature_selector_name": "SFS",
    "params": {
        "n_features_to_select": "auto", # Number of features to select (auto)
        "direction": "forward", # Forward selection ('forward') or backward elimination ('backward')
        "scoring": "accuracy", # Model performance metric
        "estimator": feature_selector_params_estimator_RandomForestClassifier_default
    },
    "filter_methods": "apply_SequentialFeatureSelector_filter"
}
feature_selector_params_sfs_default_cfg = Config.configure_data_node(id="feature_selector_params_sfs_default", storage_type="json", default_data=feature_selector_params_sfs_default)

# SFM (SelectFromModel) - Model-based feature selection
feature_selector_params_sfm_default = \
{
    "feature_selector_name": "SFM",
    "params": {
        "threshold": "median", # Feature importance threshold
        # "estimator": feature_selector_params_estimator_LGBMClassifier_default
        "estimator": feature_selector_params_estimator_RandomForestClassifier_default
    },
    "filter_methods": "apply_SelectFromModel_filter"
}

feature_selector_params_sfm_default_cfg = Config.configure_data_node(id="feature_selector_params_sfm_default", storage_type="json", default_data=feature_selector_params_sfm_default)

# In case feature selection is not performed
feature_selector_params_no_op = \
{
    "feature_selector_name": "None"
}
feature_selector_params_no_op_cfg = Config.configure_data_node(id="feature_selector_params_no_op", storage_type="json", default_data=feature_selector_params_no_op)


# FeatureFilter as the default selector
feature_selector_params_default = \
{
    "feature_selector_name": "FeatureFilter",
    "filter_methods": {
        "apply_variance_filter": False,
        "var_threshold": 0.01,
        "apply_target_linear_corr_filter": False,
        "target_linear_corr_threshold": 0.05,
        "apply_target_xicor_filter": False,
        "target_xicor_threshold": 0.0,
        "apply_feature_linear_corr_filter": False,
        "feature_linear_corr_threshold": 0.95,
        "apply_feature_xicor_filter": False,
        "feature_xicor_threshold": 0.9
    }
}
feature_selector_cfg = Config.configure_data_node(id="feature_selector", storage_type="json", default_data=feature_selector_params_default)



feature_selection_info_cfg = Config.configure_data_node(id="feature_selection_info", storage_type="json")

# Parameters for training/test data splitting and sampling.
# split_parameter (dict, optional): Parameters related to data splitting and sampling.
#     - 'test_size' (float, optional): Test dataset ratio. Defaults to 0.2.
#     - 'random_state' (int, optional): Random seed. Defaults to 42.
#     - 'var_threshold_split' (float, optional): Variance filter threshold. Defaults to 0.0.
#     - 'corr_threshold_split' (float, optional): Correlation filter threshold. Defaults to 0.98.
#     - 'sampling_ratio' (float, optional): Minority class (y=1) ratio.
#         - 1.0: Sampling with a 1:1 ratio (auto over/under-sampling applied).
#         - > 1.0: Oversampling. 1 : n_samples_majority * sampling_ratio[2, 4, ...]
#         - < 1.0: Undersampling. 0 : int(n_samples_minority / sampling_ratio[0.5, 0.25, ...])
#         - None: No sampling applied.
split_parameter_default = \
    {
        'test_size': 0.2,
        'random_state': 42,
        # Filters applied directly in create_train_test_data
        'apply_filter_split': False,
        'var_threshold_split': 0.0,
        'corr_threshold_split': 0.98,
        
        'sampling_ratio': None,
        
        'apply_feature_generation': False,
        'sum_features': False,  # Whether to create pairwise sum features
        'diff_features': False,  # Whether to create pairwise difference features
        'poly_features': False,  # Whether to create polynomial features
        'poly_degree': 2,

        # Filters applied within the feature_generator
        'apply_filter_gen': False,
        'var_threshold_gen': 0.0,
        'corr_threshold_gen': 0.1,
    }

    
split_parameter_cfg = Config.configure_data_node(
    id="split_parameter", storage_type="json",  default_data=split_parameter_default
)
split_parameter_info_cfg = Config.configure_data_node(id="split_parameter_info", storage_type="json")

train_parameters_list_default_old = \
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
                                                                          96.9]}}
             },
'logistic_regression': {'train_model_logistic_regression': {'class_weight_multiplier': 'len(y) '
                                                                                        '- '
                                                                                        'n_pos',
                                                              'function_name': 'train_model_logistic_regression',
                                                              'max_iter': 10,
                                                              'solver': 'lbfgs'},
                      'train_model_logistic_regression_optuna': {'f2_rare_scorer': {'beta': 2,
                                                                                   'name': 'fbeta_score',
                                                                                   'pos_label': 1},
                                                                 'function_name': 'train_model_logistic_regression_optuna',
                                                                 'n_trials': 50,
                                                                 'param_ranges': {'C': [0.0001,
                                                                                        20],
                                                                                  'class_weight_multiplier': [1,
                                                                                                              20],
                                                                                  'max_iter': 10,
                                                                                  'solver': ['liblinear',
                                                                                             'saga']}},
                      'train_model_logistic_regression_cv': {'cv': 3,
                                                             'f2_rare_scorer': {'beta': 2,
                                                                                'name': 'fbeta_score',
                                                                                'pos_label': 1},
                                                             'function_name': 'train_model_logistic_regression_cv',
                                                             'param_grid': {'C': [0.0001,
                                                                                  0.001,
                                                                                  0.01,
                                                                                  0.1,
                                                                                  1,
                                                                                  10,
                                                                                  20],
                                                                            'solver': ['lbfgs',
                                                                                       'liblinear']}}},
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
                                            'f2_rare_scorer': {'beta': 2,
                                                               'name': 'fbeta_score',
                                                               'pos_label': 1},
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
                                                                              'low': 100}}},
                  'train_model_rf_optuna_old': {'cv': 5,
                                                'function_name': 'train_model_rf_optuna_old',
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
            'train_model_xgboost_optuna_old': {'cv': 5,
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
                                                                          'low': 0.8}},
            'train_model_xgboost_optuna': {'cv': 5,
                                           'f2_rare_scorer': {'beta': 2, # Add f2_rare_scorer
                                                              'name': 'fbeta_score',
                                                              'pos_label': 1},
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
                                                                      'low': 0.8}}}
}

train_parameters_list_default = \
{'baseline': {'function_name': 'train_model_baseline',
              'importance_data': {'Features': ['SensorOffsetHot-Cold',
                                               'band gap dpat_ok for band gap',
                                               'Radius'],
                                  'Importance': [56.6, 4.65, 96.9]}},
 'logistic_regression_cv': {'cv': 3,
                            'f2_rare_scorer': {'beta': 2,
                                               'name': 'fbeta_score',
                                               'pos_label': 1},
                            'function_name': 'train_model_logistic_regression_cv',
                            'param_grid': {'C': [0.0001,
                                                 0.001,
                                                 0.01,
                                                 0.1,
                                                 1,
                                                 10,
                                                 20],
                                           'solver': ['lbfgs', 'liblinear']}},
 'logistic_regression': {'f2_rare_scorer': {'beta': 2,
                                            'name': 'fbeta_score',
                                            'pos_label': 1},
                         'function_name': 'train_model_logistic_regression',
                         'n_trials': 50,
                         'param_ranges': {'C': [0.0001, 20],
                                          'class_weight_multiplier': [1,
                                                                      20],
                                          'max_iter': 10,
                                          'solver': ['liblinear',
                                                     'saga']}},
 'rf_cv': {'class_weight_multiplier': 'len(y) - sum(y)',
           'cv': 3,
           'f2_rare_scorer': {'beta': 2, 'name': 'fbeta_score', 'pos_label': 1},
           'function_name': 'train_model_rf_cv',
           'param_grid': {'max_depth': [2, 5, None],
                          'min_samples_split': [2, 5],
                          'n_estimators': [20, 50, 100]},
           'verbose': 1},
 'random_forest': {'cv': 5,
                   'f2_rare_scorer': {'beta': 2,
                                      'name': 'fbeta_score',
                                      'pos_label': 1},
                   'function_name': 'train_model_rf_optuna',
                   'n_trials': 50,
                   'param_ranges': {'max_depth': {'high': 30, 'low': 10},
                                    'max_features': {'choices': ['sqrt', 0.5, 0.8]},
                                    'min_samples_leaf': {'high': 10, 'low': 1},
                                    'min_samples_split': {'high': 20, 'low': 2},
                                    'n_estimators': {'high': 300, 'low': 100}}},
 'rf_optuna': {'cv': 5,
               'f2_rare_scorer': {'beta': 2,
                                  'name': 'fbeta_score',
                                  'pos_label': 1},
               'function_name': 'train_model_rf_optuna',
               'n_trials': 50,
               'param_ranges': {'max_depth': {'high': 30, 'low': 10},
                                'max_features': {'choices': ['sqrt', 0.5, 0.8]},
                                'min_samples_leaf': {'high': 10, 'low': 1},
                                'min_samples_split': {'high': 20, 'low': 2},
                                'n_estimators': {'high': 300, 'low': 100}}},
 'xgboost_cv': {'cv': 3,
                'f2_rare_scorer': {'beta': 2,
                                   'name': 'fbeta_score',
                                   'pos_label': 1},
                'function_name': 'train_model_xgboost_cv',
                'param_grid': {'learning_rate': [0.01, 0.1, 0.2],
                               'max_depth': [2, 5],
                               'n_estimators': [30, 50, 100]},
                'scale_pos_weight_multiplier': 2,
                'verbose': 1},
 'xgboost': {'cv': 5,
             'f2_rare_scorer': {'beta': 2,
                                'name': 'fbeta_score',
                                'pos_label': 1},
             'function_name': 'train_model_xgboost_optuna',
             'n_trials': 50,
             'param_ranges': {'colsample_bytree': {'high': 1.0,
                                                   'low': 0.7},
                              'gamma': {'high': 0.5, 'low': 0.1},
                              'learning_rate': {'high': 0.2,
                                                'low': 0.01},
                              'max_depth': {'high': 20, 'low': 5},
                              'n_estimators': {'high': 300, 'low': 100},
                              'reg_alpha': {'high': 0.1, 'low': 1e-06},
                              'reg_lambda': {'high': 0.1, 'low': 1e-06},
                              'subsample': {'high': 1.0, 'low': 0.7}},
             'ratio_multiplier_range': {'high': 1.2, 'low': 0.8}},
 'xgboost_optuna': {'cv': 5,
                    'f2_rare_scorer': {'beta': 2,
                                       'name': 'fbeta_score',
                                       'pos_label': 1},
                    'function_name': 'train_model_xgboost_optuna',
                    'n_trials': 50,
                    'param_ranges': {'colsample_bytree': {'high': 1.0,
                                                          'low': 0.7},
                                     'gamma': {'high': 0.5, 'low': 0.1},
                                     'learning_rate': {'high': 0.2,
                                                       'low': 0.01},
                                     'max_depth': {'high': 20, 'low': 5},
                                     'n_estimators': {'high': 300, 'low': 100},
                                     'reg_alpha': {'high': 0.1, 'low': 1e-06},
                                     'reg_lambda': {'high': 0.1, 'low': 1e-06},
                                     'subsample': {'high': 1.0, 'low': 0.7}},
                    'ratio_multiplier_range': {'high': 1.2, 'low': 0.8}}
}


train_parameters_list_default_cfg = Config.configure_data_node(
    id="train_parameters_list_default", storage_type="json",  default_data=train_parameters_list_default
)

# Define models and their corresponding functions
models = \
{
    # "baseline": train_model_baseline,

    # "logistic_regression": train_model_logistic_regression_optuna,

    "logistic_regression": train_model_logistic_regression,
    # "logistic_regression_cv": train_model_logistic_regression_cv,
    # "logistic_regression_optuna": train_model_logistic_regression_optuna,

    # "random_forest": train_model_rf_optuna,

    # "rf_cv": train_model_rf_cv,
    # "rf_optuna": train_model_rf_optuna,

    # "xgboost": train_model_xgboost_optuna,    

    # "xgboost_cv": train_model_xgboost_cv,
    # "xgboost_optuna": train_model_xgboost_optuna,
    # "tree": train_model_decision_tree,
}



# Create data nodes for each model
data_nodes = {}
for model in models:
    function_name = models[model].__name__
    data_nodes[model] = {
        "train_dataset_proba": Config.configure_data_node(
            id=f"train_dataset_proba_{model}"
        ),
        "train_dataset_metrics": Config.configure_data_node(
            id=f"train_dataset_metrics_{model}"
        ),
        "train_parameters": Config.configure_data_node(
            id=f"train_parameters_{model}", storage_type="json", default_data=train_parameters_list_default.get(model, {})
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
    input=[train_dataset_cfg, feature_selector_cfg],
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
        feature_selector_params_FeatureFilter_default_cfg,
        feature_selector_params_rfe_default_cfg,
        feature_selector_params_sfs_default_cfg,
        feature_selector_params_sfm_default_cfg,
        feature_selector_params_no_op_cfg,
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