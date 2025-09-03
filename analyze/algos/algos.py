# Import necessary libraries.
# Import various machine learning models and utilities from scikit-learn.
from sklearn.linear_model import LogisticRegression           # Logistic Regression model
from sklearn.ensemble import RandomForestClassifier           # Random Forest Classifier model
from sklearn.model_selection import train_test_split, GridSearchCV # Tools for data splitting and hyperparameter tuning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import make_classification
from sklearn.base import BaseEstimator
# from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif # Tools for feature selection
from sklearn.feature_selection \
    import \
        RFE, \
        SequentialFeatureSelector as SFS, \
        SelectFromModel, \
        SelectKBest, \
        VarianceThreshold, \
        f_classif # Tools for feature selection
from sklearn.tree import DecisionTreeClassifier               # Decision Tree model
from sklearn.metrics import roc_auc_score, fbeta_score, make_scorer, precision_score, accuracy_score # Model performance evaluation metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # Added Imputer for handling missing values
import optuna
from xgboost import XGBClassifier                           # XGBoost Classifier model (Gradient Boosting)
from lightgbm import LGBMClassifier
import shap                                                 # SHAP (SHapley Additive exPlanations) library, provides explainability for model predictions.
import matplotlib.pyplot as plt                             # Library for data visualization

# Import libraries for data processing and other tasks.
import pandas as pd                                         # Essential library for handling DataFrame structures
import numpy as np                                          # Library for numerical operations
from numpy import array, random, arange
import datetime as dt                                       # Library for handling dates and times
import json                                                 # Library for handling JSON formatted data
import pprint
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

import uuid
import os
import shutil
from typing import Dict, List, Tuple


def rescale_df(df: pd.DataFrame, scaler_type: str = 'standard') -> pd.DataFrame:
    """
    Rescales numeric variables in a DataFrame using a specified scaler.
    Non-numeric variables are kept as is.

    Args:
        df (pd.DataFrame): The input DataFrame to rescale.
        scaler_type (str): The type of scaler to use ('standard', 'minmax', 'robust').

    Returns:
        pd.DataFrame: The rescaled DataFrame (only numeric columns are transformed).
    
    # --- Example Usage ---

    # Create a DataFrame with various data types
    data = {
        'numerical_feature_1': [10, 20, 30, 40, 50],
        'numerical_feature_2': [100, 200, 300, 400, 500],
        'categorical_feature': ['A', 'B', 'A', 'C', 'B'],
        'object_feature': ['apple', 'banana', 'orange', 'grape', 'apple']
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*30 + "\n")

    # Rescale using MinMaxScaler
    df_minmax_rescaled = rescale_df(df, scaler_type='minmax')
    print("Rescaled DataFrame (using MinMaxScaler):")
    print(df_minmax_rescaled)
    print("\n" + "="*30 + "\n")

    # Rescale using RobustScaler
    df_robust_rescaled = rescale_df(df, scaler_type='robust')
    print("Rescaled DataFrame (using RobustScaler):")
    print(df_robust_rescaled)
    print("\n" + "="*30 + "\n")

    # Input with an unsupported scaler type
    df_error = rescale_df(df, scaler_type='unsupported_scaler')    
        
    """
    # Select scaler object based on type
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        print(f"Error: Unsupported scaler type '{scaler_type}'. Please choose from 'standard', 'minmax', 'robust'.")
        return df

    # Create a copy of the original DataFrame to avoid modifying it
    df_rescaled = df.copy()

    # Select only numeric (int, float) columns
    numeric_cols = df_rescaled.select_dtypes(include=np.number).columns
    
    # Check if there are any columns to rescale
    if numeric_cols.empty:
        print("Warning: No numeric columns found for scaling.")
        return df_rescaled

    try:
        # Apply fit_transform to the selected numeric columns
        df_rescaled[numeric_cols] = scaler.fit_transform(df_rescaled[numeric_cols])

    except ValueError as e:
        print(f"Error during scaling: {e}")
        print("Check for missing values (NaN) or infinite values in your data.")
        return df

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return df

    return df_rescaled


##############################################################################################################################
# 1) Baseline Model (Unchanged in logic) but keep in mind: it returns proba[:,1] as "pass"
# 1) Baseline model (no logic change), note: it returns proba[:,1] representing 'pass'.
##############################################################################################################################

# Defines a class named 'BaselineModel'. This model is a simple rule-based baseline model.
class BaselineModel:
    def __init__(self):
        # Initializes the baseline values.
        self.radius = 70  # Radius threshold
        self.sensor_offset_hot_cold = 0.02 # Sensor offset threshold
        pass

    # Function that returns prediction probabilities (similar to a machine learning model's predict_proba)
    def predict_proba(self, X):
        # Create boolean (True/False) series for each criterion.
        radius_criteria = X["Radius"] <= self.radius # True if radius is less than or equal to the threshold
        sensor_criteria = X["SensorOffsetHot-Cold"].abs() <= self.sensor_offset_hot_cold # True if the absolute sensor offset is less than or equal to the threshold
        bandgap_criteria = X["band gap dpat_ok for band gap"] == 1 # Checks if the bandgap criterion is met in the test data
        # Determine the final prediction by checking if all criteria are met.
        y_pred_baseline = radius_criteria & sensor_criteria & bandgap_criteria

        # proba[:,1] => "pass", proba[:,0] => "fail"
        # Create a numpy array with 2 columns to store prediction probabilities.
        proba = np.zeros((len(X), 2))
        # Store 1 for 'pass' (True) and 0 for 'fail' (False) in the 'pass' column.
        proba[:, 0] = y_pred_baseline.astype(int)  # 'Pass' probability (actually 0 or 1)
        # Store the inverse of the 'pass' probability in the 'fail' column, since 'fail' is the opposite of 'pass'.
        proba[:, 1] = 1 - proba[:, 0]  # 'Fail' probability
        return proba


##############################################################################################################################
# 2) Preprocessing: now invert labels => 1 = fail, 0 = pass
# 2) Data Preprocessing: Now invert the labels. => 1 = fail, 0 = pass
##############################################################################################################################

##############################################################################################################################
# 2) Preprocessing: Now invert the labels => 1 = fail, 0 = pass
##############################################################################################################################

def preprocess_dataset(initial_dataset: pd.DataFrame):
    num_col_select = 2000 # Temporarily select a subset of columns

    print("\n      Preprocessing dataset...")

    processed_dataset = initial_dataset.copy() # Copy the original dataset

    # --- Handling Missing Values ---
    # Create and fit a SimpleImputer object.
    # We use the 'mean' strategy to replace missing values with the mean.
    # Other strategies (median, most_frequent, etc.) can also be used.
    # Before applying the Imputer, remove columns where all values are NaN.
    processed_dataset.dropna(axis=1, how='all', inplace=True) # Remove columns where all values are NaN.
    # 1. Separate numeric and non-numeric (categorical) features.
    numeric_cols = processed_dataset.select_dtypes(include=np.number).columns.tolist() # Select numeric data type columns and convert to a list.
    categorical_cols = processed_dataset.select_dtypes(exclude=np.number).columns.tolist() # Select non-numeric data type columns and convert to a list.
    # For all columns that can be converted to numeric, coerce non-convertible values to NaN.
    for col in processed_dataset.columns: # Iterate through all columns of the dataset.
        # Use pd.to_numeric with the errors='coerce' option for forced conversion.
        # This operation is applied to a copy of the original data.
        processed_dataset[col] = pd.to_numeric(processed_dataset[col], errors='coerce').fillna(processed_dataset[col]) # Replace non-numeric values with NaN, and keep original values.
    # 2. Create an object for handling missing values (numeric only).
    # Now that all numeric columns are clean, the imputer will work correctly.
    numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # Create an Imputer object to fill NaNs with the mean.
    # 3. Apply the Imputer only to the numeric group.
    if numeric_cols: # If numeric columns exist, execute the following.
        imputed_numeric_data_array = numeric_imputer.fit_transform(processed_dataset[numeric_cols]) # Fill missing values in numeric columns with the mean.
        # Ensure the number of columns matches
        if imputed_numeric_data_array.shape[1] == len(numeric_cols): # Check if the number of columns in the transformed data matches the original number of numeric columns.
            imputed_numeric_data = pd.DataFrame( # Convert the transformed data to a DataFrame.
                imputed_numeric_data_array,
                columns=numeric_cols,
                index=processed_dataset.index
            )
        else: # If the number of columns doesn't match, print an error message.
            print("Error: Number of columns in imputed data does not match numeric columns.")
            imputed_numeric_data = pd.DataFrame(index=processed_dataset.index)
    else: # If there are no numeric columns, create an empty DataFrame.
        imputed_numeric_data = pd.DataFrame(index=processed_dataset.index)
    # Use the original data for non-numeric features as is
    imputed_categorical_data = processed_dataset[categorical_cols].copy() # Copy non-numeric columns to use them.
    # 4. Recombine the processed features based on the index.
    # Ensure the indices match before joining
    imputed_categorical_data.index = imputed_numeric_data.index # Make the indices of the two DataFrames consistent.
    processed_dataset = imputed_numeric_data.join(imputed_categorical_data) # Combine the imputed numeric data and the non-numeric data.
    # Verify the final result
    # print("--- Modified Result ---")
    # print(processed_dataset)



    # NOTE: Originally, "Pass/Fail_pass=1 => pass, Pass/Fail_pass=0 => fail"
    # We invert this to make "1 => fail". In other words, "fail = 1 - old_pass_value".
    # old_pass_value = processed_dataset["Pass/Fail_pass"] (1 for pass, 0 for fail)
    # new fail => 1 - old_pass_value
    processed_dataset["Pass/Fail"] = 1 - processed_dataset["Pass/Fail_pass"] # Invert the 'Pass/Fail_pass' column to create the 'Pass/Fail' column (1=fail, 0=pass)

    # Keep only target and features
    columns_to_drop = [
        'DevID',
        'WAFER_NO',
        'Pass/Fail_pass'
    ]
    # Drop columns (use inplace=True to modify the original DataFrame)
    # Or use processed_dataset = processed_dataset.drop(...) to create a new DataFrame
    processed_dataset.drop(columns=columns_to_drop, inplace=True)


    # --- Use only a portion of features for speed, num_col_select features ---
    # 1. Define the list of columns to exclude
    excluded_columns = [
        'X',  
        'Y',  
        'Pass/Fail',  
        'Radius'
    ]
    # 2. Get the list of remaining columns after exclusion
    all_columns = processed_dataset.columns.tolist()
    remaining_columns = [col for col in all_columns if col not in excluded_columns]
    # 3. Sort the remaining columns alphabetically
    remaining_columns.sort()
    # 4. Select the first 100 columns from the sorted list
    selected_columns = remaining_columns[:num_col_select]
    # 5. Create a new DataFrame with the selected columns (or update the existing one)
    # To update the existing processed_dataset with the selected columns:
    processed_dataset = processed_dataset[excluded_columns + selected_columns]

    # Remove columns with a single value
    # processed_dataset = drop_cols_1value(processed_dataset)

    processed_dataset = pd.get_dummies(processed_dataset, drop_first=True) # One-hot encode categorical columns (drop the first category)

    # Convert all columns to numeric type
    processed_dataset = processed_dataset.apply(pd.to_numeric)

    processed_dataset.fillna(processed_dataset.mean(), inplace=True) # Fill missing values with the mean of the respective column

    # Clean up column names
    processed_dataset.columns = (
        processed_dataset.columns.str.replace("[", "_", regex=False) # Replace '[' with '_'
        .str.replace("]", "_", regex=False) # Replace ']' with '_'
        .str.replace("<", "_", regex=False) # Replace '<' with '_'
        .str.replace(">", "_", regex=False) # Replace '>' with '_'
    )

    # Reorder columns to place the target column, 'Pass/Fail', last
    reorder_cols = [c for c in processed_dataset.columns if c not in ["Pass/Fail"]] # Select all columns except 'Pass/Fail'
    processed_dataset = processed_dataset[reorder_cols + ["Pass/Fail"]] # Reorder columns by adding 'Pass/Fail' at the end

    print("      Preprocessing complete!\n")

    return processed_dataset # Return the preprocessed dataset

# Automatically generate features
def feature_generator(X, sum_features=False, diff_features=False, poly_features=False, poly_degree=2, apply_filter_gen=False, var_threshold_gen=0.0, corr_threshold_gen=0.98):
    """
    Applies various feature generation techniques to numeric features of an input DataFrame X,
    based on selected options. Includes SimpleImputer to handle missing values (NaN).
    
    Args:
        X (pd.DataFrame): The original DataFrame.
        sum_features (bool): Whether to create pairwise sum features.
        diff_features (bool): Whether to create pairwise difference features.
        poly_features (bool): Whether to create polynomial features.
        poly_degree (int): The degree for polynomial features.
        apply_filter_gen (bool): Whether to apply variance_correlation_filter within feature_generator.
        var_threshold_gen (float): The variance threshold for internal filtering.
        corr_threshold_gen (float): The correlation threshold for internal filtering.

    Returns:
        (pd.DataFrame, dict): 
            - A new DataFrame with generated features added.
            - A dictionary with detailed information on the number of generated features.
    """
    
    X_in = X.copy()
    gen_counts = {'sum': 0, 'diff': 0, 'poly': 0}
    
    if apply_filter_gen:
        print(f"    - Applying filtering within Feature Generator (Variance: {var_threshold_gen}, Correlation: {corr_threshold_gen})")
        X_filtered, _, _, _ = variance_correlation_filter(X, var_threshold=var_threshold_gen, corr_threshold=corr_threshold_gen)
        X = X_filtered
    else:
        print("    - Not applying filtering within Feature Generator.")

    numerical_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(exclude=np.number).columns
    
    if len(numerical_features) == 0:
        print("Warning: No numeric features found. Skipping feature generation.")
        return X.copy(), gen_counts
    
    # DataFrame to hold the new features that will be generated
    X_generated = pd.DataFrame(index=X.index)

    if poly_features:
        poly_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False))
        ])
        
        poly_data = poly_pipeline.fit_transform(X[numerical_features])
        poly_feature_names = poly_pipeline.named_steps['poly'].get_feature_names_out(numerical_features)
        gen_counts['poly'] = len(poly_feature_names)
        X_generated = pd.concat([X_generated, pd.DataFrame(poly_data, columns=poly_feature_names, index=X.index)], axis=1)

    if sum_features or diff_features:
        def sum_diff_transformer_func(X_array):
            n_features = X_array.shape[1]
            gen_features = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if sum_features:
                        gen_features.append(X_array[:, i] + X_array[:, j])
                    if diff_features:
                        gen_features.append(X_array[:, i] - X_array[:, j])
            if not gen_features:
                return np.empty((X_array.shape[0], 0))
            return np.column_stack(gen_features)

        sum_diff_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('sum_diff', FunctionTransformer(sum_diff_transformer_func))
        ])
        
        sum_diff_data = sum_diff_pipeline.fit_transform(X[numerical_features])
        sum_diff_feature_names = []
        for i in range(len(numerical_features)):
            for j in range(i + 1, len(numerical_features)):
                if sum_features:
                    sum_diff_feature_names.append(f'{numerical_features[i]}_{numerical_features[j]}_sum')
                if diff_features:
                    sum_diff_feature_names.append(f'{numerical_features[i]}_{numerical_features[j]}_diff')
        
        if sum_features: gen_counts['sum'] = len([f for f in sum_diff_feature_names if 'sum' in f])
        if diff_features: gen_counts['diff'] = len([f for f in sum_diff_feature_names if 'diff' in f])

        X_generated = pd.concat([X_generated, pd.DataFrame(sum_diff_data, columns=sum_diff_feature_names, index=X.index)], axis=1)
        print(f"    - Number of features generated by Feature Generator: {X_generated.shape[1]}")

    # Combine original and generated columns
    # Remove duplicate columns and reorder so that original columns come first
    X_combined = pd.concat([X_in, X_generated], axis=1)
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]
    
    return X_combined, gen_counts

##############################################################################################################################
# 3) Create Train/Test Split
# 3) Create a training/test data split
##############################################################################################################################

# Function to create training and test datasets
def create_train_test_data(
    preprocessed_dataset: pd.DataFrame,
    split_parameter: dict = None
):
    """
    Function that applies training/test data splitting and sampling.
    Feature Generation application and options are controlled by split_parameter.
    Detailed processing results are added to split_parameter_info.
    """
    print("\n\n##############################################################################################################################")
    print("# 3) Create Train/Test Split ")
    print("##############################################################################################################################")
    
    print("\n      Creating training and test datasets...")

    # Set and update default values for split_parameter
    default_params = {
        'test_size': 0.2,
        'random_state': 42,
        'sampling_ratio': None,
        'apply_feature_generation': False,
        'sum_features': False,
        'diff_features': False,
        'poly_features': False,
        'poly_degree': 2,
        'apply_filter_split': False,
        'var_threshold_split': 0.0,
        'corr_threshold_split': 0.98,
        'apply_filter_gen': False,
        'var_threshold_gen': 0.0,
        'corr_threshold_gen': 0.98,
    }
    
    if split_parameter:
        default_params.update(split_parameter)
    split_parameter = default_params

    split_parameter_info = split_parameter.copy()
    
    # ----------------------------------------------------
    # Step 1: Remove outliers
    # ----------------------------------------------------
    initial_dataset_shape = preprocessed_dataset.shape
    outlier_mask = (preprocessed_dataset["Radius"] < 32) & (preprocessed_dataset["Pass/Fail"])
    preprocessed_dataset = preprocessed_dataset[~outlier_mask].reset_index(drop=True)
    split_parameter_info['rows_after_outlier_removal'] = preprocessed_dataset.shape[0]

    X = preprocessed_dataset.iloc[:, :-1]
    y = preprocessed_dataset.iloc[:, -1]
    
    # ----------------------------------------------------
    # Step 2: Apply filtering before splitting
    # ----------------------------------------------------
    split_parameter_info['features_before_split_filter'] = X.shape[1]
    if split_parameter['apply_filter_split']:
        print(f"    - Applying filtering before splitting (Variance: {split_parameter['var_threshold_split']}, Correlation: {split_parameter['corr_threshold_split']})")
        # X, _, var_dropped, corr_dropped = variance_correlation_filter(X, var_threshold=split_parameter['var_threshold_split'], corr_threshold=split_parameter['corr_threshold_split'])
        X, _, var_dropped, corr_dropped = filter_by_variance(X, 0)
        split_parameter_info['features_after_split_filter'] = X.shape[1]
        split_parameter_info['features_dropped_by_variance_split'] = var_dropped
        split_parameter_info['features_dropped_by_correlation_split'] = corr_dropped
    else:
        print("    - Not applying filtering before splitting.")
        split_parameter_info['features_after_split_filter'] = X.shape[1]
        split_parameter_info['features_dropped_by_variance_split'] = 0
        split_parameter_info['features_dropped_by_correlation_split'] = 0
        
    # ----------------------------------------------------
    # Step 3: Apply Feature Generation
    # ----------------------------------------------------
    split_parameter_info['original_feature_count'] = X.shape[1]
    if split_parameter['apply_feature_generation']:
        print("    - Applying Feature Generation...")
        X, gen_counts = feature_generator(
            X,  
            sum_features=split_parameter['sum_features'],
            diff_features=split_parameter['diff_features'],
            poly_features=split_parameter['poly_features'],
            poly_degree=split_parameter['poly_degree'],
            apply_filter_gen=split_parameter['apply_filter_gen'],
            var_threshold_gen=split_parameter['var_threshold_gen'],
            corr_threshold_gen=split_parameter['corr_threshold_gen']
        )
        split_parameter_info['generated_feature_counts'] = gen_counts
        split_parameter_info['total_generated_features'] = sum(gen_counts.values())
        split_parameter_info['features_after_generation'] = X.shape[1]
        
        generation_types = []
        if split_parameter['sum_features']: generation_types.append('sum')
        if split_parameter['diff_features']: generation_types.append('diff')
        if split_parameter['poly_features']: generation_types.append('poly')
        split_parameter_info['generation_types_applied'] = generation_types
        
        print("    - Feature Generation complete. New feature count:", split_parameter_info['total_generated_features'])
    else:
        print("    - Not applying Feature Generation.")
        split_parameter_info['generated_feature_counts'] = {'sum': 0, 'diff': 0, 'poly': 0}
        split_parameter_info['total_generated_features'] = 0
        split_parameter_info['features_after_generation'] = X.shape[1]

    # ----------------------------------------------------
    # Step 4: Split data into training/test sets
    # ----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,  
        test_size=split_parameter['test_size'],  
        random_state=split_parameter['random_state'],  
        stratify=y
    )
    
    split_parameter_info['train_samples_before_sampling'] = len(X_train)
    split_parameter_info['test_samples'] = len(X_test)
    
    train_class_distribution_before = dict(sorted(Counter(y_train).items()))
    split_parameter_info['class_distribution_before_sampling'] = train_class_distribution_before
    print(f"\n    - Training data class distribution before splitting: {train_class_distribution_before}")

    # ----------------------------------------------------
    # Step 5: Apply sampling
    # ----------------------------------------------------
    sampling_ratio = split_parameter['sampling_ratio']
    
    if sampling_ratio is not None:
        split_parameter_info['sampling_ratio_used'] = sampling_ratio
        if sampling_ratio >= 1:
            n_samples_majority = sum(y_train == 0)
            target_minority_count = int(n_samples_majority * sampling_ratio)
            sampling_strategy = {1: target_minority_count}
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=split_parameter['random_state'])
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f"    - Oversampling applied (minority class ratio: {sampling_ratio})")
            split_parameter_info['sampling_applied'] = 'oversampling'
        else:
            n_samples_minority = sum(y_train == 1)
            target_majority_count = int(n_samples_minority / sampling_ratio)
            sampling_strategy = {0: target_majority_count}
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=split_parameter['random_state'])
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f"    - Undersampling applied (minority class ratio: {sampling_ratio})")
            split_parameter_info['sampling_applied'] = 'undersampling'
            
        train_class_distribution_after = dict(sorted(Counter(y_train).items()))
        split_parameter_info['class_distribution_after_sampling'] = train_class_distribution_after
        split_parameter_info['train_samples_after_sampling'] = len(X_train)
        print(f"    - Training data class distribution after sampling: {train_class_distribution_after}")
    else:
        print("    - No sampling applied")
        split_parameter_info['sampling_applied'] = 'None'
        split_parameter_info['train_samples_after_sampling'] = len(X_train)

    # ----------------------------------------------------
    # Step 6: Combine and return the final DataFrames
    # ----------------------------------------------------
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    split_parameter_info['final_train_feature_count'] = train_data.shape[1] - 1
    
    return train_data, test_data, split_parameter_info

##############################################################################################################################
# 4) Custom F2 scorer with pos_label=1 (since 1 = fail/rare)
# 4) Custom F2 scorer with pos_label=1 (since 1 = fail/rare)
##############################################################################################################################

# Create a custom evaluation metric that calculates the F2 score using the 'fbeta_score' function.
# beta=2 gives a higher weight to Recall.
# > By weighting Recall twice, we optimize to minimize incorrectly classifying 'fail' items as 'pass'.
# pos_label=1 means the 'fail' class (1) is considered the positive class.
# f2_rare_scorer = make_scorer(fbeta_score, beta=2, pos_label=1)
f2_rare_scorer = make_scorer(fbeta_score, beta=4, pos_label=1)

# --- New Non-linear Correlation Function (Xi Cor) ---
def xicor_old(X, Y, ties=True):
    random.seed(42)
    n = len(X)
    order = array([i[0] for i in sorted(enumerate(X), key=lambda x: x[1])])
    if ties:
        l = array([sum(y >= Y[order]) for y in Y[order]])
        r = l.copy()
        for j in range(n):
            if sum([r[j] == r[i] for i in range(n)]) > 1:
                tie_index = array([r[j] == r[i] for i in range(n)])
                r[tie_index] = random.choice(r[tie_index] - arange(0, sum([r[j] == r[i] for i in range(n)])), sum(tie_index), replace=False)
        return 1 - n*sum( abs(r[1:] - r[:n-1]) ) / (2*sum(l*(n - l)))
    else:
        r = array([sum(y >= Y[order]) for y in Y[order]])
        return 1 - 3 * sum( abs(r[1:] - r[:n-1]) ) / (n**2 - 1)
# --- Function End ---

from typing import Literal, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import stats


def xicor(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    ties: Union[bool, Literal["auto"]] = "auto",
) -> Tuple[float, float]:
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = len(y)

    if len(x) != n:
        raise IndexError(f"x, y length mismatch: {len(x)}, {len(y)}")

    if ties == "auto":
        ties = len(np.unique(y)) < n
    elif not isinstance(ties, bool):
        raise ValueError(
            f'expected ties either "auto" or boolean, '
            f"got {ties} ({type(ties)}) instead"
        )

    y = y[np.argsort(x)]
    r = stats.rankdata(y, method="ordinal")
    nominator = np.sum(np.abs(np.diff(r)))

    if ties:
        l = stats.rankdata(y, method="max")
        denominator = 2 * np.sum(l * (n - l))
        nominator *= n
    else:
        denominator = np.power(n, 2) - 1
        nominator *= 3

    statistic = 1 - nominator / denominator  # upper bound is (n - 2) / (n + 1)
    p_value = stats.norm.sf(statistic, scale=2 / 5 / np.sqrt(n))

    return statistic, p_value



# --- Step-by-step modularization function for variable filtering w. correlation ---
def filter_by_variance(X: pd.DataFrame, var_threshold: float) -> Tuple[pd.DataFrame, Dict]:
    start_time = dt.datetime.now()
    
    X = rescale_df(X)
    
    initial_cols = list(X.columns)
    
    # Calculate the variance for all features.
    features_values_checked = X.var().to_dict()
    
    vt = VarianceThreshold(threshold=var_threshold)
    X_filtered = vt.fit_transform(X)
    vt_mask = vt.get_support()
    vt_cols = X.columns[vt_mask]

    # Store whether each feature was dropped.
    features_dropped_yn = {col: not vt_mask[i] for i, col in enumerate(initial_cols)}
    
    end_time = dt.datetime.now()
    duration = end_time - start_time
    duration_str = str(duration).split('.')[0]
    
    stats = {
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': duration_str,
        'threshold_value': var_threshold,
        'original_count': len(initial_cols),
        'remaining_count': len(vt_cols),
        'features_dropped_yn': features_dropped_yn,
        'features_values_checked': features_values_checked
    }
    print(f"    - Number of features remaining after variance filtering: {stats['remaining_count']}")
    return pd.DataFrame(X_filtered, columns=vt_cols, index=X.index), stats
def filter_by_target_linear_correlation(X: pd.DataFrame, y: pd.Series, threshold: float) -> Tuple[pd.DataFrame, Dict]:
    start_time = dt.datetime.now()
    
    initial_cols = list(X.columns)
    
    # Calculate the correlation between all features and the target.
    correlations = X.corrwith(y).abs()
    features_values_checked = correlations.to_dict()
    
    low_corr_features = correlations[correlations < threshold].index.tolist()
    X_filtered = X.drop(columns=low_corr_features)

    # Store whether each feature was dropped.
    features_dropped_yn = {col: col in low_corr_features for col in initial_cols}
    
    end_time = dt.datetime.now()
    duration = end_time - start_time
    duration_str = str(duration).split('.')[0]
    
    stats = {
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': duration_str,
        'threshold_value': threshold,
        'original_count': X.shape[1],
        'remaining_count': X_filtered.shape[1],
        'features_dropped_yn': features_dropped_yn,
        'features_values_checked': features_values_checked
    }
    print(f"    - Number of features remaining after target linear correlation filtering: {stats['remaining_count']}")
    return X_filtered, stats
def filter_by_target_xicor_correlation(X: pd.DataFrame, y: pd.Series, threshold: float) -> Tuple[pd.DataFrame, Dict]:
    start_time = dt.datetime.now()
    
    # y (타겟)을 수치형 또는 0/1로 변환
    if y.dtype == 'bool':
        y_processed = y.astype(int)
    elif pd.api.types.is_numeric_dtype(y):
        y_processed = y
    else:
        # y가 수치형 또는 불리언이 아닌 경우, 처리가 불가능하므로 오류 반환
        raise TypeError("Target 'y' must be a numeric or boolean type.")

    # X 데이터프레임의 복사본을 만들어 불리언 컬럼을 0/1로 변환
    X_processed = X.copy()
    for col in X_processed.select_dtypes(include='bool').columns:
        X_processed[col] = X_processed[col].astype(int)
    
    to_drop = []
    features_dropped_yn = {}
    features_values_checked = {}
    initial_cols = list(X.columns)
    
    for col in initial_cols:
        # 컬럼이 수치형 또는 불리언인지 확인
        if pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col]):
            try:
                # xicor는 변환된 데이터(X_processed, y_processed)를 사용
                xi_corr_val, p_value = xicor(X_processed[col].values, y_processed.values)
                is_dropped = abs(xi_corr_val) <= threshold
                features_dropped_yn[col] = str(is_dropped)
                features_values_checked[col] = xi_corr_val
                if is_dropped:
                    to_drop.append(col)
            except Exception as e:
                # xicor 계산 오류 시 드롭하지 않음
                print(f"Warning: Failed to calculate xicor for feature '{col}'. Error: {e}")
                features_dropped_yn[col] = 'False'
                features_values_checked[col] = None
        else:
            # 수치형/불리언이 아닌 컬럼은 드롭하지 않고, 정보도 저장하지 않음
            features_dropped_yn[col] = 'False'
    
    X_filtered = X.drop(columns=to_drop)
    
    end_time = dt.datetime.now()
    duration = end_time - start_time
    duration_str = str(duration).split('.')[0]
    
    stats = {
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': duration_str,
        'threshold_value': threshold,
        'original_count': X.shape[1],
        'remaining_count': X_filtered.shape[1],
        'features_dropped_yn': features_dropped_yn,
        'features_values_checked': features_values_checked
    }
    
    print(f"    - 타겟 Xi Cor 필터링 후 남은 피처 수: {stats['remaining_count']}")
    return X_filtered, stats
def filter_by_feature_linear_correlation(X: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, Dict]:
    start_time = dt.datetime.now()
    
    features_dropped_yn = {col: False for col in X.columns}
    features_values_checked = {}
    initial_cols = list(X.columns)
    
    to_drop = []
    
    if len(initial_cols) > 1:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
        
        # Store correlation values for all column pairs
        for i in range(len(upper.columns)):
            for j in range(i + 1, len(upper.columns)):
                col1 = upper.columns[i]
                col2 = upper.columns[j]
                
                # Get the correlation value for 'col1' and 'col2'.
                correlation_value = upper.loc[col1, col2]
                
                # Keep consistency by using a sorted tuple as the key.
                pair = tuple(sorted((col1, col2)))
                features_values_checked[str(pair)] = correlation_value
                
                # Add to the 'to_drop' list if it exceeds the threshold.
                if correlation_value > threshold:
                    if col2 not in to_drop:
                        to_drop.append(col2)
        
        # Update features_dropped_yn based on the 'to_drop' list.
        for col in to_drop:
            features_dropped_yn[col] = True

        X_filtered = X.drop(columns=to_drop)
    else:
        X_filtered = X

    end_time = dt.datetime.now()
    duration = end_time - start_time
    duration_str = str(duration).split('.')[0]
    
    stats = {
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': duration_str,
        'threshold_value': threshold,
        'original_count': X.shape[1],
        'remaining_count': X_filtered.shape[1],
        'features_dropped_yn': features_dropped_yn,
        'features_values_checked': features_values_checked
    }
    
    print(f"    - Remaining features after filtering by linear correlation: {stats['remaining_count']}")
    return X_filtered, stats
def filter_by_feature_xicor_correlation(X: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, Dict]:
    start_time = dt.datetime.now()
    
    to_drop = []
    features_dropped_yn = {col: False for col in X.columns}
    features_values_checked = {}
    initial_cols = list(X.columns)
    
    if len(initial_cols) > 1:
        # First, calculate and store the xi correlation for all column pairs.
        for i in range(len(initial_cols)):
            for j in range(i + 1, len(initial_cols)):
                col1 = initial_cols[i]
                col2 = initial_cols[j]
                
                # Calculate the non-linear correlation value using the 'xicor' function.
                # This value is always stored regardless of the threshold.
                xi_corr_val = xicor(X[col1].values, X[col2].values)
                pair_key = str(tuple(sorted((col1, col2))))
                features_values_checked[pair_key] = xi_corr_val
        
        # Now, determine which columns to drop based on the stored values.
        # This loop finds columns with high correlation and adds them to the 'to_drop' list.
        # The logic previously skipped has been removed, and all columns are checked again.
        for i in range(len(initial_cols)):
            for j in range(i + 1, len(initial_cols)):
                col1 = initial_cols[i]
                col2 = initial_cols[j]
                
                # Do not further check features that are already scheduled to be dropped.
                if col1 in to_drop or col2 in to_drop:
                    continue
                
                pair_key = str(tuple(sorted((col1, col2))))
                xi_corr_val = features_values_checked[pair_key] # Use the already calculated value
                
                if xi_corr_val > threshold:
                    # For simplicity, this implementation drops col2.
                    to_drop.append(col2)
                    features_dropped_yn[col2] = True
        
        X_filtered = X.drop(columns=to_drop, axis=1)
    else:
        X_filtered = X
        
    end_time = dt.datetime.now()
    duration = end_time - start_time
    duration_str = str(duration).split('.')[0]
    
    stats = {
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': duration_str,
        'threshold_value': threshold,
        'original_count': X.shape[1],
        'remaining_count': X_filtered.shape[1],
        'features_dropped_yn': features_dropped_yn,
        'features_values_checked': features_values_checked
    }
    
    print(f"    - Remaining features after filtering by non-linear correlation: {stats['remaining_count']}")
    return X_filtered, stats
# --- Change integrated filtering workflow function name ---
def feature_filter(X: pd.DataFrame, y: pd.Series, params: Dict) -> Tuple[pd.DataFrame, list, Dict]:
    X_filtered = X.copy()
    filter_stats = {}
    print("\n--- Starting Feature Filtering ---")

    # 1. Variance Filtering
    if params.get('apply_variance_filter', True):
        X_filtered, stats = filter_by_variance(X_filtered, params['var_threshold'])
        filter_stats['variance'] = stats
    # 2. Linear Correlation Filtering with Target
    if params.get('apply_target_linear_corr_filter', True) and X_filtered.shape[1] > 0:
        X_filtered, stats = filter_by_target_linear_correlation(X_filtered, y, params['target_linear_corr_threshold'])
        filter_stats['target_linear_correlation'] = stats
    # 3. Non-linear Correlation (Xi Cor) Filtering with Target
    if params.get('apply_target_xicor_filter', True) and X_filtered.shape[1] > 0:
        X_filtered, stats = filter_by_target_xicor_correlation(X_filtered, y, params['target_xicor_threshold'])
        filter_stats['target_xicor_correlation'] = stats
    # 4. Linear Correlation Filtering between Features
    if params.get('apply_feature_linear_corr_filter', True) and X_filtered.shape[1] > 1:
        X_filtered, stats = filter_by_feature_linear_correlation(X_filtered, params['feature_linear_corr_threshold'])
        filter_stats['feature_linear_correlation'] = stats
    # 5. Non-linear Correlation (Xi Cor) Filtering between Features
    if params.get('apply_feature_xicor_filter', True) and X_filtered.shape[1] > 1:
        X_filtered, stats = filter_by_feature_xicor_correlation(X_filtered, params['feature_xicor_threshold'])
        filter_stats['feature_xicor_correlation'] = stats
            
    final_cols = list(X_filtered.columns)
    return X_filtered, final_cols, filter_stats

# --- Variable filtering w.model step-by-step modularization function ---
def _get_estimator_old(estimator_params: Dict) -> BaseEstimator:
    """
    Creates a Scikit-learn estimator object based on the given parameter dictionary.
    """
    estimator_name = estimator_params.get("name")
    params = estimator_params.get("params", {})
    
    if estimator_name == "LogisticRegression":
        return LogisticRegression(random_state=42, n_jobs=-1, **params)
    elif estimator_name == "RandomForestClassifier":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    elif estimator_name == "LGBMClassifier":
        return LGBMClassifier(random_state=42, n_jobs=-1, **params)
    else:
        raise ValueError(f"Unsupported estimator: {estimator_name}")
def _get_estimator(estimator_params: Dict) -> BaseEstimator:
    """
    Creates a Scikit-learn estimator object based on the given parameter dictionary.
    """
    estimator_name = estimator_params.get("name")
    params = estimator_params.get("params", {})
    
    if estimator_name == "LogisticRegression":
        return LogisticRegression(random_state=42, n_jobs=-1, **params)
    elif estimator_name == "RandomForestClassifier":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    elif estimator_name == "LGBMClassifier":
        return LGBMClassifier(random_state=42, n_jobs=-1, **params)
    else:
        raise ValueError(f"Unsupported estimator: {estimator_name}")
# --- Add helper function for extracting variable importance ---
def _get_feature_importances(estimator: BaseEstimator, feature_names: List[str]) -> Dict[str, float]:
    """
    Helper function to extract variable importances from an estimator.
    """
    importances = {}
    if hasattr(estimator, 'feature_importances_'):
        importances = {col: imp for col, imp in zip(feature_names, estimator.feature_importances_)}
    elif hasattr(estimator, 'coef_'):
        # For models like LogisticRegression, use the coef_ attribute
        coefs = estimator.coef_[0] if estimator.coef_.ndim > 1 else estimator.coef_
        importances = {col: abs(coef) for col, coef in zip(feature_names, coefs)}
    return importances


def run_model_based_feature_selection(
    X_in: pd.DataFrame, 
    y: pd.Series, 
    selector_name: str, 
    selector_params: Dict
) -> Tuple[List[str], Dict]:
    """
    **A general function to selectively run RFE, SFS, or SelectFromModel.**
    
    Args:
        X (pd.DataFrame): Feature data
        y (pd.Series): Target data
        selector_name (str): Name of the variable selector to use ('RFE', 'SFS', 'SFM')
        selector_params (Dict): Dictionary of parameters to pass to the variable selector

    Returns:
        Tuple[List[str], Dict]: A list of selected features and a dictionary of statistics
    """
    start_time = dt.datetime.now()
    
    X = X_in.copy()
    
    initial_features = list(X.columns)
    
    estimator_params = selector_params.get("estimator")
    estimator = _get_estimator(estimator_params)
    
    selected_features = []
    dropped_features = []
    stats = {}
    
    if selector_name == 'RFE':
        n_features_to_select = selector_params.get('n_features_to_select')
        step = selector_params.get('step', 1)
        
        selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
        selector.fit(X, y)
        selected_mask = selector.get_support()
        selected_features = list(X.columns[selected_mask])
        
        # RFE provides a ranking. Add the ranking to stats.
        ranking = {col: rank for col, rank in zip(initial_features, selector.ranking_)}
        stats = {'method': 'RFE', 'n_features_to_select': n_features_to_select, 'step': step, 'ranking': ranking}

        # Extract importances of selected features and add to stats.
        if ranking and any(rank == 1 for rank in ranking.values()):
            selected_estimator = _get_estimator(estimator_params)
            selected_estimator.fit(X[selected_features], y)
            importances = _get_feature_importances(selected_estimator, selected_features)
            stats['importances'] = importances
            
    elif selector_name == 'SFM':
        threshold = selector_params.get('threshold', 'median')
        
        # First, fit the model and then pass it to SelectFromModel.
        estimator.fit(X, y)
        selector = SelectFromModel(estimator, prefit=True, threshold=threshold)
        selected_mask = selector.get_support()
        selected_features = list(X.columns[selected_mask])

        # Extract variable importances from the model.
        importances = _get_feature_importances(estimator, initial_features)
        stats = {'method': 'SFM', 'threshold': threshold, 'importances': importances}

    elif selector_name == 'SFS':
        n_features_to_select = selector_params.get('n_features_to_select')
        direction = selector_params.get('direction', 'forward')
        
        selector = SFS(estimator=estimator, n_features_to_select=n_features_to_select, direction=direction, cv=5)
        
        selector.fit(X, y)
        selected_features = list(X.columns[selector.get_support()])
        
        # SFS does not directly provide importances, so re-fit the model with selected variables to extract them.
        selected_estimator = _get_estimator(estimator_params)
        selected_estimator.fit(X[selected_features], y)
        importances = _get_feature_importances(selected_estimator, selected_features)

        stats = {
            'method': 'SFS', 
            'n_features_to_select': n_features_to_select, 
            'direction': direction,
            'importances': importances
        }
        
    else:
        raise ValueError(f"Unsupported selector name: {selector_name}. Choose from 'RFE', 'SFS', 'SFM'.")

    dropped_features = [col for col in initial_features if col not in selected_features]

    end_time = dt.datetime.now()
    duration_str = str(end_time - start_time).split('.')[0]

    stats.update({
        'start_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': duration_str,
        'original_count': len(initial_features),
        'remaining_count': len(selected_features),
        'selected_features': selected_features,
        'dropped_features': dropped_features,
    })
    
    print(f"--- {selector_name} Selector completed ---")
    print(f"Number of remaining features: {stats['remaining_count']}")
    
    return selected_features, stats

# Custom class for JSON serialization
class NpEncoder(json.JSONEncoder):
    """Class for serializing Numpy types (int64, float64, etc.) to JSON"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# --- Feature Selection Function ---
def select_feature(train_data: pd.DataFrame, feature_selector_params: Dict) -> Dict:
    """
    Performs the feature selection workflow and returns the results in a dictionary.
    """
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    feature_selection_info={}
    feature_selection_info = feature_selector_params.copy()
    initial_feature_count = X_train.shape[1]
    
    selector_name = feature_selector_params.get("feature_selector_name")
    final_features = list(X_train.columns)
    stats = {}
    
    print(f"\n--- Feature Selector: {selector_name} ---")

    try:
        if selector_name == 'FeatureFilter':
            filter_params = feature_selector_params['filter_methods']
            _, final_features, stats = feature_filter(X=X_train, y=y_train, params=filter_params)
        
        elif selector_name in ['RFE', 'SFM', 'SFS']:
            # selector_params = feature_selector_params.get(f"{selector_name.lower()}_params")
            selector_params = feature_selector_params.get("params")
            if not selector_params:
                raise KeyError(f"Missing '{selector_name.lower()}_params' in configuration.")
            
            final_features, stats = run_model_based_feature_selection(
                X_train,
                y_train,
                selector_name=selector_name,
                selector_params=selector_params
            )
        else:
            print("\n--- Feature Selector: No-op ---")
            stats = {'method': 'No-op', 'original_count': initial_feature_count, 'remaining_count': initial_feature_count, 'selected_features': final_features, 'dropped_features': []}
    
    except Exception as e:
        print(f"An error occurred during feature selection: {e}")
        # In case of an error, return original features
        stats = {'method': 'Error', 'original_count': initial_feature_count, 'remaining_count': initial_feature_count, 'selected_features': final_features, 'dropped_features': [], 'error_message': str(e)}

    final_feature_count = len(final_features)
    feature_selection_info['initial_feature_count'] = initial_feature_count
    feature_selection_info['final_feature_count'] = final_feature_count
    feature_selection_info['final_features'] = final_features
    feature_selection_info['selection_details'] = stats

    # (Modify save logic)
    destination_dir = 'data/result/jsons'
    os.makedirs(destination_dir, exist_ok=True)
    current_time = dt.datetime.now()
    file_id = uuid.uuid4().hex[:8]
    filename = "feature_selection_info_" + current_time.strftime('%y%m%d_%H%M%S') + f'_{file_id}.json'
    file_path = os.path.join(destination_dir, filename)

    with open(file_path, 'w', encoding='utf-8') as f:
        # Use a class encoder to convert numpy int64 to Python int
        json.dump(feature_selection_info, f, indent=4, ensure_ascii=False, cls=NpEncoder)

    print(f"\nFeature selection results saved to '{file_path}' file.")
    print(f"\n- Final feature count: {final_feature_count}")
    
    feature_selection_info['feature_selection_info_json_path'] = file_path
    
    return feature_selection_info





##############################################################################################################################
# 5) Logistic Regression (simple) - Now up-weight class 1
# 5) Logistic Regression (simple) - Now up-weight class 1
##############################################################################################################################

# Function to train a Logistic Regression model
def train_model_logistic_regression(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    print("      Training Logistic Regression (no CV)...\n")
    
    model_parameter_info = {}

    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
    final_features = feature_selection_info['final_features']
    X = X[final_features]

    # Set parameters based on train_parameters
    if train_parameters and train_parameters.get('function_name') == 'train_model_logistic_regression':
        solver = train_parameters.get('solver', 'lbfgs')
        max_iter = train_parameters.get('max_iter', 1000)
    else:
        solver = 'lbfgs'
        max_iter = 1000
    
    # Store configured parameter information
    model_parameter_info['solver'] = solver
    model_parameter_info['max_iter'] = max_iter

    # Set class weights to handle class imbalance
    n_pos = sum(y)
    n_neg = len(y) - n_pos

    # If the value is None or the key is not in the dictionary, set to n_neg
    # if train_parameters.get('class_weight_multiplier') == '':
    #      class_weight_multiplier = n_neg
    # else:
    #      class_weight_multiplier = eval(train_parameters.get('class_weight_multiplier'))
    # class_weight = {0: 1, 1: class_weight_multiplier}

    # Assign class weights to solve the class imbalance problem
    class_weight_multiplier = train_parameters.get('class_weight_multiplier', n_neg) if train_parameters else n_neg
    class_weight = {0: 1, 1: class_weight_multiplier}


    model_parameter_info['class_weight_multiplier'] = train_parameters.get('class_weight_multiplier')    
    model_fitted = LogisticRegression(
        class_weight=class_weight,
        solver=solver,
        max_iter=max_iter
    ).fit(X, y)
    
    print("\n    LogisticRegression is trained!")

    importance_dict = {
        "Features": X.columns,
        "Importance": model_fitted.coef_[0],
        "Importance_abs": np.abs(model_fitted.coef_[0]),
    }
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    )
    
    return model_fitted, importance, model_parameter_info


##############################################################################################################################
# 5a) Logistic Regression with CV + F2 scoring
# 5a) Logistic Regression with cross-validation (CV) and F2 scoring
##############################################################################################################################
def train_model_logistic_regression_cv(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    print("      Training Logistic Regression with cross-validation & hyperparameter tuning...\n")
    
    model_parameter_info = {}
    
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
    final_features = feature_selection_info['final_features']
    X = X[final_features]

    # Set parameter grid and GridSearchCV parameters based on train_parameters
    if train_parameters and train_parameters.get('function_name') == 'train_model_logistic_regression_cv':
        param_grid = train_parameters.get('param_grid', {})
        cv = train_parameters.get('cv', 3)
        verbose = train_parameters.get('verbose', 1)

        # Apply f2_rare_scorer setting logic
        scoring_params = train_parameters.get('f2_rare_scorer', {})
        if scoring_params.get('name') == 'fbeta_score':
            beta = scoring_params.get('beta', 2)
            pos_label = scoring_params.get('pos_label', 1)
            scorer = make_scorer(fbeta_score, beta=beta, pos_label=pos_label)
        else:
            scorer = make_scorer(f2_rare_scorer, greater_is_better=True)
    else:
        # Default hyperparameter settings
        param_grid = {"C": [0.01, 0.1, 1], "penalty": ["l2"], "solver": ["lbfgs"]}
        cv = 3
        verbose = 1
        scorer = make_scorer(f2_rare_scorer, greater_is_better=True)

    # Store configured parameter information
    model_parameter_info['param_grid'] = param_grid
    model_parameter_info['cv'] = cv
    model_parameter_info['verbose'] = verbose
    if 'scorer' in locals():
        model_parameter_info['f2_rare_scorer'] = {
            'name': 'fbeta_score',
            'beta': scorer._kwargs.get('beta'),
            'pos_label': scorer._kwargs.get('pos_label')
        }

    # Assign class weights to solve the class imbalance problem
    n_neg = len(y) - sum(y)
    class_weight_multiplier = train_parameters.get('class_weight_multiplier', n_neg) if train_parameters else n_neg
    class_weight = {0: 1, 1: class_weight_multiplier}
    
    model_parameter_info['class_weight_multiplier'] = class_weight_multiplier

    lr = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)

    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        verbose=verbose,
        n_jobs=-1,
    )

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    print(f"\n    Best parameters found: {grid_search.best_params_}")
    print(f"    Best F2 (class=1) score (CV): {grid_search.best_score_:.4f}\n")
    
    model_parameter_info['best_params'] = grid_search.best_params_

    importance_dict = {
        "Features": X.columns,
        "Importance": best_model.coef_[0],
        "Importance_abs": np.abs(best_model.coef_[0]),
    }
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    )

    return best_model, importance, model_parameter_info    


##############################################################################################################################
# 6) Baseline Model (Unchanged)
# 6) Baseline Model (Unchanged)
##############################################################################################################################
# Function to train a baseline model (actually creates an object and pre-defines importances)
def train_model_baseline(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    model_parameter_info = {}
    model_fitted = BaselineModel()

    # Set feature importances based on train_parameters
    if train_parameters and train_parameters.get('function_name') == 'train_model_baseline':
        importance_data = train_parameters.get('importance_data', {})
    else:
        importance_data = {
            "Features": ["SensorOffsetHot-Cold", "band gap dpat_ok for band gap", "Radius"],
            "Importance": [56.6, 4.65, 96.9],
        }
    
    # Store configured parameter information
    model_parameter_info['importance_data'] = importance_data

    importance_dict = {
        "Features": importance_data["Features"],
        "Importance": importance_data["Importance"],
        "Importance_abs": np.abs(importance_data["Importance"]),
    }
    
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    )
    
    return model_fitted, importance, model_parameter_info


##############################################################################################################################
# 7) Random Forest (simple) - Up-weight class 1
# 7) Random Forest (simple) - Up-weight class 1
##############################################################################################################################

# (Commented out code)
# def train_model_random_forest(train_dataset: pd.DataFrame):
#      print("      Training RandomForest (no CV)...\n")
#      X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]

#      class_weight = {
#          0: 1,
#          1: sum(1 - y),
#      }

#      model_fitted = RandomForestClassifier(
#          class_weight=class_weight, random_state=42
#      ).fit(X, y)
#      print("\n    RandomForestClassifier is trained!")

#      importance_dict = {
#          "Features": X.columns,
#          "Importance": model_fitted.feature_importances_,
#          "Importance_abs": np.abs(model_fitted.feature_importances_),
#      }
#      importance = pd.DataFrame(importance_dict).sort_values(
#          by="Importance", ascending=True
#      )
#      return model_fitted, importance

##############################################################################################################################
# 7a) RandomForest with CV & GridSearch using F2 on class=1
# 7a) RandomForest with cross-validation (CV) and GridSearch using F2 on class=1
##############################################################################################################################

# Function to train a Random Forest model with cross-validation and hyperparameter tuning
def train_model_rf_cv(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    print("      Training the Random Forest model with cross-validation & hyperparameter tuning...\n")
    
    model_parameter_info = {}
    
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
    final_features = feature_selection_info['final_features']
    X = X[final_features]

    # Set parameter grid and GridSearchCV parameters based on train_parameters
    if train_parameters and train_parameters.get('function_name') == 'train_model_rf_cv':
        param_grid = train_parameters.get('param_grid', {})
        cv = train_parameters.get('cv', 3)
        verbose = train_parameters.get('verbose', 1)
        
        # Apply f2_rare_scorer setting logic
        scoring_params = train_parameters.get('f2_rare_scorer', {})
        if scoring_params.get('name') == 'fbeta_score':
            beta = scoring_params.get('beta', 2)
            pos_label = scoring_params.get('pos_label', 1)
            scorer = make_scorer(fbeta_score, beta=beta, pos_label=pos_label)
        else:
            scorer = make_scorer(f2_rare_scorer, greater_is_better=True)
            
    else:
        param_grid = {
            "n_estimators": [20, 50, 100],
            "max_depth": [2, 5, None],
            "min_samples_split": [2, 5],
        }
        cv = 3
        verbose = 1
        scorer = make_scorer(f2_rare_scorer, greater_is_better=True)
    
    # Store configured parameter information
    model_parameter_info['param_grid'] = param_grid
    model_parameter_info['cv'] = cv
    model_parameter_info['verbose'] = verbose
    if 'f2_rare_scorer' in locals():
        model_parameter_info['f2_rare_scorer'] = {
            'name': 'fbeta_score',
            'beta': scorer._kwargs.get('beta'),
            'pos_label': scorer._kwargs.get('pos_label')
        }

    # Assign class weights to solve the class imbalance problem
    # n_neg = len(y) - sum(y)
    n_neg = len(y) - sum(int(i) for i in y)

    # If the value is None or the key is not in the dictionary, set to n_neg
    if train_parameters.get('class_weight_multiplier') == '':
        class_weight_multiplier = n_neg
    else:
        class_weight_multiplier = eval(train_parameters.get('class_weight_multiplier'))
        
    class_weight = {0: 1, 1: class_weight_multiplier}

    model_parameter_info['class_weight_multiplier'] = train_parameters.get('class_weight_multiplier')    
    
    rf_model = RandomForestClassifier(class_weight=class_weight, random_state=42)

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        verbose=verbose,
        n_jobs=-1,
    )

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    print(f"\n    Best parameters found: {grid_search.best_params_}")
    print(f"    Best F2 (class=1) score (CV): {grid_search.best_score_:.4f}\n")
    
    model_parameter_info['best_params'] = grid_search.best_params_

    importance_dict = {
        "Features": X.columns,
        "Importance": best_model.feature_importances_,
        "Importance_abs": np.abs(best_model.feature_importances_),
    }
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    )
    
    return best_model, importance, model_parameter_info

##############################################################################################################################
# 8) Decision Tree
# 8) Decision Tree
##############################################################################################################################

# Function to train a Decision Tree model
def train_model_decision_tree(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
    final_features = feature_selection_info['final_features']
    X = X[final_features]
    
    model_parameter_info = {}

    # Set parameters based on train_parameters
    if train_parameters and train_parameters.get('function_name') == 'train_model_decision_tree':
        max_depth = train_parameters.get('max_depth', None)
        min_samples_split = train_parameters.get('min_samples_split', 2)
    else:
        max_depth = None
        min_samples_split = 2
        
    # Store configured parameter information
    model_parameter_info['max_depth'] = max_depth
    model_parameter_info['min_samples_split'] = min_samples_split

    # Set class weights to handle class imbalance
    n_neg = len(y) - sum(y)
    class_weight_multiplier = train_parameters.get('class_weight_multiplier', n_neg) if train_parameters else n_neg
    class_weight = {0: 1, 1: class_weight_multiplier}
    
    model_parameter_info['class_weight_multiplier'] = class_weight_multiplier
    
    model_fitted = DecisionTreeClassifier(
        class_weight=class_weight,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    ).fit(X, y)
    
    print("\n    DecisionTreeClassifier is trained!")

    importance_dict = {
        "Features": X.columns,
        "Importance": model_fitted.feature_importances_,
        "Importance_abs": np.abs(model_fitted.feature_importances_),
    }
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    )
    
    return model_fitted, importance, model_parameter_info    

##############################################################################################################################
# 9) XGBoost with CV & F2 on class=1
##############################################################################################################################

# Function to train an XGBoost model with cross-validation and hyperparameter tuning
def train_model_xgboost_cv(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    print(
        "     Training the XGBoost model with cross-validation & hyperparameter tuning...\n"
    )

    model_parameter_info = {}

    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]

    final_features = feature_selection_info['final_features']
    X = X[final_features]

    # Set parameters based on the algorithm name in train_parameters
    if train_parameters and train_parameters.get('function_name') == 'train_model_xgboost_cv':
        param_grid = train_parameters.get('param_grid', {})
        scale_pos_weight_multiplier = train_parameters.get('scale_pos_weight_multiplier', 2)
        cv = train_parameters.get('cv', 3)
        verbose = train_parameters.get('verbose', 1)
        
        # Apply logic for f2_rare_scorer
        scoring_params = train_parameters.get('f2_rare_scorer', {})
        if scoring_params.get('name') == 'fbeta_score':
            beta = scoring_params.get('beta', 2)
            pos_label = scoring_params.get('pos_label', 1)
            f2_rare_scorer = make_scorer(fbeta_score, beta=beta, pos_label=pos_label)
        else:
            # Default F2 score
            f2_rare_scorer = make_scorer(lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=2, pos_label=1))
            
    else:
        # Default hyperparameter settings
        param_grid = {
            "n_estimators": [30, 50, 100, 200],
            "max_depth": [2, 5],
            "learning_rate": [0.01, 0.1, 0.2],
        }
        scale_pos_weight_multiplier = 2
        cv = 3
        verbose = 1
        f2_rare_scorer = make_scorer(lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=2, pos_label=1))

    # Save the configured parameter information
    model_parameter_info['param_grid'] = param_grid
    model_parameter_info['scale_pos_weight_multiplier'] = scale_pos_weight_multiplier
    model_parameter_info['cv'] = cv
    model_parameter_info['verbose'] = verbose
    model_parameter_info['f2_rare_scorer'] = {
        'name': 'fbeta_score',
        'beta': f2_rare_scorer._kwargs.get('beta'),
        'pos_label': f2_rare_scorer._kwargs.get('pos_label')
    }

    # Calculate 'scale_pos_weight' for class imbalance
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos * scale_pos_weight_multiplier if n_pos > 0 else 1

    # Create the XGBoost model object
    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )

    # Configure GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring=f2_rare_scorer,
        cv=cv,
        verbose=verbose,
        n_jobs=-1,
    )

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    print(f"\n     Best parameters found: {grid_search.best_params_}")
    print(f"     Best F2 (class=1) score: {grid_search.best_score_:.4f}\n")
    
    model_parameter_info['best_params'] = grid_search.best_params_

    # Calculate feature importances for the best model
    importance_dict = {
        "Features": X.columns,
        "Importance": best_model.feature_importances_,
        "Importance_abs": np.abs(best_model.feature_importances_),
    }
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    )

    return best_model, importance, model_parameter_info


##############################################################################################################################
# Function to train a Logistic Regression model optimized with Optuna
##############################################################################################################################
def train_model_logistic_regression_optuna(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    """
    Function to optimize Logistic Regression model hyperparameters using Optuna.
    Dynamically sets Optuna search ranges using the train_parameters dictionary.
    """
    print("     Training Logistic Regression with Optuna hyperparameter tuning...\n")
    
    # model_parameter_info = {}
    model_parameter_info = train_parameters
    
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
    final_features = feature_selection_info['final_features']
    X = X[final_features]
    
    n_pos = sum(y)
    n_neg = len(y) - n_pos

    # Set Optuna parameters and study based on train_parameters
    if train_parameters and train_parameters.get('function_name') == 'train_model_logistic_regression_optuna':
        n_trials = train_parameters.get('n_trials', 30)
        param_ranges = train_parameters.get('param_ranges', {})
        
        # Extract search ranges (use defaults if none provided)
        c_range = param_ranges.get('C', [1e-3, 10])
        solver_list = param_ranges.get('solver', ['liblinear', 'lbfgs'])
        max_iter_val = param_ranges.get('max_iter', 1000)
        class_weight_multiplier_range = param_ranges.get('class_weight_multiplier', [1, n_neg])
        
        # Set F2 scorer
        scoring_params = train_parameters.get('f2_rare_scorer', {})
        if scoring_params.get('name') == 'fbeta_score':
            beta = scoring_params.get('beta', 2)
            pos_label = scoring_params.get('pos_label', 1)
            scorer = make_scorer(fbeta_score, beta=beta, pos_label=pos_label)
        else:
            scorer = f2_rare_scorer
            
    else:
        # Default settings (suitable for quick execution)
        n_trials = 30
        c_range = [1e-3, 10]
        solver_list = ['liblinear', 'lbfgs']
        max_iter_val = 1000
        class_weight_multiplier_range = [1, n_neg]
        scorer = f2_rare_scorer

    # Save configured parameter information
    model_parameter_info['n_trials'] = n_trials
    model_parameter_info['param_ranges'] = {
        'C': c_range,
        'solver': solver_list,
        'max_iter': max_iter_val,
        'class_weight_multiplier': class_weight_multiplier_range
    }
    
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', c_range[0], c_range[1], log=True),
            'solver': trial.suggest_categorical('solver', solver_list),
            'class_weight_multiplier': trial.suggest_int(
                'class_weight_multiplier', 
                class_weight_multiplier_range[0], 
                class_weight_multiplier_range[1]
            )
        }
        
        class_weight = {0: 1, 1: params.pop('class_weight_multiplier')}
        
        lr = LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter_val,
            random_state=42,
            **params
        )

        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        score = cross_val_score(lr, X, y, cv=kf, scoring=scorer, n_jobs=-1).mean()
        
        return score
    
    # Create a study to maximize the F2 score
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_class_weight_multiplier = best_params.pop('class_weight_multiplier')
    best_class_weight = {0: 1, 1: best_class_weight_multiplier}
    
    best_model = LogisticRegression(
        class_weight=best_class_weight,
        max_iter=max_iter_val,
        random_state=42,
        **best_params
    ).fit(X, y)

    print(f"\n     Best parameters found (Optuna): {study.best_params}")
    print(f"     Best F2 (class=1) score (CV): {study.best_value:.4f}\n")
    
    model_parameter_info['best_params'] = study.best_params
    
    importance_dict = {
        "Features": X.columns,
        "Importance": best_model.coef_[0],
        "Importance_abs": np.abs(best_model.coef_[0]),
    }
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    )
    
    return best_model, importance, model_parameter_info


##############################################################################################################################
# Function to train a Random Forest model optimized with Optuna
##############################################################################################################################
def train_model_rf_optuna(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    print("     Training the Random Forest model with Optuna hyperparameter tuning...\n")
    
    model_parameter_info = {}
    
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
    final_features = feature_selection_info['final_features']
    X = X[final_features]
    
    n_neg = len(y) - sum(y)

    # Get Optuna-related settings from train_parameters
    if train_parameters and train_parameters.get('function_name') == 'train_model_rf_optuna':
        n_trials = train_parameters.get('n_trials', 30)
        cv = train_parameters.get('cv', 3)
        param_ranges = train_parameters.get('param_ranges', {
            'n_estimators': {'low': 50, 'high': 150},
            'max_depth': {'low': 5, 'high': 20},
            'min_samples_split': {'low': 2, 'high': 10},
            'min_samples_leaf': {'low': 1, 'high': 5},
            'max_features': {'choices': ['sqrt', 'log2', 0.8]}
        })
        if 'class_weight_multiplier' not in param_ranges:
            param_ranges['class_weight_multiplier'] = {'low': 1, 'high': n_neg}
        
        # Add f2_rare_scorer setting logic
        scoring_params = train_parameters.get('f2_rare_scorer', {})
        if scoring_params.get('name') == 'fbeta_score':
            beta = scoring_params.get('beta', 2)
            pos_label = scoring_params.get('pos_label', 1)
            scorer = make_scorer(fbeta_score, beta=beta, pos_label=pos_label)
        else:
            scorer = make_scorer(f2_rare_scorer, greater_is_better=True)
            
    else:
        n_trials = 30
        cv = 3
        param_ranges = {
            'n_estimators': {'low': 50, 'high': 150},
            'max_depth': {'low': 5, 'high': 20},
            'min_samples_split': {'low': 2, 'high': 10},
            'min_samples_leaf': {'low': 1, 'high': 5},
            'max_features': {'choices': ['sqrt', 'log2', 0.8]},
            'class_weight_multiplier': {'low': 1, 'high': n_neg}
        }
        scorer = make_scorer(f2_rare_scorer, greater_is_better=True)

    model_parameter_info['n_trials'] = n_trials
    model_parameter_info['cv'] = cv
    model_parameter_info['param_ranges'] = param_ranges
    if 'f2_rare_scorer' in locals():
        model_parameter_info['f2_rare_scorer'] = {
            'name': 'fbeta_score',
            'beta': scorer._kwargs.get('beta'),
            'pos_label': scorer._kwargs.get('pos_label')
        }

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', param_ranges['n_estimators']['low'], param_ranges['n_estimators']['high']),
            'max_depth': trial.suggest_int('max_depth', param_ranges['max_depth']['low'], param_ranges['max_depth']['high']),
            'min_samples_split': trial.suggest_int('min_samples_split', param_ranges['min_samples_split']['low'], param_ranges['min_samples_split']['high']),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', param_ranges['min_samples_leaf']['low'], param_ranges['min_samples_leaf']['high']),
            'max_features': trial.suggest_categorical('max_features', param_ranges['max_features']['choices']),
            'class_weight_multiplier': trial.suggest_int('class_weight_multiplier', param_ranges['class_weight_multiplier']['low'], param_ranges['class_weight_multiplier']['high'])
        }

        class_weight = {0: 1, 1: params.pop('class_weight_multiplier')}
        
        rf_model = RandomForestClassifier(
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            **params
        )

        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        score = cross_val_score(rf_model, X, y, cv=kf, scoring=scorer, n_jobs=-1).mean()
        
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_class_weight_multiplier = best_params.pop('class_weight_multiplier')
    best_class_weight = {0: 1, 1: best_class_weight_multiplier}

    best_model = RandomForestClassifier(
        class_weight=best_class_weight,
        random_state=42,
        n_jobs=-1,
        **best_params
    ).fit(X, y)

    print(f"\n      Best parameters found (Optuna): {study.best_params}")
    print(f"      Best F2 (class=1) score (CV): {study.best_value:.4f}\n")
    
    model_parameter_info['best_params'] = study.best_params
    model_parameter_info['best_class_weight_multiplier'] = best_class_weight_multiplier

    importance_dict = {
        "Features": X.columns,
        "Importance": best_model.feature_importances_,
        "Importance_abs": np.abs(best_model.feature_importances_),
    }
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    )
    
    return best_model, importance, model_parameter_info


##############################################################################################################################
# Function to train an XGBoost model optimized with Optuna
##############################################################################################################################
def train_model_xgboost_optuna(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    print("     Training the XGBoost model with Optuna hyperparameter tuning...\n")
    
    model_parameter_info = {}
    
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
    final_features = feature_selection_info['final_features']
    X = X[final_features]
    
    n_pos = sum(y)
    n_neg = len(y) - n_pos

    # Get Optuna-related settings from train_parameters
    if train_parameters and train_parameters.get('function_name') == 'train_model_xgboost_optuna':
        n_trials = train_parameters.get('n_trials', 30)
        cv = train_parameters.get('cv', 3)
        param_ranges = train_parameters.get('param_ranges', {
            'n_estimators': {'low': 50, 'high': 150},
            'max_depth': {'low': 3, 'high': 10},
            'learning_rate': {'low': 0.01, 'high': 0.2},
            'subsample': {'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'low': 0.6, 'high': 1.0},
            'gamma': {'low': 0.0, 'high': 0.2},
            'reg_alpha': {'low': 1e-8, 'high': 1.0},
            'reg_lambda': {'low': 1e-8, 'high': 1.0}
        })
        # Complement scale_pos_weight to be searchable dynamically using a ratio_multiplier
        ratio_multiplier_range = train_parameters.get('ratio_multiplier_range', {'low': 0.5, 'high': 2.0})

        # Apply logic for f2_rare_scorer
        scoring_params = train_parameters.get('f2_rare_scorer', {})
        if scoring_params.get('name') == 'fbeta_score':
            beta = scoring_params.get('beta', 2)
            pos_label = scoring_params.get('pos_label', 1)
            f2_rare_scorer = make_scorer(fbeta_score, beta=beta, pos_label=pos_label)
        else:
            # Default F2 score
            f2_rare_scorer = make_scorer(lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=2, pos_label=1))
            
    else:
        n_trials = 30
        cv = 3
        param_ranges = {
            'n_estimators': {'low': 50, 'high': 150},
            'max_depth': {'low': 3, 'high': 10},
            'learning_rate': {'low': 0.01, 'high': 0.2},
            'subsample': {'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'low': 0.6, 'high': 1.0},
            'gamma': {'low': 0.0, 'high': 0.2},
            'reg_alpha': {'low': 1e-8, 'high': 1.0},
            'reg_lambda': {'low': 1e-8, 'high': 1.0}
        }
        ratio_multiplier_range = {'low': 0.5, 'high': 2.0}
        f2_rare_scorer = make_scorer(lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=2, pos_label=1))
        
    model_parameter_info['n_trials'] = n_trials
    model_parameter_info['cv'] = cv
    model_parameter_info['param_ranges'] = param_ranges
    model_parameter_info['ratio_multiplier_range'] = ratio_multiplier_range
    model_parameter_info['f2_rare_scorer'] = {
        'name': 'fbeta_score',
        'beta': f2_rare_scorer._kwargs.get('beta'),
        'pos_label': f2_rare_scorer._kwargs.get('pos_label')
    }


    def objective(trial):
        # Set search ranges for Optuna parameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', param_ranges['n_estimators']['low'], param_ranges['n_estimators']['high']),
            'max_depth': trial.suggest_int('max_depth', param_ranges['max_depth']['low'], param_ranges['max_depth']['high']),
            'learning_rate': trial.suggest_float('learning_rate', param_ranges['learning_rate']['low'], param_ranges['learning_rate']['high'], log=True),
            'subsample': trial.suggest_float('subsample', param_ranges['subsample']['low'], param_ranges['subsample']['high']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', param_ranges['colsample_bytree']['low'], param_ranges['colsample_bytree']['high']),
            'gamma': trial.suggest_float('gamma', param_ranges['gamma']['low'], param_ranges['gamma']['high']),
            'reg_alpha': trial.suggest_float('reg_alpha', param_ranges['reg_alpha']['low'], param_ranges['reg_alpha']['high'], log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', param_ranges['reg_lambda']['low'], param_ranges['reg_lambda']['high'], log=True),
        }

        # Search for scale_pos_weight for class imbalance
        base_scale = n_neg / n_pos if n_pos > 0 else 1
        ratio_multiplier = trial.suggest_float('ratio_multiplier', ratio_multiplier_range['low'], ratio_multiplier_range['high'])
        scale_pos_weight = base_scale * ratio_multiplier
        
        xgb_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            **params
        )

        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        score = cross_val_score(xgb_model, X, y, cv=kf, scoring=f2_rare_scorer, n_jobs=-1).mean()
        
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    
    # Separate ratio_multiplier from best_params to calculate scale_pos_weight
    best_ratio_multiplier = best_params.pop('ratio_multiplier')
    base_scale = n_neg / n_pos if n_pos > 0 else 1
    best_scale_pos_weight = base_scale * best_ratio_multiplier

    best_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=best_scale_pos_weight,
        n_jobs=-1,
        **best_params
    ).fit(X, y)

    print(f"\n     Best parameters found (Optuna): {study.best_params}")
    print(f"     Best F2 (class=1) score (CV): {study.best_value:.4f}\n")
    
    model_parameter_info['best_params'] = study.best_params
    model_parameter_info['best_ratio_multiplier'] = best_ratio_multiplier
    model_parameter_info['best_scale_pos_weight'] = best_scale_pos_weight
    
    importance_dict = {
        "Features": X.columns,
        "Importance": best_model.feature_importances_,
        "Importance_abs": np.abs(best_model.feature_importances_),
    }
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    )
    
    return best_model, importance, model_parameter_info

##############################################################################################################################
# 10) Prediction & Evaluation Helper Functions
##############################################################################################################################

# Function to generate labels for the Confusion Matrix
def _confusion_label(row):
    # Now, "1" is 'fail', which is considered positive.
    # row["Historical"] = actual label, row["Forecast"] = predicted label
    if row["Historical"] == 1 and row["Forecast"] == 1:
        return "True Fail (TP)" # Correctly predicted a fail as a fail
    elif row["Historical"] == 0 and row["Forecast"] == 0:
        return "True Pass (TN)" # Correctly predicted a pass as a pass
    elif row["Historical"] == 0 and row["Forecast"] == 1:
        return "False Fail (FP)" # Incorrectly predicted a pass as a fail (error)
    else:  # row["Historical"] == 1 and row["Forecast"] == 0
        return "Missed Fail (FN)" # Incorrectly predicted a fail as a pass (missed)


# Function to find the optimal threshold that maximizes the F2 score
def find_best_threshold(best_model, train_dataset, feature_selection_info: dict):
    """
    Finds the optimal threshold for classification that maximizes the F2 score.

    Parameters:
    - best_model: A trained classifier model with a `predict_proba` method.
    - train_dataset: DataFrame containing features and the target.

    Returns:
    - best_threshold: The optimal threshold that maximizes the F2 score.
    """
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]

    # If not a baseline model, reconstruct X by loading features from 'final_features.json'
    if not isinstance(best_model, BaselineModel):
        # with open("final_features.json", "r") as f:
        #     final_features = json.load(f)
        final_features = feature_selection_info['final_features']
        X = X[final_features]

    # Get the probability of class 1 (fail) predicted by the model.
    prob_class1 = best_model.predict_proba(X)[:, 1]

    # Try 100 threshold candidates from 0 to 1.
    thresholds = np.linspace(0, 1, 100)
    f2_scores = []

    for threshold in thresholds:
        y_pred = (prob_class1 >= threshold).astype(int) # Generate predicted labels based on the threshold
        # score = fbeta_score(y, y_pred, beta=2, pos_label=1) # Calculate F2 score
        score = fbeta_score(y, y_pred, beta=4, pos_label=1) # Calculate F2 score
        f2_scores.append(score)

    # Find the threshold that recorded the highest F2 score.
    best_idx = np.argmax(f2_scores)
    best_threshold = thresholds[best_idx]
    best_f2_score = f2_scores[best_idx]

    print(
        f"Best threshold for F2 score: {best_threshold:.4f} with F2 score: {best_f2_score:.4f}"
    )

    # ------------------------------------------------------------------
    # Apply the user-selected threshold.
    # ------------------------------------------------------------------

    # Add 'Probability' and 'Historical' columns to the training dataset.
    train_dataset["Probability"] = prob_class1
    train_dataset["Historical"] = y
    return train_dataset, best_threshold

# Function to create confusion matrix metrics on the training dataset
def create_metrics_on_train(train_dataset, threshold):
    """
    After training, predicts on the training dataset with a given threshold (for class 1, fail).
    """
    # ------------------------------------------------------------------
    # Apply the user-selected threshold.
    # ------------------------------------------------------------------
    # Predict 1 if 'Probability' is greater than or equal to the threshold, otherwise 0.
    forecast = (train_dataset["Probability"] >= threshold).astype(int)

    train_dataset["Forecast"] = forecast
    # Apply confusion matrix labels.
    train_dataset["True/False/Positive/Negative"] = train_dataset.apply(
        _confusion_label, axis=1
    )
    return train_dataset


# Function to perform predictions on the test dataset
def forecast(test_dataset: pd.DataFrame, trained_model, feature_selection_info: dict):
    print("     Forecasting the test dataset...")
    X = test_dataset.iloc[:, :-1]

    # If not a baseline model, reconstruct X by loading features from 'final_features.json'.
    if not isinstance(trained_model, BaselineModel):
        # with open("final_features.json", "r") as f:
        #     final_features = json.load(f)
        final_features = feature_selection_info['final_features']
        X = X[final_features]


    # Get the prediction probabilities for class 1.
    predictions = trained_model.predict_proba(X)[:, 1]
    print("     Forecasting done!")

    # Use SHAP to analyze the explainability of model predictions.
    # Use TreeExplainer for tree-based models, otherwise use KernelExplainer.
    if hasattr(trained_model, "feature_importances_"):
        explainer = shap.TreeExplainer(trained_model)
    elif not isinstance(trained_model, BaselineModel):
        explainer = shap.Explainer(trained_model, X)
    
    # Calculate SHAP values only if it's not a baseline model.
    if not isinstance(trained_model, BaselineModel):
        shap_values = explainer(X)
        # (Commented-out code) Plot SHAP summary plot.
        # plt.figure(figsize=(10, 5))
        # shap.summary_plot(shap_values, X, max_display=10, show=False)
        # plt.show()
    else:
        shap_values = None

    return predictions, [shap_values, X]


# Function to calculate the ROC curve from scratch
def roc_from_scratch(probabilities, test_dataset, partitions=100):
    print("     Calculation of the ROC curve...")
    y_test = test_dataset.iloc[:, -1] # Actual labels of the test data

    roc = []
    # Iterate through 101 thresholds from 0 to 1.
    for i in range(partitions + 1):
        thr = i / partitions
        threshold_vector = (probabilities >= thr).astype(int) # Predict based on the threshold
        tpr, fpr = true_false_positive(threshold_vector, y_test) # Calculate TPR and FPR
        roc.append([fpr, tpr])

    # Create a DataFrame with the calculated TPR and FPR.
    roc_data = pd.DataFrame(roc, columns=["False positive rate", "True positive rate"])
    print("     Calculation done")
    print("     Scoring...")

    # Calculate the AUC score using scikit-learn's 'roc_auc_score'.
    auc_score = roc_auc_score(y_test, probabilities)
    print("     Scoring done\n")
    return roc_data, auc_score


# Function to calculate TPR (True Positive Rate) and FPR (False Positive Rate)
def true_false_positive(threshold_vector: np.array, y_test: np.array):
    # "1" is 'fail', which is positive.
    true_positive = (threshold_vector == 1) & (y_test == 1) # TP: pred=1 & actual=1
    false_positive = (threshold_vector == 1) & (y_test == 0) # FP: pred=1 & actual=0
    true_negative = (threshold_vector == 0) & (y_test == 0) # TN: pred=0 & actual=0
    false_negative = (threshold_vector == 0) & (y_test == 1) # FN: pred=0 & actual=1

    # Calculate TPR: TP / (TP + FN)
    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum() + 1e-9)
    # Calculate FPR: FP / (FP + TN)
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum() + 1e-9)
    return tpr, fpr


# Function to generate various performance metrics based on prediction results
def create_metrics(
    predictions: np.array, test_dataset: pd.DataFrame, auc_score, threshold
):
    print("     Creating the metrics...")
    # Generate final predicted labels based on the threshold.
    threshold_vector = (predictions >= threshold).astype(int)

    y_test = test_dataset.iloc[:, -1]

    # Calculate TP, TN, FP, FN values.
    tp = ((threshold_vector == 1) & (y_test == 1)).sum()
    tn = ((threshold_vector == 0) & (y_test == 0)).sum()
    fp = ((threshold_vector == 1) & (y_test == 0)).sum()
    fn = ((threshold_vector == 0) & (y_test == 1)).sum()

    # Calculate F1 score (for class 1)
    denom = 2 * tp + fp + fn
    if denom == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * tp / denom
    f1_score = np.around(f1_score, 2) # Round to two decimal places

    # Calculate Accuracy
    accuracy = np.around((tp + tn) / (tp + tn + fp + fn + 1e-9), 2)
    # Round AUC score
    auc_score = np.around(auc_score, 2)

    # Store TP, TN, FP, FN values in a dictionary.
    dict_ftpn = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    number_of_good_predictions = tp + tn
    number_of_false_predictions = fp + fn

    # Calculate Precision and Recall
    if (tp + fp) == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    precision = np.around(precision, 2)

    if (tp + fn) == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    recall = np.around(recall, 2)

    # Return all metrics in a dictionary.
    metrics = {
        "f1_score": f1_score,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
        "auc_score": auc_score,
        "dict_ftpn": dict_ftpn,
        "number_of_predictions": len(predictions),
        "number_of_good_predictions": number_of_good_predictions,
        "number_of_false_predictions": number_of_false_predictions,
    }

    return metrics


# Function to organize prediction results into a DataFrame
def create_results(forecast_values, test_dataset, threshold):
    # Create a series of predicted probabilities, rounded to two decimal places.
    forecast_series_proba = pd.Series(
        np.around(forecast_values, decimals=2),
        index=test_dataset.index,
        name="Probability",
    )
    # Create a series of predicted labels (0 or 1) based on the threshold.
    forecast_series = pd.Series(
        (forecast_values > threshold).astype(int),
        index=test_dataset.index,
        name="Forecast",
    )
    # Create a series of actual labels.
    true_series = pd.Series(
        test_dataset.iloc[:, -1], name="Historical", index=test_dataset.index
    )
    # Create a series containing the index numbers.
    index_series = pd.Series(
        range(len(true_series)), index=test_dataset.index, name="Id"
    )

    # Combine all series into a single DataFrame.
    results = pd.concat(
        [index_series, forecast_series_proba, forecast_series, true_series], axis=1
    )
    # Add confusion matrix labels.
    results["True/False/Positive/Negative"] = results.apply(_confusion_label, axis=1)
    return results

##### util function for data processing

def filter_features_user(features):
    """
    Returns a sorted list of features from the given list, excluding certain items and including others.
    """
    # Load data from data/features_user_excluded.csv into a features_user_excluded list
    features_user_excluded_df = pd.read_csv('data/features_user_excluded.csv')
    features_user_excluded = features_user_excluded_df.iloc[:, 0].tolist()

    # Load data from data/features_user_included.csv into a features_user_included list
    features_user_included_df = pd.read_csv('data/features_user_included.csv')
    features_user_included = features_user_included_df.iloc[:, 0].tolist()

    # Convert to sets to remove and add items
    features_set = set(features)
    excluded_set = set(features_user_excluded)
    included_set = set(features_user_included)

    # 1. Remove items from 'features' that are in 'features_user_excluded'
    features_filtered_set = features_set - excluded_set

    # 2. Add items from 'features_user_included' to 'features_filtered_set'
    features_filtered_set.update(included_set)

    # --- Add sorting logic from here ---
    
    # Extract and sort included_set items that are in the final result
    included_sorted = sorted(list(features_filtered_set.intersection(included_set)))
    
    # Extract the remaining items (not in included_set)
    remaining_features = features_filtered_set - included_set
    
    # Sort the remaining items
    remaining_sorted = sorted(list(remaining_features))
    
    # Combine the included_sorted and remaining_sorted lists to create the final list
    features_filtered = included_sorted + remaining_sorted
    
    return features_filtered

from functools import reduce # Import the reduce function.

def export_features_json(final_features, fn_json):
    features_filtered = filter_features_user(final_features)
    with open(fn_json, "w") as f:
        json.dump(features_filtered, f, indent=4) # Save the final_features list to a JSON file (4-space indent)
    return features_filtered # Return the final_features list.
    
def print_object_attributes(obj):
    """
    Function to print all attributes of an object, one per line.
    
    Args:
        obj: The object whose attributes are to be printed.
    """
    if not hasattr(obj, '__dict__'):
        print(f"'{type(obj).__name__}' object does not have attributes.")
        return

    print(f"--- {type(obj).__name__} Object Attributes ---")
    attributes = vars(obj)
    
    # Iterate through the dictionary and print the attribute name and value on each line
    for key, value in attributes.items():
        print(f"{key}: {value}")
    
    return attributes