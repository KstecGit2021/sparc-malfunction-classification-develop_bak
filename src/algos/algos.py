# 필요한 라이브러리들을 불러옵니다.
# scikit-learn에서 여러 머신러닝 모델과 유틸리티를 가져옵니다.
from sklearn.linear_model import LogisticRegression         # 로지스틱 회귀 모델
from sklearn.ensemble import RandomForestClassifier         # 랜덤 포레스트 분류 모델
from sklearn.model_selection import train_test_split, GridSearchCV # 데이터 분할 및 하이퍼파라미터 튜닝을 위한 도구
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import make_classification
from sklearn.base import BaseEstimator
# from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif # 특징(변수) 선택을 위한 도구
from sklearn.feature_selection \
    import \
        RFE, \
        SequentialFeatureSelector as SFS, \
        SelectFromModel, \
        SelectKBest, \
        VarianceThreshold, \
        f_classif # 특징(변수) 선택을 위한 도구
from sklearn.tree import DecisionTreeClassifier             # 의사결정나무 모델
from sklearn.metrics import roc_auc_score, fbeta_score, make_scorer, precision_score, accuracy_score # 모델 성능 평가 지표
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # 누락된 값 처리를 위한 Imputer 추가
import optuna
from xgboost import XGBClassifier                           # XGBoost 분류 모델 (경사 부스팅)
from lightgbm import LGBMClassifier
import shap                                                 # SHAP (SHapley Additive exPlanations) 라이브러리, 모델 예측에 대한 설명력을 제공합니다.
import matplotlib.pyplot as plt                             # 데이터 시각화를 위한 라이브러리

# 데이터 처리 및 기타 작업을 위한 라이브러리들을 불러옵니다.
import pandas as pd                                         # 데이터프레임 구조를 다루는 데 필수적인 라이브러리
import numpy as np                                          # 숫자 연산을 위한 라이브러리
from numpy import array, random, arange
import datetime as dt                                       # 날짜와 시간을 다루는 라이브러리
import json                                                 # JSON 형식의 데이터를 처리하는 라이브러리
import pprint
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

import uuid
import os
import shutil
from typing import Dict, List, Tuple


def rescale_df(df: pd.DataFrame, scaler_type: str = 'standard') -> pd.DataFrame:
    """
    지정된 스케일러를 사용하여 데이터프레임의 숫자형 변수를 리스케일링합니다.
    숫자형이 아닌 변수들은 그대로 유지합니다.

    Args:
        df (pd.DataFrame): 리스케일링할 입력 데이터프레임
        scaler_type (str): 사용할 스케일러 타입 ('standard', 'minmax', 'robust')

    Returns:
        pd.DataFrame: 리스케일링된 데이터프레임 (숫자형 컬럼만 변환됨)
    
    # --- 사용 예시 ---

    # 다양한 타입의 데이터프레임 생성
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

    # MinMaxScaler를 사용하여 리스케일링
    df_minmax_rescaled = rescale_df(df, scaler_type='minmax')
    print("Rescaled DataFrame (using MinMaxScaler):")
    print(df_minmax_rescaled)
    print("\n" + "="*30 + "\n")

    # RobustScaler를 사용하여 리스케일링
    df_robust_rescaled = rescale_df(df, scaler_type='robust')
    print("Rescaled DataFrame (using RobustScaler):")
    print(df_robust_rescaled)
    print("\n" + "="*30 + "\n")

    # 지원하지 않는 스케일러 타입 입력 시
    df_error = rescale_df(df, scaler_type='unsupported_scaler')    
    
        
    """
    # 스케일러 타입에 따라 객체 선택
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        print(f"Error: Unsupported scaler type '{scaler_type}'. Please choose from 'standard', 'minmax', 'robust'.")
        return df

    # 원본 데이터프레임의 복사본을 만들어 원본이 수정되지 않도록 함
    df_rescaled = df.copy()

    # 숫자형(int, float) 컬럼만 선택
    numeric_cols = df_rescaled.select_dtypes(include=np.number).columns
    
    # 리스케일링할 컬럼이 없는 경우
    if numeric_cols.empty:
        print("Warning: No numeric columns found for scaling.")
        return df_rescaled

    try:
        # 선택된 숫자형 컬럼에 대해 fit_transform 적용
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
# 1) 기준 모델 (논리 변경 없음), 주의: 'pass'를 나타내는 proba[:,1]을 반환합니다.
##############################################################################################################################

# 'BaselineModel'이라는 클래스를 정의합니다. 이 모델은 간단한 규칙 기반의 기준 모델입니다.
class BaselineModel:
    def __init__(self):
        # 기준이 되는 값들을 초기화합니다.
        self.radius = 70  # 반지름(Radius) 기준값
        self.sensor_offset_hot_cold = 0.02 # 센서 오프셋 기준값
        pass

    # 예측 확률을 반환하는 함수 (머신러닝 모델의 predict_proba와 유사)
    def predict_proba(self, X):
        # 각 기준에 따라 불리언(True/False) 시리즈를 생성합니다.
        radius_criteria = X["Radius"] <= self.radius # 반지름이 기준값 이하면 True
        sensor_criteria = X["SensorOffsetHot-Cold"].abs() <= self.sensor_offset_hot_cold # 센서 오프셋의 절대값이 기준값 이하면 True
        bandgap_criteria = X["band gap dpat_ok for band gap"] == 1 # 테스트 데이터에서 밴드갭 기준 충족 여부
        # 모든 기준을 만족하는지 확인하여 최종 예측값을 결정합니다.
        y_pred_baseline = radius_criteria & sensor_criteria & bandgap_criteria

        # proba[:,1] => "pass", proba[:,0] => "fail"
        # 예측 확률을 저장할 2열짜리 넘파이 배열을 생성합니다.
        proba = np.zeros((len(X), 2))
        # 기준을 통과(True)한 경우 1, 불통과(False)한 경우 0을 'pass' 열에 저장합니다.
        proba[:, 0] = y_pred_baseline.astype(int)  # 통과(pass) 확률 (실제로는 0 또는 1)
        # 'pass'가 아닌 경우 'fail'이므로, 'pass' 확률의 역을 'fail' 열에 저장합니다.
        proba[:, 1] = 1 - proba[:, 0]  # 불통과(fail) 확률
        return proba


##############################################################################################################################
# 2) Preprocessing: now invert labels => 1 = fail, 0 = pass
# 2) 데이터 전처리: 이제 레이블을 반전시킵니다. => 1 = 불합격(fail), 0 = 합격(pass)
##############################################################################################################################

##############################################################################################################################
# 2) 전처리: 이제 레이블을 반전합니다 => 1 = 실패, 0 = 통과
##############################################################################################################################

def preprocess_dataset(initial_dataset: pd.DataFrame):
    num_col_select = 2000 # 임시로 일부 컬럼만 선택하여 진행

    print("\n     데이터셋 전처리 중...")

    processed_dataset = initial_dataset.copy() # 원본 데이터셋 복사

    # --- 결측치 대체 ---
    # SimpleImputer 객체 생성 및 fit
    # 'mean' 전략을 사용하여 평균값으로 결측치를 대체합니다.
    # 다른 전략(median, most_frequent 등)을 사용할 수도 있습니다.
    # Imputer 적용 전, 모든 값이 NaN인 열을 제거합니다.
    processed_dataset.dropna(axis=1, how='all', inplace=True) # 모든 값이 NaN인 열을 제거합니다.
    # 1. 수치형 피처와 비 수치형(범주형) 피처 분리
    numeric_cols = processed_dataset.select_dtypes(include=np.number).columns.tolist() # 숫자 데이터 타입 열을 선택하여 리스트로 만듭니다.
    categorical_cols = processed_dataset.select_dtypes(exclude=np.number).columns.tolist() # 숫자가 아닌 데이터 타입 열을 선택하여 리스트로 만듭니다.
    # 숫자형으로 변환 가능한 모든 열에 대해, 변환할 수 없는 값은 NaN으로 처리
    for col in processed_dataset.columns: # 데이터셋의 모든 열을 반복합니다.
        # pd.to_numeric을 errors='coerce' 옵션과 함께 사용하여 강제 변환
        # 이 작업은 원본 데이터의 복사본에 적용됨
        processed_dataset[col] = pd.to_numeric(processed_dataset[col], errors='coerce').fillna(processed_dataset[col]) # 숫자로 변환할 수 없는 값을 NaN으로 바꾸고, 원본 값은 유지합니다.
    # 2. 결측치 처리를 위한 객체 생성 (수치형만)
    # 이제 모든 수치형 열이 깨끗해졌으므로 imputer가 정상적으로 작동합니다.
    numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # NaN을 평균값으로 채우는 Imputer 객체를 생성합니다.
    # 3. 수치형 그룹에만 Imputer 적용
    if numeric_cols: # 수치형 열이 존재하면 다음을 실행합니다.
        imputed_numeric_data_array = numeric_imputer.fit_transform(processed_dataset[numeric_cols]) # 수치형 열의 결측치를 평균으로 채웁니다.
        # Ensure the number of columns matches
        if imputed_numeric_data_array.shape[1] == len(numeric_cols): # 변환된 데이터의 열 개수가 원래 수치형 열 개수와 일치하는지 확인합니다.
            imputed_numeric_data = pd.DataFrame( # 변환된 데이터를 DataFrame으로 변환합니다.
                imputed_numeric_data_array,
                columns=numeric_cols,
                index=processed_dataset.index
            )
        else: # 열 개수가 일치하지 않으면 에러 메시지를 출력합니다.
            print("Error: Number of columns in imputed data does not match numeric columns.")
            imputed_numeric_data = pd.DataFrame(index=processed_dataset.index)
    else: # 수치형 열이 없으면 빈 DataFrame을 만듭니다.
        imputed_numeric_data = pd.DataFrame(index=processed_dataset.index)
    # 비 수치형 피처는 원본 데이터를 그대로 사용
    imputed_categorical_data = processed_dataset[categorical_cols].copy() # 비 수치형 열은 복사하여 사용합니다.
    # 4. 처리된 피처들을 인덱스 기준으로 다시 결합
    # Ensure the indices match before joining
    imputed_categorical_data.index = imputed_numeric_data.index # 두 DataFrame의 인덱스를 일치시킵니다.
    processed_dataset = imputed_numeric_data.join(imputed_categorical_data) # 결측치가 채워진 수치형 데이터와 비 수치형 데이터를 합칩니다.
    # 최종 결과 확인
    # print("--- 수정된 결과 ---")
    # print(processed_dataset)



    # NOTE: 원래 "Pass/Fail_pass=1 => 통과, Pass/Fail_pass=0 => 실패"
    # 우리는 "1 => 실패"가 되도록 반전합니다. 즉, "실패 = 1 - old_pass_value" 입니다.
    # old_pass_value = processed_dataset["Pass/Fail_pass"] (통과이면 1, 실패이면 0)
    # new fail => 1 - old_pass_value
    processed_dataset["Pass/Fail"] = 1 - processed_dataset["Pass/Fail_pass"] # 'Pass/Fail_pass' 컬럼을 반전하여 'Pass/Fail' 컬럼 생성 (1=실패, 0=통과)

    # keep cols only target and features
    columns_to_drop = [
        'DevID',
        'WAFER_NO',
        'Pass/Fail_pass'
    ]
    # 컬럼 drop (원본 DataFrame을 변경하려면 inplace=True 사용)
    # 또는 새로운 DataFrame을 만들려면 processed_dataset = processed_dataset.drop(...) 사용
    processed_dataset.drop(columns=columns_to_drop, inplace=True)


    # --- 속도위해 일부 피처만 사용 , num_col_select 개수의 피처 ---
    # 1. 제외할 컬럼 목록 정의
    excluded_columns = [
        'X', 
        'Y', 
        'Pass/Fail', 
        'Radius'
    ]
    # 2. 제외할 컬럼을 뺀 나머지 컬럼 목록 가져오기
    all_columns = processed_dataset.columns.tolist()
    remaining_columns = [col for col in all_columns if col not in excluded_columns]
    # 3. 나머지 컬럼들을 알파벳 순서로 정렬
    remaining_columns.sort()
    # 4. 정렬된 컬럼 중 처음 100개 선택
    selected_columns = remaining_columns[:num_col_select]
    # 5. 선택된 컬럼들로 새로운 DataFrame 생성 (또는 기존 DataFrame 업데이트)
    # 기존 processed_dataset을 선택된 컬럼들로 업데이트하려면:
    processed_dataset = processed_dataset[excluded_columns + selected_columns]

    # 단일 값의 컬럼들 제거
    # processed_dataset = drop_cols_1value(processed_dataset)

    processed_dataset = pd.get_dummies(processed_dataset, drop_first=True) # 범주형 컬럼을 원-핫 인코딩 (첫 번째 카테고리 드롭)

    # 모든 컬럼을 숫자형으로 변환
    processed_dataset = processed_dataset.apply(pd.to_numeric)

    processed_dataset.fillna(processed_dataset.mean(), inplace=True) # 누락된 값을 해당 컬럼의 평균으로 채움

    # 컬럼 이름 정리
    processed_dataset.columns = (
        processed_dataset.columns.str.replace("[", "_", regex=False) # '['를 '_'로 대체
        .str.replace("]", "_", regex=False) # ']'를 '_'로 대체
        .str.replace("<", "_", regex=False) # '<'를 '_'로 대체
        .str.replace(">", "_", regex=False) # '>'를 '_'로 대체
    )

    # 최종 컬럼이 타겟컬럼 즉 Pass/Fail이 되도록 순서 재정렬
    reorder_cols = [c for c in processed_dataset.columns if c not in ["Pass/Fail"]] # 'Pass/Fail'을 제외한 모든 컬럼 선택
    processed_dataset = processed_dataset[reorder_cols + ["Pass/Fail"]] # 'Pass/Fail'을 마지막에 추가하여 컬럼 순서 재정렬

    print("     전처리 완료!\n")

    return processed_dataset # 전처리된 데이터셋 반환

# 자동으로 피처를 생성
def feature_generator(X, sum_features=False, diff_features=False, poly_features=False, poly_degree=2, apply_filter_gen=False, var_threshold_gen=0.0, corr_threshold_gen=0.98):
    """
    입력 데이터프레임 X에 대해 수치형 피처에 다양한 피처 생성 기법을 선택적으로 적용합니다.
    누락된 값(NaN)을 처리하기 위해 SimpleImputer를 포함합니다.
    
    Args:
        X (pd.DataFrame): 원본 데이터프레임.
        sum_features (bool): 쌍별 합 피처를 생성할지 여부.
        diff_features (bool): 쌍별 차 피처를 생성할지 여부.
        poly_features (bool): 다항식 피처를 생성할지 여부.
        poly_degree (int): 다항식 피처의 차수.
        apply_filter_gen (bool): feature_generator 내에서 variance_correlation_filter 적용 여부.
        var_threshold_gen (float): 내부 필터링을 위한 분산 임계값.
        corr_threshold_gen (float): 내부 필터링을 위한 상관관계 임계값.

    Returns:
        (pd.DataFrame, dict): 
            - 생성된 피처가 추가된 새로운 데이터프레임.
            - 생성된 피처 수에 대한 상세 정보 딕셔너리.
    """
    
    X_in = X.copy()
    gen_counts = {'sum': 0, 'diff': 0, 'poly': 0}
    
    if apply_filter_gen:
        print(f"    - Feature Generator 내부에서 필터링 적용 (분산: {var_threshold_gen}, 상관관계: {corr_threshold_gen})")
        X_filtered, _, _, _ = variance_correlation_filter(X, var_threshold=var_threshold_gen, corr_threshold=corr_threshold_gen)
        X = X_filtered
    else:
        print("    - Feature Generator 내부 필터링 미적용")

    numerical_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(exclude=np.number).columns
    
    if len(numerical_features) == 0:
        print("경고: 수치형 피처가 없습니다. 피처 생성을 건너뜁니다.")
        return X.copy(), gen_counts
    
    # 생성될 새로운 피처들을 담을 데이터프레임
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
        print(f"    - Feature Generator에서 생성된 피처 수: {X_generated.shape[1]}")

    # 원본 컬럼과 생성된 컬럼을 결합
    # 중복되는 컬럼을 제거하고 원본 컬럼이 앞에 오도록 재정렬
    X_combined = pd.concat([X_in, X_generated], axis=1)
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]
    
    return X_combined, gen_counts

##############################################################################################################################
# 3) Create Train/Test Split
# 3) 훈련/테스트 데이터 분할
##############################################################################################################################

# 훈련 및 테스트 데이터셋을 생성하는 함수
def create_train_test_data_old(
    preprocessed_dataset: pd.DataFrame,
    split_parameter: dict = None
):
    """
    훈련/테스트 데이터 분할 및 샘플링을 적용하는 함수.
    Feature Generation 적용 여부 및 옵션을 split_parameter에서 제어합니다.
    split_parameter_info에 처리 결과 정보를 세부적으로 추가합니다.
    """
    print("\n\n##############################################################################################################################")
    print("# 3) Create Train/Test Split (훈련/테스트 데이터 분할) ")
    print("##############################################################################################################################")
    
    print("\n    훈련 및 테스트 데이터셋 생성 중...")

    # split_parameter의 기본값 설정 및 업데이트
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
    # Step 1: 아웃라이어 제거
    # ----------------------------------------------------
    initial_dataset_shape = preprocessed_dataset.shape
    outlier_mask = (preprocessed_dataset["Radius"] < 32) & (preprocessed_dataset["Pass/Fail"])
    preprocessed_dataset = preprocessed_dataset[~outlier_mask].reset_index(drop=True)
    split_parameter_info['rows_after_outlier_removal'] = preprocessed_dataset.shape[0]

    X = preprocessed_dataset.iloc[:, :-1]
    y = preprocessed_dataset.iloc[:, -1]
    
    # ----------------------------------------------------
    # Step 2: 분할 전 필터링 적용
    # ----------------------------------------------------
    split_parameter_info['features_before_split_filter'] = X.shape[1]
    if split_parameter['apply_filter_split']:
        print(f"    - 분할 전 필터링 적용 (분산: {split_parameter['var_threshold_split']}, 상관관계: {split_parameter['corr_threshold_split']})")
        # X, _, var_dropped, corr_dropped = variance_correlation_filter(X, var_threshold=split_parameter['var_threshold_split'], corr_threshold=split_parameter['corr_threshold_split'])
        X, _, var_dropped, corr_dropped = filter_by_variance(X, 0)
        split_parameter_info['features_after_split_filter'] = X.shape[1]
        split_parameter_info['features_dropped_by_variance_split'] = var_dropped
        split_parameter_info['features_dropped_by_correlation_split'] = corr_dropped
    else:
        print("    - 분할 전 필터링 미적용.")
        split_parameter_info['features_after_split_filter'] = X.shape[1]
        split_parameter_info['features_dropped_by_variance_split'] = 0
        split_parameter_info['features_dropped_by_correlation_split'] = 0
        
    # ----------------------------------------------------
    # Step 3: Feature Generation 적용
    # ----------------------------------------------------
    split_parameter_info['original_feature_count'] = X.shape[1]
    if split_parameter['apply_feature_generation']:
        print("    - Feature Generation 적용...")
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
        
        print("    - Feature Generation 완료. 새로운 피쳐 수:", split_parameter_info['total_generated_features'])
    else:
        print("    - Feature Generation 미적용.")
        split_parameter_info['generated_feature_counts'] = {'sum': 0, 'diff': 0, 'poly': 0}
        split_parameter_info['total_generated_features'] = 0
        split_parameter_info['features_after_generation'] = X.shape[1]

    # ----------------------------------------------------
    # Step 4: 훈련/테스트 데이터 분할
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
    print(f"\n    - 분할 전 훈련 데이터 클래스 분포: {train_class_distribution_before}")

    # ----------------------------------------------------
    # Step 5: 샘플링 적용
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
            print(f"    - 오버샘플링 적용 완료 (소수 클래스 비율: {sampling_ratio})")
            split_parameter_info['sampling_applied'] = 'oversampling'
        else:
            n_samples_minority = sum(y_train == 1)
            target_majority_count = int(n_samples_minority / sampling_ratio)
            sampling_strategy = {0: target_majority_count}
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=split_parameter['random_state'])
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f"    - 언더샘플링 적용 완료 (소수 클래스 비율: {sampling_ratio})")
            split_parameter_info['sampling_applied'] = 'undersampling'
            
        train_class_distribution_after = dict(sorted(Counter(y_train).items()))
        split_parameter_info['class_distribution_after_sampling'] = train_class_distribution_after
        split_parameter_info['train_samples_after_sampling'] = len(X_train)
        print(f"    - 샘플링 적용 후 훈련 데이터 클래스 분포: {train_class_distribution_after}")
    else:
        print("    - 샘플링 미적용")
        split_parameter_info['sampling_applied'] = 'None'
        split_parameter_info['train_samples_after_sampling'] = len(X_train)

    # ----------------------------------------------------
    # Step 6: 최종 데이터프레임 병합 및 반환
    # ----------------------------------------------------
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    split_parameter_info['final_train_feature_count'] = train_data.shape[1] - 1
    
    return train_data, test_data, split_parameter_info

def create_train_test_data(
    preprocessed_dataset: pd.DataFrame,
    split_parameter: dict = None
):
    """
    훈련/테스트 데이터 분할 및 샘플링을 적용하는 함수.
    Feature Generation 적용 여부 및 옵션을 split_parameter에서 제어합니다.
    split_parameter_info에 처리 결과 정보를 세부적으로 추가합니다.
    """
    print("\n\n##############################################################################################################################")
    print("# 3) Create Train/Test Split (훈련/테스트 데이터 분할) ")
    print("##############################################################################################################################")
    
    print("\n     훈련 및 테스트 데이터셋 생성 중...")

    # split_parameter의 기본값 설정 및 업데이트
    default_params = {
        'test_size': 0.2,
        'random_state': 42,
        'sampling_method': None,  # Add sampling_method parameter
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
    # Step 1: 아웃라이어 제거
    # ----------------------------------------------------
    initial_dataset_shape = preprocessed_dataset.shape
    # Check for 'Radius' and 'Pass/Fail' columns before applying mask to avoid KeyError
    if "Radius" in preprocessed_dataset.columns and "Pass/Fail" in preprocessed_dataset.columns:
      outlier_mask = (preprocessed_dataset["Radius"] < 32) & (preprocessed_dataset["Pass/Fail"])
      preprocessed_dataset = preprocessed_dataset[~outlier_mask].reset_index(drop=True)
      split_parameter_info['rows_after_outlier_removal'] = preprocessed_dataset.shape[0]
    else:
      print("Warning: 'Radius' or 'Pass/Fail' column not found. Skipping outlier removal.")
      split_parameter_info['rows_after_outlier_removal'] = initial_dataset_shape[0]

    X = preprocessed_dataset.iloc[:, :-1]
    y = preprocessed_dataset.iloc[:, -1]
    
    # ----------------------------------------------------
    # Step 2: 분할 전 필터링 적용
    # ----------------------------------------------------
    split_parameter_info['features_before_split_filter'] = X.shape[1]
    if split_parameter['apply_filter_split']:
        print(f"     - 분할 전 필터링 적용 (분산: {split_parameter['var_threshold_split']}, 상관관계: {split_parameter['corr_threshold_split']})")
        # Assuming variance_correlation_filter is defined elsewhere
        X, _, var_dropped, corr_dropped = filter_by_variance(X, split_parameter['var_threshold_split'])
        split_parameter_info['features_after_split_filter'] = X.shape[1]
        split_parameter_info['features_dropped_by_variance_split'] = var_dropped
        split_parameter_info['features_dropped_by_correlation_split'] = corr_dropped
    else:
        print("     - 분할 전 필터링 미적용.")
        split_parameter_info['features_after_split_filter'] = X.shape[1]
        split_parameter_info['features_dropped_by_variance_split'] = 0
        split_parameter_info['features_dropped_by_correlation_split'] = 0
        
    # ----------------------------------------------------
    # Step 3: Feature Generation 적용
    # ----------------------------------------------------
    split_parameter_info['original_feature_count'] = X.shape[1]
    if split_parameter['apply_feature_generation']:
        print("     - Feature Generation 적용...")
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
        
        print("     - Feature Generation 완료. 새로운 피쳐 수:", split_parameter_info['total_generated_features'])
    else:
        print("     - Feature Generation 미적용.")
        split_parameter_info['generated_feature_counts'] = {'sum': 0, 'diff': 0, 'poly': 0}
        split_parameter_info['total_generated_features'] = 0
        split_parameter_info['features_after_generation'] = X.shape[1]

    # ----------------------------------------------------
    # Step 4: 훈련/테스트 데이터 분할
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
    print(f"\n     - 분할 전 훈련 데이터 클래스 분포: {train_class_distribution_before}")

    # ----------------------------------------------------
    # Step 5: 샘플링 적용 (오버샘플링/언더샘플링)
    # ----------------------------------------------------
    sampling_method = split_parameter['sampling_method']
    sampling_ratio = split_parameter['sampling_ratio']
    
    split_parameter_info['sampling_method_used'] = sampling_method
    split_parameter_info['sampling_ratio_used'] = sampling_ratio
    
    sampler = None
    if sampling_method == 'SMOTE':
        if sampling_ratio is None:
            sampling_strategy = 'auto'
        elif sampling_ratio >= 1:
            n_samples_majority = sum(y_train == 0)
            target_minority_count = int(n_samples_majority * sampling_ratio)
            sampling_strategy = {1: target_minority_count}
        else:
            print("Warning: SMOTE is an oversampling method. For a sampling_ratio < 1, you might want to use RandomUnderSampler or a different method.")
            sampling_strategy = 'auto' # Fallback to auto
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=split_parameter['random_state'])
        print(f"     - SMOTE 오버샘플링 적용 (sampling_ratio: {sampling_ratio})")
    elif sampling_method == 'ROS':
        if sampling_ratio is None:
            sampling_strategy = 'auto'
        elif sampling_ratio >= 1:
            n_samples_majority = sum(y_train == 0)
            target_minority_count = int(n_samples_majority * sampling_ratio)
            sampling_strategy = {1: target_minority_count}
        else:
            print("Warning: RandomOverSampler is an oversampling method. For a sampling_ratio < 1, you might want to use RandomUnderSampler or a different method.")
            sampling_strategy = 'auto' # Fallback to auto
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=split_parameter['random_state'])
        print(f"     - ROS(RandomOverSampler) 오버샘플링 적용 (sampling_ratio: {sampling_ratio})")
    elif sampling_method == 'ADASYN':
        if sampling_ratio is None:
            sampling_strategy = 'auto'
        elif sampling_ratio >= 1:
            n_samples_majority = sum(y_train == 0)
            target_minority_count = int(n_samples_majority * sampling_ratio)
            sampling_strategy = {1: target_minority_count}
        else:
            print("Warning: ADASYN is an oversampling method. For a sampling_ratio < 1, you might want to use RandomUnderSampler or a different method.")
            sampling_strategy = 'auto' # Fallback to auto
        sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=split_parameter['random_state'])
        print(f"     - ADASYN 오버샘플링 적용 (sampling_ratio: {sampling_ratio})")
    elif sampling_method == 'RandomUnderSampler':
        if sampling_ratio is not None and sampling_ratio < 1:
            n_samples_minority = sum(y_train == 1)
            target_majority_count = int(n_samples_minority / sampling_ratio)
            sampling_strategy = {0: target_majority_count}
        else:
            print("Warning: RandomUnderSampler is an undersampling method. The sampling_ratio should be < 1. Using 'auto' strategy.")
            sampling_strategy = 'auto'
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=split_parameter['random_state'])
        print(f"     - RandomUnderSampler 적용 (sampling_ratio: {sampling_ratio})")
    elif sampling_method is None or sampling_method.lower() == 'none':
        print("     - 샘플링 미적용")
        split_parameter_info['sampling_applied'] = 'None'
    else:
        print(f"Warning: Unknown sampling method '{sampling_method}'. No sampling will be applied.")
        split_parameter_info['sampling_applied'] = 'None'

    if sampler is not None:
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        train_class_distribution_after = dict(sorted(Counter(y_train).items()))
        split_parameter_info['class_distribution_after_sampling'] = train_class_distribution_after
        split_parameter_info['train_samples_after_sampling'] = len(X_train)
        print(f"     - 샘플링 적용 후 훈련 데이터 클래스 분포: {train_class_distribution_after}")
        split_parameter_info['sampling_applied'] = sampling_method
    else:
        split_parameter_info['sampling_applied'] = 'None'
        split_parameter_info['train_samples_after_sampling'] = len(X_train)

    # ----------------------------------------------------
    # Step 6: 최종 데이터프레임 병합 및 반환
    # ----------------------------------------------------
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    split_parameter_info['final_train_feature_count'] = train_data.shape[1] - 1
    
    return train_data, test_data, split_parameter_info


##############################################################################################################################
# 4) Custom F2 scorer with pos_label=1 (since 1 = fail/rare)
# 4) pos_label=1 (1은 불합격/드문 클래스이므로)을 사용하는 사용자 정의 F2 평가 지표
##############################################################################################################################

# 'fbeta_score' 함수를 사용하여 F2 스코어를 계산하는 사용자 정의 평가 지표를 만듭니다.
# beta=2는 재현율(Recall)에 더 높은 가중치를 부여합니다.
# > 재현율에 2배의 가중치를 둠으로써, 불합격품을 합격으로 잘못 판단하는 것을 최소화하도록 최적화.
# pos_label=1은 불합격(fail) 클래스(1)를 긍정(positive) 클래스로 간주함을 의미합니다.
# f2_rare_scorer = make_scorer(fbeta_score, beta=2, pos_label=1)
f2_rare_scorer = make_scorer(fbeta_score, beta=4, pos_label=1)

# --- 새로운 비선형 상관관계 함수 (Xi Cor) ---
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
# --- 함수 끝 ---

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



# --- 변수 필터링 w.상관도 단계별 모듈화 함수 ---
def filter_by_variance(X: pd.DataFrame, var_threshold: float) -> Tuple[pd.DataFrame, Dict]:
    start_time = dt.datetime.now()
    
    X = rescale_df(X)
    
    initial_cols = list(X.columns)
    
    # 모든 피처의 분산 값을 계산합니다.
    features_values_checked = X.var().to_dict()
    
    vt = VarianceThreshold(threshold=var_threshold)
    X_filtered = vt.fit_transform(X)
    vt_mask = vt.get_support()
    vt_cols = X.columns[vt_mask]

    # 각 피처의 드롭 여부를 저장합니다.
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
    print(f"    - 분산 필터링 후 남은 피처 수: {stats['remaining_count']}")
    return pd.DataFrame(X_filtered, columns=vt_cols, index=X.index), stats
def filter_by_target_linear_correlation(X: pd.DataFrame, y: pd.Series, threshold: float) -> Tuple[pd.DataFrame, Dict]:
    start_time = dt.datetime.now()
    
    initial_cols = list(X.columns)
    
    # 모든 피처와 타겟 간의 상관관계를 계산합니다.
    correlations = X.corrwith(y).abs()
    features_values_checked = correlations.to_dict()
    
    low_corr_features = correlations[correlations < threshold].index.tolist()
    X_filtered = X.drop(columns=low_corr_features)

    # 각 피처의 드롭 여부를 저장합니다.
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
    print(f"    - 타겟 선형 상관관계 필터링 후 남은 피처 수: {stats['remaining_count']}")
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
        
        # 모든 컬럼 쌍의 상관관계 값을 저장
        for i in range(len(upper.columns)):
            for j in range(i + 1, len(upper.columns)):
                col1 = upper.columns[i]
                col2 = upper.columns[j]
                
                # 'col1'과 'col2'의 상관관계 값을 가져옵니다.
                correlation_value = upper.loc[col1, col2]
                
                # 키를 정렬된 튜플로 만들어 일관성을 유지합니다.
                pair = tuple(sorted((col1, col2)))
                features_values_checked[str(pair)] = correlation_value
                
                # threshold를 넘는 경우 'to_drop' 리스트에 추가합니다.
                if correlation_value > threshold:
                    if col2 not in to_drop:
                        to_drop.append(col2)
        
        # 'to_drop' 리스트를 바탕으로 features_dropped_yn을 업데이트합니다.
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
    
    print(f"    - 피처 간 선형 상관관계 필터링 후 남은 피처 수: {stats['remaining_count']}")
    return X_filtered, stats
def filter_by_feature_xicor_correlation(X: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, Dict]:
    start_time = dt.datetime.now()
    
    to_drop = []
    features_dropped_yn = {col: False for col in X.columns}
    features_values_checked = {}
    initial_cols = list(X.columns)
    
    if len(initial_cols) > 1:
        # 먼저 모든 컬럼 쌍의 xi 상관관계를 계산하고 저장합니다.
        for i in range(len(initial_cols)):
            for j in range(i + 1, len(initial_cols)):
                col1 = initial_cols[i]
                col2 = initial_cols[j]
                
                # 'xicor' 함수를 사용하여 비선형 상관관계 값을 계산합니다.
                # 이 값은 threshold와 관계없이 항상 저장됩니다.
                xi_corr_val = xicor(X[col1].values, X[col2].values)
                pair_key = str(tuple(sorted((col1, col2))))
                features_values_checked[pair_key] = xi_corr_val
        
        # 이제 저장된 값을 바탕으로 드롭할 컬럼을 결정합니다.
        # 이 루프에서는 상관관계가 높은 컬럼을 찾아 'to_drop' 리스트에 추가합니다.
        # 이전에 건너뛰었던 로직을 제거하고 전체 컬럼을 다시 확인합니다.
        for i in range(len(initial_cols)):
            for j in range(i + 1, len(initial_cols)):
                col1 = initial_cols[i]
                col2 = initial_cols[j]
                
                # 이미 드롭 예정인 피처는 추가로 검토하지 않습니다.
                if col1 in to_drop or col2 in to_drop:
                    continue
                
                pair_key = str(tuple(sorted((col1, col2))))
                xi_corr_val = features_values_checked[pair_key] # 이미 계산된 값을 사용
                
                if xi_corr_val > threshold:
                    # 여기서는 간단하게 col2를 드롭하도록 구현합니다.
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
    
    print(f"    - 피처 간 비선형 상관관계 필터링 후 남은 피처 수: {stats['remaining_count']}")
    return X_filtered, stats
# --- 통합 필터링 워크플로 함수명 변경 ---
def feature_filter(X: pd.DataFrame, y: pd.Series, params: Dict) -> Tuple[pd.DataFrame, list, Dict]:
    X_filtered = X.copy()
    filter_stats = {}
    print("\n--- 피처 필터링 시작 ---")

    # 1. 분산 필터링
    if params.get('apply_variance_filter', True):
        X_filtered, stats = filter_by_variance(X_filtered, params['var_threshold'])
        filter_stats['variance'] = stats
    # 2. 타겟과의 선형 상관관계 필터링
    if params.get('apply_target_linear_corr_filter', True) and X_filtered.shape[1] > 0:
        X_filtered, stats = filter_by_target_linear_correlation(X_filtered, y, params['target_linear_corr_threshold'])
        filter_stats['target_linear_correlation'] = stats
    # 3. 타겟과의 비선형 상관관계 (Xi Cor) 필터링
    if params.get('apply_target_xicor_filter', True) and X_filtered.shape[1] > 0:
        X_filtered, stats = filter_by_target_xicor_correlation(X_filtered, y, params['target_xicor_threshold'])
        filter_stats['target_xicor_correlation'] = stats
    # 4. 피처 간 선형 상관관계 필터링
    if params.get('apply_feature_linear_corr_filter', True) and X_filtered.shape[1] > 1:
        X_filtered, stats = filter_by_feature_linear_correlation(X_filtered, params['feature_linear_corr_threshold'])
        filter_stats['feature_linear_correlation'] = stats
    # 5. 피처 간 비선형 상관관계 (Xi Cor) 필터링
    if params.get('apply_feature_xicor_filter', True) and X_filtered.shape[1] > 1:
        X_filtered, stats = filter_by_feature_xicor_correlation(X_filtered, params['feature_xicor_threshold'])
        filter_stats['feature_xicor_correlation'] = stats
            
    final_cols = list(X_filtered.columns)
    return X_filtered, final_cols, filter_stats

# --- 변수 필터링 w.모델 단계별 모듈화 함수 ---
def _get_estimator_old(estimator_params: Dict) -> BaseEstimator:
    """
    주어진 파라미터 딕셔너리를 기반으로 Scikit-learn 추정기 객체를 생성합니다.
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
        raise ValueError(f"지원되지 않는 추정기(estimator): {estimator_name}")
def _get_estimator(estimator_params: Dict) -> BaseEstimator:
    """
    주어진 파라미터 딕셔너리를 기반으로 Scikit-learn 추정기 객체를 생성합니다.
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
        raise ValueError(f"지원되지 않는 추정기(estimator): {estimator_name}")
# --- 변수 중요도 추출 헬퍼 함수 추가 ---
def _get_feature_importances(estimator: BaseEstimator, feature_names: List[str]) -> Dict[str, float]:
    """
    추정기(estimator)에서 변수 중요도를 추출하는 헬퍼 함수
    """
    importances = {}
    if hasattr(estimator, 'feature_importances_'):
        importances = {col: imp for col, imp in zip(feature_names, estimator.feature_importances_)}
    elif hasattr(estimator, 'coef_'):
        # LogisticRegression과 같은 모델의 경우 coef_ 속성을 사용
        coefs = estimator.coef_[0] if estimator.coef_.ndim > 1 else estimator.coef_
        importances = {col: abs(coef) for col, coef in zip(feature_names, coefs)}
    return importances


def run_model_based_feature_selection_old(
    X_in: pd.DataFrame, 
    y: pd.Series, 
    selector_name: str, 
    selector_params: Dict
) -> Tuple[List[str], Dict]:
    """
    **RFE, SFS, SelectFromModel을 선택적으로 실행하는 범용 함수입니다.**
    
    Args:
        X (pd.DataFrame): 피처 데이터
        y (pd.Series): 타겟 데이터
        selector_name (str): 사용할 변수 선택기 이름 ('RFE', 'SFS', 'SFM')
        selector_params (Dict): 변수 선택기에 전달할 파라미터 딕셔너리

    Returns:
        Tuple[List[str], Dict]: 선택된 피처 리스트와 통계 정보 딕셔너리
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
        # The code snippet you provided is not valid Python code. It seems like there is a placeholder
        # "initial_features" followed by some comment lines denoted by "
        # initial_features = list(X[0].columns)
        n_features_to_select = selector_params.get('n_features_to_select')
        step = selector_params.get('step', 1)
        
        selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
        selector.fit(X, y)
        selected_mask = selector.get_support()
        selected_features = list(X.columns[selected_mask])
        ranking = {col: rank for col, rank in zip(initial_features, selector.ranking_)}
        stats = {'method': 'RFE', 'n_features_to_select': n_features_to_select, 'step': step, 'ranking': ranking}

    elif selector_name == 'SFM':
        threshold = selector_params.get('threshold', 'median')
        
        # 모델을 먼저 학습하고 SelectFromModel에 전달
        estimator.fit(X, y)
        selector = SelectFromModel(estimator, prefit=True, threshold=threshold)
        selected_mask = selector.get_support()
        selected_features = list(X.columns[selected_mask])
        importances = {col: imp for col, imp in zip(initial_features, estimator.feature_importances_)}
        stats = {'method': 'SFM', 'threshold': threshold, 'importances': importances}

    elif selector_name == 'SFS':
        # SFS는 estimator를 fit하지 않고 바로 인자로 사용
        # n_features_to_select가 정수 또는 'auto'일 수 있음
        n_features_to_select = selector_params.get('n_features_to_select')
        direction = selector_params.get('direction', 'forward')
        
        selector = SFS(estimator=estimator, n_features_to_select=n_features_to_select, direction=direction, cv=5)
        
        selector.fit(X, y)
        selected_features = list(X.columns[selector.get_support()])
        
        # SFS는 특성 순위를 직접 제공하지 않으므로, 선택된 피처 리스트만 반환합니다.
        stats = {'method': 'SFS', 'n_features_to_select': n_features_to_select, 'direction': direction}
        
    else:
        raise ValueError(f"지원되지 않는 선택기(selector) 이름: {selector_name}. 'RFE', 'SFS', 'SFM' 중 선택하세요.")

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
    
    print(f"--- {selector_name} 선택기 완료 ---")
    print(f"남은 피처 수: {stats['remaining_count']}")
    
    return selected_features, stats
def run_model_based_feature_selection(
    X_in: pd.DataFrame, 
    y: pd.Series, 
    selector_name: str, 
    selector_params: Dict
) -> Tuple[List[str], Dict]:
    """
    **RFE, SFS, SelectFromModel을 선택적으로 실행하는 범용 함수입니다.**
    
    Args:
        X (pd.DataFrame): 피처 데이터
        y (pd.Series): 타겟 데이터
        selector_name (str): 사용할 변수 선택기 이름 ('RFE', 'SFS', 'SFM')
        selector_params (Dict): 변수 선택기에 전달할 파라미터 딕셔너리

    Returns:
        Tuple[List[str], Dict]: 선택된 피처 리스트와 통계 정보 딕셔너리
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
        
        # RFE는 순위를 제공. 랭킹을 stats에 추가
        ranking = {col: rank for col, rank in zip(initial_features, selector.ranking_)}
        stats = {'method': 'RFE', 'n_features_to_select': n_features_to_select, 'step': step, 'ranking': ranking}

        # 선택된 피처의 중요도를 추출하여 stats에 추가
        if ranking and any(rank == 1 for rank in ranking.values()):
            selected_estimator = _get_estimator(estimator_params)
            selected_estimator.fit(X[selected_features], y)
            importances = _get_feature_importances(selected_estimator, selected_features)
            stats['importances'] = importances
            
    elif selector_name == 'SFM':
        threshold = selector_params.get('threshold', 'median')
        
        # 모델을 먼저 학습하고 SelectFromModel에 전달
        estimator.fit(X, y)
        selector = SelectFromModel(estimator, prefit=True, threshold=threshold)
        selected_mask = selector.get_support()
        selected_features = list(X.columns[selected_mask])

        # 모델에서 변수 중요도 추출
        importances = _get_feature_importances(estimator, initial_features)
        stats = {'method': 'SFM', 'threshold': threshold, 'importances': importances}

    elif selector_name == 'SFS':
        n_features_to_select = selector_params.get('n_features_to_select')
        direction = selector_params.get('direction', 'forward')
        
        selector = SFS(estimator=estimator, n_features_to_select=n_features_to_select, direction=direction, cv=5)
        
        selector.fit(X, y)
        selected_features = list(X.columns[selector.get_support()])
        
        # SFS는 직접 중요도를 제공하지 않으므로, 선택된 변수로 모델을 다시 학습하여 중요도를 추출
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
        raise ValueError(f"지원되지 않는 선택기(selector) 이름: {selector_name}. 'RFE', 'SFS', 'SFM' 중 선택하세요.")

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
    
    print(f"--- {selector_name} 선택기 완료 ---")
    print(f"남은 피처 수: {stats['remaining_count']}")
    
    return selected_features, stats

# JSON 직렬화에 사용될 커스텀 클래스
class NpEncoder(json.JSONEncoder):
    """Numpy 타입(int64, float64 등)을 JSON으로 직렬화하기 위한 클래스"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# --- 변수 선택 함수 ---
def select_feature_old(train_data: pd.DataFrame, feature_selector_params: Dict) -> Dict:
    """
    개선된 피처 선택 워크플로를 수행하고 결과를 딕셔너리로 반환합니다.
    """
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    feature_selection_info = feature_selector_params.copy()
    initial_feature_count = X_train.shape[1]
    
    # feature_selector_name을 기반으로 분기
    if feature_selector_params.get("feature_selector_name") == 'FeatureFilter':
        print("\n--- 피처 선택기: Filter Method ---")
        filter_params = feature_selector_params['filter_methods']
        X_train_filtered_df, final_features, stats = feature_filter(
            X=X_train,
            y=y_train,
            params=filter_params
        )
        final_feature_count = len(final_features)
        feature_selection_info['initial_feature_count'] = initial_feature_count
        feature_selection_info['final_feature_count'] = final_feature_count
        feature_selection_info['final_features'] = final_features
        feature_selection_info['filter_details'] = stats
    elif feature_selector_params.get("feature_selector_name") == 'EmbeddedMethod':
        print("\n--- 피처 선택기: Embedded Method (SelectFromModel) ---")
        model_params = feature_selector_params['model_based']['random_forest']
        threshold = feature_selector_params['model_based']['threshold']
        
        model = RandomForestClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', None),
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        sfm = SelectFromModel(model, prefit=True, threshold=threshold)
        sfm_mask = sfm.get_support()
        
        # 피처 중요도 정보 추출 및 저장
        feature_importances = model.feature_importances_
        feature_importance_dict = {col: imp for col, imp in zip(X_train.columns, feature_importances)}
        
        selected_features = list(X_train.columns[sfm_mask])
        dropped_features = [col for col in X_train.columns if col not in selected_features]
        final_feature_count = len(selected_features)
        
        # 상세 통계 정보 딕셔너리 구성
        stats = {
            'method': 'SelectFromModel',
            'threshold_type': threshold,
            'original_count': initial_feature_count,
            'remaining_count': final_feature_count,
            'selected_features': selected_features,
            'dropped_features': dropped_features,
            'feature_importances': feature_importance_dict
        }
        
        feature_selection_info['initial_feature_count'] = initial_feature_count
        feature_selection_info['final_feature_count'] = final_feature_count
        feature_selection_info['final_features'] = selected_features
        feature_selection_info['filter_details'] = stats
    else:
        print("\n--- 피처 선택기: No-op ---")
        final_features = list(X_train.columns)
        final_feature_count = initial_feature_count
        stats = {
            'method': 'No-op',
            'original_count': initial_feature_count,
            'remaining_count': final_feature_count,
            'selected_features': final_features,
            'dropped_features': []
        }
        feature_selection_info['initial_feature_count'] = initial_feature_count
        feature_selection_info['final_feature_count'] = final_feature_count
        feature_selection_info['final_features'] = final_features
        feature_selection_info['filter_details'] = stats
    
    # 결과 및 정보 저장
    
    # 저장할 디렉토리
    destination_dir = 'data/result/jsons'

    # 대상 디렉토리가 없으면 생성
    os.makedirs(destination_dir, exist_ok=True)

    current_time = dt.datetime.now()
    file_id = uuid.uuid4().hex[:8]
    filename = "feature_selection_info_" + current_time.strftime('%y%m%d_%H%M%S') + f'_{file_id}.json'

    file_path = os.path.join(destination_dir, filename)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(feature_selection_info, f, indent=4, ensure_ascii=False)

    print(f"\n피처 선택 결과가 '{file_path}' 파일에 저장되었습니다.")
    
    print(f"\n- 최종 피처 수: {final_feature_count}")
    return feature_selection_info
def select_feature_old2(train_data: pd.DataFrame, feature_selector_params: Dict) -> Dict:
    """
    피처 선택 워크플로를 수행하고 결과를 딕셔너리로 반환합니다.
    """
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    feature_selection_info = feature_selector_params.copy()
    initial_feature_count = X_train.shape[1]
    
    selector_name = feature_selector_params.get("feature_selector_name")
    final_features = list(X_train.columns)
    stats = {}
    
    print(f"\n--- 피처 선택기: {selector_name} ---")

    try:
        if selector_name == 'FeatureFilter':
            filter_params = feature_selector_params['filter_methods']
            _, final_features, stats = feature_filter(X=X_train, y=y_train, params=filter_params)
        
        elif selector_name in ['RFE', 'SFM', 'SFS']:
            selector_params = feature_selector_params.get(f"{selector_name.lower()}_params")
            if not selector_params:
                raise KeyError(f"설정(configuration)에 '{selector_name.lower()}_params'가 누락되었습니다.")
            
            final_features, stats = run_model_based_feature_selection(
                # X=X_train,
                # y=y_train,
                X_train,
                y_train,
                selector_name=selector_name,
                selector_params=selector_params
            )
        else:
            print("\n--- 피처 선택기: No-op ---")
            stats = {'method': 'No-op', 'original_count': initial_feature_count, 'remaining_count': initial_feature_count, 'selected_features': final_features, 'dropped_features': []}
    
    except Exception as e:
        print(f"피처 선택 중 오류가 발생했습니다: {e}")
        # 오류 발생 시 원본 피처를 반환
        stats = {'method': 'Error', 'original_count': initial_feature_count, 'remaining_count': initial_feature_count, 'selected_features': final_features, 'dropped_features': [], 'error_message': str(e)}

    final_feature_count = len(final_features)
    feature_selection_info['initial_feature_count'] = initial_feature_count
    feature_selection_info['final_feature_count'] = final_feature_count
    feature_selection_info['final_features'] = final_features
    feature_selection_info['selection_details'] = stats

    # (저장 로직은 동일)
    destination_dir = 'data/result/jsons'
    os.makedirs(destination_dir, exist_ok=True)
    current_time = dt.datetime.now()
    file_id = uuid.uuid4().hex[:8]
    filename = "feature_selection_info_" + current_time.strftime('%y%m%d_%H%M%S') + f'_{file_id}.json'
    file_path = os.path.join(destination_dir, filename)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(feature_selection_info, f, indent=4, ensure_ascii=False)

    print(f"\n피처 선택 결과가 '{file_path}' 파일에 저장되었습니다.")
    print(f"\n- 최종 피처 수: {final_feature_count}")
    
    feature_selection_info['feature_selection_info_json_path'] = file_path
    
    return feature_selection_info
def select_feature(train_data: pd.DataFrame, feature_selector_params: Dict) -> Dict:
    """
    피처 선택 워크플로를 수행하고 결과를 딕셔너리로 반환합니다.
    """
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    feature_selection_info={}
    feature_selection_info = feature_selector_params.copy()
    initial_feature_count = X_train.shape[1]
    
    selector_name = feature_selector_params.get("feature_selector_name")
    final_features = list(X_train.columns)
    stats = {}
    
    print(f"\n--- 피처 선택기: {selector_name} ---")

    try:
        if selector_name == 'FeatureFilter':
            filter_params = feature_selector_params['filter_methods']
            _, final_features, stats = feature_filter(X=X_train, y=y_train, params=filter_params)
        
        elif selector_name in ['RFE', 'SFM', 'SFS']:
            # selector_params = feature_selector_params.get(f"{selector_name.lower()}_params")
            selector_params = feature_selector_params.get("params")
            if not selector_params:
                raise KeyError(f"설정(configuration)에 '{selector_name.lower()}_params'가 누락되었습니다.")
            
            final_features, stats = run_model_based_feature_selection(
                X_train,
                y_train,
                selector_name=selector_name,
                selector_params=selector_params
            )
        else:
            print("\n--- 피처 선택기: No-op ---")
            stats = {'method': 'No-op', 'original_count': initial_feature_count, 'remaining_count': initial_feature_count, 'selected_features': final_features, 'dropped_features': []}
    
    except Exception as e:
        print(f"피처 선택 중 오류가 발생했습니다: {e}")
        # 오류 발생 시 원본 피처를 반환
        stats = {'method': 'Error', 'original_count': initial_feature_count, 'remaining_count': initial_feature_count, 'selected_features': final_features, 'dropped_features': [], 'error_message': str(e)}

    final_feature_count = len(final_features)
    feature_selection_info['initial_feature_count'] = initial_feature_count
    feature_selection_info['final_feature_count'] = final_feature_count
    feature_selection_info['final_features'] = final_features
    feature_selection_info['selection_details'] = stats

    # (저장 로직 수정)
    destination_dir = 'data/result/jsons'
    os.makedirs(destination_dir, exist_ok=True)
    current_time = dt.datetime.now()
    file_id = uuid.uuid4().hex[:8]
    filename = "feature_selection_info_" + current_time.strftime('%y%m%d_%H%M%S') + f'_{file_id}.json'
    file_path = os.path.join(destination_dir, filename)

    with open(file_path, 'w', encoding='utf-8') as f:
        # 클래스 인코더를 사용하여 numpy int64를 파이썬 int로 변환
        json.dump(feature_selection_info, f, indent=4, ensure_ascii=False, cls=NpEncoder)

    print(f"\n피처 선택 결과가 '{file_path}' 파일에 저장되었습니다.")
    print(f"\n- 최종 피처 수: {final_feature_count}")
    
    feature_selection_info['feature_selection_info_json_path'] = file_path
    
    return feature_selection_info





##############################################################################################################################
# 5) Logistic Regression (simple) - Now up-weight class 1
# 5) 로지스틱 회귀 (단순) - 이제 클래스 1에 가중치를 부여합니다.
##############################################################################################################################

# 로지스틱 회귀 모델을 훈련하는 함수
def train_model_logistic_regression(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    print("      Training Logistic Regression (no CV)...\n")
    
    model_parameter_info = {}

    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
    final_features = feature_selection_info['final_features']
    X = X[final_features]

    # train_parameters에 따라 파라미터 설정
    if train_parameters and train_parameters.get('function_name') == 'train_model_logistic_regression':
        solver = train_parameters.get('solver', 'lbfgs')
        max_iter = train_parameters.get('max_iter', 1000)
    else:
        solver = 'lbfgs'
        max_iter = 1000
    
    # 설정된 파라미터 정보 저장
    model_parameter_info['solver'] = solver
    model_parameter_info['max_iter'] = max_iter

    # 클래스 불균형 해결을 위한 클래스 가중치 설정
    n_pos = sum(y)
    n_neg = len(y) - n_pos

    # 값이 None이거나 딕셔너리에 키가 없으면 n_neg로 설정
    # if train_parameters.get('class_weight_multiplier') == '':
    #     class_weight_multiplier = n_neg
    # else:
    #     class_weight_multiplier = eval(train_parameters.get('class_weight_multiplier'))
    # class_weight = {0: 1, 1: class_weight_multiplier}

    # 클래스 불균형 문제를 해결하기 위해 클래스 가중치 부여
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
# 5a) 교차 검증(CV) 및 F2 스코어링을 사용한 로지스틱 회귀
##############################################################################################################################
def train_model_logistic_regression_cv(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    print("      Training Logistic Regression with cross-validation & hyperparameter tuning...\n")
    
    model_parameter_info = {}
    
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
    final_features = feature_selection_info['final_features']
    X = X[final_features]

    # train_parameters에 따라 파라미터 그리드 및 GridSearchCV 파라미터 설정
    if train_parameters and train_parameters.get('function_name') == 'train_model_logistic_regression_cv':
        param_grid = train_parameters.get('param_grid', {})
        cv = train_parameters.get('cv', 3)
        verbose = train_parameters.get('verbose', 1)

        # f2_rare_scorer 설정 로직 반영
        scoring_params = train_parameters.get('f2_rare_scorer', {})
        if scoring_params.get('name') == 'fbeta_score':
            beta = scoring_params.get('beta', 2)
            pos_label = scoring_params.get('pos_label', 1)
            scorer = make_scorer(fbeta_score, beta=beta, pos_label=pos_label)
        else:
            scorer = make_scorer(f2_rare_scorer, greater_is_better=True)
    else:
        # 기본 하이퍼파라미터 설정
        param_grid = {"C": [0.01, 0.1, 1], "penalty": ["l2"], "solver": ["lbfgs"]}
        cv = 3
        verbose = 1
        scorer = make_scorer(f2_rare_scorer, greater_is_better=True)

    # 설정된 파라미터 정보 저장
    model_parameter_info['param_grid'] = param_grid
    model_parameter_info['cv'] = cv
    model_parameter_info['verbose'] = verbose
    if 'scorer' in locals():
        model_parameter_info['f2_rare_scorer'] = {
            'name': 'fbeta_score',
            'beta': scorer._kwargs.get('beta'),
            'pos_label': scorer._kwargs.get('pos_label')
        }

    # 클래스 불균형 문제를 해결하기 위해 클래스 가중치 부여
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
# 6) 기준 모델 (변경 없음)
##############################################################################################################################
# 기준 모델을 훈련하는 함수 (실제로 훈련하는 것이 아니라 객체를 생성하고 중요도를 미리 정의)
def train_model_baseline(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    model_parameter_info = {}
    model_fitted = BaselineModel()

    # train_parameters에 따라 특징 중요도 설정
    if train_parameters and train_parameters.get('function_name') == 'train_model_baseline':
        importance_data = train_parameters.get('importance_data', {})
    else:
        importance_data = {
            "Features": ["SensorOffsetHot-Cold", "band gap dpat_ok for band gap", "Radius"],
            "Importance": [56.6, 4.65, 96.9],
        }
    
    # 설정된 파라미터 정보 저장
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
# 7) 랜덤 포레스트 (단순) - 클래스 1에 가중치 부여
##############################################################################################################################

# (주석 처리된 코드)
# def train_model_random_forest(train_dataset: pd.DataFrame):
#     print("      Training RandomForest (no CV)...\n")
#     X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]

#     class_weight = {
#         0: 1,
#         1: sum(1 - y),
#     }

#     model_fitted = RandomForestClassifier(
#         class_weight=class_weight, random_state=42
#     ).fit(X, y)
#     print("\n    RandomForestClassifier is trained!")

#     importance_dict = {
#         "Features": X.columns,
#         "Importance": model_fitted.feature_importances_,
#         "Importance_abs": np.abs(model_fitted.feature_importances_),
#     }
#     importance = pd.DataFrame(importance_dict).sort_values(
#         by="Importance", ascending=True
#     )
#     return model_fitted, importance

##############################################################################################################################
# 7a) RandomForest with CV & GridSearch using F2 on class=1
# 7a) 클래스 1에 대한 F2를 사용한 교차 검증 및 그리드 서치를 이용한 랜덤 포레스트
##############################################################################################################################

# 교차 검증 및 하이퍼파라미터 튜닝을 통해 랜덤 포레스트 모델을 훈련하는 함수
def train_model_rf_cv(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    print("      Training the Random Forest model with cross-validation & hyperparameter tuning...\n")
    
    model_parameter_info = {}
    
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
    final_features = feature_selection_info['final_features']
    X = X[final_features]

    # train_parameters에 따라 파라미터 그리드 및 GridSearchCV 파라미터 설정
    if train_parameters and train_parameters.get('function_name') == 'train_model_rf_cv':
        param_grid = train_parameters.get('param_grid', {})
        cv = train_parameters.get('cv', 3)
        verbose = train_parameters.get('verbose', 1)
        
        # f2_rare_scorer 설정 로직 반영
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
    
    # 설정된 파라미터 정보 저장
    model_parameter_info['param_grid'] = param_grid
    model_parameter_info['cv'] = cv
    model_parameter_info['verbose'] = verbose
    if 'f2_rare_scorer' in locals():
        model_parameter_info['f2_rare_scorer'] = {
            'name': 'fbeta_score',
            'beta': scorer._kwargs.get('beta'),
            'pos_label': scorer._kwargs.get('pos_label')
        }

    # 클래스 불균형 문제를 해결하기 위해 클래스 가중치 부여
    # n_neg = len(y) - sum(y)
    n_neg = len(y) - sum(int(i) for i in y)

    # 값이 None이거나 딕셔너리에 키가 없으면 n_neg로 설정
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
# 8) 의사결정나무
##############################################################################################################################

# 의사결정나무 모델을 훈련하는 함수
def train_model_decision_tree(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
    final_features = feature_selection_info['final_features']
    X = X[final_features]
    
    model_parameter_info = {}

    # train_parameters에 따라 파라미터 설정
    if train_parameters and train_parameters.get('function_name') == 'train_model_decision_tree':
        max_depth = train_parameters.get('max_depth', None)
        min_samples_split = train_parameters.get('min_samples_split', 2)
    else:
        max_depth = None
        min_samples_split = 2
        
    # 설정된 파라미터 정보 저장
    model_parameter_info['max_depth'] = max_depth
    model_parameter_info['min_samples_split'] = min_samples_split

    # 클래스 불균형 해결을 위한 클래스 가중치 설정
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
# 9) 클래스 1에 대한 F2를 사용한 교차 검증 XGBoost
##############################################################################################################################

# 교차 검증 및 하이퍼파라미터 튜닝을 통해 XGBoost 모델을 훈련하는 함수
def train_model_xgboost_cv(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    print(
        "    Training the XGBoost model with cross-validation & hyperparameter tuning...\n"
    )

    model_parameter_info = {}

    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]

    final_features = feature_selection_info['final_features']
    X = X[final_features]

    # train_parameters에서 알고리즘 이름에 따라 파라미터 설정
    if train_parameters and train_parameters.get('function_name') == 'train_model_xgboost_cv':
        param_grid = train_parameters.get('param_grid', {})
        scale_pos_weight_multiplier = train_parameters.get('scale_pos_weight_multiplier', 2)
        cv = train_parameters.get('cv', 3)
        verbose = train_parameters.get('verbose', 1)
        
        # f2_rare_scorer 설정 로직 반영
        scoring_params = train_parameters.get('f2_rare_scorer', {})
        if scoring_params.get('name') == 'fbeta_score':
            beta = scoring_params.get('beta', 2)
            pos_label = scoring_params.get('pos_label', 1)
            f2_rare_scorer = make_scorer(fbeta_score, beta=beta, pos_label=pos_label)
        else:
            # 기본 F2 스코어
            f2_rare_scorer = make_scorer(lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=2, pos_label=1))
            
    else:
        # 기본 하이퍼파라미터 설정
        param_grid = {
            "n_estimators": [30, 50, 100, 200],
            "max_depth": [2, 5],
            "learning_rate": [0.01, 0.1, 0.2],
        }
        scale_pos_weight_multiplier = 2
        cv = 3
        verbose = 1
        f2_rare_scorer = make_scorer(lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=2, pos_label=1))

    # 설정된 파라미터 정보를 저장합니다.
    model_parameter_info['param_grid'] = param_grid
    model_parameter_info['scale_pos_weight_multiplier'] = scale_pos_weight_multiplier
    model_parameter_info['cv'] = cv
    model_parameter_info['verbose'] = verbose
    model_parameter_info['f2_rare_scorer'] = {
        'name': 'fbeta_score',
        'beta': f2_rare_scorer._kwargs.get('beta'),
        'pos_label': f2_rare_scorer._kwargs.get('pos_label')
    }

    # 클래스 불균형을 위한 'scale_pos_weight'를 계산합니다.
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos * scale_pos_weight_multiplier if n_pos > 0 else 1

    # XGBoost 모델 객체를 생성합니다.
    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )

    # GridSearchCV를 설정합니다.
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

    print(f"\n    Best parameters found: {grid_search.best_params_}")
    print(f"    Best F2 (class=1) score: {grid_search.best_score_:.4f}\n")
    
    model_parameter_info['best_params'] = grid_search.best_params_

    # 최적 모델의 특징 중요도를 계산합니다.
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
# Optuna를 사용하여 최적화하는 로지스틱 회귀 모델 훈련 함수
##############################################################################################################################
def train_model_logistic_regression_optuna(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    """
    Optuna를 활용하여 로지스틱 회귀 모델의 하이퍼파라미터를 최적화하는 함수.
    train_parameters 딕셔너리를 사용하여 Optuna의 탐색 범위를 동적으로 설정합니다.
    """
    print("      Training Logistic Regression with Optuna hyperparameter tuning...\n")
    
    # model_parameter_info = {}
    model_parameter_info = train_parameters
    
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
    final_features = feature_selection_info['final_features']
    X = X[final_features]
    
    n_pos = sum(y)
    n_neg = len(y) - n_pos

    # train_parameters에 따라 Optuna 파라미터 및 스터디 설정
    if train_parameters and train_parameters.get('function_name') == 'train_model_logistic_regression_optuna':
        n_trials = train_parameters.get('n_trials', 30)
        param_ranges = train_parameters.get('param_ranges', {})
        
        # 탐색 범위 추출 (제공된 값이 없으면 기본값 사용)
        c_range = param_ranges.get('C', [1e-3, 10])
        solver_list = param_ranges.get('solver', ['liblinear', 'lbfgs'])
        max_iter_val = param_ranges.get('max_iter', 1000)
        class_weight_multiplier_range = param_ranges.get('class_weight_multiplier', [1, n_neg])
        
        # F2 스코어러 설정
        scoring_params = train_parameters.get('f2_rare_scorer', {})
        if scoring_params.get('name') == 'fbeta_score':
            beta = scoring_params.get('beta', 2)
            pos_label = scoring_params.get('pos_label', 1)
            scorer = make_scorer(fbeta_score, beta=beta, pos_label=pos_label)
        else:
            scorer = f2_rare_scorer
            
    else:
        # 기본 설정 (빠른 수행에 적합한 범위)
        n_trials = 30
        c_range = [1e-3, 10]
        solver_list = ['liblinear', 'lbfgs']
        max_iter_val = 1000
        class_weight_multiplier_range = [1, n_neg]
        scorer = f2_rare_scorer

    # 설정된 파라미터 정보 저장
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
    
    # F2 score를 최대화하는 방향으로 스터디 생성
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

    print(f"\n    Best parameters found (Optuna): {study.best_params}")
    print(f"    Best F2 (class=1) score (CV): {study.best_value:.4f}\n")
    
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
# Optuna를 사용하여 최적화하는 랜덤 포레스트 모델 훈련 함수
##############################################################################################################################
def train_model_rf_optuna(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    print("      Training the Random Forest model with Optuna hyperparameter tuning...\n")
    
    model_parameter_info = {}
    
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
    final_features = feature_selection_info['final_features']
    X = X[final_features]
    
    n_neg = len(y) - sum(y)

    # train_parameters에서 Optuna 관련 설정 가져오기
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
        
        # f2_rare_scorer 설정 로직 추가
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
# Optuna를 사용하여 최적화하는 XGBoost 모델 훈련 함수
##############################################################################################################################
def train_model_xgboost_optuna(train_dataset: pd.DataFrame, feature_selection_info: dict, train_parameters: dict = None):
    print("    Training the XGBoost model with Optuna hyperparameter tuning...\n")
    
    model_parameter_info = {}
    
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
    final_features = feature_selection_info['final_features']
    X = X[final_features]
    
    n_pos = sum(y)
    n_neg = len(y) - n_pos

    # train_parameters에서 Optuna 관련 설정 가져오기
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
        # scale_pos_weight는 ratio_multiplier를 사용하여 동적 탐색 가능하도록 보완
        ratio_multiplier_range = train_parameters.get('ratio_multiplier_range', {'low': 0.5, 'high': 2.0})

        # f2_rare_scorer 설정 로직 반영
        scoring_params = train_parameters.get('f2_rare_scorer', {})
        if scoring_params.get('name') == 'fbeta_score':
            beta = scoring_params.get('beta', 2)
            pos_label = scoring_params.get('pos_label', 1)
            f2_rare_scorer = make_scorer(fbeta_score, beta=beta, pos_label=pos_label)
        else:
            # 기본 F2 스코어
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
        # Optuna를 위한 파라미터 탐색 범위 설정
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

        # class imbalance를 위한 scale_pos_weight를 탐색
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
    
    # best_params에서 ratio_multiplier를 분리하여 scale_pos_weight 계산
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

    print(f"\n    Best parameters found (Optuna): {study.best_params}")
    print(f"    Best F2 (class=1) score (CV): {study.best_value:.4f}\n")
    
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
# 10) Prediction & Evaluation Helpers
# 10) 예측 및 평가 도우미 함수들
##############################################################################################################################

# 혼동 행렬(Confusion Matrix)의 레이블을 생성하는 함수
def _confusion_label(row):
    # 이제 "1"은 불합격(fail)이며, 이는 긍정(positive)으로 간주됩니다.
    # row["Historical"] = 실제 레이블, row["Forecast"] = 예측 레이블
    if row["Historical"] == 1 and row["Forecast"] == 1:
        return "True Fail (TP)" # 실제 불합격을 불합격으로 올바르게 예측
    elif row["Historical"] == 0 and row["Forecast"] == 0:
        return "True Pass (TN)" # 실제 합격을 합격으로 올바르게 예측
    elif row["Historical"] == 0 and row["Forecast"] == 1:
        return "False Fail (FP)" # 실제 합격을 불합격으로 잘못 예측 (오류)
    else:  # row["Historical"] == 1 and row["Forecast"] == 0
        return "Missed Fail (FN)" # 실제 불합격을 합격으로 잘못 예측 (놓침)


# F2 스코어를 최대화하는 최적의 임계값(threshold)을 찾는 함수
def find_best_threshold(best_model, train_dataset, feature_selection_info: dict):
    """
    F2 스코어를 최대화하는 분류를 위한 최적의 임계값을 찾습니다.

    매개변수:
    - best_model: `predict_proba` 메서드가 있는 훈련된 분류 모델.
    - train_dataset: 특징과 타겟이 포함된 데이터프레임.

    반환값:
    - best_threshold: F2 스코어를 최대화하는 최적의 임계값.
    """
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]

    # 기준 모델이 아닌 경우, 'final_features.json'에서 특징을 불러와서 X를 재구성
    if not isinstance(best_model, BaselineModel):
        # with open("final_features.json", "r") as f:
        #     final_features = json.load(f)
        final_features = feature_selection_info['final_features']
        X = X[final_features]

    # 모델이 예측한 클래스 1(불합격)에 대한 확률을 가져옵니다.
    prob_class1 = best_model.predict_proba(X)[:, 1]

    # 0부터 1까지 100개의 임계값 후보를 시도합니다.
    thresholds = np.linspace(0, 1, 100)
    f2_scores = []

    for threshold in thresholds:
        y_pred = (prob_class1 >= threshold).astype(int) # 임계값을 기준으로 예측 레이블을 생성
        # score = fbeta_score(y, y_pred, beta=2, pos_label=1) # F2 스코어 계산
        score = fbeta_score(y, y_pred, beta=4, pos_label=1) # F2 스코어 계산
        f2_scores.append(score)

    # 가장 높은 F2 스코어를 기록한 임계값을 찾습니다.
    best_idx = np.argmax(f2_scores)
    best_threshold = thresholds[best_idx]
    best_f2_score = f2_scores[best_idx]

    print(
        f"Best threshold for F2 score: {best_threshold:.4f} with F2 score: {best_f2_score:.4f}"
    )

    # ------------------------------------------------------------------
    # 사용자가 선택한 임계값을 적용합니다.
    # ------------------------------------------------------------------

    # 훈련 데이터셋에 'Probability'와 'Historical' 컬럼을 추가합니다.
    train_dataset["Probability"] = prob_class1
    train_dataset["Historical"] = y
    return train_dataset, best_threshold

# 훈련 데이터셋에 대한 혼동 행렬 지표를 생성하는 함수
def create_metrics_on_train(train_dataset, threshold):
    """
    훈련 후, 주어진 임계값(클래스 1, 불합격)으로 훈련 데이터셋에 대해 예측합니다.
    """
    # ------------------------------------------------------------------
    # 사용자가 선택한 임계값을 적용합니다.
    # ------------------------------------------------------------------
    # 'Probability'가 임계값보다 크거나 같으면 1, 아니면 0으로 예측합니다.
    forecast = (train_dataset["Probability"] >= threshold).astype(int)

    train_dataset["Forecast"] = forecast
    # 혼동 행렬 레이블을 적용합니다.
    train_dataset["True/False/Positive/Negative"] = train_dataset.apply(
        _confusion_label, axis=1
    )
    return train_dataset


# 테스트 데이터셋에 대한 예측을 수행하는 함수
def forecast(test_dataset: pd.DataFrame, trained_model, feature_selection_info: dict):
    print("      Forecasting the test dataset...")
    X = test_dataset.iloc[:, :-1]

    # 기준 모델이 아닌 경우, 'final_features.json'에서 특징을 불러와서 X를 재구성합니다.
    if not isinstance(trained_model, BaselineModel):
        # with open("final_features.json", "r") as f:
        #     final_features = json.load(f)
        final_features = feature_selection_info['final_features']
        X = X[final_features]


    # 클래스 1에 대한 예측 확률을 가져옵니다.
    predictions = trained_model.predict_proba(X)[:, 1]
    print("      Forecasting done!")

    # SHAP을 사용하여 모델 예측의 설명력을 분석합니다.
    # 트리 기반 모델인 경우 TreeExplainer를, 그 외에는 KernelExplainer를 사용합니다.
    if hasattr(trained_model, "feature_importances_"):
        explainer = shap.TreeExplainer(trained_model)
    elif not isinstance(trained_model, BaselineModel):
        explainer = shap.Explainer(trained_model, X)
    
    # 기준 모델이 아닌 경우에만 SHAP 값을 계산합니다.
    if not isinstance(trained_model, BaselineModel):
        # shap_values = explainer(X)
        shap_values = explainer(X, check_additivity=False) 
        # (주석 처리된 코드) SHAP 요약 플롯을 그립니다.
        # plt.figure(figsize=(10, 5))
        # shap.summary_plot(shap_values, X, max_display=10, show=False)
        # plt.show()
    else:
        shap_values = None

    return predictions, [shap_values, X]


# ROC 곡선을 처음부터 계산하는 함수
def roc_from_scratch(probabilities, test_dataset, partitions=100):
    print("      Calculation of the ROC curve...")
    y_test = test_dataset.iloc[:, -1] # 테스트 데이터의 실제 레이블

    roc = []
    # 0부터 1까지 101개의 임계값을 순회합니다.
    for i in range(partitions + 1):
        thr = i / partitions
        threshold_vector = (probabilities >= thr).astype(int) # 임계값을 기준으로 예측
        tpr, fpr = true_false_positive(threshold_vector, y_test) # TPR과 FPR을 계산
        roc.append([fpr, tpr])

    # 계산된 TPR과 FPR을 데이터프레임으로 만듭니다.
    roc_data = pd.DataFrame(roc, columns=["False positive rate", "True positive rate"])
    print("      Calculation done")
    print("      Scoring...")

    # scikit-learn의 'roc_auc_score'를 사용하여 AUC 점수를 계산합니다.
    auc_score = roc_auc_score(y_test, probabilities)
    print("      Scoring done\n")
    return roc_data, auc_score


# TPR(True Positive Rate)과 FPR(False Positive Rate)을 계산하는 함수
def true_false_positive(threshold_vector: np.array, y_test: np.array):
    # "1"은 불합격(fail)이며, 이는 긍정(positive)입니다.
    true_positive = (threshold_vector == 1) & (y_test == 1) # TP: 예측=1 & 실제=1
    false_positive = (threshold_vector == 1) & (y_test == 0) # FP: 예측=1 & 실제=0
    true_negative = (threshold_vector == 0) & (y_test == 0) # TN: 예측=0 & 실제=0
    false_negative = (threshold_vector == 0) & (y_test == 1) # FN: 예측=0 & 실제=1

    # TPR 계산: TP / (TP + FN)
    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum() + 1e-9)
    # FPR 계산: FP / (FP + TN)
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum() + 1e-9)
    return tpr, fpr


# 예측 결과를 바탕으로 다양한 성능 지표를 생성하는 함수
def create_metrics(
    predictions: np.array, test_dataset: pd.DataFrame, auc_score, threshold
):
    print("      Creating the metrics...")
    # 임계값을 기준으로 최종 예측 레이블을 생성합니다.
    threshold_vector = (predictions >= threshold).astype(int)

    y_test = test_dataset.iloc[:, -1]

    # TP, TN, FP, FN 값을 계산합니다.
    tp = ((threshold_vector == 1) & (y_test == 1)).sum()
    tn = ((threshold_vector == 0) & (y_test == 0)).sum()
    fp = ((threshold_vector == 1) & (y_test == 0)).sum()
    fn = ((threshold_vector == 0) & (y_test == 1)).sum()

    # F1 스코어 계산 (클래스 1에 대한)
    denom = 2 * tp + fp + fn
    if denom == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * tp / denom
    f1_score = np.around(f1_score, 2) # 소수점 둘째 자리까지 반올림

    # 정확도(Accuracy) 계산
    accuracy = np.around((tp + tn) / (tp + tn + fp + fn + 1e-9), 2)
    # AUC 스코어 반올림
    auc_score = np.around(auc_score, 2)

    # TP, TN, FP, FN 값을 딕셔너리로 저장합니다.
    dict_ftpn = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    number_of_good_predictions = tp + tn
    number_of_false_predictions = fp + fn

    # 정밀도(Precision)와 재현율(Recall) 계산
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

    # 모든 지표를 딕셔너리에 담아 반환합니다.
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


# 예측 결과를 데이터프레임으로 정리하는 함수
def create_results(forecast_values, test_dataset, threshold):
    # 예측 확률을 소수점 둘째 자리까지 반올림하여 시리즈로 만듭니다.
    forecast_series_proba = pd.Series(
        np.around(forecast_values, decimals=2),
        index=test_dataset.index,
        name="Probability",
    )
    # 임계값을 기준으로 예측 레이블(0 또는 1)을 시리즈로 만듭니다.
    forecast_series = pd.Series(
        (forecast_values > threshold).astype(int),
        index=test_dataset.index,
        name="Forecast",
    )
    # 실제 레이블을 시리즈로 만듭니다.
    true_series = pd.Series(
        test_dataset.iloc[:, -1], name="Historical", index=test_dataset.index
    )
    # 인덱스 번호를 담는 시리즈를 만듭니다.
    index_series = pd.Series(
        range(len(true_series)), index=test_dataset.index, name="Id"
    )

    # 모든 시리즈를 하나의 데이터프레임으로 합칩니다.
    results = pd.concat(
        [index_series, forecast_series_proba, forecast_series, true_series], axis=1
    )
    # 혼동 행렬 레이블을 추가합니다.
    results["True/False/Positive/Negative"] = results.apply(_confusion_label, axis=1)
    return results

##### util function for data processing

def filter_features_user(features):
    """
    주어진 features 리스트에서 특정 항목을 제외하고 포함하여 정렬된 리스트를 반환합니다.
    """
    # data/features_user_excluded.csv 의 데이터를 features_user_excluded list 로 저장
    features_user_excluded_df = pd.read_csv('data/features_user_excluded.csv')
    features_user_excluded = features_user_excluded_df.iloc[:, 0].tolist()

    # data/features_user_included.csv 의 데이터를 features_user_included list 로 저장
    features_user_included_df = pd.read_csv('data/features_user_included.csv')
    features_user_included = features_user_included_df.iloc[:, 0].tolist()

    # 집합으로 변환하여 항목 제거 및 추가
    features_set = set(features)
    excluded_set = set(features_user_excluded)
    included_set = set(features_user_included)

    # 1. 'features'에서 'features_user_excluded' 항목 제거
    features_filtered_set = features_set - excluded_set

    # 2. 'features_filtered_set'에 'features_user_included' 항목 추가
    features_filtered_set.update(included_set)

    # --- 여기서부터 정렬 로직 추가 ---
    
    # included_set 항목 중 최종 결과에 포함된 항목만 추출하여 정렬
    included_sorted = sorted(list(features_filtered_set.intersection(included_set)))
    
    # 나머지 항목 추출 (included_set에 없는 항목)
    remaining_features = features_filtered_set - included_set
    
    # 나머지 항목 정렬
    remaining_sorted = sorted(list(remaining_features))
    
    # included_sorted 리스트와 remaining_sorted 리스트를 합쳐 최종 리스트 생성
    features_filtered = included_sorted + remaining_sorted
    
    return features_filtered

def drop_cols_1value(df):
    # Identify columns where all values are the same
    columns_to_drop = []
    for col in df.columns:
        # Check if the number of unique values in the column is 1
        if df[col].nunique() == 1:
            columns_to_drop.append(col)

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"\nDropped columns (all values identical): {columns_to_drop}")
    else:
        print("\nNo columns found where all values are identical.")
    return df

from functools import reduce # reduce 함수를 임포트합니다.

def multi_maximum(series_list):
    """
    주어진 Series 리스트에서 각 요소별 최댓값을 계산합니다.

    Parameters:
    series_list (list of pandas.Series): 최댓값을 계산할 Series 객체들의 리스트.
                                         모든 Series는 동일한 인덱스와 길이를 가져야 합니다.

    Returns:
    pandas.Series: 각 위치별 최댓값을 담고 있는 새로운 Series.
    """
    if not series_list:
        raise ValueError("Series 리스트는 비어 있을 수 없습니다.")
    
    # 첫 번째 Series를 초기 값으로 설정하고, reduce를 사용하여 순차적으로 np.maximum을 적용합니다.
    # reduce(function, iterable, initializer)
    # initializer가 주어지지 않으면, iterable의 첫 번째 항목이 initializer가 되고
    # 두 번째 항목부터 function에 적용됩니다.
    return reduce(np.maximum, series_list)

def export_features_json(final_features, fn_json):
    features_filtered = filter_features_user(final_features)
    with open(fn_json, "w") as f:
        json.dump(features_filtered, f, indent=4) # final_features 리스트를 JSON 파일로 저장 (들여쓰기 4칸)
    return features_filtered # final_features 리스트를 반환합니다.
    
def print_object_attributes(obj):
    """
    객체의 모든 속성을 한 줄씩 출력하는 함수.
    
    Args:
        obj: 속성을 출력할 객체.
    """
    if not hasattr(obj, '__dict__'):
        print(f"'{type(obj).__name__}' 객체는 속성을 가지고 있지 않습니다.")
        return

    print(f"--- {type(obj).__name__} 객체 속성 ---")
    attributes = vars(obj)
    
    # 딕셔너리를 반복하며 속성 이름과 값을 한 줄씩 출력
    for key, value in attributes.items():
        print(f"{key}: {value}")
    
    return attributes

def get_train_parameters_default(function_name: str):
    """
    지정된 알고리즘에 대한 기본 파라미터 설정을 반환합니다.

    Args:
        function_name (str): 알고리즘 함수 이름.

    Returns:
        dict: 알고리즘 기본 파라미터 설정.
    """
    if function_name == 'train_model_xgboost_cv':
        return {
            "function_name": "train_model_xgboost_cv",
            "param_grid": {
                "n_estimators": [30, 50, 100],
                "max_depth": [2, 5],
                "learning_rate": [0.01, 0.1, 0.2],
            },
            "scale_pos_weight_multiplier": 2,
            "cv": 3,
            "verbose": 1,
            "f2_rare_scorer": {
                "name": "fbeta_score",
                "beta": 2,
                "pos_label": 1,
            }
        }
    elif function_name == 'train_model_logistic_regression':
        return {
            "function_name": "train_model_logistic_regression",
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight_multiplier": "len(y) - n_pos"
        }
    elif function_name == 'train_model_baseline':
        return {
            "function_name": "train_model_baseline",
            "importance_data": {
                "Features": ["SensorOffsetHot-Cold", "band gap dpat_ok for band gap", "Radius"],
                "Importance": [56.6, 4.65, 96.9],
            }
        }
    elif function_name == 'train_model_rf_cv':
        return {
            "function_name": "train_model_rf_cv",
            "param_grid": {
                "n_estimators": [20, 50, 100],
                "max_depth": [2, 5, None],
                "min_samples_split": [2, 5],
            },
            "class_weight_multiplier": "len(y) - sum(y)",
            "cv": 3,
            "verbose": 1,
            "f2_rare_scorer": {
                "name": "fbeta_score",
                "beta": 2,
                "pos_label": 1,
            }
        }
    elif function_name == 'train_model_decision_tree':
        return {
            "function_name": "train_model_decision_tree",
            "max_depth": None,
            "min_samples_split": 2,
            "class_weight_multiplier": "auto"
        }
    elif function_name == 'train_model_logistic_regression_cv':
        return {
            "function_name": "train_model_logistic_regression_cv",
            "param_grid": {
                "C": [0.01, 0.1, 1],
                "penalty": ["l2"],
                "solver": ["lbfgs"]
            },
            "class_weight_multiplier": "len(y) - sum(y)",
            "cv": 3,
            "verbose": 1,
            "f2_rare_scorer": {
                "name": "fbeta_score",
                "beta": 2,
                "pos_label": 1,
            }
        }
    else:
        # 다른 알고리즘에 대한 기본 설정
        return {}
    

