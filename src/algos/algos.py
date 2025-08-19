# 필요한 라이브러리들을 불러옵니다.
# scikit-learn에서 여러 머신러닝 모델과 유틸리티를 가져옵니다.
from sklearn.linear_model import LogisticRegression         # 로지스틱 회귀 모델
from sklearn.ensemble import RandomForestClassifier         # 랜덤 포레스트 분류 모델
from sklearn.model_selection import train_test_split, GridSearchCV # 데이터 분할 및 하이퍼파라미터 튜닝을 위한 도구
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif # 특징(변수) 선택을 위한 도구
from sklearn.tree import DecisionTreeClassifier             # 의사결정나무 모델
from sklearn.metrics import roc_auc_score, fbeta_score, make_scorer, precision_score # 모델 성능 평가 지표
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.compose import ColumnTransformer

import optuna

from xgboost import XGBClassifier                           # XGBoost 분류 모델 (경사 부스팅)
import shap                                                 # SHAP (SHapley Additive exPlanations) 라이브러리, 모델 예측에 대한 설명력을 제공합니다.
import matplotlib.pyplot as plt                             # 데이터 시각화를 위한 라이브러리

# 데이터 처리 및 기타 작업을 위한 라이브러리들을 불러옵니다.
import pandas as pd                                         # 데이터프레임 구조를 다루는 데 필수적인 라이브러리
import numpy as np                                          # 숫자 연산을 위한 라이브러리
import datetime as dt                                       # 날짜와 시간을 다루는 라이브러리
import json                                                 # JSON 형식의 데이터를 처리하는 라이브러리
import pprint

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


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
    processed_dataset = drop_cols_1value(processed_dataset)

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


# 분산 및 상관관계 필터를 적용하는 함수입니다.
def variance_correlation_filter_old(
    X: pd.DataFrame, var_threshold=0.0, corr_threshold=0.98
):
    """
    1) 분산이 var_threshold 이하인 특징을 제거합니다.
    2) 절대 상관관계가 corr_threshold를 초과하는 특징 쌍 중 하나를 제거합니다.
    반환값:
      (필터링된 X_filtered, 최종 컬럼 리스트 final_cols)
    """
    # 분산 임계값(VarianceThreshold)을 사용하여 분산이 0 이하인 컬럼을 제거합니다.
    vt = VarianceThreshold(threshold=var_threshold)
    X_vt = vt.fit_transform(X) # 데이터를 변환
    vt_mask = vt.get_support() # 제거되지 않고 남은 컬럼의 마스크를 가져옴
    vt_cols = X.columns[vt_mask] # 남은 컬럼 이름들을 가져옴
    print("Number of features kept after Variance threshold", sum(vt_mask))

    # 필터링된 데이터로 새로운 데이터프레임을 만듭니다.
    X_vt_df = pd.DataFrame(X_vt, columns=vt_cols)

    # 상관관계 행렬을 계산하고, 절대값으로 변환합니다.
    corr_matrix = X_vt_df.corr().abs()
    # 상관관계 행렬의 위쪽 삼각형 부분만 선택합니다 (중복 방지).
    upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    # 상관관계가 임계값(0.98)보다 높은 컬럼들을 찾아서 제거할 리스트에 담습니다.
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    # 해당 컬럼들을 데이터프레임에서 제거합니다.
    X_filtered = X_vt_df.drop(to_drop, axis=1)
    # 최종적으로 남은 컬럼 리스트를 저장합니다.
    final_cols = list(X_filtered.columns)

    return X_filtered, final_cols

def variance_correlation_filter(
    X: pd.DataFrame, var_threshold=0.0, corr_threshold=0.98
):
    """
    1) 분산이 var_threshold 이하인 특징을 제거합니다.
    2) 절대 상관관계가 corr_threshold를 초과하는 특징 쌍 중 하나를 제거합니다.
    
    Args:
        X (pd.DataFrame): 원본 데이터프레임.
        var_threshold (float): 분산 임계값.
        corr_threshold (float): 상관관계 임계값.

    Returns:
        (pd.DataFrame, list, int, int):
            - 필터링된 데이터프레임
            - 최종 컬럼 리스트
            - 분산 필터링으로 제거된 피처 수
            - 상관관계 필터링으로 제거된 피처 수
    """
    # 분산 임계값(VarianceThreshold)을 사용하여 분산이 0 이하인 컬럼을 제거합니다.
    initial_feature_count = X.shape[1]
    vt = VarianceThreshold(threshold=var_threshold)
    X_vt = vt.fit_transform(X) # 데이터를 변환
    vt_mask = vt.get_support() # 제거되지 않고 남은 컬럼의 마스크를 가져옴
    vt_cols = X.columns[vt_mask] # 남은 컬럼 이름들을 가져옴
    features_dropped_by_variance = initial_feature_count - len(vt_cols)
    print(f"    - 분산 필터링 후 남은 피처 수: {len(vt_cols)}")

    # 필터링된 데이터로 새로운 데이터프레임을 만듭니다.
    X_vt_df = pd.DataFrame(X_vt, columns=vt_cols)

    # 상관관계 행렬을 계산하고, 절대값으로 변환합니다.
    corr_matrix = X_vt_df.corr().abs()
    # 상관관계 행렬의 위쪽 삼각형 부분만 선택합니다 (중복 방지).
    upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    # 상관관계가 임계값(0.98)보다 높은 컬럼들을 찾아서 제거할 리스트에 담습니다.
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    features_dropped_by_correlation = len(to_drop)
    # 해당 컬럼들을 데이터프레임에서 제거합니다.
    X_filtered = X_vt_df.drop(to_drop, axis=1)
    # 최종적으로 남은 컬럼 리스트를 저장합니다.
    final_cols = list(X_filtered.columns)
    
    print(f"    - 상관관계 필터링 후 남은 피처 수: {len(final_cols)}")

    return X_filtered, final_cols, features_dropped_by_variance, features_dropped_by_correlation


# 자동으로 피처를 생성

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer  # 누락된 값 처리를 위한 Imputer 추가

def feature_generator_old(X):
    """
    입력 데이터프레임 X에 대해 수치형 피쳐에만 다양한 피쳐 생성 기법을 적용합니다.
    누락된 값(NaN)을 처리하기 위해 SimpleImputer를 포함합니다.

    Args:
        X (pd.DataFrame): 원본 데이터프레임. 수치형 및 범주형 열을 포함할 수 있습니다.

    Returns:
        pd.DataFrame: 생성된 피쳐가 추가된 새로운 데이터프레임.
    """
    
    # 1. 수치형 피쳐와 범주형 피쳐 식별
    numerical_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(exclude=np.number).columns
    
    if len(numerical_features) == 0:
        print("경고: 수치형 피쳐가 없습니다. 피쳐 생성을 건너뜁니다.")
        return X.copy()
        
    # 2. 수치형 피쳐를 위한 변환 파이프라인
    # 쌍별 합과 차를 생성하는 함수 정의
    def sum_diff_transformer_func(X_array):
        n_features = X_array.shape[1]
        sum_features = []
        diff_features = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                sum_features.append(X_array[:, i] + X_array[:, j])
                diff_features.append(X_array[:, i] - X_array[:, j])
        if not sum_features and not diff_features:
            return np.empty((X_array.shape[0], 0))
        return np.column_stack(sum_features + diff_features)

    # 3. ColumnTransformer를 사용하여 수치형 피쳐에만 변환 적용
    preprocessor = ColumnTransformer(
        transformers=[
            # PolynomialFeatures 이전에 Imputer를 적용하는 파이프라인 구성
            ('poly_pipeline', 
             Pipeline([
                 ('imputer', SimpleImputer(strategy='mean')),
                 ('poly', PolynomialFeatures(degree=2, include_bias=False))
             ]),
             numerical_features),
            
            # FunctionTransformer 이전에 Imputer를 적용하는 파이프라인 구성
            ('sum_diff_pipeline',
             Pipeline([
                 ('imputer', SimpleImputer(strategy='mean')),
                 ('sum_diff', FunctionTransformer(sum_diff_transformer_func))
             ]),
             numerical_features)
        ],
        remainder='passthrough'  # 범주형 피쳐는 그대로 통과
    )
    
    # 변환 실행 및 피쳐 이름 생성
    transformed_data = preprocessor.fit_transform(X)
    
    # 4. 변환된 피쳐 이름 얻기
    # Pipeline 내의 변환기 접근
    poly_feature_names = preprocessor.named_transformers_['poly_pipeline'].named_steps['poly'].get_feature_names_out(numerical_features)
    
    sum_diff_feature_names = []
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            sum_diff_feature_names.append(f'{numerical_features[i]}_{numerical_features[j]}_sum')
            sum_diff_feature_names.append(f'{numerical_features[i]}_{numerical_features[j]}_diff')
    
    # 모든 피쳐 이름 결합: [poly, sum_diff, 원본 범주형]
    all_feature_names = list(poly_feature_names) + sum_diff_feature_names + list(categorical_features)
    
    X_gen = pd.DataFrame(transformed_data, columns=all_feature_names)
    
    # ColumnTransformer의 'remainder'가 원본 열을 마지막에 추가하므로,
    # 원본 수치형 열이 중복되어 포함됩니다. 이를 제거해야 합니다.
    # PolynomialFeatures가 원본 열을 이미 포함하므로, 원본 수치형 열은 따로 추가하지 않습니다.
    X_gen = X_gen.loc[:, ~X_gen.columns.duplicated()]

    return X_gen

def feature_generator_old2(X, sum_features=False, diff_features=False, poly_features=False, poly_degree=2):
    """
    입력 데이터프레임 X에 대해 수치형 피처에 다양한 피처 생성 기법을 선택적으로 적용합니다.
    누락된 값(NaN)을 처리하기 위해 SimpleImputer를 포함합니다.

    Args:
        X (pd.DataFrame): 원본 데이터프레임.
        sum_features (bool): 쌍별 합 피처를 생성할지 여부.
        diff_features (bool): 쌍별 차 피처를 생성할지 여부.
        poly_features (bool): 다항식 피처를 생성할지 여부.
        poly_degree (int): 다항식 피처의 차수.

    Returns:
        pd.DataFrame: 생성된 피처가 추가된 새로운 데이터프레임.
    """
    numerical_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(exclude=np.number).columns
    
    if len(numerical_features) == 0:
        print("경고: 수치형 피처가 없습니다. 피처 생성을 건너뜁니다.")
        return X.copy()
    
    transformers = []
    
    if poly_features:
        # 다항식 피처 생성 파이프라인
        poly_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False))
        ])
        transformers.append(('poly_pipeline', poly_pipeline, numerical_features))

    if sum_features or diff_features:
        # 쌍별 합과 차를 생성하는 함수 정의
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

        # 합/차 피처 생성 파이프라인
        sum_diff_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('sum_diff', FunctionTransformer(sum_diff_transformer_func))
        ])
        transformers.append(('sum_diff_pipeline', sum_diff_pipeline, numerical_features))
        
    # 아무것도 선택되지 않았을 경우, 원본 수치형 피처를 그대로 통과시킵니다.
    if not transformers:
        return X.copy()
        
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )
    
    transformed_data = preprocessor.fit_transform(X)
    
    # 변환된 피처 이름 생성 및 결합
    all_feature_names = []
    
    if poly_features:
        poly_feature_names = preprocessor.named_transformers_['poly_pipeline'].named_steps['poly'].get_feature_names_out(numerical_features)
        all_feature_names.extend(poly_feature_names)
        
    if sum_features or diff_features:
        sum_diff_feature_names = []
        for i in range(len(numerical_features)):
            for j in range(i + 1, len(numerical_features)):
                if sum_features:
                    sum_diff_feature_names.append(f'{numerical_features[i]}_{numerical_features[j]}_sum')
                if diff_features:
                    sum_diff_feature_names.append(f'{numerical_features[i]}_{numerical_features[j]}_diff')
        all_feature_names.extend(sum_diff_feature_names)
    
    all_feature_names.extend(categorical_features)
    
    X_gen = pd.DataFrame(transformed_data, columns=all_feature_names)
    
    # 중복된 컬럼 제거 (ColumnTransformer의 remainder='passthrough'로 인해 발생)
    X_gen = X_gen.loc[:, ~X_gen.columns.duplicated()]

    return X_gen	

def feature_generator_old3(X, sum_features=False, diff_features=False, poly_features=False, poly_degree=2, apply_filter_gen=False, var_threshold_gen=0.0, corr_threshold_gen=0.98):
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
        pd.DataFrame: 생성된 피처가 추가된 새로운 데이터프레임.
    """
    
    X_in = X.copy()
    
    if apply_filter_gen:
        print(f"    - Feature Generator 내부에서 필터링 적용 (분산: {var_threshold_gen}, 상관관계: {corr_threshold_gen})")
        X_filtered, _ = variance_correlation_filter(X, var_threshold=var_threshold_gen, corr_threshold=corr_threshold_gen)
        X = X_filtered
    else:
        print("    - Feature Generator 내부 필터링 미적용")

    numerical_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(exclude=np.number).columns
    
    if len(numerical_features) == 0:
        print("경고: 수치형 피처가 없습니다. 피처 생성을 건너뜁니다.")
        return X.copy()
    
    # 생성될 새로운 피처들을 담을 데이터프레임
    X_generated = pd.DataFrame(index=X.index)

    if poly_features:
        poly_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False))
        ])
        
        poly_data = poly_pipeline.fit_transform(X[numerical_features])
        poly_feature_names = poly_pipeline.named_steps['poly'].get_feature_names_out(numerical_features)
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
        
        X_generated = pd.concat([X_generated, pd.DataFrame(sum_diff_data, columns=sum_diff_feature_names, index=X.index)], axis=1)

        print(f"    - Feature Generator에서 생성된 피처 수: {X_generated.shape[1]}")

    # 원본 컬럼과 생성된 컬럼을 결합
    # 중복되는 컬럼을 제거하고 원본 컬럼이 앞에 오도록 재정렬
    # X_combined = pd.concat([X, X_generated], axis=1)
    X_combined = pd.concat([X_in, X_generated], axis=1)
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]
    

    return X_combined

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
def create_train_test_data_old(preprocessed_dataset: pd.DataFrame):
    print("\n     훈련 및 테스트 데이터셋 생성 중...")

    outlier_mask = (preprocessed_dataset["Radius"] < 32) & (
        preprocessed_dataset["Pass/Fail"]
    ) # "Radius"가 32 미만이고 "Pass/Fail"이 True인 이상치 마스크 생성

    # ~를 사용하여 반전합니다. 즉, 이상치가 아닌 행을 유지합니다.
    preprocessed_dataset = preprocessed_dataset[~outlier_mask].reset_index(drop=True) # 이상치를 제거하고 인덱스 재설정

    X = preprocessed_dataset.iloc[:, :-1] # 마지막 컬럼을 제외한 모든 컬럼 (특성)
    y = preprocessed_dataset.iloc[:, -1] # 마지막 컬럼 (레이블)
    X, kept_cols = variance_correlation_filter(
        X, var_threshold=0.0, corr_threshold=0.99
    ) # 분산 및 상관 관계 필터 적용

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    ) # 훈련 및 테스트 데이터셋으로 분할 (80% 훈련, 20% 테스트)
    train_data = pd.concat([X_train, y_train], axis=1) # 훈련 특성과 레이블을 결합
    test_data = pd.concat([X_test, y_test], axis=1) # 테스트 특성과 레이블을 결합

    return train_data, test_data # 훈련 및 테스트 데이터 반환


def create_train_test_data_old2(
    preprocessed_dataset: pd.DataFrame,
    split_parameter: dict = None
):
    """
    훈련/테스트 데이터 분할 및 샘플링을 적용하는 함수.

    Args:
        preprocessed_dataset (pd.DataFrame): 전처리된 데이터셋.
        split_parameter (dict, optional): 데이터 분할 및 샘플링 관련 파라미터.
            - 'test_size' (float, optional): 테스트 데이터셋 비율. Defaults to 0.2.
            - 'random_state' (int, optional): 난수 시드. Defaults to 42.
            - 'var_threshold' (float, optional): 분산 필터 임계값. Defaults to 0.0.
            - 'corr_threshold' (float, optional): 상관 관계 필터 임계값. Defaults to 0.98.
            - 'sampling_ratio' (float, optional): 소수 클래스(y=1) 비율.
                - 1.0: 1:1 비율로 샘플링 (오버/언더 자동 적용).
                - 1.0 초과: 오버샘플링. 1 : n_samples_majority * sampling_ratio[2, 4, ...]
                - 1.0 미만: 언더샘플링. 0 : int(n_samples_minority / sampling_ratio[0.5, 0.25, ...])
                - None: 샘플링 미적용.
            - 'apply_feature_generation': True # 특성 생성을 적용하도록 설정

    Returns:
        tuple: (훈련 데이터, 테스트 데이터, 파라미터 정보 딕셔너리).
    """
    print("\n\n##############################################################################################################################")
    print("# 3) Create Train/Test Split (훈련/테스트 데이터 분할) ")
    print("##############################################################################################################################")
    
    print("\n    훈련 및 테스트 데이터셋 생성 중...")

    # 기본 파라미터 설정
    default_params = {
        'test_size': 0.2,
        'random_state': 42,
        'var_threshold': 0.0,
        'corr_threshold': 0.98,
        'sampling_ratio': None
    }
    
    # 입력 파라미터로 기본값 업데이트
    if split_parameter:
        default_params.update(split_parameter)
        
    split_parameter = default_params

    # 이상치 제거
    outlier_mask = (preprocessed_dataset["Radius"] < 32) & (preprocessed_dataset["Pass/Fail"])
    preprocessed_dataset = preprocessed_dataset[~outlier_mask].reset_index(drop=True)

    # 특성과 레이블 분리
    X = preprocessed_dataset.iloc[:, :-1]
    y = preprocessed_dataset.iloc[:, -1]
    
    
    # Feature generation
    X = feature_generator(X)
    

    # 분산 및 상관 관계 필터 적용
    X, kept_cols = variance_correlation_filter(
        X, 
        var_threshold=split_parameter['var_threshold'], 
        corr_threshold=split_parameter['corr_threshold']
    )

    # 훈련/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=split_parameter['test_size'], 
        random_state=split_parameter['random_state'], 
        stratify=y
    )
    
    print("\n    - 분할 전 훈련 데이터 클래스 분포:", sorted(Counter(y_train).items()))

    # 샘플링 적용
    sampling_ratio = split_parameter['sampling_ratio']
    
    if sampling_ratio is not None:
        if sampling_ratio >= 1: # 오버샘플링 (또는 1:1 균형)
            # y=1이 소수 클래스임을 가정하고 오버샘플링
            # 소수 클래스(y=1)의 수를 다수 클래스(y=0) 수에 비례하여 조정
            n_samples_majority = sum(y_train == 0)
            target_minority_count = int(n_samples_majority * sampling_ratio)
            sampling_strategy = {1: target_minority_count}
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=split_parameter['random_state'])
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f"    - 오버샘플링 적용 완료 (소수 클래스 비율: {sampling_ratio})")
        
        else: # 언더샘플링
            # y=1이 소수 클래스임을 가정하고 언더샘플링
            # 다수 클래스(y=0)의 수를 소수 클래스(y=1) 수에 비례하여 조정
            n_samples_minority = sum(y_train == 1)
            target_majority_count = int(n_samples_minority / sampling_ratio)
            sampling_strategy = {0: target_majority_count}
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=split_parameter['random_state'])
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f"    - 언더샘플링 적용 완료 (소수 클래스 비율: {sampling_ratio})")
            
        print(f"    - 샘플링 적용 후 훈련 데이터 클래스 분포: {sorted(Counter(y_train).items())}")
    else:
        print("    - 샘플링 미적용")

    # 훈련 및 테스트 데이터 결합
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # 파라미터 정보 저장
    split_parameter_info = split_parameter.copy()
    split_parameter_info['sampling_applied'] = 'None' if sampling_ratio is None else ('oversampling' if sampling_ratio >= 1 else 'undersampling')
    
    return train_data, test_data, split_parameter_info

def create_train_test_data_old3(
    preprocessed_dataset: pd.DataFrame,
    split_parameter: dict = None
):
    """
    훈련/테스트 데이터 분할 및 샘플링을 적용하는 함수.
    Feature Generation 적용 여부를 split_parameter에서 제어합니다.
    """
    print("\n\n##############################################################################################################################")
    print("# 3) Create Train/Test Split (훈련/테스트 데이터 분할) ")
    print("##############################################################################################################################")
    
    print("\n    훈련 및 테스트 데이터셋 생성 중...")

    # 기본 파라미터 설정
    default_params = {
        'test_size': 0.2,
        'random_state': 42,
        'var_threshold': 0.0,
        'corr_threshold': 0.98,
        'sampling_ratio': None,
        'apply_feature_generation': False  # 기본값으로 False 설정
    }
    
    # 입력 파라미터로 기본값 업데이트
    if split_parameter:
        default_params.update(split_parameter)
        
    split_parameter = default_params

    # 이상치 제거
    outlier_mask = (preprocessed_dataset["Radius"] < 32) & (preprocessed_dataset["Pass/Fail"])
    preprocessed_dataset = preprocessed_dataset[~outlier_mask].reset_index(drop=True)

    # 특성과 레이블 분리
    X = preprocessed_dataset.iloc[:, :-1]
    y = preprocessed_dataset.iloc[:, -1]
    
    # 분산 및 상관 관계 필터 적용
    # 이 부분은 변경되지 않았으므로 기존 코드 그대로 사용
    # X, kept_cols = variance_correlation_filter(...)
    # 해당 함수 정의가 없으므로 주석 처리
    X, kept_cols = variance_correlation_filter(
        X, 
        var_threshold=split_parameter['var_threshold'], 
        corr_threshold=split_parameter['corr_threshold']
    )

    # Feature generation 적용 여부 확인
    if split_parameter['apply_feature_generation']:
        print("    - Feature Generation 적용...")
        X = feature_generator(X)
        print("    - Feature Generation 완료. 새로운 피쳐 수:", X.shape[1] - (preprocessed_dataset.shape[1] - 1))
    else:
        print("    - Feature Generation 미적용.")


    # 훈련/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=split_parameter['test_size'], 
        random_state=split_parameter['random_state'], 
        stratify=y
    )
    
    print("\n    - 분할 전 훈련 데이터 클래스 분포:", sorted(Counter(y_train).items()))

    # 샘플링 적용
    sampling_ratio = split_parameter['sampling_ratio']
    
    if sampling_ratio is not None:
        if sampling_ratio >= 1: # 오버샘플링 (또는 1:1 균형)
            n_samples_majority = sum(y_train == 0)
            target_minority_count = int(n_samples_majority * sampling_ratio)
            sampling_strategy = {1: target_minority_count}
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=split_parameter['random_state'])
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f"    - 오버샘플링 적용 완료 (소수 클래스 비율: {sampling_ratio})")
        else: # 언더샘플링
            n_samples_minority = sum(y_train == 1)
            target_majority_count = int(n_samples_minority / sampling_ratio)
            sampling_strategy = {0: target_majority_count}
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=split_parameter['random_state'])
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f"    - 언더샘플링 적용 완료 (소수 클래스 비율: {sampling_ratio})")
            
        print(f"    - 샘플링 적용 후 훈련 데이터 클래스 분포: {sorted(Counter(y_train).items())}")
    else:
        print("    - 샘플링 미적용")

    # 훈련 및 테스트 데이터 결합
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # 파라미터 정보 저장
    split_parameter_info = split_parameter.copy()
    split_parameter_info['sampling_applied'] = 'None' if sampling_ratio is None else ('oversampling' if sampling_ratio >= 1 else 'undersampling')
    
    return train_data, test_data, split_parameter_info

def create_train_test_data_old4(
    preprocessed_dataset: pd.DataFrame,
    split_parameter: dict = None
):
    """
    훈련/테스트 데이터 분할 및 샘플링을 적용하는 함수.
    Feature Generation 적용 여부 및 옵션을 split_parameter에서 제어합니다.
    """
    print("\n\n##############################################################################################################################")
    print("# 3) Create Train/Test Split (훈련/테스트 데이터 분할) ")
    print("##############################################################################################################################")
    
    print("\n    훈련 및 테스트 데이터셋 생성 중...")

    # 기본 파라미터 설정
    default_params = {
        'test_size': 0.2,
        'random_state': 42,
        'var_threshold': 0.0,
        'corr_threshold': 0.98,
        'sampling_ratio': None,
        'apply_feature_generation': False,
        'sum_features': False,
        'diff_features': False,
        'poly_features': False,
        'poly_degree': 2
    }
    
    # 입력 파라미터로 기본값 업데이트
    if split_parameter:
        default_params.update(split_parameter)
        
    split_parameter = default_params

    # 이상치 제거
    outlier_mask = (preprocessed_dataset["Radius"] < 32) & (preprocessed_dataset["Pass/Fail"])
    preprocessed_dataset = preprocessed_dataset[~outlier_mask].reset_index(drop=True)

    # 특성과 레이블 분리
    X = preprocessed_dataset.iloc[:, :-1]
    y = preprocessed_dataset.iloc[:, -1]
    
    # 분산 및 상관 관계 필터 적용
    # (variance_correlation_filter 함수가 정의되어 있다고 가정)
    X, kept_cols = variance_correlation_filter(
        X, 
        var_threshold=split_parameter['var_threshold'], 
        corr_threshold=split_parameter['corr_threshold']
    )
    
    # Feature generation 적용 여부 확인
    if split_parameter['apply_feature_generation']:
        print("    - Feature Generation 적용...")
        X = feature_generator(
            X, 
            sum_features=split_parameter['sum_features'],
            diff_features=split_parameter['diff_features'],
            poly_features=split_parameter['poly_features'],
            poly_degree=split_parameter['poly_degree']
        )
        print("    - Feature Generation 완료. 새로운 피처 수:", X.shape[1] - (preprocessed_dataset.shape[1] - 1))
    else:
        print("    - Feature Generation 미적용.")

    # 훈련/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=split_parameter['test_size'], 
        random_state=split_parameter['random_state'], 
        stratify=y
    )
    
    print("\n    - 분할 전 훈련 데이터 클래스 분포:", sorted(Counter(y_train).items()))

    # 샘플링 적용
    sampling_ratio = split_parameter['sampling_ratio']
    
    if sampling_ratio is not None:
        if sampling_ratio >= 1: # 오버샘플링 (또는 1:1 균형)
            n_samples_majority = sum(y_train == 0)
            target_minority_count = int(n_samples_majority * sampling_ratio)
            sampling_strategy = {1: target_minority_count}
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=split_parameter['random_state'])
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f"    - 오버샘플링 적용 완료 (소수 클래스 비율: {sampling_ratio})")
        else: # 언더샘플링
            n_samples_minority = sum(y_train == 1)
            target_majority_count = int(n_samples_minority / sampling_ratio)
            sampling_strategy = {0: target_majority_count}
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=split_parameter['random_state'])
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f"    - 언더샘플링 적용 완료 (소수 클래스 비율: {sampling_ratio})")
            
        print(f"    - 샘플링 적용 후 훈련 데이터 클래스 분포: {sorted(Counter(y_train).items())}")
    else:
        print("    - 샘플링 미적용")

    # 훈련 및 테스트 데이터 결합
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # 파라미터 정보 저장
    split_parameter_info = split_parameter.copy()
    split_parameter_info['sampling_applied'] = 'None' if sampling_ratio is None else ('oversampling' if sampling_ratio >= 1 else 'undersampling')
    
    return train_data, test_data, split_parameter_info    

def create_train_test_data_old5(
    preprocessed_dataset: pd.DataFrame,
    split_parameter: dict = None
):
    """
    훈련/테스트 데이터 분할 및 샘플링을 적용하는 함수.
    Feature Generation 적용 여부 및 옵션을 split_parameter에서 제어합니다.
    """
    print("\n\n##############################################################################################################################")
    print("# 3) Create Train/Test Split (훈련/테스트 데이터 분할) ")
    print("##############################################################################################################################")
    
    print("\n    훈련 및 테스트 데이터셋 생성 중...")

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

    outlier_mask = (preprocessed_dataset["Radius"] < 32) & (preprocessed_dataset["Pass/Fail"])
    preprocessed_dataset = preprocessed_dataset[~outlier_mask].reset_index(drop=True)

    X = preprocessed_dataset.iloc[:, :-1]
    y = preprocessed_dataset.iloc[:, -1]
    
    # 분할 전 필터링 적용 (첫 번째 필터링 단계)
    if split_parameter['apply_filter_split']:
        print(f"    - 분할 전 필터링 적용 (분산: {split_parameter['var_threshold_split']}, 상관관계: {split_parameter['corr_threshold_split']})")
        X, _ = variance_correlation_filter(X, var_threshold=split_parameter['var_threshold_split'], corr_threshold=split_parameter['corr_threshold_split'])
    else:
        print("    - 분할 전 필터링 미적용.")
        
    
    # Feature generation 적용 여부 확인
    if split_parameter['apply_feature_generation']:
        print("    - Feature Generation 적용...")
        X = feature_generator(
            X, 
            sum_features=split_parameter['sum_features'],
            diff_features=split_parameter['diff_features'],
            poly_features=split_parameter['poly_features'],
            poly_degree=split_parameter['poly_degree'],
            apply_filter_gen=split_parameter['apply_filter_gen'],
            var_threshold_gen=split_parameter['var_threshold_gen'],
            corr_threshold_gen=split_parameter['corr_threshold_gen']
        )
        print("    - Feature Generation 완료. 새로운 피쳐 수:", X.shape[1] - (preprocessed_dataset.shape[1] - 1))
    else:
        print("    - Feature Generation 미적용.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=split_parameter['test_size'], 
        random_state=split_parameter['random_state'], 
        stratify=y
    )
    
    print("\n    - 분할 전 훈련 데이터 클래스 분포:", sorted(Counter(y_train).items()))

    sampling_ratio = split_parameter['sampling_ratio']
    
    if sampling_ratio is not None:
        if sampling_ratio >= 1:
            n_samples_majority = sum(y_train == 0)
            target_minority_count = int(n_samples_majority * sampling_ratio)
            sampling_strategy = {1: target_minority_count}
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=split_parameter['random_state'])
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f"    - 오버샘플링 적용 완료 (소수 클래스 비율: {sampling_ratio})")
        else:
            n_samples_minority = sum(y_train == 1)
            target_majority_count = int(n_samples_minority / sampling_ratio)
            sampling_strategy = {0: target_majority_count}
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=split_parameter['random_state'])
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f"    - 언더샘플링 적용 완료 (소수 클래스 비율: {sampling_ratio})")
            
        print(f"    - 샘플링 적용 후 훈련 데이터 클래스 분포: {sorted(Counter(y_train).items())}")
    else:
        print("    - 샘플링 미적용")

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    split_parameter_info = split_parameter.copy()
    split_parameter_info['sampling_applied'] = 'None' if sampling_ratio is None else ('oversampling' if sampling_ratio >= 1 else 'undersampling')
    
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
        X, _, var_dropped, corr_dropped = variance_correlation_filter(X, var_threshold=split_parameter['var_threshold_split'], corr_threshold=split_parameter['corr_threshold_split'])
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


# # --- 입출력 파라미터 샘플 ---
# # 1. 예제 데이터 생성
# # 실제 사용 시에는 전처리된 데이터프레임을 사용해야 합니다.
# from sklearn.datasets import make_classification
# data = make_classification(n_samples=500, n_features=10, weights=[0.9, 0.1], random_state=42)
# preprocessed_dataset = pd.DataFrame(data[0], columns=[f'feature_{i}' for i in range(10)])
# preprocessed_dataset['Radius'] = np.random.rand(500) * 50
# preprocessed_dataset['Pass/Fail'] = data[1]

# # 2. 함수 호출 예시
# # 모든 파라미터가 포함된 split_parameter 딕셔너리를 정의합니다.
# full_parameters = {
#     'test_size': 0.3,
#     'random_state': 100,
#     'var_threshold': 0.01,
#     'corr_threshold': 0.95,
#     'sampling_ratio': 1.0 # 1.0이면 1:1 비율로 오버샘플링/언더샘플링 자동 적용
# }

# print("\n[예제 1] 오버샘플링 적용 (소수 클래스 비율 1:1)")
# print(f"    입력 파라미터: {{'sampling_ratio': 1.0}}")
# train_data_full, test_data_full, info_full = create_train_test_data(
#     preprocessed_dataset, 
#     split_parameter={'sampling_ratio': 1.0}
# )
# print("반환된 파라미터 정보:", info_full)

# print("\n\n[예제 2] 언더샘플링 적용 (소수 클래스 비율 0.5)")
# print(f"    입력 파라미터: {{'sampling_ratio': 0.5}}")
# train_data_undersampling, test_data_undersampling, info_undersampling = create_train_test_data(
#     preprocessed_dataset, 
#     split_parameter={'sampling_ratio': 0.5}
# )
# print("반환된 파라미터 정보:", info_undersampling)

# print("\n\n[예제 3] 샘플링 미적용 (sampling_ratio=None)")
# print(f"    입력 파라미터: {{'sampling_ratio': None}}")
# train_data_no_sampling, test_data_no_sampling, info_no_sampling = create_train_test_data(
#     preprocessed_dataset, 
#     split_parameter={'sampling_ratio': None}
# )
# print("반환된 파라미터 정보:", info_no_sampling)




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


def select_feature(test_data, train_data, feature_selector):

    # X를 분리한 다음 필터링합니다.
    X_train = train_data.iloc[:, :-1]  # 레이블을 제외한 훈련 특성
    # print("X_train shape:", X_train.shape)
    y_train = train_data.iloc[:, -1] # 훈련 레이블

    # 테스트 세트에도 동일하게 적용합니다.
    X_test = test_data.iloc[:, :-1]
    # print("X_test shape:", X_test.shape)
    y_test = test_data.iloc[:, -1]

    # JSON 변수에 저장할 정보 초기화
    feature_selection_info = feature_selector
    
    # if feature_selector["feature_selector_name"] == 'CorrelationsClassifier':
    if 'CorrelationsClassifier' in feature_selector["feature_selector_name"]:    
        # print("feature selector: ", feature_selector)
        print("feature selector: ", feature_selector['feature_selector_name'])
        
        # feature_selection_info에 기본 정보 저장
        # feature_selection_info['feature_selector_name'] = feature_selector['feature_selector_name']

        # 초기 특성 필터링
        initial_feature_count = X_train.shape[1]
        print("- initial_feature_count:", initial_feature_count)

        # F2 스코어러 정의 (pos_label=1 => 실패 클래스)
        # beta=4로 재현율에 더 큰 가중치      
        # f2_scorer = make_scorer(fbeta_score, beta=4, pos_label=1)
        if feature_selector["f2_scorer"]["name"] == "fbeta_score":
            f2_scorer = make_scorer( \
                fbeta_score, 
                beta=feature_selector["f2_scorer"]["beta"], 
                pos_label=feature_selector["f2_scorer"]["pos_label"]
                )
        
        # 1. 분산 임곗값(Variance Threshold) 필터링
        # 분산이 0인 특성(모든 값이 동일한 특성)을 제거합니다.
        vt_threshold = feature_selector["variance_threshold_filter"]["threshold"]
        selector_vt = VarianceThreshold(threshold=vt_threshold)
        X_train_filtered_vt = selector_vt.fit_transform(X_train)
        X_train_filtered_vt = pd.DataFrame(X_train_filtered_vt, columns=X_train.columns[selector_vt.get_support()])

        # 2. 상관 관계 필터링
        # 타겟 변수(y_train)와의 상관 관계가 낮은 특성 제거
        # 여기서는 예시로 절대 상관 관계가 0.01 미만인 특성을 제거합니다.
        corr_threshold = feature_selector["correlation_filter"]["threshold"]
        correlations = X_train_filtered_vt.corrwith(y_train).abs()
        low_corr_features = correlations[correlations < corr_threshold].index
        X_train_final = X_train_filtered_vt.drop(columns=low_corr_features)

        # 3. 사용자 필터링
        # 사용자가 지정한 피처를 추가/제거
        features_filtered_user = filter_features_user(X_train_final.columns)

        # 최종 특성 목록
        # final_feature_count = X_train_final.shape[1]
        final_feature_count = len(features_filtered_user)
        # X_train_final_list = list(X_train_final.columns)
        X_train_final_list = features_filtered_user
        
        print("- final_feature_count:", final_feature_count)

        # feature_selection_info에 최종 정보 저장
        feature_selection_info['initial_feature_count'] = initial_feature_count
        feature_selection_info['final_feature_count'] = final_feature_count
        feature_selection_info['final_features'] = X_train_final_list
        
        # JSON 변수 출력
        # print("\n--- feature_selection_info JSON ---")
        # print(json.dumps(feature_selection_info, indent=4))    

    else:
        # print("feature selector: ", feature_selector)
        print("feature selector: ", feature_selector['feature_selector_name'])       

        # 초기 특성 필터링
        initial_feature_count = X_train.shape[1]
        print("- initial_feature_count:", initial_feature_count)

        X_train_final = X_train  # 전체 특성 사용

        # 3. 사용자 필터링
        # 사용자가 지정한 피처를 추가/제거
        features_filtered_user = filter_features_user(X_train_final.columns)

        # 최종 특성 목록
        # final_feature_count = X_train_final.shape[1]
        final_feature_count = len(features_filtered_user)
        # X_train_final_list = list(X_train_final.columns)
        X_train_final_list = features_filtered_user
       

        print("- final_feature_count:", final_feature_count)

        feature_selection_info['initial_feature_count'] = initial_feature_count
        feature_selection_info['final_feature_count'] = final_feature_count
        feature_selection_info['final_features'] = X_train_final_list

        # JSON 변수 출력
        # print("\n--- feature_selection_info JSON ---")
        # print(json.dumps(feature_selection_info, indent=4))

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
    if train_parameters.get('class_weight_multiplier') == '':
        class_weight_multiplier = n_neg
    else:
        class_weight_multiplier = eval(train_parameters.get('class_weight_multiplier'))
        
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
    n_neg = len(y) - sum(y)

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
        # param_ranges 딕셔너리를 직접 가져와서 수정
        param_ranges = train_parameters.get('param_ranges', {
            'n_estimators': {'low': 50, 'high': 150},
            'max_depth': {'low': 5, 'high': 20},
            'min_samples_split': {'low': 2, 'high': 10},
            'min_samples_leaf': {'low': 1, 'high': 5},
            'max_features': {'choices': ['sqrt', 'log2', 0.8]}
        })
        # 'class_weight_multiplier' 파라미터가 없으면 n_neg를 high 값으로 추가
        if 'class_weight_multiplier' not in param_ranges:
            param_ranges['class_weight_multiplier'] = {'low': 1, 'high': n_neg}
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
    
    model_parameter_info['n_trials'] = n_trials
    model_parameter_info['cv'] = cv
    model_parameter_info['param_ranges'] = param_ranges

    def objective(trial):
        # Optuna를 위한 파라미터 탐색 범위 설정
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
        score = cross_val_score(rf_model, X, y, cv=kf, scoring=f2_rare_scorer, n_jobs=-1).mean()
        
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

    print(f"\n    Best parameters found (Optuna): {study.best_params}")
    print(f"    Best F2 (class=1) score (CV): {study.best_value:.4f}\n")
    
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
def train_model_xgboost_optuna_old(train_dataset: pd.DataFrame, feature_selection_info: dict):
    print("    Training the XGBoost model with Optuna hyperparameter tuning...\n")
    
    model_parameter_info = {}
    
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
    final_features = feature_selection_info['final_features']
    X = X[final_features]
    
    n_pos = sum(y)
    n_neg = len(y) - n_pos

    def objective(trial):
        # Optuna를 위한 파라미터 탐색 범위 설정
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.2),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        }

        # XGBoost는 scale_pos_weight를 사용
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
        
        xgb_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            **params
        )

        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        score = cross_val_score(xgb_model, X, y, cv=kf, scoring=f2_rare_scorer, n_jobs=-1).mean()
        
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    
    best_params = study.best_params

    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

    best_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        **best_params
    ).fit(X, y)

    print(f"\n    Best parameters found (Optuna): {study.best_params}")
    print(f"    Best F2 (class=1) score (CV): {study.best_value:.4f}\n")
    
    model_parameter_info['best_params'] = study.best_params
    
    importance_dict = {
        "Features": X.columns,
        "Importance": best_model.feature_importances_,
        "Importance_abs": np.abs(best_model.feature_importances_),
    }
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    )
    
    return best_model, importance, model_parameter_info

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
        
    model_parameter_info['n_trials'] = n_trials
    model_parameter_info['cv'] = cv
    model_parameter_info['param_ranges'] = param_ranges
    model_parameter_info['ratio_multiplier_range'] = ratio_multiplier_range

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
        shap_values = explainer(X)
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