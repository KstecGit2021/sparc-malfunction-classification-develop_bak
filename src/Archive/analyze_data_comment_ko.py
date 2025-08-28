# %%
!pip install xgboost # xgboost 라이브러리 설치

# %%
# 필요한 라이브러리들을 임포트합니다.
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 모델
from sklearn.ensemble import RandomForestClassifier # 랜덤 포레스트 분류기
from sklearn.model_selection import train_test_split, GridSearchCV # 훈련/테스트 데이터 분할 및 그리드 서치를 위한 모듈
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif # 특성 선택을 위한 모듈 (최고 K개 선택, 분산 임계값, ANOVA F-값)
from sklearn.tree import DecisionTreeClassifier # 결정 트리 분류기
from sklearn.metrics import roc_auc_score, fbeta_score, make_scorer # ROC AUC 점수, F-베타 점수, 커스텀 스코어러 생성
from xgboost import XGBClassifier # XGBoost 분류기
import shap # SHAP(SHapley Additive exPlanations) 라이브러리 (모델 예측 설명)
import matplotlib.pyplot as plt # 데이터 시각화를 위한 라이브러리

import pandas as pd # 데이터 조작 및 분석을 위한 라이브러리
import numpy as np # 수치 계산을 위한 라이브러리
import datetime as dt # 날짜 및 시간 처리를 위한 라이브러리
import json # JSON 데이터 처리를 위한 라이브러리

# %%

##############################################################################################################################
# 1) 기준 모델 (논리 변경 없음) 하지만, proba[:,1]이 "pass"를 의미한다는 점을 유의하십시오.
##############################################################################################################################


def check_band_gap(wafer_no, bias_value):
    """
    원래 로직을 재현합니다:
    wafer_no == 9 이면:
        - bias_value < 1.21492 이면 -> 'bandGapFail' (밴드갭 실패)
        - 그 외                   -> 'ok for band gap' (밴드갭 통과)
    wafer_no == 10 이면:
        - bias_value < 1.2045 이면 -> 'bandGapFail' (밴드갭 실패)
        - 그 외                  -> 'ok for band gap' (밴드갭 통과)
    그 외의 경우:
        - 'impossible Wafer' (불가능한 웨이퍼)
    """
    if wafer_no == 9: # 웨이퍼 번호가 9인 경우
        if bias_value < 1.21492: # bias_value가 특정 값보다 작으면
            return False # 실패 (False) 반환
        else:
            return True # 통과 (True) 반환
    elif wafer_no == 10: # 웨이퍼 번호가 10인 경우
        if bias_value < 1.2045: # bias_value가 특정 값보다 작으면
            return False # 실패 (False) 반환
        else:
            return True # 통과 (True) 반환
    else: # 그 외의 웨이퍼 번호인 경우
        return False # 실패 (False) 반환 (또는 'impossible Wafer' 상황으로 간주)

class BaselineModel:
    def __init__(self):
        self.radius = 70 # 반지름 기준값 초기화
        self.sensor_offset_hot_cold = 0.02 # 센서 오프셋 기준값 초기화
        pass # 초기화 메서드

    def check_band_gap(self, wafer_no, bias_value):
        # 위의 전역 check_band_gap 함수와 동일한 로직을 클래스 메서드로 재정의
        if wafer_no == 9:
            if bias_value < 1.21492:
                return False
            else:
                return True
        elif wafer_no == 10:
            if bias_value < 1.2045:
                return False
            else:
                return True
        else:
            return False

    def _check_data(self, X):
        # 입력 DataFrame X에 필요한 컬럼이 있는지 확인하는 비공개 메서드
        if "Radius" not in X.columns:
            raise ValueError("Radius column not found in X") # "Radius" 컬럼이 없으면 에러 발생
        if "SensorOffsetHot-Cold" not in X.columns:
            raise ValueError("SensorOffsetHot-Cold column not found in X") # "SensorOffsetHot-Cold" 컬럼이 없으면 에러 발생
        if "Bias_Ref_test for Criteria FT2" not in X.columns:
            raise ValueError(
                "Bias_Ref_test for Criteria FT2 of FT2 column not found in X"
            ) # "Bias_Ref_test for Criteria FT2" 컬럼이 없으면 에러 발생
        if "WAFER_NO" not in X.columns:
            raise ValueError("WAFER_NO column not found in X") # "WAFER_NO" 컬럼이 없으면 에러 발생

    def predict_proba(self, X):
        # 기준 모델의 예측 확률을 반환하는 메서드
        radius_criteria = X["Radius"] <= self.radius # 반지름 기준을 충족하는지 확인
        sensor_criteria = X["SensorOffsetHot-Cold"].abs() <= self.sensor_offset_hot_cold # 센서 오프셋 기준을 충족하는지 확인 (절대값)
        bandgap_criteria = X.apply(
            lambda row: self.check_band_gap(
                row["WAFER_NO"], row["Bias_Ref_test for Criteria FT2"]
            ),
            axis=1,
        ) # 밴드갭 기준을 충족하는지 각 행에 대해 확인
        y_pred_baseline = radius_criteria & sensor_criteria & bandgap_criteria # 모든 기준을 충족하는 경우 (True)

        # proba[:,1] => "pass" (통과)
        # proba[:,0] => "fail" (실패)
        proba = np.zeros((len(X), 2)) # (데이터 수, 2) 크기의 0으로 채워진 배열 생성
        proba[:, 0] = y_pred_baseline.astype(int)  # 통과 확률 (불리언을 정수로 변환)
        proba[:, 1] = 1 - proba[:, 0]  # 실패 확률 (1 - 통과 확률)
        return proba # 예측 확률 반환


##############################################################################################################################
# 2) 전처리: 이제 레이블을 반전합니다 => 1 = 실패, 0 = 통과
##############################################################################################################################


def preprocess_dataset(initial_dataset: pd.DataFrame):
    """이 함수는 데이터셋을 전처리하여 최종 'Pass/Fail' 컬럼이 1 = 실패 (드문 경우), 0 = 통과가 되도록 합니다."""

    print("\n     데이터셋 전처리 중...")

    processed_dataset = initial_dataset.copy() # 원본 데이터셋 복사

    # 유지할 컬럼의 예시:
    ft1_2_3 = [
        col
        for col in processed_dataset.columns
        if ("FT1" in col or "FT2" in col or "FT3" in col) and "Parameter" not in col
    ] # 컬럼 이름에 'FT1', 'FT2', 'FT3' 중 하나가 포함되어 있고 'Parameter'가 없는 컬럼들을 선택

    processed_dataset = processed_dataset[
        ft1_2_3 # 위에서 선택한 FT1, FT2, FT3 관련 컬럼들
        + [
            "Radius", # 반지름
            "X", # X 좌표
            "Y", # Y 좌표
            "band gap dpat", # 밴드갭 관련 데이터
            "SensorOffsetHot-Cold", # 센서 오프셋 (Hot-Cold)
            "SensorOffsetHot", # 센서 오프셋 (Hot)
            "SensorOffsetCold", # 센서 오프셋 (Cold)
            "WAFER_NO", # 웨이퍼 번호
            # "all Criteria together", # 모든 기준 (주석 처리됨)
            "Pass/Fail", # 통과/실패 레이블
        ]
    ]
    processed_dataset["SensorOffsetHot-Cold-Abs"] = processed_dataset[
        "SensorOffsetHot-Cold"
    ].abs() # "SensorOffsetHot-Cold" 컬럼의 절대값으로 새로운 컬럼 생성

    processed_dataset = pd.get_dummies(processed_dataset, drop_first=True) # 범주형 컬럼을 원-핫 인코딩 (첫 번째 카테고리 드롭)

    # 모든 컬럼을 숫자형으로 변환
    processed_dataset = processed_dataset.apply(pd.to_numeric)

    # NOTE: 원래 "Pass/Fail_pass=1 => 통과, Pass/Fail_pass=0 => 실패"
    # 우리는 "1 => 실패"가 되도록 반전합니다. 즉, "실패 = 1 - old_pass_value" 입니다.
    # old_pass_value = processed_dataset["Pass/Fail_pass"] (통과이면 1, 실패이면 0)
    # new fail => 1 - old_pass_value
    processed_dataset["Pass/Fail"] = 1 - processed_dataset["Pass/Fail_pass"] # 'Pass/Fail_pass' 컬럼을 반전하여 'Pass/Fail' 컬럼 생성 (1=실패, 0=통과)
    processed_dataset.drop(["Pass/Fail_pass"], axis=1, inplace=True) # 원본 'Pass/Fail_pass' 컬럼 삭제

    processed_dataset.fillna(processed_dataset.mean(), inplace=True) # 누락된 값을 해당 컬럼의 평균으로 채움

    processed_dataset["Bias_Ref_test for Criteria FT2"] = processed_dataset[
        "Bias_Ref_test:VR1V2D@Bias_Ref_test[1] of FT2"
    ] # 긴 이름의 컬럼을 짧은 이름으로 복사

    # 컬럼 이름 정리
    processed_dataset.columns = (
        processed_dataset.columns.str.replace("[", "_", regex=False) # '['를 '_'로 대체
        .str.replace("]", "_", regex=False) # ']'를 '_'로 대체
        .str.replace("<", "_", regex=False) # '<'를 '_'로 대체
        .str.replace(">", "_", regex=False) # '>'를 '_'로 대체
    )

    # 최종 컬럼이 Pass/Fail이 되도록 순서 재정렬
    reorder_cols = [c for c in processed_dataset.columns if c not in ["Pass/Fail"]] # 'Pass/Fail'을 제외한 모든 컬럼 선택
    processed_dataset = processed_dataset[reorder_cols + ["Pass/Fail"]] # 'Pass/Fail'을 마지막에 추가하여 컬럼 순서 재정렬

    print("     전처리 완료!\n")
    return processed_dataset # 전처리된 데이터셋 반환


def variance_correlation_filter(
    X: pd.DataFrame, var_threshold=0.0, corr_threshold=0.98
):
    """
    1) 분산이 var_threshold 이하인 특성을 제거합니다.
    2) 절대 상관 관계가 corr_threshold보다 큰 특성 쌍 중 하나를 제거합니다.
    반환 값:
      (X_filtered, final_cols)
    """
    vt = VarianceThreshold(threshold=var_threshold) # 분산 임계값 필터 초기화
    X_vt = vt.fit_transform(X) # X에 필터를 적용하여 분산이 낮은 특성 제거
    vt_mask = vt.get_support() # 유지된 특성의 마스크
    vt_cols = X.columns[vt_mask] # 유지된 특성 컬럼 이름
    print("분산 임계값 적용 후 유지된 특성 수", sum(vt_mask))

    X_vt_df = pd.DataFrame(X_vt, columns=vt_cols) # 필터링된 특성으로 새 DataFrame 생성

    corr_matrix = X_vt_df.corr().abs() # 절대 상관 관계 행렬 계산
    upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)) # 상관 관계 행렬의 상삼각 부분 (대각선 제외)
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)] # 상관 관계 임계값을 초과하는 특성들을 제거할 목록에 추가
    X_filtered = X_vt_df.drop(to_drop, axis=1) # 해당 특성들 제거
    final_cols = list(X_filtered.columns) # 최종 특성 컬럼 이름

    return X_filtered, final_cols # 필터링된 데이터와 최종 컬럼 목록 반환


##############################################################################################################################
# 3) 훈련/테스트 데이터 분할 생성
##############################################################################################################################


def create_train_test_data(preprocessed_dataset: pd.DataFrame):
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
    X["WAFER_NO"] = preprocessed_dataset["WAFER_NO"] # 필터링 후 "WAFER_NO" 컬럼 다시 추가
    X["Bias_Ref_test for Criteria FT2"] = preprocessed_dataset[
        "Bias_Ref_test for Criteria FT2"
    ] # 필터링 후 "Bias_Ref_test for Criteria FT2" 컬럼 다시 추가

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    ) # 훈련 및 테스트 데이터셋으로 분할 (80% 훈련, 20% 테스트)
    train_data = pd.concat([X_train, y_train], axis=1) # 훈련 특성과 레이블을 결합
    test_data = pd.concat([X_test, y_test], axis=1) # 테스트 특성과 레이블을 결합

    return train_data, test_data # 훈련 및 테스트 데이터 반환


##############################################################################################################################
# 4) pos_label=1인 사용자 정의 F2 스코어러 (1 = 실패/드문 경우이므로)
##############################################################################################################################

f2_rare_scorer = make_scorer(fbeta_score, beta=2, pos_label=1) # F2 스코어러 생성 (beta=2는 재현율에 더 큰 가중치를 줌, pos_label=1은 실패 클래스가 긍정 클래스임을 나타냄)


##############################################################################################################################
# 5) 로지스틱 회귀 (단순) - 이제 클래스 1에 가중치 부여
##############################################################################################################################


def train_model_logistic_regression(train_dataset: pd.DataFrame):
    print("     로지스틱 회귀 모델 훈련 중 (CV 없음)...\n")
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1] # 훈련 데이터셋에서 특성과 레이블 분리

    with open("final_features.json", "r") as f:
        final_features = json.load(f) # "final_features.json" 파일에서 최종 특성 목록 로드
    X = X[final_features] # 최종 특성만 사용하여 X 필터링

    # 이제 1이 드문 클래스입니다 => 가중치를 부여합니다.
    # sum(y)는 1의 개수입니다. sum(1-y)는 0의 개수입니다.
    class_weight = {
        0: 1, # 클래스 0 (통과)에 대한 가중치
        1: sum(1 - y),  # 클래스 1 (실패)에 대한 가중치 (0의 개수)
    }

    model_fitted = LogisticRegression(class_weight=class_weight, max_iter=1000).fit(
        X, y
    ) # 로지스틱 회귀 모델 훈련 (클래스 가중치 및 최대 반복 횟수 설정)
    print("\n    LogisticRegression 모델이 훈련되었습니다!")

    importance_dict = {
        "Features": X.columns, # 특성 이름
        "Importance": model_fitted.coef_[0], # 모델 계수 (중요도)
        "Importance_abs": np.abs(model_fitted.coef_[0]), # 모델 계수의 절대값
    }
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    ) # 중요도 데이터프레임 생성 및 정렬
    return model_fitted, importance # 훈련된 모델과 특성 중요도 반환


##############################################################################################################################
# 5a) CV + F2 스코어링을 사용한 로지스틱 회귀
##############################################################################################################################


def train_model_logistic_regression_cv(train_dataset: pd.DataFrame):
    print(
        "     교차 검증 및 하이퍼파라미터 튜닝을 사용하여 로지스틱 회귀 모델 훈련 중...\n"
    )
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1] # 훈련 데이터셋에서 특성과 레이블 분리

    with open("final_features.json", "r") as f:
        final_features = json.load(f) # "final_features.json" 파일에서 최종 특성 목록 로드
    X = X[final_features] # 최종 특성만 사용하여 X 필터링

    # 파라미터 그리드
    param_grid = {"C": [0.01, 0.1, 1], "penalty": ["l2"], "solver": ["lbfgs"]} # 그리드 서치를 위한 하이퍼파라미터 그리드 정의

    # 실패 클래스 = 1에 가중치 부여
    class_weight = {
        0: 1, # 클래스 0 (통과)에 대한 가중치
        1: sum(1 - y), # 클래스 1 (실패)에 대한 가중치 (0의 개수)
    }

    lr = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42) # 로지스틱 회귀 모델 초기화

    grid_search = GridSearchCV(
        estimator=lr, # 추정기 (로지스틱 회귀 모델)
        param_grid=param_grid, # 파라미터 그리드
        scoring=f2_rare_scorer,  # 클래스 1에 대한 F2 스코어링 사용
        cv=3, # 3-겹 교차 검증
        verbose=1, # 자세한 출력 활성화
        n_jobs=-1, # 모든 CPU 코어 사용
    )

    grid_search.fit(X, y) # 그리드 서치 수행
    best_model = grid_search.best_estimator_ # 최적의 모델 선택

    print(f"\n    최적 파라미터: {grid_search.best_params_}") # 최적 파라미터 출력
    print(f"    최적 F2 (클래스=1) 점수 (CV): {grid_search.best_score_:.4f}\n") # 최적 F2 점수 출력

    importance_dict = {
        "Features": X.columns,
        "Importance": best_model.coef_[0],
        "Importance_abs": np.abs(best_model.coef_[0]),
    }
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    )
    return best_model, importance # 최적 모델과 특성 중요도 반환


##############################################################################################################################
# 6) 기준 모델 (변경 없음)
##############################################################################################################################


def train_model_baseline(train_dataset: pd.DataFrame):
    # 기준 모델 훈련 함수 (실제 훈련 없이 미리 정의된 중요도 반환)
    model_fitted = BaselineModel() # BaselineModel 인스턴스 생성
    importance_dict = {
        "Features": [
            "SensorOffsetHot-Cold",
            "Bias_Ref_test:VR1V2D@Bias_Ref_test[1] of FT2",
            "Radius",
        ], # 특성 목록
        "Importance": [56.6, 4.65, 96.9], # 중요도 값
        "Importance_abs": [56.6, 4.65, 96.9], # 중요도 절대값
    }
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    ) # 중요도 데이터프레임 생성 및 정렬
    return model_fitted, importance # 모델과 특성 중요도 반환




##############################################################################################################################
# 7a) 클래스=1에 F2를 사용하는 CV & 그리드 서치를 사용한 랜덤 포레스트
##############################################################################################################################


def train_model_rf_cv(train_dataset: pd.DataFrame):
    print(
        "     교차 검증 및 하이퍼파라미터 튜닝을 사용하여 랜덤 포레스트 모델 훈련 중...\n"
    )

    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1] # 훈련 데이터셋에서 특성과 레이블 분리

    with open("final_features.json", "r") as f:
        final_features = json.load(f) # "final_features.json" 파일에서 최종 특성 목록 로드
    X = X[final_features] # 최종 특성만 사용하여 X 필터링

    param_grid = {
        "n_estimators": [20, 50, 100, 200], # 트리의 개수
        "max_depth": [2, 5, None], # 트리의 최대 깊이
        "min_samples_split": [2, 5], # 노드를 분할하기 위한 최소 샘플 수
    }

    class_weight = {
        0: 1,
        1: sum(1 - y),
    } # 클래스 0 (통과)에 대한 가중치 1, 클래스 1 (실패)에 대한 가중치는 0의 개수로 설정
    rf_model = RandomForestClassifier(class_weight=class_weight, random_state=42) # 랜덤 포레스트 분류기 초기화

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring=f2_rare_scorer,  # 클래스 1에 대한 F2 스코어링 사용
        cv=3,
        verbose=1,
        n_jobs=-1,
    )

    grid_search.fit(X, y) # 그리드 서치 수행
    best_model = grid_search.best_estimator_ # 최적의 모델 선택

    print(f"\n    최적 파라미터: {grid_search.best_params_}")
    print(f"    최적 F2 (클래스=1) 점수 (CV): {grid_search.best_score_:.4f}\n")

    importance_dict = {
        "Features": X.columns,
        "Importance": best_model.feature_importances_, # 특성 중요도 (랜덤 포레스트의 경우)
        "Importance_abs": np.abs(best_model.feature_importances_), # 특성 중요도 절대값
    }
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    )
    return best_model, importance # 최적 모델과 특성 중요도 반환


##############################################################################################################################
# 8) 결정 트리
##############################################################################################################################


def train_model_decision_tree(train_dataset: pd.DataFrame):
    print("     결정 트리 모델 훈련 중 (CV 없음)...\n")
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1] # 훈련 데이터셋에서 특성과 레이블 분리

    with open("final_features.json", "r") as f:
        final_features = json.load(f) # "final_features.json" 파일에서 최종 특성 목록 로드
    X = X[final_features] # 최종 특성만 사용하여 X 필터링

    class_weight = {
        0: 1,
        1: sum(1 - y),
    } # 클래스 0 (통과)에 대한 가중치 1, 클래스 1 (실패)에 대한 가중치는 0의 개수로 설정
    model_fitted = DecisionTreeClassifier(
        class_weight=class_weight, random_state=42
    ).fit(X, y) # 결정 트리 분류기 훈련 (클래스 가중치 및 랜덤 시드 설정)
    print("\n    DecisionTreeClassifier 모델이 훈련되었습니다!")

    importance_dict = {
        "Features": X.columns,
        "Importance": model_fitted.feature_importances_, # 특성 중요도 (결정 트리의 경우)
        "Importance_abs": np.abs(model_fitted.feature_importances_), # 특성 중요도 절대값
    }
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    )
    return model_fitted, importance # 훈련된 모델과 특성 중요도 반환


##############################################################################################################################
# 9) CV & 클래스=1에 F2를 사용하는 XGBoost
##############################################################################################################################


def train_model_xgboost_cv(train_dataset: pd.DataFrame):
    print(
        "     교차 검증 및 하이퍼파라미터 튜닝을 사용하여 XGBoost 모델 훈련 중...\n"
    )

    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1] # 훈련 데이터셋에서 특성과 레이블 분리

    with open("final_features.json", "r") as f:
        final_features = json.load(f) # "final_features.json" 파일에서 최종 특성 목록 로드
    X = X[final_features] # 최종 특성만 사용하여 X 필터링

    param_grid = {
        "n_estimators": [30, 50, 100, 200], # 부스팅 라운드 또는 트리의 개수
        "max_depth": [2, 5], # 트리의 최대 깊이
        "learning_rate": [0.01, 0.1, 0.2], # 학습률
    }

    # 1이 5%이면 => sum(y)가 작습니다.
    # scale_pos_weight = (음성 클래스 개수 / 양성 클래스 개수) => 우리는 0의 개수 / 1의 개수를 사용합니다.
    n_pos = sum(y) # 양성 클래스 (실패)의 개수
    n_neg = len(y) - n_pos # 음성 클래스 (통과)의 개수
    scale_pos_weight = n_neg / n_pos * 2 if n_pos > 0 else 1 # 양성 클래스에 대한 가중치 계산 (불균형 데이터셋 처리)

    xgb_model = XGBClassifier(
        use_label_encoder=False, # label encoder 사용 경고 비활성화
        eval_metric="logloss", # 평가 지표를 logloss로 설정
        random_state=42, # 랜덤 시드 설정
        scale_pos_weight=scale_pos_weight, # 양성 클래스 가중치 적용
    )

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring=f2_rare_scorer,  # 클래스 1에 대한 F2 스코어링 사용
        cv=3,
        verbose=1,
        n_jobs=-1,
    )

    grid_search.fit(X, y) # 그리드 서치 수행
    best_model = grid_search.best_estimator_ # 최적의 모델 선택

    print(f"\n    최적 파라미터: {grid_search.best_params_}")
    print(f"    최적 F2 (클래스=1) 점수: {grid_search.best_score_:.4f}\n")

    importance_dict = {
        "Features": X.columns,
        "Importance": best_model.feature_importances_, # 특성 중요도 (XGBoost의 경우)
        "Importance_abs": np.abs(best_model.feature_importances_), # 특성 중요도 절대값
    }
    importance = pd.DataFrame(importance_dict).sort_values(
        by="Importance", ascending=True
    )
    return best_model, importance # 최적 모델과 특성 중요도 반환


##############################################################################################################################
# 10) 예측 및 평가 도우미 함수
##############################################################################################################################


def _confusion_label(row):
    # 이제 "1"은 실패 => 혼동 행렬에서 "양성"입니다.
    # row["Historical"] = 실제 레이블, row["Forecast"] = 예측된 레이블
    if row["Historical"] == 1 and row["Forecast"] == 1:
        return "True Fail (TP)" # 실제 실패, 예측 실패 (참 양성)
    elif row["Historical"] == 0 and row["Forecast"] == 0:
        return "True Pass (TN)" # 실제 통과, 예측 통과 (참 음성)
    elif row["Historical"] == 0 and row["Forecast"] == 1:
        return "False Fail (FP)" # 실제 통과, 예측 실패 (거짓 양성)
    else:  # row["Historical"] == 1 and row["Forecast"] == 0
        return "Missed Fail (FN)" # 실제 실패, 예측 통과 (거짓 음성)


def find_best_threshold(best_model, train_dataset):
    """
    F2 점수를 최대화하는 분류를 위한 최적의 임계값을 찾습니다.

    매개변수:
    - best_model: `predict_proba` 메서드가 있는 훈련된 분류 모델.
    - train_dataset: 특성과 마지막 컬럼이 타겟인 DataFrame.

    반환 값:
    - best_threshold: F2 점수를 최대화하는 최적의 임계값.
    """
    X, y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1] # 훈련 데이터셋에서 특성과 레이블 분리

    if not isinstance(best_model, BaselineModel): # 모델이 BaselineModel이 아닌 경우
        with open("final_features.json", "r") as f:
            final_features = json.load(f) # "final_features.json" 파일에서 최종 특성 목록 로드
        X = X[final_features] # 최종 특성만 사용하여 X 필터링

    prob_class1 = best_model.predict_proba(X)[:, 1] # 클래스 1 (실패)에 대한 예측 확률

    # 0과 1 사이의 여러 임계값 시도
    thresholds = np.linspace(0, 1, 100) # 0부터 1까지 100개의 균등 간격 임계값 생성
    f2_scores = [] # F2 점수를 저장할 리스트

    for threshold in thresholds: # 각 임계값에 대해 반복
        y_pred = (prob_class1 >= threshold).astype(int) # 임계값을 기준으로 예측 (정수형으로 변환)
        score = fbeta_score(y, y_pred, beta=2, pos_label=1) # F2 점수 계산
        f2_scores.append(score) # F2 점수 추가

    # 최적의 임계값 찾기
    best_idx = np.argmax(f2_scores) # F2 점수가 가장 높은 인덱스 찾기
    best_threshold = thresholds[best_idx] # 최적의 임계값
    best_f2_score = f2_scores[best_idx] # 최적의 F2 점수

    print(
        f"F2 점수를 위한 최적의 임계값: {best_threshold:.4f} (F2 점수: {best_f2_score:.4f})"
    )

    # ------------------------------------------------------------------
    # 사용자가 선택한 임계값 적용
    # ------------------------------------------------------------------

    train_dataset["Probability"] = prob_class1 # 예측 확률을 'Probability' 컬럼에 저장
    train_dataset["Historical"] = y # 실제 레이블을 'Historical' 컬럼에 저장
    return train_dataset, best_threshold # 업데이트된 훈련 데이터셋과 최적 임계값 반환


def create_metrics_on_train(train_dataset, threshold):
    """
    훈련 후, 주어진 임계값 (클래스=1, 실패)으로 훈련 데이터셋에 대해 예측합니다.
    또한 데이터셋 전체에서 예측된 실패가 0 또는 1인 경우의 임계값을 출력합니다.
    """
    # ------------------------------------------------------------------
    # 사용자가 선택한 임계값 적용
    # ------------------------------------------------------------------
    forecast = (train_dataset["Probability"] >= threshold).astype(int) # 임계값을 기준으로 예측 (정수형으로 변환)

    train_dataset["Forecast"] = forecast # 예측 결과를 'Forecast' 컬럼에 저장
    train_dataset["True/False/Positive/Negative"] = train_dataset.apply(
        _confusion_label, axis=1
    ) # 혼동 행렬 레이블 생성
    return train_dataset # 업데이트된 훈련 데이터셋 반환


def forecast(test_dataset: pd.DataFrame, trained_model):
    print("     테스트 데이터셋 예측 중...")
    X = test_dataset.iloc[:, :-1] # 테스트 데이터셋에서 특성 분리

    if not isinstance(trained_model, BaselineModel): # 훈련된 모델이 BaselineModel이 아닌 경우
        with open("final_features.json", "r") as f:
            final_features = json.load(f) # "final_features.json" 파일에서 최종 특성 목록 로드
        X = X[final_features] # 최종 특성만 사용하여 X 필터링

    # 클래스=1에 대한 확률
    predictions = trained_model.predict_proba(X)[:, 1] # 클래스 1 (실패)에 대한 예측 확률
    print("     예측 완료!")

    # 트리 기반 모델의 경우 SHAP의 TreeExplainer를 사용하고, 그렇지 않으면 KernelExplainer를 사용합니다.
    if hasattr(trained_model, "feature_importances_"): # 모델이 feature_importances_ 속성을 가지고 있다면 (트리 기반 모델)
        explainer = shap.TreeExplainer(trained_model) # TreeExplainer 사용
    elif not isinstance(trained_model, BaselineModel): # BaselineModel이 아닌 경우
        explainer = shap.Explainer(trained_model, X) # KernelExplainer 사용 (X는 배경 데이터)
    if not isinstance(trained_model, BaselineModel):

        shap_values = explainer(X) # SHAP 값 계산

        # SHAP 요약 플롯 그리기 (주석 처리됨)
        # plt.figure(figsize=(10, 5))
        # shap.summary_plot(shap_values, X, max_display=10, show=False)
        # plt.show()
    else:
        shap_values = None # BaselineModel의 경우 SHAP 값 없음

    return predictions, [shap_values, X] # 예측 확률과 SHAP 값, 특성 데이터 반환


def roc_from_scratch(probabilities, test_dataset, partitions=100):
    print("     ROC 곡선 계산 중...")
    y_test = test_dataset.iloc[:, -1] # 테스트 데이터셋에서 실제 레이블 가져오기

    roc = [] # ROC 곡선 데이터를 저장할 리스트
    for i in range(partitions + 1): # 0부터 partitions까지 반복하여 임계값 생성
        thr = i / partitions # 임계값 계산
        threshold_vector = (probabilities >= thr).astype(int) # 임계값을 기준으로 예측 (정수형)
        tpr, fpr = true_false_positive(threshold_vector, y_test) # 참 양성률(TPR)과 거짓 양성률(FPR) 계산
        roc.append([fpr, tpr]) # ROC 데이터에 (FPR, TPR) 추가

    roc_data = pd.DataFrame(roc, columns=["False positive rate", "True positive rate"]) # ROC 데이터를 DataFrame으로 변환
    print("     계산 완료")
    print("     점수 계산 중...")

    auc_score = roc_auc_score(y_test, probabilities) # ROC AUC 점수 계산
    print("     점수 계산 완료\n")
    return roc_data, auc_score # ROC 데이터와 AUC 점수 반환


def true_false_positive(threshold_vector: np.array, y_test: np.array):
    # "1"은 실패 => "양성"
    true_positive = (threshold_vector == 1) & (y_test == 1) # 참 양성: 예측 1, 실제 1
    false_positive = (threshold_vector == 1) & (y_test == 0) # 거짓 양성: 예측 1, 실제 0
    true_negative = (threshold_vector == 0) & (y_test == 0) # 참 음성: 예측 0, 실제 0
    false_negative = (threshold_vector == 0) & (y_test == 1) # 거짓 음성: 예측 0, 실제 1

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum() + 1e-9) # 참 양성률 계산 (분모에 작은 값 추가하여 0으로 나누는 오류 방지)
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum() + 1e-9) # 거짓 양성률 계산
    return tpr, fpr # TPR과 FPR 반환


def create_metrics(
    predictions: np.array, test_dataset: pd.DataFrame, auc_score, threshold
):
    print("     메트릭 생성 중...")
    threshold_vector = (predictions >= threshold).astype(int) # 임계값을 기준으로 예측 (정수형)

    y_test = test_dataset.iloc[:, -1] # 테스트 데이터셋에서 실제 레이블 가져오기

    tp = ((threshold_vector == 1) & (y_test == 1)).sum() # 참 양성 수
    tn = ((threshold_vector == 0) & (y_test == 0)).sum() # 참 음성 수
    fp = ((threshold_vector == 1) & (y_test == 0)).sum() # 거짓 양성 수
    fn = ((threshold_vector == 0) & (y_test == 1)).sum() # 거짓 음성 수

    # 클래스=1에 대한 표준 F1 점수
    denom = 2 * tp + fp + fn # F1 점수 분모
    if denom == 0:
        f1_score = 0.0 # 분모가 0이면 F1 점수 0.0
    else:
        f1_score = 2 * tp / denom # F1 점수 계산
    f1_score = np.around(f1_score, 2) # F1 점수를 소수점 둘째 자리까지 반올림

    accuracy = np.around((tp + tn) / (tp + tn + fp + fn + 1e-9), 2) # 정확도 계산 및 반올림
    auc_score = np.around(auc_score, 2) # AUC 점수 반올림

    dict_ftpn = {"tp": tp, "tn": tn, "fp": fp, "fn": fn} # TP, TN, FP, FN 값을 딕셔너리에 저장
    number_of_good_predictions = tp + tn # 올바른 예측 수 (TP + TN)
    number_of_false_predictions = fp + fn # 잘못된 예측 수 (FP + FN)

    # 정밀도와 재현율
    if (tp + fp) == 0:
        precision = 0.0 # 분모가 0이면 정밀도 0.0
    else:
        precision = tp / (tp + fp) # 정밀도 계산
    precision = np.around(precision, 2) # 정밀도 반올림

    if (tp + fn) == 0:
        recall = 0.0 # 분모가 0이면 재현율 0.0
    else:
        recall = tp / (tp + fn) # 재현율 계산
    recall = np.around(recall, 2) # 재현율 반올림

    metrics = {
        "f1_score": f1_score, # F1 점수
        "recall": recall, # 재현율
        "precision": precision, # 정밀도
        "accuracy": accuracy, # 정확도
        "auc_score": auc_score, # AUC 점수
        "dict_ftpn": dict_ftpn, # TP, TN, FP, FN 딕셔너리
        "number_of_predictions": len(predictions), # 총 예측 수
        "number_of_good_predictions": number_of_good_predictions, # 올바른 예측 수
        "number_of_false_predictions": number_of_false_predictions, # 잘못된 예측 수
    }

    return metrics # 메트릭 딕셔너리 반환


def create_results(forecast_values, test_dataset, threshold):
    forecast_series_proba = pd.Series(
        np.around(forecast_values, decimals=2), # 예측 확률을 소수점 둘째 자리까지 반올림
        index=test_dataset.index, # 테스트 데이터셋 인덱스 사용
        name="Probability", # 컬럼 이름 'Probability'
    )
    forecast_series = pd.Series(
        (forecast_values > threshold).astype(int), # 임계값을 기준으로 예측 (정수형)
        index=test_dataset.index,
        name="Forecast", # 컬럼 이름 'Forecast'
    )
    true_series = pd.Series(
        test_dataset.iloc[:, -1], name="Historical", index=test_dataset.index
    ) # 실제 레이블을 'Historical' 컬럼으로 생성
    index_series = pd.Series(
        range(len(true_series)), index=test_dataset.index, name="Id"
    ) # 인덱스에 기반한 'Id' 컬럼 생성

    results = pd.concat(
        [index_series, forecast_series_proba, forecast_series, true_series], axis=1
    ) # 모든 시리즈를 결합하여 결과 데이터프레임 생성
    results["True/False/Positive/Negative"] = results.apply(_confusion_label, axis=1) # 혼동 행렬 레이블 생성
    return results # 결과 데이터프레임 반환

# ----------------------------------------------------------------
# 6) F2 스코어러를 사용한 XGB + RFECV
# ----------------------------------------------------------------
def run_rfecv_xgb(
    X_train_filtered,
    y_train,
    xgb_clf,
    scorer,
    cv=5,
    step=1,
    min_features_to_select=1
):
    """
    제공된 XGBClassifier (xgb_clf) 및 사용자 정의 F2 스코어러를 사용하여 RFECV를 실행합니다.

    반환 값:
        rfecv: 피팅된 RFECV 객체
        kept_features: 유지된 특성 이름 목록
    """
    from sklearn.feature_selection import RFECV # RFECV 임포트

    rfecv = RFECV(
        estimator=xgb_clf, # 추정기 (XGBoost 분류기)
        step=step, # 각 반복에서 제거할 특성 수
        cv=cv, # 교차 검증 폴드 수
        scoring=scorer, # 스코어링 지표
        min_features_to_select=min_features_to_select # 선택할 최소 특성 수
    )
    rfecv.fit(X_train_filtered, y_train) # RFECV 피팅

    optimal_num_features = rfecv.n_features_ # 최적의 특성 수
    ranking_array = rfecv.ranking_ # 특성 순위 배열
    support_mask = rfecv.support_ # 유지된 특성의 부울 마스크

    print(f"최적의 특성 수: {optimal_num_features}")

    kept_features = X_train_filtered.columns[support_mask] # 유지된 특성 이름
    print("\nRFECV를 통해 선택된 특성:")
    for feat in kept_features:
        print("  ", feat)

    print("\n특성 순위 (1=유지, 높을수록 일찍 제거):")
    for feat, rank in zip(X_train_filtered.columns, ranking_array):
        print(f"  {feat:20s} => {rank}")

    return rfecv, list(kept_features) # RFECV 객체와 유지된 특성 목록 반환



# %%
import warnings # 경고 메시지 처리를 위한 모듈

# 'use_label_encoder' 경고만 무시합니다.
warnings.filterwarnings("ignore")


data = pd.read_csv("data/REl data and Cp data joined dec20_gg.csv") # CSV 파일로부터 데이터 로드

print("전처리 전:", data.shape) # 전처리 전 데이터의 형태 출력

# (B) 전처리
df_processed = preprocess_dataset(data)  # 전처리 함수 호출 (레이블을 1=실패로 반전)

print("전처리 후:", df_processed.shape) # 전처리 후 데이터의 형태 출력


# (C) 훈련/테스트 분할
train_data, test_data = create_train_test_data(df_processed) # 훈련 및 테스트 데이터셋 생성

# 선택 사항: 특성 필터링 (분산 + 상관 관계)
# X를 분리한 다음 필터링합니다.
X_train = train_data.iloc[:, :-1]  # 레이블을 제외한 훈련 특성
y_train = train_data.iloc[:, -1] # 훈련 레이블

# 테스트 세트에도 동일하게 적용합니다.
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# (D) F2 스코어러 정의 (pos_label=1 => 실패 클래스)
f2_scorer = make_scorer(fbeta_score, beta=4, pos_label=1) # F2 스코어러 정의 (beta=4로 재현율에 더 큰 가중치)

# (E) XGBoost 설정
num_pos = y_train.sum() # 훈련 데이터셋의 양성 클래스 (실패) 개수
num_neg = len(y_train) - num_pos # 훈련 데이터셋의 음성 클래스 (통과) 개수
scale_pos_weight = num_neg / num_pos * 5 if num_pos > 0 else 1.0 # 양성 클래스 가중치 계산 (불균형 데이터셋 처리)
xgb_clf = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
) # XGBoost 분류기 초기화

# (F) 최적의 특성 부분집합을 찾기 위한 RFECV
rfecv_obj, final_features = run_rfecv_xgb(
    X_train,
    y_train,
    xgb_clf=xgb_clf,
    scorer=f2_scorer,
    cv=5, # 5-겹 교차 검증
    step=5, # 한 번에 5개 특성 제거
    min_features_to_select=1 # 최소 1개의 특성 선택
)

print("\n최종 선택된 특성:", final_features) # 최종 선택된 특성 출력




# %%
import warnings

# 'use_label_encoder' 경고만 무시합니다.
warnings.filterwarnings("ignore")


data = pd.read_csv("data/REl data and Cp data joined dec20_gg.csv")

print("전처리 전:", data.shape)

# (B) 전처리
df_processed = preprocess_dataset(data)  # 레이블을 1=실패로 반전

print("전처리 후:", df_processed.shape)


# (C) 훈련/테스트 분할
train_data, test_data = create_train_test_data(df_processed)

# 선택 사항: 특성 필터링 (분산 + 상관 관계)
# X를 분리한 다음 필터링합니다.
X_train = train_data.iloc[:, :-1]  # 레이블 제외
y_train = train_data.iloc[:, -1]

# 테스트 세트에도 동일하게 적용합니다.
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# (D) F2 스코어러 정의 (pos_label=1 => 실패 클래스)
f2_scorer = make_scorer(fbeta_score, beta=4, pos_label=1)


# 실패 클래스 = 1에 가중치 부여
class_weight = {
    0: 1,
    1: sum(1 - y_train),
}

lr = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42) # 로지스틱 회귀 모델 초기화

# (F) 최적의 특성 부분집합을 찾기 위한 RFECV
rfecv_obj, final_features_lr = run_rfecv_xgb(
    X_train,
    y_train,
    xgb_clf=lr, # XGBoost 분류기 대신 로지스틱 회귀 모델 사용
    scorer=f2_scorer,
    cv=5,
    step=2, # 한 번에 2개 특성 제거
    min_features_to_select=1
)

print("\n최종 선택된 특성:", final_features) # 최종 선택된 특성 출력




# %%
import json # JSON 모듈 임포트

with open("final_features_lr.json", "w") as f:
    json.dump(final_features_lr, f, indent=4) # final_features_lr 리스트를 JSON 파일로 저장 (들여쓰기 4칸)


final_features_lr # final_features_lr 변수 출력

# %%
lr = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42) # 로지스틱 회귀 모델 다시 초기화 및 설정

X_train_lr = X_train[final_features_lr] # 훈련 데이터에서 final_features_lr에 해당하는 특성만 선택
X_test_lr = X_test[final_features_lr] # 테스트 데이터에서 final_features_lr에 해당하는 특성만 선택

lr.fit(X_train_lr, y_train) # 필터링된 훈련 데이터로 로지스틱 회귀 모델 훈련

# %%
# final_features = ["Radius"] # 주석 처리된 라인
from matplotlib import pyplot as plt # matplotlib.pyplot 임포트
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix # ConfusionMatrixDisplay, confusion_matrix 임포트

y_pred_lr = lr.predict_proba(X_test_lr)[:, 1] > 0.98 # 로지스틱 회귀 모델의 예측 확률 (클래스 1)이 0.98보다 큰지 여부로 예측 (이진)
# y_mixed = (best_model.predict_proba(X_test)[:, 1] > 0.54) | (1-bandgap_criteria_test.astype(int)) # 주석 처리된 복합 예측 로직


cm_lr = confusion_matrix(y_test, y_pred_lr) # 실제 레이블과 예측 레이블을 사용하여 혼동 행렬 계산

print("혼동 행렬 (모든 특성):")
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr) # 혼동 행렬 시각화 객체 생성
disp_lr.plot(cmap=plt.cm.Blues) # 혼동 행렬 플롯 생성 (파란색 색상 맵 사용)
plt.show() # 플롯 표시


# %%
import warnings # 경고 메시지 처리를 위한 모듈

# 'use_label_encoder' 경고만 무시합니다.
warnings.filterwarnings("ignore")


data = pd.read_csv("data/REl data and Cp data joined dec20_gg.csv") # CSV 파일로부터 데이터 로드

print("전처리 전:", data.shape) # 전처리 전 데이터의 형태 출력

# (B) 전처리
df_processed = preprocess_dataset(data)  # 전처리 함수 호출 (레이블을 1=실패로 반전)

print("전처리 후:", df_processed.shape) # 전처리 후 데이터의 형태 출력


# (C) 훈련/테스트 분할
train_data, test_data = create_train_test_data(df_processed) # 훈련 및 테스트 데이터셋 생성

# 선택 사항: 특성 필터링 (분산 + 상관 관계)
# X를 분리한 다음 필터링합니다.
X_train = train_data.iloc[:, :-1]  # 레이블을 제외한 훈련 특성
y_train = train_data.iloc[:, -1] # 훈련 레이블

# 테스트 세트에도 동일하게 적용합니다.
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

X_test_for_baseline = X_test.copy() # 기준 모델을 위한 테스트 데이터 복사

# X_test_for_baseline["WAFER_NO"] = X_test["WAFER_NO"] # 주석 처리된 라인
# X_test_for_baseline["Band gap criteria"] = X_test["Band gap criteria"] # 주석 처리된 라인

X_train = X_train["Radius"] # 훈련 특성으로 "Radius" 컬럼만 선택
X_test = X_test["Radius"] # 테스트 특성으로 "Radius" 컬럼만 선택


# %%

radius_criteria_test = X_test_for_baseline["Radius"] <= 70 # 테스트 데이터에서 "Radius" 기준 충족 여부
sensor_criteria_test = X_test_for_baseline["SensorOffsetHot-Cold"] <= 0.02 # 테스트 데이터에서 "SensorOffsetHot-Cold" 기준 충족 여부
bandgap_criteria_test = X_test_for_baseline.apply(lambda row: check_band_gap(row['WAFER_NO'], row['Band gap criteria']), axis=1) # 테스트 데이터에서 밴드갭 기준 충족 여부


# %%
from sklearn.model_selection import GridSearchCV # GridSearchCV 임포트
# final_features=["all Criteria together"] # 주석 처리된 라인

param_grid = {
    "n_estimators": [30, 50, 100, 200], # 부스팅 라운드 또는 트리의 개수
    "max_depth": [2, 5], # 트리의 최대 깊이
    "learning_rate": [0.01, 0.1, 0.2], # 학습률
}

# 1이 5%이면 => sum(y)가 작습니다.
# scale_pos_weight = (음성 클래스 개수 / 양성 클래스 개수) => 우리는 0의 개수 / 1의 개수를 사용합니다.
n_pos = sum(y_train) # 훈련 데이터셋의 양성 클래스 (실패) 개수
n_neg = len(y_train) - n_pos # 훈련 데이터셋의 음성 클래스 (통과) 개수
scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1 # 양성 클래스 가중치 계산

xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale_pos_weight,
) # XGBoost 분류기 초기화

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring=f2_scorer,  # 클래스 1에 대한 F2 스코어링 사용
    cv=3,
    verbose=1,
    n_jobs=-1,
)

grid_search.fit(X_train.to_frame(), y_train) # 그리드 서치 수행 (X_train이 Series이므로 DataFrame으로 변환)
best_model = grid_search.best_estimator_ # 최적의 모델 선택

# %%
y_mixed # y_mixed 변수 출력 (이 변수는 정의되지 않아 오류가 발생할 수 있습니다.)

# %%
# final_features = ["Radius"] # 주석 처리된 라인


from matplotlib import pyplot as plt # matplotlib.pyplot 임포트
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix # ConfusionMatrixDisplay, confusion_matrix 임포트

y_pred = best_model.predict_proba(X_test.to_frame())[:, 1] > 0.51 # 최적 모델의 예측 확률 (클래스 1)이 0.51보다 큰지 여부로 예측 (X_test가 Series이므로 DataFrame으로 변환)
# y_mixed = (best_model.predict_proba(X_test)[:, 1] > 0.54) | (1-bandgap_criteria_test.astype(int)) # 주석 처리된 복합 예측 로직

cm = confusion_matrix(y_test, y_pred) # 실제 레이블과 예측 레이블을 사용하여 혼동 행렬 계산

print("혼동 행렬 (모든 특성):")
disp_all = ConfusionMatrixDisplay(confusion_matrix=cm) # 혼동 행렬 시각화 객체 생성
disp_all.plot(cmap=plt.cm.Blues) # 혼동 행렬 플롯 생성
plt.show() # 플롯 표시

# %%
importance_dict = {
    "Features": X.columns, # 특성 이름
    "Importance": best_model.feature_importances_, # 모델의 특성 중요도
    "Importance_abs": np.abs(best_model.feature_importances_), # 모델 특성 중요도의 절대값
}
importance = pd.DataFrame(importance_dict).sort_values(
    by="Importance", ascending=True
) # 중요도 데이터프레임 생성 및 중요도 기준으로 오름차순 정렬

# %%
from sklearn.preprocessing import StandardScaler # StandardScaler 임포트 (데이터 스케일링)
from sklearn.svm import SVC # SVC 임포트 (서포트 벡터 분류기)

scaler = StandardScaler() # StandardScaler 객체 생성
X_train_scaled = scaler.fit_transform(X_train_filtered) # 훈련 데이터를 스케일링하고 피팅
X_test_scaled = scaler.transform(X_test_filtered) # 테스트 데이터를 스케일링

# --------------------------------------------------------
# 3) 클래스=1에 대한 사용자 정의 F2 스코어러 정의
# --------------------------------------------------------
f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=1) # F2 스코어러 정의 (beta=2, pos_label=1)

# --------------------------------------------------------
# 4) SVC 설정 (class_weight='balanced', probability=True)
# - 'balanced' => 클래스 빈도에 자동으로 반비례
# - probability=True => 사용자 정의 임계값을 위해 .predict_proba()를 호출할 수 있음
# --------------------------------------------------------
svm_clf = SVC(
    kernel='rbf', # RBF 커널 사용
    class_weight='balanced', # 클래스 불균형 처리를 위한 가중치 자동 조정
    probability=True, # predict_proba 메서드 활성화
    random_state=42 # 랜덤 시드 설정
)
# --------------------------------------------------------
# 5) SVM + F2 스코어링을 사용한 RFECV
# --------------------------------------------------------
# rfecv = RFECV( # 주석 처리된 RFECV 객체 초기화
# estimator=svm_clf,
# step=5,
# cv=5,
# scoring=f2_scorer,
# min_features_to_select=1
# )
svm_clf.fit(X_train_scaled, y_train) # 스케일링된 훈련 데이터로 SVM 모델 훈련

# %%
# %%
from sklearn.metrics import ConfusionMatrixDisplay # ConfusionMatrixDisplay 임포트

y_pred_all = svm_clf.predict_proba(X_test_scaled)[:, 1] > 0.021 # SVM 모델의 예측 확률 (클래스 1)이 0.021보다 큰지 여부로 예측
cm_all = confusion_matrix(y_test, y_pred_all) # 실제 레이블과 예측 레이블을 사용하여 혼동 행렬 계산

print("혼동 행렬 (모든 특성):")
disp_all = ConfusionMatrixDisplay(confusion_matrix=cm_all) # 혼동 행렬 시각화 객체 생성
disp_all.plot(cmap=plt.cm.Blues) # 혼동 행렬 플롯 생성
plt.show() # 플롯 표시

# %%
from sklearn.metrics import ConfusionMatrixDisplay # ConfusionMatrixDisplay 임포트

y_pred_all = model_all_xgb.predict_proba(X_test["Radius"])[:, 1] > 0.02 # XGBoost 모델 (model_all_xgb)의 예측 확률이 0.02보다 큰지 여부로 예측 (X_test["Radius"]만 사용)
cm_all = confusion_matrix(y_test, y_pred_all) # 실제 레이블과 예측 레이블을 사용하여 혼동 행렬 계산

print("혼동 행렬 (모든 특성):")
disp_all = ConfusionMatrixDisplay(confusion_matrix=cm_all) # 혼동 행렬 시각화 객체 생성
disp_all.plot(cmap=plt.cm.Blues) # 혼동 행렬 플롯 생성
plt.show() # 플롯 표시

# %%
import warnings # 경고 메시지 처리를 위한 모듈
from sklearn.model_selection import StratifiedKFold # StratifiedKFold 임포트 (층화 k-겹 교차 검증)

# 'use_label_encoder' 경고만 무시합니다.
warnings.filterwarnings("ignore")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 5-겹 층화 교차 검증 객체 생성

data = pd.read_csv("data/REl data and Cp data joined dec20_gg.csv") # CSV 파일로부터 데이터 로드

print("전처리 전:", data.shape)

# (B) 전처리
df_processed = preprocess_dataset(data)  # 레이블을 1=실패로 반전

print("전처리 후:", df_processed.shape)


# (C) 훈련/테스트 분할
train_data, test_data = create_train_test_data(df_processed)

# 선택 사항: 특성 필터링 (분산 + 상관 관계)
# X를 분리한 다음 필터링합니다.
X_train = train_data.iloc[:, :-1]  # 레이블 제외
y_train = train_data.iloc[:, -1]

# 테스트 세트에도 동일하게 적용합니다.
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

X = pd.concat([X_train, X_test]) # 훈련 및 테스트 특성 결합
y = pd.concat([y_train, y_test]) # 훈련 및 테스트 레이블 결합

param_grid = {
    "n_estimators": [20, 50, 100, 200, 300], # 트리의 개수
    "max_depth": [2, 3, 4, 5, 10, None], # 트리의 최대 깊이
    "min_samples_split": [2, 3, 4, 5], # 노드를 분할하기 위한 최소 샘플 수
}

class_weight = {
    0: 1,
    1: sum(1 - y),
} # 클래스 가중치 설정
rf_model = RandomForestClassifier(class_weight=class_weight, random_state=42) # 랜덤 포레스트 분류기 초기화

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring=f2_rare_scorer,  # 클래스 1에 대한 F2 스코어링 사용
    cv=cv, # 층화 k-겹 교차 검증 사용
    verbose=1,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train) # 그리드 서치 수행
best_model = grid_search.best_estimator_ # 최적의 모델 선택
best_model.fit(X_train, y_train) # 최적의 모델로 훈련 데이터에 다시 피팅

y_pred_all = best_model.predict_proba(X)[:, 1] > 0.05 # 최적 모델의 예측 확률이 0.05보다 큰지 여부로 예측 (전체 X 사용)
cm_baseline_rf = confusion_matrix(y, y_pred_all) # 실제 레이블과 예측 레이블을 사용하여 혼동 행렬 계산
disp_baseline_rf = ConfusionMatrixDisplay(confusion_matrix=cm_baseline_rf) # 혼동 행렬 시각화 객체 생성
disp_baseline_rf.plot(cmap=plt.cm.Blues) # 혼동 행렬 플롯 생성
plt.show() # 플롯 표시

# %%
y_pred_all = best_model.predict_proba(X)[:, 1] > 0.4 # 최적 모델의 예측 확률이 0.4보다 큰지 여부로 예측 (전체 X 사용)
cm_baseline_rf = confusion_matrix(y, y_pred_all) # 실제 레이블과 예측 레이블을 사용하여 혼동 행렬 계산
disp_baseline_rf = ConfusionMatrixDisplay(confusion_matrix=cm_baseline_rf) # 혼동 행렬 시각화 객체 생성
disp_baseline_rf.plot(cmap=plt.cm.Blues) # 혼동 행렬 플롯 생성
plt.show() # 플롯 표시

# %%
importance = pd.DataFrame([best_model.feature_importances_, X_train.columns]) # 특성 중요도와 특성 이름을 포함하는 DataFrame 생성
# plot # 주석 처리된 라인
importance = importance.T # DataFrame 전치
importance.columns = ['importance', 'feature'] # 컬럼 이름 변경
importance.sort_values(by='importance', inplace=True, ascending=False) # 중요도 기준으로 내림차순 정렬
importance.reset_index(drop=True, inplace=True) # 인덱스 재설정
importance.set_index('feature', inplace=True) # 'feature' 컬럼을 인덱스로 설정
importance # 중요도 DataFrame 출력

# %% [markdown]
# ## 기준 모델 및 규칙 (Baseline model and rules)
# %%
X = data # 원본 데이터 전체를 특성으로 사용
y = data["Pass/Fail"] == "pass" # "Pass/Fail" 컬럼이 "pass"이면 True (통과), 아니면 False (실패)

radius_criteria = X["Radius"] <= 70 # "Radius"가 70 이하인지 확인
sensor_criteria = X["SensorOffsetHot-Cold"].abs() <= 0.02 # "SensorOffsetHot-Cold"의 절대값이 0.02 이하인지 확인
bandgap_criteria = data.apply(lambda row: check_band_gap(row['WAFER_NO'], row['Bias_Ref_test:VR1V2D@Bias_Ref_test[1] of FT2']), axis=1) # "WAFER_NO"와 "Bias_Ref_test..."를 사용하여 밴드갭 기준 확인

X_train, X_test, y_train, y_test = train_test_split(
    data, data["Pass/Fail"] == "pass", test_size=0.2, random_state=42
) # 데이터셋을 훈련 및 테스트 세트로 분할

radius_criteria_test = X_test["Radius"] <= 70 # 테스트 세트에서 "Radius" 기준 확인
sensor_criteria_test = X_test["SensorOffsetHot-Cold"] <= 0.02 # 테스트 세트에서 "SensorOffsetHot-Cold" 기준 확인
bandgap_criteria_test = X_test.apply(lambda row: check_band_gap(row['WAFER_NO'], row['Bias_Ref_test:VR1V2D@Bias_Ref_test[1] of FT2']), axis=1) # 테스트 세트에서 밴드갭 기준 확인

print(f"Radius 기준: {radius_criteria.sum()}") # Radius 기준을 충족하는 데이터 수 출력
print(f"Sensor 기준: {sensor_criteria.sum()}") # Sensor 기준을 충족하는 데이터 수 출력
print(f"Bandgap 기준: {bandgap_criteria.sum()}") # Bandgap 기준을 충족하는 데이터 수 출력
print(f"총계: {sum(radius_criteria & sensor_criteria & bandgap_criteria)}") # 모든 기준을 충족하는 데이터 수 출력

y_pred_baseline = radius_criteria & sensor_criteria & bandgap_criteria # 모든 기준을 결합한 기준선 예측 (훈련 데이터)
y_pred_baseline_test = radius_criteria_test & sensor_criteria_test & bandgap_criteria_test # 모든 기준을 결합한 기준선 예측 (테스트 데이터)

# %%
import pandas as pd # pandas 임포트
from scipy.stats import chi2_contingency # chi2_contingency 임포트 (카이제곱 독립성 검정)

def criterion_significance_test(criterion_series, outcome_series, criterion_name="Criterion"):
    """
    부울 Series 'criterion_series'와 부울 Series 'outcome_series'가 주어지면,
    카이제곱 독립성 검정을 수행하고 테스트 결과를 반환합니다.
    """
    # 1. 분할표 생성
    # 행 = 기준 True/False, 열 = 결과 Pass/Fail
    contingency_table = pd.crosstab(criterion_series, outcome_series) # 분할표 생성 (기준과 결과 간의 관계)

    # 2. 카이제곱 검정 실행
    chi2, p_value, dof, expected = chi2_contingency(contingency_table) # 카이제곱 통계량, p-값, 자유도, 기대 빈도 계산

    # 3. 결과 출력 또는 반환
    print(f"=== {criterion_name} ===") # 기준 이름 출력
    print("분할표:\n", contingency_table) # 분할표 출력
    print(f"카이제곱 통계량 = {chi2:.4f}, p-값 = {p_value:.4e}, 자유도 = {dof}") # 카이제곱 통계량, p-값, 자유도 출력
    print("기대 빈도:\n", expected, "\n") # 기대 빈도 출력
    return chi2, p_value, dof, expected # 카이제곱 통계량, p-값, 자유도, 기대 빈도 반환

# %%
chi2_sensor, p_sensor, *_ = criterion_significance_test(
    sensor_criteria, y, criterion_name="SensorOffsetHot-Cold <= 0.02"
) # 센서 오프셋 기준에 대한 유의성 검정 수행

# %%
chi2_radius, p_radius, *_ = criterion_significance_test(
    radius_criteria, y, criterion_name="Radius <= 70"
) # Radius 기준에 대한 유의성 검정 수행

# %%
chi2_bandgap, p_bandgap, *_ = criterion_significance_test(
    bandgap_criteria, y, criterion_name="BandGap Criterion"
) # 밴드갭 기준에 대한 유의성 검정 수행

# %%
cm_baseline_rf = confusion_matrix(y, y_pred_baseline) # 실제 레이블과 기준선 예측을 사용하여 혼동 행렬 계산 (훈련 데이터)
disp_baseline_rf = ConfusionMatrixDisplay(confusion_matrix=cm_baseline_rf) # 혼동 행렬 시각화 객체 생성
disp_baseline_rf.plot(cmap=plt.cm.Blues) # 혼동 행렬 플롯 생성
plt.show() # 플롯 표시

cm_baseline_rf_test = confusion_matrix((1-y_test), y_pred_baseline_test) # 실제 레이블 (1-y_test로 반전)과 기준선 예측 (테스트 데이터)을 사용하여 혼동 행렬 계산
disp_baseline_rf_test = ConfusionMatrixDisplay(confusion_matrix=cm_baseline_rf_test) # 혼동 행렬 시각화 객체 생성
disp_baseline_rf_test.plot(cmap=plt.cm.Blues) # 혼동 행렬 플롯 생성
plt.show() # 플롯 표시

# %%
import numpy as np # 넘파이(numpy) 라이브러리를 임포트합니다. 주로 숫자 계산, 배열 처리에 사용됩니다.
import pandas as pd # 판다스(pandas) 라이브러리를 임포트합니다. 데이터프레임과 같은 데이터 구조를 다루는 데 유용합니다.
from sklearn.model_selection import StratifiedKFold # 사이킷런(sklearn)의 모델 선택 모듈에서 StratifiedKFold를 임포트합니다. 이는 클래스 비율을 유지하면서 데이터를 K-겹 교차 검증으로 분할하는 데 사용됩니다.
from sklearn.metrics import fbeta_score, confusion_matrix, roc_auc_score # 사이킷런의 메트릭(metrics) 모듈에서 fbeta_score, confusion_matrix, roc_auc_score를 임포트합니다. 모델의 성능을 평가하는 데 사용되는 지표들입니다.
from sklearn.linear_model import LogisticRegression # 사이킷런의 선형 모델(linear_model) 모듈에서 LogisticRegression을 임포트합니다. 로지스틱 회귀 모델입니다.

def run_stratified_kfold_threshold_search(X, y, model, beta=2, n_splits=5, bandgap_criteria=None, sensor_offset=None):
    """
    주어진 `model`을 사용하여 (X, y)에 대해 계층별 K-겹 교차 검증을 수행합니다.
    F2 점수를 기반으로 각 폴드에 대한 최적의 임계값(0..1)을 검색합니다.
    평균 지표와 폴드별 결과를 반환합니다.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) # StratifiedKFold 객체를 초기화합니다. n_splits는 폴드 수, shuffle=True는 데이터를 섞을지 여부, random_state는 재현성을 위한 난수 시드입니다.

    # 폴드 전반에 걸쳐 지표를 추적하기 위한 컨테이너
    best_thresholds = [] # 각 폴드의 최적 임계값을 저장할 리스트
    f2_scores = [] # 각 폴드의 F2 점수를 저장할 리스트
    roc_aucs = [] # 각 폴드의 ROC AUC 점수를 저장할 리스트
    confusion_matrices = [] # 각 폴드의 혼동 행렬을 저장할 리스트

    all_preds = [] # 모든 폴드의 예측값을 저장할 리스트
    all_probs = [] # 모든 폴드의 예측 확률을 저장할 리스트
    all_true = [] # 모든 폴드의 실제값을 저장할 리스트

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)): # StratifiedKFold를 사용하여 데이터를 분할하고 각 폴드에 대해 반복합니다. train_idx는 훈련 세트 인덱스, val_idx는 검증 세트 인덱스입니다.
        # -------------------------
        # 1) 훈련/검증 세트로 분할
        # -------------------------
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx] # 훈련 세트 특성과 레이블을 추출합니다.
        X_val, y_val     = X.iloc[val_idx], y.iloc[val_idx] # 검증 세트 특성과 레이블을 추출합니다.

        # -------------------------
        # 2) 모델 훈련
        # -------------------------
        model.fit(X_train, y_train) # 훈련 세트로 모델을 훈련합니다.

        # -------------------------
        # 3) 확률 예측
        # -------------------------
        probs_class1 = model.predict_proba(X_val)[:, 1]  # 클래스=1에 대한 확률을 예측합니다.
        if bandgap_criteria is not None: # bandgap_criteria가 제공된 경우
            bandgap_criteria_val = bandgap_criteria[val_idx] # 검증 세트의 bandgap_criteria를 추출합니다.
        else: # bandgap_criteria가 제공되지 않은 경우
            bandgap_criteria_val = pd.Series([0]*len(X_val)) # 모든 값이 0인 판다스 시리즈를 생성합니다.

        if sensor_offset is not None: # sensor_offset이 제공된 경우
            sensor_offset_val = sensor_offset[val_idx] # 검증 세트의 sensor_offset을 추출합니다.
        else: # sensor_offset이 제공되지 않은 경우
            sensor_offset_val = pd.Series([0]*len(X_val)) # 모든 값이 0인 판다스 시리즈를 생성합니다.

        probs_class1[sensor_offset_val] = 1.0 # sensor_offset 조건이 충족되면 해당 샘플의 클래스1 확률을 1.0으로 설정합니다.
        probs_class1[bandgap_criteria_val] = 1.0 # bandgap_criteria 조건이 충족되면 해당 샘플의 클래스1 확률을 1.0으로 설정합니다.

        # -------------------------
        # 4) F2를 통한 최적 임계값 찾기
        # -------------------------
        thresholds = np.linspace(0, 1, 101)  # 0.00, 0.01, ... 1.00 범위의 임계값을 생성합니다.
        best_thr = 0.0 # 최적 임계값 초기화
        best_f2 = -1.0 # 최적 F2 점수 초기화
        
        for thr in thresholds: # 각 임계값에 대해 반복합니다.
            y_pred_temp = (probs_class1 >= thr).astype(int) # 현재 임계값을 기준으로 예측값을 이진화합니다.
            f2_temp = fbeta_score(y_val, y_pred_temp, beta=beta, pos_label=1) # F-beta 점수를 계산합니다.
            if f2_temp > best_f2: # 현재 F-beta 점수가 최적 점수보다 높으면
                best_f2 = f2_temp # 최적 F-beta 점수를 업데이트합니다.
                best_thr = thr # 최적 임계값을 업데이트합니다.

        best_thresholds.append(best_thr) # 현재 폴드의 최적 임계값을 리스트에 추가합니다.

        # 최적 임계값으로 최종 예측 수행
        y_pred = (probs_class1 >= best_thr).astype(int) # 최적 임계값을 사용하여 최종 예측값을 얻습니다.

        # -------------------------
        # 5) 지표 계산 및 저장
        # -------------------------
        f2_scores.append(best_f2) # 현재 폴드의 최적 F2 점수를 리스트에 추가합니다.
        roc_aucs.append(roc_auc_score(y_val, probs_class1)) # 현재 폴드의 ROC AUC 점수를 계산하여 리스트에 추가합니다.

        # 클래스=1을 "양성"으로 간주하는 혼동 행렬
        cm = confusion_matrix(y_val, y_pred, labels=[1,0]) # 혼동 행렬을 계산합니다.
        confusion_matrices.append(cm) # 현재 폴드의 혼동 행렬을 리스트에 추가합니다.

        print(f"\nFold {fold_idx+1}: Best threshold={best_thr:.2f}, F2={best_f2:.4f}, ROC AUC={roc_aucs[-1]:.4f}") # 현재 폴드의 결과 출력
        print("Confusion matrix:\n", cm) # 현재 폴드의 혼동 행렬 출력

        all_preds.extend(y_pred) # 모든 예측값을 리스트에 추가합니다.
        all_probs.extend(probs_class1) # 모든 예측 확률을 리스트에 추가합니다.
        all_true.extend(y_val) # 모든 실제값을 리스트에 추가합니다.

    # -------------------------
    # 6) 결과 평균화
    # -------------------------
    avg_f2 = np.mean(f2_scores) # 모든 폴드의 F2 점수 평균을 계산합니다.
    avg_roc = np.mean(roc_aucs) # 모든 폴드의 ROC AUC 점수 평균을 계산합니다.

    # 폴드 전체의 혼동 행렬 합산 (결합된 행렬)
    sum_confusion = sum(confusion_matrices) # 모든 폴드의 혼동 행렬을 합산합니다.

    print("\n======== CROSS-VALIDATION SUMMARY ========") # 교차 검증 요약 제목 출력
    print(f"Average best threshold : {np.mean(best_thresholds):.3f}") # 평균 최적 임계값 출력
    print(f"Average F2 score       : {avg_f2:.4f}") # 평균 F2 점수 출력
    print(f"Average ROC AUC        : {avg_roc:.4f}") # 평균 ROC AUC 점수 출력
    print("Sum of confusion matrices across folds:\n", sum_confusion) # 합산된 혼동 행렬 출력

    return { # 결과를 딕셔너리 형태로 반환합니다.
        "thresholds": best_thresholds,
        "f2_scores": f2_scores,
        "roc_aucs": roc_aucs,
        "avg_f2": avg_f2,
        "avg_roc": avg_roc,
        "sum_confusion": sum_confusion,
        "all_preds": np.array(all_preds),
        "all_probs": np.array(all_probs),
        "all_true": np.array(all_true),
    }

# %%
import json # JSON 라이브러리를 임포트합니다.

with open("final_features.json", "r") as f: # "final_features.json" 파일을 읽기 모드로 엽니다.
    final_features_lr = json.load(f) # JSON 파일에서 데이터를 로드하여 final_features_lr 변수에 저장합니다.
# final_model_rf = ["Radius", "SensorOffsetHot-Cold", "SensorOffsetHot-Cold-Abs", "Magnetic_DC:V_VPP8MTFVBI_MAG_Tx@Magnetic_DC_1_ of FT1", "Magnetic_DC_after_CAL:V_V0FVBI_MAG2_CAL_Tx@Magnetic_DC_after_CAL_1_ of FT1",  "Leakage:LEAK_h1@Leakage_1_ of FT1", "Output_Buffer:OBVOH@Output_Buffer_Test_1_ of FT1"]

# 예시: 최종 데이터셋은 preprocessed_df
preprocessed_df_with_outlier = preprocess_dataset(data) # 'data'를 전처리하는 함수를 호출하여 아웃라이어가 포함된 데이터프레임을 생성합니다.
outlier_mask = (preprocessed_df_with_outlier["Radius"] < 32) & ( # 'Radius'가 32 미만이고 'Pass/Fail'이 True인 경우를 아웃라이어로 정의하는 마스크를 생성합니다.
    preprocessed_df_with_outlier["Pass/Fail"]
)



# ~를 사용하여 마스크를 반전시켜 아웃라이어가 아닌 행만 유지합니다.
preprocessed_dataset = preprocessed_df_with_outlier[~outlier_mask].reset_index(drop=True) # 아웃라이어가 아닌 데이터를 선택하고 인덱스를 재설정합니다.

bandgap_fail = 1 - preprocessed_dataset["band gap dpat_ok for band gap"] # "band gap dpat_ok for band gap" 컬럼을 사용하여 bandgap_fail 기준을 정의합니다. (1 - 값)
sensor_offset = preprocessed_dataset[ "SensorOffsetHot-Cold-Abs"] > 0.02 # "SensorOffsetHot-Cold-Abs" 컬럼이 0.02보다 큰 경우를 sensor_offset 기준으로 정의합니다.


X_full = preprocessed_dataset.iloc[:, :-1] # 전처리된 데이터셋의 마지막 컬럼을 제외한 모든 컬럼을 특성(X_full)으로 사용합니다.
y_full = preprocessed_dataset.iloc[:,  -1] # 전처리된 데이터셋의 마지막 컬럼을 레이블(y_full)로 사용합니다.
X_full = X_full # X_full을 그대로 사용합니다. (이 라인은 사실상 변경 없음)


# 예시: 로지스틱 회귀
num_pos = y_full.sum() # 양성 클래스(1)의 개수를 계산합니다.
num_neg = len(y_full) - num_pos # 음성 클래스(0)의 개수를 계산합니다.
# scale_pos_weight = num_neg / num_pos * 10 if num_pos > 0 else 1.0
weights = [1, 10, 25, 50, 75, 99, 100, 1000] # XGBoost의 scale_pos_weight에 사용할 가중치 리스트를 정의합니다.

model_xgboost = XGBClassifier( # XGBoost 분류기 모델을 초기화합니다.
    use_label_encoder=False, # 라벨 인코더 사용을 비활성화합니다.
    eval_metric="logloss", # 평가 지표로 로그 손실을 사용합니다.
    # scale_pos_weight=scale_pos_weight,
    random_state=42 # 재현성을 위한 난수 시드를 설정합니다.
)


param_grid = { # GridSearchCV에 사용할 하이퍼파라미터 그리드를 정의합니다.
    "n_estimators": [10, 30, 50, 100, 200], # 부스팅 트리의 개수
    "max_depth": [2, 3, 4, 5, 10], # 각 트리의 최대 깊이
    "learning_rate": [0.01, 0.1, 0.2, 0.3], # 학습률
    "scale_pos_weight" : weights # 양성 클래스 가중치
}

grid_search = GridSearchCV( # GridSearchCV 객체를 초기화합니다.
    estimator=xgb_model, # 사용할 모델 (xgb_model은 정의되지 않았으므로 model_xgboost로 추정됩니다.)
    param_grid=param_grid, # 하이퍼파라미터 그리드
    scoring=f2_scorer,  # 클래스=1에 대한 F2 점수 (f2_scorer는 정의되지 않았으므로 make_scorer로 생성된 것으로 추정됩니다.)
    cv=5, # 5겹 교차 검증
    verbose=1, # 자세한 출력 활성화
    n_jobs=-1, # 모든 CPU 코어 사용
)

grid_search.fit(X_train, y_train) # 훈련 데이터로 그리드 탐색을 수행하여 최적의 하이퍼파라미터를 찾습니다. (X_train, y_train은 정의되지 않았으므로 X_full, y_full로 추정됩니다.)
best_model_xgb = grid_search.best_estimator_ # 그리드 탐색을 통해 찾은 최적의 모델을 저장합니다.
print(best_model_xgb) # 최적 모델 출력

results_xgb = run_stratified_kfold_threshold_search( # run_stratified_kfold_threshold_search 함수를 호출하여 XGBoost 모델의 교차 검증을 수행합니다.
    X_full, 
    y_full, 
    model=best_model_xgb,
    beta=5,           # F2 점수를 위한 베타 값 (F5)
    n_splits=20,       # 20-겹 교차 검증
    sensor_offset=sensor_offset, # sensor_offset 기준 전달
    bandgap_criteria=bandgap_fail # bandgap_criteria 기준 전달
)

print("CV Results:\n", results_xgb) # 교차 검증 결과 출력

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score # 혼동 행렬, ROC 곡선, ROC AUC 스코어를 임포트합니다.

global_cm = confusion_matrix(results_xgb["all_true"], results_xgb["all_preds"], labels=[1,0]) # 전체 예측 및 실제값에 대한 혼동 행렬을 계산합니다.
fpr, tpr, _ = roc_curve(results_xgb["all_true"], results_xgb["all_probs"], pos_label=1) # ROC 곡선을 위한 FPR, TPR을 계산합니다.
global_auc = roc_auc_score(results_xgb["all_true"], results_xgb["all_probs"]) # 전체 ROC AUC 스코어를 계산합니다.

print("Global confusion matrix:\n", global_cm) # 전체 혼동 행렬 출력
print(f"Global ROC AUC: {global_auc:.4f}, Best threshold: {np.mean(results_xgb["thresholds"])}") # 전체 ROC AUC 및 평균 최적 임계값 출력


# %%
grid_search.best_params_ # 그리드 탐색으로 찾은 최적의 하이퍼파라미터 출력

# %%
import json # JSON 라이브러리를 임포트합니다.

with open("final_features.json", "r") as f: # "final_features.json" 파일을 읽기 모드로 엽니다.
    final_features_lr = json.load(f) # JSON 파일에서 데이터를 로드하여 final_features_lr 변수에 저장합니다.

# 예시: 최종 데이터셋은 preprocessed_df
preprocessed_dataset = preprocess_dataset(data) # 'data'를 전처리하는 함수를 호출하여 데이터프레임을 생성합니다.

outlier_mask = (preprocessed_dataset["Radius"] < 32) & ( # 'Radius'가 32 미만이고 'Pass/Fail'이 True인 경우를 아웃라이어로 정의하는 마스크를 생성합니다.
    preprocessed_dataset["Pass/Fail"]
)

# ~를 사용하여 마스크를 반전시켜 아웃라이어가 아닌 행만 유지합니다.
preprocessed_dataset = preprocessed_dataset[~outlier_mask].reset_index(drop=True) # 아웃라이어가 아닌 데이터를 선택하고 인덱스를 재설정합니다.

bandgap_fail = 1 - preprocessed_dataset["band gap dpat_ok for band gap"] # "band gap dpat_ok for band gap" 컬럼을 사용하여 bandgap_fail 기준을 정의합니다.
sensor_offset = preprocessed_dataset[ "SensorOffsetHot-Cold-Abs"] > 0.02 # "SensorOffsetHot-Cold-Abs" 컬럼이 0.02보다 큰 경우를 sensor_offset 기준으로 정의합니다.


X_full = preprocessed_dataset.iloc[:, :-1] # 전처리된 데이터셋의 마지막 컬럼을 제외한 모든 컬럼을 특성(X_full)으로 사용합니다.
y_full = preprocessed_dataset.iloc[:,  -1] # 전처리된 데이터셋의 마지막 컬럼을 레이블(y_full)로 사용합니다.
X_full = X_full # X_full을 그대로 사용합니다.
# (선택 사항) 여기에 또는 각 폴드 내에서 특성 필터링
# X_full, final_cols = variance_correlation_filter(X_full, 0.0, 0.99)
# X_full = X_full[final_cols]

# 예시: 로지스틱 회귀
model_logreg = LogisticRegression( # 로지스틱 회귀 모델을 초기화합니다.
    class_weight={0: 1, 1: sum(1 - y_full)}, # 클래스 가중치를 설정합니다. (음성 클래스: 1, 양성 클래스: 음성 샘플 수)
    max_iter=1000, # 최대 반복 횟수
    random_state=42 # 재현성을 위한 난수 시드를 설정합니다.
)

results_log = run_stratified_kfold_threshold_search( # run_stratified_kfold_threshold_search 함수를 호출하여 로지스틱 회귀 모델의 교차 검증을 수행합니다.
    X_full, 
    y_full, 
    model=model_logreg,
    beta=5,          # F2 점수를 위한 베타 값 (F5)
    n_splits=20,       # 20-겹 교차 검증
    # bandgap_criteria=bandgap_fail,
    # sensor_offset=sensor_offset
)

print("CV Results:\n", results_log) # 교차 검증 결과 출력

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score # 혼동 행렬, ROC 곡선, ROC AUC 스코어를 임포트합니다.

global_cm = confusion_matrix(results_log["all_true"], results_log["all_preds"], labels=[1,0]) # 전체 예측 및 실제값에 대한 혼동 행렬을 계산합니다.
fpr, tpr, _ = roc_curve(results_log["all_true"], results_log["all_probs"], pos_label=1) # ROC 곡선을 위한 FPR, TPR을 계산합니다.
global_auc = roc_auc_score(results_log["all_true"], results_log["all_probs"]) # 전체 ROC AUC 스코어를 계산합니다.

print("Global confusion matrix:\n", global_cm) # 전체 혼동 행렬 출력
print(f"Global ROC AUC: {global_auc:.4f}, Best threshold: {np.mean(results_log["thresholds"]):.4f}") # 전체 ROC AUC 및 평균 최적 임계값 출력


# %%


# %%
import json # JSON 라이브러리를 임포트합니다.
import numpy as np # 넘파이(numpy) 라이브러리를 임포트합니다.
import pandas as pd # 판다스(pandas) 라이브러리를 임포트합니다.

from sklearn.ensemble import IsolationForest # 사이킷런의 앙상블(ensemble) 모듈에서 IsolationForest를 임포트합니다. 고립 포레스트 모델입니다.
from sklearn.metrics import ( # 사이킷런의 메트릭(metrics) 모듈에서 여러 평가 지표를 임포트합니다.
    fbeta_score, # F-beta 스코어
    make_scorer, # 사용자 정의 스코어러를 만들기 위한 함수
    confusion_matrix, # 혼동 행렬
    roc_curve, # ROC 곡선
    roc_auc_score # ROC AUC 스코어
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold # 사이킷런의 모델 선택 모듈에서 GridSearchCV와 StratifiedKFold를 임포트합니다.

# ============ STEP 1: 최종 특성 로드 (필요한 경우) ============
with open("final_features.json", "r") as f: # "final_features.json" 파일을 읽기 모드로 엽니다.
    final_features_if = json.load(f) # JSON 파일에서 데이터를 로드하여 final_features_if 변수에 저장합니다.

# 예를 들어, final_features_if = ["Radius", "SensorOffsetHot-Cold-Abs", ...]
# 전처리된 데이터프레임에 이 컬럼들이 있는지 확인하세요.

# 예시: 최종 데이터셋은 preprocessed_df
preprocessed_df_with_outlier = preprocess_dataset(data) # 'data'를 전처리하는 함수를 호출하여 아웃라이어가 포함된 데이터프레임을 생성합니다.

# 예시 아웃라이어 마스크 (관련된 경우, 이전 스니펫과 동일)
outlier_mask = (preprocessed_df_with_outlier["Radius"] < 32) & ( # 'Radius'가 32 미만이고 'Pass/Fail'이 True인 경우를 아웃라이어로 정의하는 마스크를 생성합니다.
    preprocessed_df_with_outlier["Pass/Fail"]
)

# ~를 사용하여 마스크를 반전시킵니다 (원하는 경우 아웃라이어가 아닌 행 유지)
preprocessed_dataset = preprocessed_df_with_outlier[~outlier_mask].reset_index(drop=True) # 아웃라이어가 아닌 데이터를 선택하고 인덱스를 재설정합니다.


# ============ STEP 3: 보조 기준 정의 (bandgap, sensor_offset 등) ============
# 여전히 이 조건들을 추적하고 싶다면, 이전과 동일하게 유지합니다.
bandgap_fail = 1 - preprocessed_dataset["band gap dpat_ok for band gap"] # bandgap_fail 기준 정의
sensor_offset = preprocessed_dataset["SensorOffsetHot-Cold-Abs"] > 0.02 # sensor_offset 기준 정의

# ============ STEP 4: X와 y 준비 ============
# 일반적인 아웃라이어 감지 설정에서:
#   - y = 1 => "Fail" => 아웃라이어
#   - y = 0 => "Pass" => 인라이어
# 자신의 데이터셋에 따라 슬라이스를 조정하세요.

# 스니펫에서 마지막 컬럼은 Pass/Fail
X_full = preprocessed_dataset.iloc[:, :-1] # 마지막 컬럼을 제외한 모든 컬럼을 특성(X_full)으로 사용합니다.
y_full = preprocessed_dataset.iloc[:,  -1] # 마지막 컬럼을 레이블(y_full)로 사용합니다.

# y가 0/1이고 1 = Fail (아웃라이어), 0 = Pass (인라이어)인지 확인합니다.
# ("Pass/Fail" 컬럼이 이미 {True/False}인 경우, 필요에 따라 변환합니다.)
y_full = y_full.astype(int) # y_full을 정수형으로 변환합니다.

print("Number of samples:", len(X_full)) # 샘플 개수 출력
print("Number of fails (outliers):", sum(y_full)) # 불합격(아웃라이어) 개수 출력
print("Number of passes (inliers):", len(y_full) - sum(y_full)) # 합격(인라이어) 개수 출력

# ============ STEP 5: 고립 포레스트를 위한 사용자 정의 F-beta 스코어러 정의 ============
# IsolationForest.predict(X)는 인라이어에 대해 +1을, 아웃라이어에 대해 -1을 반환합니다.
# "Fail" = 1로 간주한다면, 아웃라이어가 1로 예측되기를 원합니다.
def isolation_fbeta_scorer(estimator, X, y_true, beta=2.0):
    """
    y_true=1 => 아웃라이어, 0 => 인라이어라는 가정하에 F-beta (예: F2)를 평가하기 위한
    사용자 정의 스코어링 함수입니다.
    """
    # 고립 포레스트로 예측
    y_pred_raw = estimator.predict(X) # 고립 포레스트의 원시 예측값을 얻습니다. (+1 또는 -1)
    # +1 (인라이어) => 0, -1 (아웃라이어) => 1로 변환
    y_pred = np.where(y_pred_raw == -1, 1, 0) # 예측값을 0과 1로 변환합니다.
    return fbeta_score(y_true, y_pred, beta=beta) # F-beta 스코어를 반환합니다.

# GridSearchCV에 적합한 스코어러 객체 생성
f2_scorer_iforest = make_scorer( # make_scorer를 사용하여 사용자 정의 스코어러를 만듭니다.
    isolation_fbeta_scorer,
    beta=2.0,  # F2
    greater_is_better=True, # 값이 클수록 좋음을 나타냅니다.
    needs_proba=False, # 확률이 필요 없음을 나타냅니다.
    needs_threshold=False # 임계값이 필요 없음을 나타냅니다.
)

# ============ STEP 6: 고립 포레스트 및 하이퍼파라미터 그리드 정의 ============
model_if = IsolationForest(random_state=42) # IsolationForest 모델을 초기화합니다.

param_grid_if = { # GridSearchCV에 사용할 하이퍼파라미터 그리드를 정의합니다.
    "n_estimators":   [50, 100, 200], # 고립 트리의 개수
    "max_samples":    ["auto", 0.8, 0.9], # 각 트리에 그릴 샘플의 수
    "contamination":  [0.01, 0.05, 0.1],  # 아웃라이어의 예상 비율 (데이터셋의 아웃라이어 비율에 따라 조정)
    "max_features":   [1.0, 0.5, 0.8], # 각 트리에 그릴 특성의 수
    "bootstrap":      [False, True], # 샘플링 시 복원 추출 여부
}

grid_search_if = GridSearchCV( # GridSearchCV 객체를 초기화합니다.
    estimator=model_if, # 사용할 모델
    param_grid=param_grid_if, # 하이퍼파라미터 그리드
    scoring=f2_scorer_iforest, # 스코어링 함수
    cv=5, # 5겹 교차 검증
    verbose=1, # 자세한 출력 활성화
    n_jobs=-1 # 모든 CPU 코어 사용
)

# IsolationForest는 비지도 학습이지만, 알려진 레이블이 있으므로
# "y_full"을 타겟처럼 취급합니다. GridSearchCV는
# X, y를 분할하고, X_train에서 훈련한 다음 X_test에서 예측을 평가하여
# 스코어를 매깁니다.
grid_search_if.fit(X_full, y_full) # 훈련 데이터로 그리드 탐색을 수행하여 최적의 하이퍼파라미터를 찾습니다.
best_model_if = grid_search_if.best_estimator_ # 그리드 탐색을 통해 찾은 최적의 모델을 저장합니다.

print("Best Isolation Forest Model:", best_model_if) # 최적 고립 포레스트 모델 출력
print("Best CV Score (F2):", grid_search_if.best_score_) # 최적 교차 검증 F2 스코어 출력


# ============ STEP 7: 전체 데이터셋에 대해 최종 모델 교차 검증 ============

# 최적 모델로 교차 검증 실행
results_if = run_stratified_kfold_threshold_search(    X_full, # run_stratified_kfold_threshold_search 함수를 호출하여 모델의 교차 검증을 수행합니다. (여기서 model=model_logreg로 되어 있어 잘못된 호출일 수 있습니다. best_model_if가 되어야 합니다.)
    y_full, 
    model=model_logreg, # 이 부분은 위에서 정의된 model_logreg를 사용하고 있습니다. IsolationForest를 사용하려면 best_model_if로 변경해야 합니다.
    beta=5,          # F2
    n_splits=20,       # 10-겹 CV)

# ============ STEP 8: 결과 평가 ============

# 혼동 행렬
cm_if = confusion_matrix(results_if["all_true"], results_if["all_preds"], labels=[1,0]) # 혼동 행렬을 계산합니다.
print("Isolation Forest global confusion matrix (label order = [1,0]):\n", cm_if) # 고립 포레스트 전체 혼동 행렬 출력

# ROC 곡선 및 AUC (pos_label=1 => 아웃라이어 필요)
fpr_if, tpr_if, thresholds_if = roc_curve(results_if["all_true"], -results_if["all_scores"], pos_label=1) # ROC 곡선을 위한 FPR, TPR을 계산합니다. (results_if["all_scores"]는 정의되지 않았으므로 오류가 발생할 수 있습니다. 일반적으로 decision_function의 음수 값을 사용합니다.)
# 참고: decision_function의 음수 값을 사용했습니다.
#       점수가 높을수록 인라이어를 의미하므로 "아웃라이어"를 양성으로 취급하기 위해 반전합니다.

auc_if = roc_auc_score(results_if["all_true"], -results_if["all_scores"]) # ROC AUC 스코어를 계산합니다. (results_if["all_scores"]는 정의되지 않았으므로 오류가 발생할 수 있습니다.)
print(f"Isolation Forest ROC AUC: {auc_if:.4f}") # 고립 포레스트 ROC AUC 출력

# 결합된 예측에 대한 F2 스코어:
f2_if = fbeta_score(results_if["all_true"], results_if["all_preds"], beta=2) # F2 스코어를 계산합니다.
print(f"Combined F2 score across folds: {f2_if:.4f}") # 폴드 전체의 결합된 F2 스코어 출력


# %%
best_model_xgb.fit(X_train, y_train) # 최적 XGBoost 모델을 훈련 세트로 훈련합니다. (X_train, y_train은 정의되지 않았으므로 X_full, y_full로 추정됩니다.)
y_pred = (best_model_xgb.predict_proba(X)[:, 1]>0.5) #| sensor_offset | bandgap_fail # X에 대한 클래스1 예측 확률이 0.5보다 큰 경우를 예측값으로 사용합니다. (X는 정의되지 않았으므로 X_full로 추정됩니다.)

cm_baseline_rf_test = confusion_matrix(y, y_pred) # 실제값과 예측값에 대한 혼동 행렬을 계산합니다. (y는 정의되지 않았으므로 y_full로 추정됩니다.)

disp_baseline_rf_test = ConfusionMatrixDisplay(confusion_matrix=cm_baseline_rf_test) # 혼동 행렬 디스플레이 객체를 생성합니다.

disp_baseline_rf_test.plot(cmap=plt.cm.Blues) # 혼동 행렬을 플로팅합니다.
plt.show() # 플롯을 보여줍니다.

# %%
model_logreg.fit(X_full, y_full) # 로지스틱 회귀 모델을 전체 데이터(X_full, y_full)로 훈련합니다.
y_pred = (model_logreg.predict_proba(X_full)[:, 1]>0.9) # | sensor_offset | bandgap_fail # X_full에 대한 클래스1 예측 확률이 0.9보다 큰 경우를 예측값으로 사용합니다.

cm_baseline_rf_test = confusion_matrix(y_full, y_pred) # 실제값과 예측값에 대한 혼동 행렬을 계산합니다.

disp_baseline_rf_test = ConfusionMatrixDisplay(confusion_matrix=cm_baseline_rf_test) # 혼동 행렬 디스플레이 객체를 생성합니다.

disp_baseline_rf_test.plot(cmap=plt.cm.Blues) # 혼동 행렬을 플로팅합니다.
plt.show() # 플롯을 보여줍니다.

# %%
print(np.mean(results_log["thresholds"])) # 로지스틱 회귀 모델의 평균 최적 임계값을 출력합니다.
y_pred = (results_log["all_probs"]>np.mean(results_log["thresholds"])) #| bandgap_fail | sensor_offset # 모든 예측 확률이 평균 최적 임계값보다 큰 경우를 예측값으로 사용합니다.
y_true = results_log["all_true"] # 실제값을 가져옵니다.

cm_baseline_log = confusion_matrix(y_true, y_pred) # 실제값과 예측값에 대한 혼동 행렬을 계산합니다.

disp_baseline_log = ConfusionMatrixDisplay(confusion_matrix=cm_baseline_log) # 혼동 행렬 디스플레이 객체를 생성합니다.

disp_baseline_log.plot(cmap=plt.cm.Blues) # 혼동 행렬을 플로팅합니다.
plt.show() # 플롯을 보여줍니다.

# %%
best_model_xgb.fit(X_full, y_full) # 최적 XGBoost 모델을 전체 데이터(X_full, y_full)로 훈련합니다.
y_pred = (best_model_xgb.predict_proba(X_full)[:, 1]>0.2) # X_full에 대한 클래스1 예측 확률이 0.2보다 큰 경우를 예측값으로 사용합니다.

cm_baseline_rf_test = confusion_matrix(y_full, y_pred) # 실제값과 예측값에 대한 혼동 행렬을 계산합니다.

disp_baseline_rf_test = ConfusionMatrixDisplay(confusion_matrix=cm_baseline_rf_test) # 혼동 행렬 디스플레이 객체를 생성합니다.

disp_baseline_rf_test.plot(cmap=plt.cm.Blues) # 혼동 행렬을 플로팅합니다.
plt.show() # 플롯을 보여줍니다.

# %%
import matplotlib.pyplot as plt # Matplotlib의 pyplot 모듈을 임포트합니다.

# 1) X_full을 복사하여 원본을 변경하지 않고 컬럼을 추가할 수 있도록 합니다.
df_vis = X_full.copy() # X_full 데이터프레임을 복사합니다.

# 2) 실제 및 예측 레이블 추가
df_vis["y_true"] = y_full # 실제 레이블 컬럼을 추가합니다.
df_vis["y_pred"] = y_pred # 예측 레이블 컬럼을 추가합니다.

# 3) 각 행을 TP/FP/TN/FN으로 레이블링하는 함수 정의
def confusion_label(row): # 혼동 행렬 레이블을 반환하는 함수를 정의합니다.
    if row["y_true"] == 1 and row["y_pred"] == 1: # 실제 1, 예측 1 (True Positive)
        return "TP"
    elif row["y_true"] == 0 and row["y_pred"] == 0: # 실제 0, 예측 0 (True Negative)
        return "TN"
    elif row["y_true"] == 0 and row["y_pred"] == 1: # 실제 0, 예측 1 (False Positive)
        return "FP"
    else:  # row["y_true"] == 1 and row["y_pred"] == 0 # 실제 1, 예측 0 (False Negative)
        return "FN"

df_vis["conf_label"] = df_vis.apply(confusion_label, axis=1) # 각 행에 대해 confusion_label 함수를 적용하여 'conf_label' 컬럼을 생성합니다.

# 4) 각 혼동 레이블을 색상에 매핑
color_map = { # 혼동 레이블과 색상 매핑 딕셔너리를 정의합니다.
    "TP": "blue",
    "FP": "red",
    "TN": "green",
    "FN": "orange",
}

# 5) 산점도 플롯
plt.figure(figsize=(8,6)) # 플롯의 크기를 설정합니다.

for label, color in color_map.items(): # 각 혼동 레이블과 색상에 대해 반복합니다.
    subset = df_vis[df_vis["conf_label"] == label] # 현재 레이블에 해당하는 데이터 서브셋을 선택합니다.
    plt.scatter( # 산점도를 그립니다.
        subset["Radius"], 
        subset["SensorOffsetHot-Cold"], 
        c=color, 
        label=label, 
        alpha=0.7,
    )

plt.xlabel("Radius") # X축 레이블을 설정합니다.
plt.ylabel("SensorOffsetHot-Cold") # Y축 레이블을 설정합니다.
plt.title("Scatter Plot: Radius vs SensorOffsetHot-Cold (TP, FP, TN, FN)") # 플롯 제목을 설정합니다.
plt.legend() # 범례를 표시합니다.
plt.show() # 플롯을 보여줍니다.


# %%
print(np.mean(results_xgb["thresholds"])) # XGBoost 모델의 평균 최적 임계값을 출력합니다.
y_pred = (results_xgb["all_probs"]>0.2) # 모든 예측 확률이 0.2보다 큰 경우를 예측값으로 사용합니다.
y_true = results_xgb["all_true"] # 실제값을 가져옵니다.

cm_baseline_rf_test = confusion_matrix(y_true, y_pred) # 실제값과 예측값에 대한 혼동 행렬을 계산합니다.

disp_baseline_rf_test = ConfusionMatrixDisplay(confusion_matrix=cm_baseline_rf_test) # 혼동 행렬 디스플레이 객체를 생성합니다.

disp_baseline_rf_test.plot(cmap=plt.cm.Blues) # 혼동 행렬을 플로팅합니다.
plt.show() # 플롯을 보여줍니다.

# %%
import json # JSON 라이브러리를 임포트합니다.

with open("final_features.json", "r") as f: # "final_features.json" 파일을 읽기 모드로 엽니다.
    final_features_lr = json.load(f) # JSON 파일에서 데이터를 로드하여 final_features_lr 변수에 저장합니다.

# 예시: 최종 데이터셋은 preprocessed_df
preprocessed_dataset = preprocess_dataset(data) # 'data'를 전처리하는 함수를 호출하여 데이터프레임을 생성합니다.

outlier_mask = (preprocessed_dataset["Radius"] < 32) & ( # 'Radius'가 32 미만이고 'Pass/Fail'이 True인 경우를 아웃라이어로 정의하는 마스크를 생성합니다.
    preprocessed_dataset["Pass/Fail"]
)

# ~를 사용하여 마스크를 반전시켜 아웃라이어가 아닌 행만 유지합니다.
preprocessed_dataset = preprocessed_dataset[~outlier_mask].reset_index(drop=True) # 아웃라이어가 아닌 데이터를 선택하고 인덱스를 재설정합니다.
outlier = preprocessed_dataset[outlier_mask].reset_index(drop=True) # 아웃라이어 데이터를 별도로 저장합니다.

bandgap_fail = 1 - preprocessed_dataset["band gap dpat_ok for band gap"] # bandgap_fail 기준 정의
sensor_offset = preprocessed_dataset[ "SensorOffsetHot-Cold-Abs"] > 0.02 # sensor_offset 기준 정의


X_full = preprocessed_dataset.iloc[:, :-1] # 전처리된 데이터셋의 마지막 컬럼을 제외한 모든 컬럼을 특성(X_full)으로 사용합니다.
y_full = preprocessed_dataset.iloc[:,  -1] # 전처리된 데이터셋의 마지막 컬럼을 레이블(y_full)로 사용합니다.
X_full = X_full[final_features_lr] # X_full을 final_features_lr에 지정된 특성만 포함하도록 필터링합니다.

X_outlier = outlier.iloc[:, :-1] # 아웃라이어 데이터셋에서 마지막 컬럼을 제외한 모든 컬럼을 특성(X_outlier)으로 사용합니다.
X_outlier = X_outlier[final_features_lr] # X_outlier를 final_features_lr에 지정된 특성만 포함하도록 필터링합니다.

# (선택 사항) 여기에 또는 각 폴드 내에서 특성 필터링
# X_full, final_cols = variance_correlation_filter(X_full, 0.0, 0.99)
# X_full = X_full[final_cols]

# 예시: 로지스틱 회귀
model_logreg = LogisticRegression( # 로지스틱 회귀 모델을 초기화합니다.
    class_weight={0: 1, 1: sum(1 - y_full)}, # 클래스 가중치를 설정합니다.
    max_iter=1000, # 최대 반복 횟수
    random_state=42 # 재현성을 위한 난수 시드를 설정합니다.
)
model_logreg.fit(X_full, y_full) # 로지스틱 회귀 모델을 X_full, y_full로 훈련합니다.
model_logreg.predict_proba(X_outlier)[:, 1] # 훈련된 모델로 X_outlier에 대한 클래스1 예측 확률을 계산합니다.


# %%
import pandas as pd # 판다스(pandas) 라이브러리를 임포트합니다.
import numpy as np # 넘파이(numpy) 라이브러리를 임포트합니다.

from xgboost import XGBRegressor # XGBoost 회귀 모델을 임포트합니다.
from sklearn.metrics import ( # 사이킷런의 메트릭(metrics) 모듈에서 여러 평가 지표를 임포트합니다.
    confusion_matrix, classification_report, accuracy_score, fbeta_score
)
from sklearn.model_selection import train_test_split # 사이킷런의 모델 선택 모듈에서 train_test_split을 임포트합니다.

# ----------------------------------------------------------------
# 1) 데이터 읽기
# ----------------------------------------------------------------
data = pd.read_csv("data/REl data and Cp data joined dec20_gg.csv") # CSV 파일을 읽어 데이터프레임으로 로드합니다.
print("Data shape:", data.shape) # 데이터의 형상(행, 열)을 출력합니다.

# (선택 사항) 사용자 정의 전처리 함수가 있는 경우:
df_processed = preprocess_dataset(data) # 'data'를 전처리하는 함수를 호출하여 데이터프레임을 생성합니다.

# 'V_V0_SA_MAG_CAL_DIFF of PostStress'가 df_processed에 있는지 확인하거나,
# 필요한 경우 복사합니다.
df_processed["V_V0_SA_MAG_CAL_DIFF of PostStress"] = data["V_V0_SA_MAG_CAL_DIFF of PostStress"] # 원본 데이터에서 해당 컬럼을 복사합니다.

# 이제 컬럼이 있습니다:
#    - 'Pass/Fail' (0=pass, 1=fail)  [코드에서]
#    - 'V_V0_SA_MAG_CAL_DIFF of PostStress' (숫자 타겟)
#    - 회귀에 사용할 다른 특성 컬럼

# ----------------------------------------------------------------
# 2) 특성, 타겟 분리
# ----------------------------------------------------------------
# y_reg = 회귀 모델이 예측할 숫자 컬럼
y_reg = df_processed["V_V0_SA_MAG_CAL_DIFF of PostStress"] # 회귀 모델의 타겟 변수를 설정합니다.

# y_clf = 실제 pass/fail 레이블 (최종 비교용)
y_clf = df_processed["Pass/Fail"] # 분류 모델의 타겟 변수를 설정합니다.

# X = 'V_V0_SA_MAG_CAL_DIFF of PostStress' 및 'Pass/Fail'을 제외한 나머지 모든 컬럼
X = df_processed.drop(["V_V0_SA_MAG_CAL_DIFF of PostStress", "Pass/Fail"], axis=1, errors='ignore') # 타겟 컬럼을 제외한 모든 컬럼을 특성(X)으로 사용합니다.

X = X[final_features_lr] # X를 final_features_lr에 지정된 특성만 포함하도록 필터링합니다.
# ----------------------------------------------------------------
# 3) 훈련/테스트 분할
# ----------------------------------------------------------------
# 숫자 타겟과 pass/fail 레이블을 최종 평가를 위해 유지하고 싶으므로,
# "다중 출력" 분할을 수행하거나 두 단계로 수행합니다.
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split( # 데이터를 훈련 세트와 테스트 세트로 분할합니다.
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

print("Shapes => X_train:", X_train.shape, "y_reg_train:", y_reg_train.shape, "y_clf_train:", y_clf_train.shape) # 각 분할된 데이터셋의 형상을 출력합니다.

# ----------------------------------------------------------------
# 4) 회귀 모델 훈련
# ----------------------------------------------------------------
xgb_reg = XGBRegressor( # XGBoost 회귀 모델을 초기화합니다.
    n_estimators=100, # 부스팅 트리의 개수
    max_depth=3, # 각 트리의 최대 깊이
    learning_rate=0.1, # 학습률
    random_state=42 # 재현성을 위한 난수 시드
)
xgb_reg.fit(X_train, y_reg_train) # 훈련 데이터로 회귀 모델을 훈련합니다.

print("\nXGBRegressor training complete!\n") # 훈련 완료 메시지 출력

# ----------------------------------------------------------------
# 5) 연속 값 예측
# ----------------------------------------------------------------
y_pred_reg = xgb_reg.predict(X_test)  # shape=(n_test_samples,) # 테스트 세트에 대한 연속 값을 예측합니다.

# ----------------------------------------------------------------
# 6) 예측 변환 => Pass/Fail
# ----------------------------------------------------------------
# 규칙:
#   예측 < 2.48 또는 예측 > 2.52 => fail=1
#   그 외 => pass=0
#
y_pred_class = np.where((y_pred_reg < 2.495) | (y_pred_reg > 2.505), 1, 0) # 예측된 연속 값을 이진 Pass/Fail 레이블로 변환합니다.

# ----------------------------------------------------------------
# 7) 실제 Pass/Fail과 비교하여 평가
# ----------------------------------------------------------------
# 혼동 행렬 => labels=[1,0]은 행=fail, 행=pass를 보장합니다.
cm = confusion_matrix(y_clf_test, y_pred_class, labels=[1, 0]) # 실제 분류 레이블과 예측된 분류 레이블에 대한 혼동 행렬을 계산합니다.
print("Confusion Matrix (labels=[1, 0]):\n", cm) # 혼동 행렬 출력

# 분류 보고서 => 정밀도, 재현율, F1 포함
report = classification_report(y_clf_test, y_pred_class, labels=[1, 0]) # 분류 보고서를 생성합니다.
print("\nClassification Report:\n", report) # 분류 보고서 출력

# 정확도
accuracy = accuracy_score(y_clf_test, y_pred_class) # 정확도를 계산합니다.
print("Accuracy:", round(accuracy, 3)) # 정확도 출력

# 예시: F2 점수 (pos_label=1 => fail)
f2_score = fbeta_score(y_clf_test, y_pred_class, beta=2, pos_label=1) # F2 점수를 계산합니다.
print("F2 score (fail=1):", round(f2_score, 3)) # F2 점수 출력

# 재현율 및 정밀도가 별도로 필요한 경우:
# classification_report에서 구문 분석하거나 직접 계산할 수 있습니다:
# from sklearn.metrics import precision_score, recall_score
# precision = precision_score(y_clf_test, y_pred_class, pos_label=1)
# recall = recall_score(y_clf_test, y_pred_class, pos_label=1)


# %%
X_full.to_numpy(dtype=np.float32).shape # X_full을 넘파이 배열로 변환하고 형상(shape)을 출력합니다.

# %%
import torch # PyTorch 라이브러리를 임포트합니다.
import torch.nn as nn # PyTorch의 신경망 모듈을 임포트합니다.
import torch.optim as optim # PyTorch의 최적화 모듈을 임포트합니다.
import torch.nn.functional as F # PyTorch의 함수형 API 모듈을 임포트합니다.
from torch.utils.data import TensorDataset, DataLoader # PyTorch의 데이터 유틸리티에서 TensorDataset과 DataLoader를 임포트합니다.
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay # 사이킷런의 메트릭 모듈에서 혼동 행렬, 분류 보고서, 혼동 행렬 디스플레이를 임포트합니다.
from sklearn.model_selection import train_test_split # 사이킷런의 모델 선택 모듈에서 train_test_split을 임포트합니다.
import numpy as np # 넘파이(numpy) 라이브러리를 임포트합니다.
import matplotlib.pyplot as plt # Matplotlib의 pyplot 모듈을 임포트합니다.
from sklearn.preprocessing import StandardScaler # 사이킷런의 전처리 모듈에서 StandardScaler를 임포트합니다.

# 합성 데이터 또는 실제 데이터
np.random.seed(42) # 넘파이 난수 시드 설정
torch.manual_seed(42) # PyTorch 난수 시드 설정

scaler = StandardScaler() # StandardScaler 객체를 초기화합니다.

X_scaled = scaler.fit_transform(X_full) # X_full 데이터를 표준화합니다.

X = X_scaled.astype(dtype=np.float32) # 표준화된 데이터를 float32 타입으로 변환합니다.
y = y_full.to_numpy(dtype=np.float32) # y_full 데이터를 넘파이 배열로 변환하고 float32 타입으로 변환합니다.

n_features = X.shape[1] # 특성의 개수를 가져옵니다.

# --------------------------------------------------
# 1) 훈련/테스트로 분할
# --------------------------------------------------
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split( # 데이터를 훈련 세트와 테스트 세트로 분할합니다.
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Torch 텐서로 변환
X_train_tensor = torch.from_numpy(X_train_np) # 훈련 특성 데이터를 PyTorch 텐서로 변환합니다.
y_train_tensor = torch.from_numpy(y_train_np).view(-1, 1) # 훈련 레이블 데이터를 PyTorch 텐서로 변환하고 형상을 조정합니다.

X_test_tensor = torch.from_numpy(X_test_np) # 테스트 특성 데이터를 PyTorch 텐서로 변환합니다.
y_test_tensor = torch.from_numpy(y_test_np).view(-1, 1) # 테스트 레이블 데이터를 PyTorch 텐서로 변환하고 형상을 조정합니다.

train_dataset = TensorDataset(X_train_tensor, y_train_tensor) # 훈련 데이터를 TensorDataset으로 만듭니다.
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True) # 훈련 데이터로 DataLoader를 생성합니다.

test_dataset = TensorDataset(X_test_tensor, y_test_tensor) # 테스트 데이터를 TensorDataset으로 만듭니다.
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) # 테스트 데이터로 DataLoader를 생성합니다.

# --------------------------------------------------
# 2) pos_weight 계산 (neg/pos 비율 또는 선택 사항)
# --------------------------------------------------
num_pos = (y_train_np == 1).sum() # 훈련 세트에서 양성 클래스(1)의 개수를 계산합니다.
num_neg = (y_train_np == 0).sum() # 훈련 세트에서 음성 클래스(0)의 개수를 계산합니다.

# 일반적인 선택 => 비율 = num_neg / num_pos
ratio = float(num_neg) / float(num_pos) * 1000 # 양성 클래스 가중치 비율을 계산합니다.
pos_weight_tensor = torch.tensor([ratio], dtype=torch.float32) # 계산된 비율을 PyTorch 텐서로 변환합니다.

print(f"num_pos={num_pos}, num_neg={num_neg}, pos_weight={pos_weight_tensor.item():.2f}") # 양성/음성 개수 및 pos_weight 출력

# --------------------------------------------------
# 3) 모델 정의 (출력 = 로짓, 최종 시그모이드 없음)
# --------------------------------------------------
class AnomalyDetector(nn.Module): # AnomalyDetector 클래스를 정의하고 nn.Module을 상속합니다.
    def __init__(self, input_dim=10): # 생성자: 입력 차원을 인자로 받습니다.
        super().__init__() # 부모 클래스 생성자 호출
        self.fc1 = nn.Linear(input_dim, 128) # 첫 번째 완전 연결 레이어
        self.fc2 = nn.Linear(128, 32) # 두 번째 완전 연결 레이어

        self.fc3 = nn.Linear(32, 16) # 세 번째 완전 연결 레이어
        self.fc4 = nn.Linear(16, 4) # 네 번째 완전 연결 레이어
        self.fc5 = nn.Linear(4, 1) # 다섯 번째 완전 연결 레이어 (출력 레이어)

    def forward(self, x): # 순전파 메서드
        x = F.relu(self.fc1(x)) # 첫 번째 레이어에 ReLU 활성화 함수 적용
        x = F.relu(self.fc2(x)) # 두 번째 레이어에 ReLU 활성화 함수 적용
        x = F.relu(self.fc3(x)) # 세 번째 레이어에 ReLU 활성화 함수 적용
        x = F.dropout(x, p=0.2) # 드롭아웃 적용 (p=0.2)
        x = F.relu(self.fc4(x)) # 네 번째 레이어에 ReLU 활성화 함수 적용
        x = self.fc5(x) # 다섯 번째 레이어

        return x # 최종 출력 반환

model = AnomalyDetector(input_dim=n_features) # AnomalyDetector 모델을 초기화합니다.

# --------------------------------------------------
# 4) 손실 및 최적화 (BCEWithLogits + pos_weight 사용)
# --------------------------------------------------
# 함수형 API 또는 nn.Module 버전을 사용할 수 있습니다.
# 명확성을 위해 nn.Module을 사용합니다.
optimizer = optim.Adam(model.parameters(), lr=1e-3) # Adam 최적화 도구를 초기화합니다. 학습률은 1e-3입니다.

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) # BCEWithLogitsLoss를 손실 함수로 사용하고 pos_weight를 적용합니다.
scheduler = optim.lr_scheduler.ExponentialLR(gamma=0.99, optimizer=optimizer) # 학습률 스케줄러를 초기화합니다.

# --------------------------------------------------
# 5) 모델 훈련
# --------------------------------------------------
num_epochs = 10 # 에포크(epoch) 수를 설정합니다.
for epoch in range(num_epochs): # 각 에포크에 대해 반복합니다.
    model.train() # 모델을 훈련 모드로 설정합니다.
    running_loss = 0.0 # 실행 손실을 초기화합니다.
    for batch_x, batch_y in train_loader: # 훈련 데이터 로더에서 배치 데이터를 가져옵니다.
        logits = model(batch_x)                        # shape=[batch_size,1] # 모델에 입력 데이터를 전달하여 로짓(logits)을 얻습니다.
        # print(logits)

        loss = criterion(logits, batch_y)    # 로짓과 실제 레이블을 사용하여 손실을 계산합니다.

        # mask = ((batch_y != 0) | (torch.rand_like(batch_y)<0.1)).float()
        # target_w = batch_y.clone()
        # target_w[batch_y==1] = ratio
        # target_w[batch_y==0] = 0.1
        # weight = target_w * mask

        # loss = nn.functional.binary_cross_entropy_with_logits(input=logits, target=batch_y, reduction='none', weight=weight) 
        # loss = loss * mask
        # print(loss)

        # loss = loss.sum() / mask.sum()
        # loss = criterion(logits, batch_y)              # BCEWithLogitsLoss
        optimizer.zero_grad() # 옵티마이저의 기울기를 0으로 초기화합니다.
        loss.backward() # 역전파를 수행하여 기울기를 계산합니다.
        optimizer.step() # 옵티마이저를 사용하여 모델 파라미터를 업데이트합니다.
        running_loss += loss.item() # 현재 배치의 손실을 running_loss에 더합니다.

    scheduler.step() # 학습률 스케줄러를 업데이트합니다.
    epoch_loss = running_loss / len(train_dataset) # 에포크의 평균 손실을 계산합니다.
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}") # 에포크 번호와 손실을 출력합니다.

# --------------------------------------------------
# 6) 테스트 세트에서 평가
# --------------------------------------------------
model.eval() # 모델을 평가 모드로 설정합니다.
all_preds = [] # 모든 예측값을 저장할 리스트
all_labels = [] # 모든 실제 레이블을 저장할 리스트
with torch.no_grad(): # 기울기 계산을 비활성화합니다.
    for batch_x, batch_y in test_loader: # 테스트 데이터 로더에서 배치 데이터를 가져옵니다.
        logits = model(batch_x)                       # raw logits # 모델에 입력 데이터를 전달하여 로짓을 얻습니다.
        probs = torch.sigmoid(logits)                 # convert to probabilities # 로짓을 시그모이드 함수를 통해 확률로 변환합니다.
        preds = (probs >= 0.5).float()                # threshold # 확률이 0.5보다 크거나 같으면 1, 아니면 0으로 이진화합니다.
        all_preds.extend(preds.squeeze().cpu().numpy()) # 예측값을 넘파이 배열로 변환하여 all_preds에 추가합니다.
        all_labels.extend(batch_y.squeeze().cpu().numpy()) # 실제 레이블을 넘파이 배열로 변환하여 all_labels에 추가합니다.

all_preds = np.array(all_preds) # all_preds 리스트를 넘파이 배열로 변환합니다.
all_labels = np.array(all_labels) # all_labels 리스트를 넘파이 배열로 변환합니다.

cm = confusion_matrix(all_labels, all_preds, labels=[0,1]) # 실제 레이블과 예측 레이블에 대한 혼동 행렬을 계산합니다.
print("\nConfusion Matrix (labels=[0,1])\n", cm) # 혼동 행렬 출력
print("\nClassification Report (pos_label=1):\n", 
      classification_report(all_labels, all_preds, labels=[1,0])) # 분류 보고서 출력

disp_nn = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1]) # 혼동 행렬 디스플레이 객체를 생성합니다.
disp_nn.plot(cmap=plt.cm.Blues) # 혼동 행렬을 플로팅합니다.
plt.title("Confusion Matrix with pos_weight") # 플롯 제목을 설정합니다.
plt.show() # 플롯을 보여줍니다.


# %%
import numpy as np # 넘파이(numpy) 라이브러리를 임포트합니다.
import torch # PyTorch 라이브러리를 임포트합니다.
import torch.nn as nn # PyTorch의 신경망 모듈을 임포트합니다.
import torch.optim as optim # PyTorch의 최적화 모듈을 임포트합니다.
import torch.nn.functional as F # PyTorch의 함수형 API 모듈을 임포트합니다.
from torch.utils.data import TensorDataset, DataLoader # PyTorch의 데이터 유틸리티에서 TensorDataset과 DataLoader를 임포트합니다.
from sklearn.metrics import confusion_matrix, classification_report # 사이킷런의 메트릭 모듈에서 혼동 행렬, 분류 보고서를 임포트합니다.
from sklearn.model_selection import train_test_split # 사이킷런의 모델 선택 모듈에서 train_test_split을 임포트합니다.
import matplotlib.pyplot as plt # Matplotlib의 pyplot 모듈을 임포트합니다.

# -------------------------------------
# A) 합성 또는 실제 데이터셋
# -------------------------------------
# X (특성)와 y (레이블 {0,1})가 있다고 가정합니다. X의 형상 => (n_samples, n_features)

# 시연을 위해 합성 데이터셋을 생성합니다:
np.random.seed(42) # 넘파이 난수 시드 설정
torch.manual_seed(42) # PyTorch 난수 시드 설정

# n_samples = 2000
# n_features = 10

# # 1) "정상" (레이블=0)에 대한 무작위 정규 데이터 생성:
# X_normal = np.random.normal(loc=0.0, scale=1.0, size=(1800, n_features)).astype(np.float32)
# y_normal = np.zeros(1800, dtype=np.float32)

# # 2) "이상" (레이블=1)에 대한 무작위 데이터 생성:
# X_anomaly = np.random.normal(loc=5.0, scale=1.0, size=(200, n_features)).astype(np.float32)
# y_anomaly = np.ones(200, dtype=np.float32)

# # 결합
# X_all = np.vstack([X_normal, X_anomaly])
# y_all = np.concatenate([y_normal, y_anomaly])

scaler = StandardScaler() # StandardScaler 객체를 초기화합니다.

X_scaled = scaler.fit_transform(X_full) # X_full 데이터를 표준화합니다.

X_all = X_scaled.astype(dtype=np.float32) # 표준화된 데이터를 float32 타입으로 변환합니다.
y_all = y_full.to_numpy(dtype=np.float32) # y_full 데이터를 넘파이 배열로 변환하고 float32 타입으로 변환합니다.

n_features = X.shape[1] # 특성의 개수를 가져옵니다.


# 훈련/테스트 분할 (y를 기준으로 계층화하여 비율 유지)
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split( # 데이터를 훈련 세트와 테스트 세트로 분할합니다.
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# -------------------------------------
# B) AE 훈련을 위해 정상 데이터만 필터링
# -------------------------------------
normal_mask = (y_train_np == 0) # 훈련 세트에서 정상(레이블=0) 데이터만 선택하는 마스크를 생성합니다.
X_train_normal = X_train_np[normal_mask]  # shape => (# of normal samples, n_features) # 정상 데이터만 추출합니다.

print("Training set has shape:", X_train_normal.shape, "(only normal data)") # 훈련 세트의 형상과 "정상 데이터만"임을 출력합니다.

# 텐서로 변환
X_train_tensor = torch.from_numpy(X_train_normal) # 훈련 정상 데이터를 PyTorch 텐서로 변환합니다.
train_dataset = TensorDataset(X_train_tensor) # 훈련 정상 데이터를 TensorDataset으로 만듭니다.
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True) # 훈련 정상 데이터로 DataLoader를 생성합니다.

# 테스트 세트 (정상 + 이상 포함)
X_test_tensor = torch.from_numpy(X_test_np) # 테스트 특성 데이터를 PyTorch 텐서로 변환합니다.
y_test_tensor = torch.from_numpy(y_test_np) # 테스트 레이블 데이터를 PyTorch 텐서로 변환합니다.

# -------------------------------------
# C) 간단한 오토인코더 정의
# -------------------------------------
class AutoEncoder(nn.Module): # AutoEncoder 클래스를 정의하고 nn.Module을 상속합니다.
    def __init__(self, input_dim=483, latent_dim=32): # 생성자: 입력 차원과 잠재 공간 차원을 인자로 받습니다.
        super(AutoEncoder, self).__init__() # 부모 클래스 생성자 호출
        
        # ENCODER
        self.encoder = nn.Sequential( # 인코더를 시퀀셜(Sequential) 모델로 정의합니다.
            nn.Linear(input_dim, 256), # 첫 번째 선형 레이어
            nn.BatchNorm1d(256), # 배치 정규화
            nn.ReLU(True), # ReLU 활성화 함수
            nn.Dropout(0.2),          # 선택적 드롭아웃

            nn.Linear(256, 128), # 두 번째 선형 레이어
            nn.BatchNorm1d(128), # 배치 정규화
            nn.ReLU(True), # ReLU 활성화 함수
            nn.Dropout(0.2), # 선택적 드롭아웃

            nn.Linear(128, latent_dim)  # 최종 잠재 표현
        )

        # DECODER (미러와 유사한 구조)
        self.decoder = nn.Sequential( # 디코더를 시퀀셜(Sequential) 모델로 정의합니다.
            nn.Linear(latent_dim, 128), # 첫 번째 선형 레이어
            nn.BatchNorm1d(128), # 배치 정규화
            nn.ReLU(True), # ReLU 활성화 함수
            nn.Dropout(0.2), # 선택적 드롭아웃

            nn.Linear(128, 256), # 두 번째 선형 레이어
            nn.BatchNorm1d(256), # 배치 정규화
            nn.ReLU(True), # ReLU 활성화 함수
            nn.Dropout(0.2), # 선택적 드롭아웃

            nn.Linear(256, input_dim)   # 활성화 함수 없음 => 실수 값 재구성
        )

    def forward(self, x): # 순전파 메서드
        # x shape: [batch_size, 483]
        z = self.encoder(x)         # shape: [batch_size, latent_dim] # 인코더를 통해 잠재 표현을 얻습니다.
        out = self.decoder(z)       # shape: [batch_size, 483] # 디코더를 통해 재구성된 출력을 얻습니다.
        return out # 재구성된 출력 반환

model = AutoEncoder(input_dim=n_features, latent_dim=8) # AutoEncoder 모델을 초기화합니다.

# -------------------------------------
# D) 오토인코더 훈련
# -------------------------------------
criterion = nn.MSELoss()  # 재구성 오류를 위한 MSELoss를 손실 함수로 사용합니다.
optimizer = optim.Adam(model.parameters(), lr=1e-3) # Adam 최적화 도구를 초기화합니다.

num_epochs = 100 # 에포크 수를 설정합니다.
model.train() # 모델을 훈련 모드로 설정합니다.
for epoch in range(num_epochs): # 각 에포크에 대해 반복합니다.
    running_loss = 0.0 # 실행 손실을 초기화합니다.
    for (batch_x,) in train_loader:  # TensorDataset에서 단일 튜플 (batch_x,) 가져옴
        # 순전파
        batch_x = batch_x.float()  # float 타입으로 변환
        reconstructed = model(batch_x) # 모델에 입력 데이터를 전달하여 재구성된 출력을 얻습니다.
        loss = criterion(reconstructed, batch_x) # 재구성된 출력과 원본 입력을 사용하여 손실을 계산합니다.

        # 역전파
        optimizer.zero_grad() # 옵티마이저의 기울기를 0으로 초기화합니다.
        loss.backward() # 역전파를 수행하여 기울기를 계산합니다.
        optimizer.step() # 옵티마이저를 사용하여 모델 파라미터를 업데이트합니다.

        running_loss += loss.item() * len(batch_x) # 현재 배치의 손실을 running_loss에 더합니다.

    epoch_loss = running_loss / len(train_dataset) # 에포크의 평균 손실을 계산합니다.
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}") # 에포크 번호와 손실을 출력합니다.

# -------------------------------------
# E) 테스트 세트에서 평가 (정상 + 이상)
# -------------------------------------
model.eval() # 모델을 평가 모드로 설정합니다.
with torch.no_grad(): # 기울기 계산을 비활성화합니다.
    # float로 변환
    X_test_tensor_f = X_test_tensor.float() # 테스트 특성 텐서를 float 타입으로 변환합니다.
    reconstructed_test = model(X_test_tensor_f) # 모델에 테스트 특성 데이터를 전달하여 재구성된 출력을 얻습니다.
    # 재구성 오류 계산
    errors = torch.mean((reconstructed_test - X_test_tensor_f)**2, dim=1).cpu().numpy() # 재구성 오류(MSE)를 계산하고 넘파이 배열로 변환합니다.

# 테스트 세트의 모든 샘플(y_test_np)에 대한 `errors`를 가집니다.

# -------------------------------------
# F) 임계값 선택 및 이상 예측
# -------------------------------------
# 1) 한 가지 간단한 접근 방식: 훈련 정상 데이터의 재구성 오류의 95번째 백분위수와 같은 임계값 선택
#    또는 일부 도메인 지식 사용. 시연을 위해 테스트 세트의 "정상" 부분의 백분위수를 선택하거나
#    더 간단한 접근 방식을 취하여 추측합니다.

# 빠른 접근 방식:
#   훈련 정상 세트의 재구성 오류를 가져와 백분위수=95를 선택합니다.
model.eval() # 모델을 평가 모드로 설정합니다.
with torch.no_grad(): # 기울기 계산을 비활성화합니다.
    train_recon = model(torch.from_numpy(X_train_normal).float()) # 훈련 정상 데이터에 대한 재구성된 출력을 얻습니다.
    train_errors = torch.mean((train_recon - torch.from_numpy(X_train_normal).float())**2, dim=1).numpy() # 훈련 정상 데이터에 대한 재구성 오류를 계산합니다.

threshold = np.percentile(train_errors, 95)  # 예: 95번째 백분위수 # 훈련 정상 데이터의 재구성 오류의 95번째 백분위수를 임계값으로 설정합니다.
print(f"Chosen anomaly threshold: {threshold:.4f}") # 선택된 이상 임계값 출력

# 2) 오류 > 임계값인 경우 이상 예측
y_pred_anomaly = (errors > threshold).astype(np.float32)  # 1=이상, 0=정상 # 재구성 오류가 임계값보다 크면 이상(1), 아니면 정상(0)으로 예측합니다.

# -------------------------------------
# G) 혼동 행렬 및 지표
# -------------------------------------
from sklearn.metrics import confusion_matrix, classification_report # 혼동 행렬, 분류 보고서를 임포트합니다.

cm = confusion_matrix(y_test_np, y_pred_anomaly, labels=[0,1]) # 실제 레이블과 예측된 이상 레이블에 대한 혼동 행렬을 계산합니다.
print("\nConfusion Matrix (labels=[0,1]):\n", cm) # 혼동 행렬 출력
print("\nClassification Report (pos_label=1):\n",
      classification_report(y_test_np, y_pred_anomaly, labels=[1,0])) # 분류 보고서 출력

# 선택 사항: 재구성 오류 시각화
plt.hist(errors[y_test_np==0], bins=30, alpha=0.5, label="Normal") # 정상 데이터의 재구성 오류 히스토그램을 그립니다.
plt.hist(errors[y_test_np==1], bins=30, alpha=0.5, label="Anomaly") # 이상 데이터의 재구성 오류 히스토그램을 그립니다.
plt.axvline(x=threshold, color="red", linestyle="--", label="Threshold") # 임계값을 나타내는 수직선을 그립니다.
plt.title("Reconstruction Error Distribution") # 플롯 제목을 설정합니다.
plt.xlabel("Error") # X축 레이블을 설정합니다.
plt.ylabel("Frequency") # Y축 레이블을 설정합니다.
plt.legend() # 범례를 표시합니다.
plt.show() # 플롯을 보여줍니다.

disp_nn = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1]) # 혼동 행렬 디스플레이 객체를 생성합니다.
disp_nn.plot(cmap=plt.cm.Blues) # 혼동 행렬을 플로팅합니다.
plt.title("Confusion Matrix with pos_weight") # 플롯 제목을 설정합니다.
plt.show() # 플롯을 보여줍니다.


# %%
X_full # X_full 데이터프레임을 출력합니다.

# %%
from sklearn.manifold import TSNE # 사이킷런의 매니폴드(manifold) 모듈에서 TSNE를 임포트합니다.
from mpl_toolkits.mplot3d import Axes3D # Matplotlib의 3D 플로팅을 위한 Axes3D를 임포트합니다.

def tsne_scatter(features, labels, dimensions=2, save_as='graph.png'):
    if dimensions not in (2, 3): # 차원이 2 또는 3이 아닌 경우 오류를 발생시킵니다.
        raise ValueError('tsne_scatter can only plot in 2d or 3d (What are you? An alien that can visualise >3d?). Make sure the "dimensions" argument is in (2, 3)')

    # t-SNE 차원 축소
    features_embedded = TSNE(n_components=dimensions, random_state=42).fit_transform(features) # t-SNE를 사용하여 특성을 지정된 차원으로 축소합니다.
    
    # 플롯 초기화
    fig, ax = plt.subplots(figsize=(8,8)) # 플롯과 축을 생성합니다.
    
    # 차원 계산
    if dimensions == 3: ax = fig.add_subplot(111, projection='3d') # 3D 플롯인 경우 3D 축을 추가합니다.

    # 데이터 플로팅
    ax.scatter( # 레이블이 1인 데이터를 산점도로 그립니다.
        *zip(*features_embedded[np.where(labels==1)]),
        marker='o',
        color='r',
        s=2,
        alpha=0.7,
        label='Fraud'
    )
    ax.scatter( # 레이블이 0인 데이터를 산점도로 그립니다.
        *zip(*features_embedded[np.where(labels==0)]),
        marker='o',
        color='g',
        s=2,
        alpha=0.3,
        label='Clean'
    )

    # 나중에 표시하기 위해 저장
    plt.legend(loc='best') # 범례를 최적의 위치에 표시합니다.
    plt.savefig(save_as); # 플롯을 이미지 파일로 저장합니다.
    plt.show; # 플롯을 보여줍니다.

X_scaled = pd.DataFrame(scaler.fit_transform(X_full[final_features_lr])) # X_full을 표준화하고 final_features_lr에 지정된 특성만 포함하여 데이터프레임으로 변환합니다.

RATIO_TO_FRAUD = 10 # 사기(fraud) 비율에 대한 비율을 설정합니다.
# 클래스별 분할
fraud = X_scaled[y_full == 1] # y_full이 1인 (사기) 데이터를 선택합니다.
clean = X_scaled[y_full == 0] # y_full이 0인 (정상) 데이터를 선택합니다.

# 정상 거래 언더샘플링
clean_undersampled = clean.sample( # 정상 데이터를 언더샘플링합니다.
    int(len(fraud) * RATIO_TO_FRAUD),
    random_state=42
)

y_fraud = y_full[y_full == 1] # y_full이 1인 레이블을 선택합니다.
y_clean = y_full[y_full == 0] # y_full이 0인 레이블을 선택합니다.

# 정상 거래 언더샘플링
y_clean_undersampled = y_clean.sample( # 정상 레이블을 언더샘플링합니다.
    int(len(y_fraud) * RATIO_TO_FRAUD),
    random_state=42
)

# 사기 거래와 연결하여 단일 데이터프레임으로
X_undersampled = pd.concat([fraud, clean_undersampled]) # 사기 데이터와 언더샘플링된 정상 데이터를 연결합니다.
y_undersampled = pd.concat([y_fraud, y_clean_undersampled]) # 사기 레이블과 언더샘플링된 정상 레이블을 연결합니다.


tsne_scatter(X_undersampled, y_undersampled, dimensions=2, save_as='tsne_initial_2d_under.png') # 언더샘플링된 데이터에 대해 2D t-SNE 산점도를 그립니다.


# %%
import torch # PyTorch 라이브러리를 임포트합니다.
import torch.nn as nn # PyTorch의 신경망 모듈을 임포트합니다.
import torch.optim as optim # PyTorch의 최적화 모듈을 임포트합니다.
import torch.nn.functional as F # PyTorch의 함수형 API 모듈을 임포트합니다.
from torch.utils.data import DataLoader, TensorDataset # PyTorch의 데이터 유틸리티에서 DataLoader와 TensorDataset을 임포트합니다.
from sklearn.metrics import confusion_matrix # 사이킷런의 메트릭 모듈에서 혼동 행렬을 임포트합니다.
import numpy as np # 넘파이(numpy) 라이브러리를 임포트합니다.

# 간단한 합성 데이터: 1000개 샘플, 2개 특성
# x1 + x2 > 0 이면 Label=1, 아니면 0 (단순한 선)
np.random.seed(42) # 넘파이 난수 시드 설정
X = X_full.to_numpy(dtype=np.float32) # X_full을 넘파이 배열로 변환하고 float32 타입으로 변환합니다.
y = y_full.to_numpy(dtype=np.float32) # y_full을 넘파이 배열로 변환하고 float32 타입으로 변환합니다.

X_tensor = torch.from_numpy(X) # X를 PyTorch 텐서로 변환합니다.
y_tensor = torch.from_numpy(y).view(-1,1) # y를 PyTorch 텐서로 변환하고 형상을 조정합니다.

dataset = TensorDataset(X_tensor, y_tensor) # X와 y 텐서를 TensorDataset으로 만듭니다.
loader = DataLoader(dataset, batch_size=32, shuffle=True) # DataLoader를 생성합니다.

class Net(nn.Module): # Net 클래스를 정의하고 nn.Module을 상속합니다.
    def __init__(self): # 생성자
        super().__init__() # 부모 클래스 생성자 호출
        self.fc1 = nn.Linear(483, 8) # 첫 번째 완전 연결 레이어 (입력 483, 출력 8)
        self.fc2 = nn.Linear(8, 1) # 두 번째 완전 연결 레이어 (입력 8, 출력 1)
    def forward(self, x): # 순전파 메서드
        x = F.relu(self.fc1(x)) # 첫 번째 레이어에 ReLU 활성화 함수 적용
        # 최종 => Sigmoid
        x = torch.sigmoid(self.fc2(x)) # 두 번째 레이어에 시그모이드 활성화 함수 적용
        return x # 최종 출력 반환

model = Net() # Net 모델을 초기화합니다.
criterion = nn.BCELoss() # BCELoss를 손실 함수로 사용합니다.
optimizer = optim.Adam(model.parameters(), lr=1e-3) # Adam 최적화 도구를 초기화합니다.

for epoch in range(200): # 200 에포크에 대해 반복합니다.
    total_loss = 0.0 # 총 손실을 초기화합니다.
    for bx, by in loader: # 데이터 로더에서 배치 데이터를 가져옵니다.
        out = model(bx) # 모델에 입력 데이터를 전달하여 출력을 얻습니다.
        loss = criterion(out, by) # 출력과 실제 레이블을 사용하여 손실을 계산합니다.
        optimizer.zero_grad() # 옵티마이저의 기울기를 0으로 초기화합니다.
        loss.backward() # 역전파를 수행하여 기울기를 계산합니다.
        optimizer.step() # 옵티마이저를 사용하여 모델 파라미터를 업데이트합니다.
        total_loss += loss.item() * bx.size(0) # 현재 배치의 손실을 total_loss에 더합니다.
    print(f"Epoch {epoch+1} Loss={total_loss/len(dataset):.4f}") # 에포크 번호와 평균 손실을 출력합니다.

# 빠르게 평가
with torch.no_grad(): # 기울기 계산을 비활성화합니다.
    out = model(X_tensor) # 모델에 전체 X 텐서를 전달하여 출력을 얻습니다.
    preds = (out >= 0.5).float().view(-1) # 출력이 0.5보다 크거나 같으면 1, 아니면 0으로 이진화합니다.
    acc = (preds == y_tensor.view(-1)).float().mean() # 예측과 실제 레이블을 비교하여 정확도를 계산합니다.
    print("Accuracy:", acc.item()) # 정확도를 출력합니다.