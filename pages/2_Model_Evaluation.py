import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import streamlit as st

# ==============================================================================
# 1. 페이지 설정
# ==============================================================================
st.set_page_config(
    page_title="모델 성능 평가 및 통계적 검증", page_icon="🎯", layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/jimnny1230/TestDASC/main/bio.csv"
)


# ==============================================================================
# 2. 데이터 로드 및 정제 함수
# ==============================================================================
@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv(DATA_URL, encoding="utf-8", sep="\t")
        if len(df.columns) <= 1:
            df = pd.read_csv(DATA_URL, encoding="utf-8")
    except Exception:
        df = pd.read_csv(DATA_URL, encoding="utf-8")

    numeric_cols = [
        "Age",
        "BMI",
        "A1c PDL (Lab)",
        "Fasting GLU - PDL (Lab)",
        "Insulin",
        "#1 Contour Fingerstick GLU",
        "#2 Contour Fingerstick GLU",
        "#3 Contour Fingerstick GLU",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    finger_cols = [
        c
        for c in [
            "#1 Contour Fingerstick GLU",
            "#2 Contour Fingerstick GLU",
            "#3 Contour Fingerstick GLU",
        ]
        if c in df.columns
    ]
    if finger_cols:
        df["Observed_Peak_GLU"] = df[finger_cols].max(axis=1)
    else:
        df["Observed_Peak_GLU"] = df["Fasting GLU - PDL (Lab)"] * 1.45

    feature_cols = [
        "Fasting GLU - PDL (Lab)",
        "A1c PDL (Lab)",
        "BMI",
        "Insulin",
    ]
    existing_features = [c for c in feature_cols if c in df.columns]

    target_subset = existing_features + ["Observed_Peak_GLU"]
    clean_df = df.dropna(subset=target_subset).copy()
    return clean_df, existing_features


clean_df, feature_cols = load_and_prepare_data()

# ==============================================================================
# 3. 페이지 UI 및 모델 검증 로직
# ==============================================================================
st.title("🎯 다중선형회귀 모델 성능 평가 및 통계적 검증")
st.markdown(
    "학습된 다중선형회귀 모델의 **예측 정확도, 일반화 성능, 잔차(오차)의 통계적 타당성**을 다각도로 검증합니다."
)
st.markdown("---")

# 사용자 설정: 학습/테스트 분할 비율 조정 가능
st.subheader("⚙️ 1. 데이터 분할(Train / Test Split) 및 교차 검증 설정")
test_ratio = st.slider(
    "테스트 데이터셋(검증용) 비율 선택",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05,
    help="전체 임상 데이터 중 모델 학습에 사용하지 않고 성능 평가용으로 남겨둘 데이터 비율입니다.",
)

X = clean_df[feature_cols]
y = clean_df["Observed_Peak_GLU"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_ratio, random_state=42
)

# 모델 학습 및 예측
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 전체 데이터 기준 K-Fold 교차 검증 (과적합 여부 확인)
kf = KFold(n_splits=min(5, len(clean_df)), shuffle=True, random_state=42)
cv_r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.metric("학습 데이터 샘플 수 (Train)", f"{len(X_train)} 건")
with col_s2:
    st.metric("평가 데이터 샘플 수 (Test)", f"{len(X_test)} 건")
with col_s3:
    st.metric(
        f"{kf.get_n_splits()}-Fold 교차검증 평균 R²",
        f"{cv_r2_scores.mean():.3f}",
        help="데이터를 여러 조각으로 나누어 반복 검증한 평균 설명력입니다.",
    )

st.markdown("---")

# 2. 핵심 회귀 성능 지표 (Metrics)
st.subheader("📏 2. 핵심 회귀 평가 지표 (Test Set 기준)")
st.markdown(
    "모델이 한 번도 보지 못한 검증 데이터(`Test Set`)에 대해 얼마나 정확하게 최고 혈당을 예측하는지 나타냅니다."
)

r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(
        "결정계수 (R² Score)",
        f"{r2:.3f}",
        help="1에 가까울수록 모델이 실제 혈당 변동을 완벽히 설명함을 의미합니다.",
    )
with m2:
    st.metric(
        "평균 절대 오차 (MAE)",
        f"{mae:.2f} mg/dL",
        help="예측값과 실제 최고 혈당 간의 평균적인 차이(mg/dL)입니다.",
    )
with m3:
    st.metric(
        "평균 제곱근 오차 (RMSE)",
        f"{rmse:.2f} mg/dL",
        help="큰 오차에 페널티를 부여한 오차 지표로, 값이 작을수록 예측이 안정적입니다.",
    )
with m4:
    st.metric(
        "평균 절대 백분율 오차 (MAPE)",
        f"{mape:.2f} %",
        help="실제 혈당 수치 대비 예측 오차가 차지하는 평균 비율입니다.",
    )

st.markdown("---")

# 3. 시각화 분석: 실제값 vs 예측값 & 잔차 분석
st.subheader("📉 3. 예측 정확도 및 잔차(Residual) 진단 시각화")
col_v1, col_v2 = st.columns(2)

with col_v1:
    st.markdown("**① 실제 최고 혈당 vs 모델 예측 최고 혈당 (Actual vs Predicted)**")
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.scatter(
        y_test,
        y_test_pred,
        color="teal",
        alpha=0.75,
        edgecolors="k",
        label="Test Samples",
    )
    # 완벽한 예측 기준선 (y = x)
    min_val = min(y_test.min(), y_test_pred.min()) - 5
    max_val = max(y_test.max(), y_test_pred.max()) + 5
    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        lw=2,
        label="Ideal Prediction (y = x)",
    )
    ax1.set_xlabel("Actual Peak GLU (mg/dL)")
    ax1.set_ylabel("Predicted Peak GLU (mg/dL)")
    ax1.set_title("Actual vs Predicted Peak Glucose", fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig1)
    st.caption(
        "점들이 빨간색 점선(y=x)에 가깝게 분포할수록 모델의 예측 정확도가 높음을 의미합니다."
    )

with col_v2:
    st.markdown("**② 잔차(오차) 분포 및 등분산성 진단 (Residual Plot)**")
    residuals = y_test - y_test_pred
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(
        y_test_pred,
        residuals,
        color="coral",
        alpha=0.75,
        edgecolors="k",
    )
    ax2.axhline(y=0, color="navy", linestyle="--", lw=2, label="Zero Error Line")
    ax2.set_xlabel("Predicted Peak GLU (mg/dL)")
    ax2.set_ylabel("Residuals (Actual - Predicted)")
    ax2.set_title("Residual Analysis Plot", fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig2)
    st.caption(
        "잔차(실제값 - 예측값)가 0을 기준으로 위아래에 특정 패턴 없이 무작위로 고르게 퍼져 있을수록 선형회귀 모델의 가정이 잘 성립함을 뜻합니다."
    )

st.markdown("---")

# 4. 테스트 샘플별 상세 예측 오차 비교표
st.subheader("📋 4. 검증 샘플별 실제값 · 예측값 · 오차 상세 비교")
eval_df = X_test.copy()
eval_df["실제 최고 혈당 (Actual)"] = y_test.values
eval_df["예측 최고 혈당 (Predicted)"] = np.round(y_test_pred, 1)
eval_df["예측 오차 (Residual)"] = np.round(y_test.values - y_test_pred, 1)
eval_df["절대 오차 (|Error|)"] = np.round(np.abs(y_test.values - y_test_pred), 1)

st.dataframe(eval_df, use_container_width=True)
