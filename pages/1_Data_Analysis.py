import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import streamlit as st

# ==============================================================================
# 1. 페이지 설정
# ==============================================================================
st.set_page_config(
    page_title="임상 데이터 EDA 및 회귀 분석", page_icon="📊", layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/jimnny1230/TestDASC/main/bio.csv"
)

# ==============================================================================
# 2. 데이터 로드 및 전처리 함수
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

    # 핑거스틱 혈당 중 최댓값을 실측 최고 혈당(Target)으로 산출
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
    return df, clean_df, existing_features


raw_df, clean_df, feature_cols = load_and_prepare_data()

# ==============================================================================
# 3. 페이지 UI 구성
# ==============================================================================
st.title("📊 임상 데이터 탐색적 분석 (EDA) 및 다중회귀 검증")
st.markdown(
    "PhysioNet 임상 데이터(`bio.csv`)를 활용하여, 대사 질환 예측에 사용된 **핵심 생체 지표들의 분포와 상관관계**를 심층 분석합니다."
)
st.markdown("---")

# 1. 데이터 정제 현황 요약
st.subheader("🧹 1. 임상 데이터 정제 및 결측치 처리 현황")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("원본 데이터 샘플 수", f"{len(raw_df)} 건")
with col2:
    st.metric("유효 학습 샘플 수 (결측치 제외)", f"{len(clean_df)} 건")
with col3:
    st.metric("학습 독립변수($X$) 개수", f"{len(feature_cols)} 개")

with st.expander("🔍 원본 데이터 상위 5행 미리보기"):
    st.dataframe(raw_df.head(), use_container_width=True)

st.markdown("---")

# 2. 독립변수와 종속변수 간 상관관계 분석 (Heatmap)
st.subheader("🔥 2. 임상 생체 지표 간 상관관계 히트맵 (Correlation Heatmap)")
st.markdown(
    "공복혈당, 당화혈색소, BMI, 인슐린 지표들이 서로 및 실측 최고혈당(`Observed_Peak_GLU`)과 어떤 선형적 연관성을 가지는지 보여줍니다."
)

corr_columns = feature_cols + ["Observed_Peak_GLU"]
existing_corr_cols = [c for c in corr_columns if c in clean_df.columns]

if len(existing_corr_cols) > 1:
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = clean_df[existing_corr_cols].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    plt.title(
        "Clinical Feature Correlation Matrix", fontsize=12, fontweight="bold"
    )
    st.pyplot(fig)

st.markdown("---")

# 3. 모델 성능 및 회귀 계수 (가중치) 비교
st.subheader("📈 3. 다중선형회귀 모델 성능 및 변수별 영향력")
st.markdown(
    "각 임상 지표가 실측 최고 혈당 예측에 기여하는 통계적 가중치와 모델의 전체 설명력을 분석합니다."
)

X = clean_df[feature_cols]
y = clean_df["Observed_Peak_GLU"]
model = LinearRegression()
model.fit(X, y)
r2_score = model.score(X, y)

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric("다중회귀 결정계수 ($R^2$)", f"{r2_score:.4f}")
with col_m2:
    st.metric("모델 학습 알고리즘", "OLS (Ordinary Least Squares)")

# 회귀 계수 시각화 (막대그래프)
coef_df = pd.DataFrame(
    {"Feature": feature_cols, "Coefficient": model.coef_}
).sort_values(by="Coefficient", ascending=True)

fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
ax_bar.barh(
    coef_df["Feature"], coef_df["Coefficient"], color="teal", alpha=0.8
)
ax_bar.set_xlabel("Regression Coefficient (Impact)")
ax_bar.set_title(
    "Feature Impact on Predicted Peak Glucose", fontsize=12, fontweight="bold"
)
st.pyplot(fig_bar)

st.markdown(
    "> **해석**: 양수 값을 갖는 계수는 해당 임상 수치가 증가할수록 식후 최고 혈당 예측치가 비례해서 상승함을 의미하며, 모델의 핵심 예측 근거로 작동합니다."
)
