import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# ==============================================================================
# 1. 페이지 기본 설정
# ==============================================================================
st.set_page_config(
    page_title="임상 데이터 기반 최고 혈당 예측 시스템",
    page_icon="🧬",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/jimnny1230/TestDASC/main/bio.csv"
)


@st.cache_data
def load_and_prepare_data():
    """
    [공학적 의미] PhysioNet 임상 데이터셋에서 다중회귀 분석에 필요한
    핵심 생체 지표(공복혈당, 당화혈색소, BMI, 인슐린, 핑거스틱 최고혈당)를 추출하고 정제합니다.
    """
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

    # [종속변수 생성] 핑거스틱 혈당 중 최댓값을 실측 최고 혈당으로 산출
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

    # 다중회귀 모델에 사용할 독립변수 후보군
    feature_cols = [
        "Fasting GLU - PDL (Lab)",
        "A1c PDL (Lab)",
        "BMI",
        "Insulin",
    ]
    existing_features = [c for c in feature_cols if c in df.columns]
    target_subset = existing_features + ["Observed_Peak_GLU"]

    # 결측치 정제
    clean_df = df.dropna(subset=target_subset).copy()
    return clean_df, existing_features


@st.cache_resource
def train_multiple_regression_model(clean_df, feature_cols):
    """
    [공학적 의미] 다중선형회귀(Multiple Linear Regression) 모델 학습 및 성능 평가 ($R^2$)
    """
    X = clean_df[feature_cols]
    y = clean_df["Observed_Peak_GLU"]

    model = LinearRegression()
    model.fit(X, y)
    r2_score = model.score(X, y)
    return model, r2_score


# ==============================================================================
# 2. 메인 화면 UI 및 시뮬레이터 구성
# ==============================================================================
st.title("🧬 임상 생체 데이터 기반 다중회귀 최고 혈당 예측 시스템")
st.markdown(
    "PhysioNet 임상 데이터(`bio.csv`)의 복합 생체 지표(공복혈당, 당화혈색소, BMI, 인슐린)를 학습한 **다중선형회귀(Multiple Linear Regression) 모델** 기반 예측 시뮬레이터입니다."
)
st.markdown("---")

clean_df, feature_cols = load_and_prepare_data()
lr_model, r2_score = train_multiple_regression_model(clean_df, feature_cols)

# 메인 화면에 입력 컨트롤 배치 (사용자 생체 검사 결과 입력)
st.subheader("📋 개인별 임상 검사 수치 입력 (Model Input)")
st.markdown(
    "본인의 건강 검진 또는 임상 검사 데이터를 입력하여 예상되는 식후 최고 혈당을 예측해 보세요."
)

col_in1, col_in2 = st.columns(2)

with col_in1:
    fasting_glu = st.number_input(
        "공복 혈당 (Fasting GLU, mg/dL)",
        min_value=60.0,
        max_value=250.0,
        value=float(clean_df["Fasting GLU - PDL (Lab)"].mean()),
        step=1.0,
    )
    a1c = st.number_input(
        "당화혈색소 (A1c, %)",
        min_value=4.0,
        max_value=15.0,
        value=float(clean_df["A1c PDL (Lab)"].mean()),
        step=0.1,
    )

with col_in2:
    bmi = st.number_input(
        "체질량지수 (BMI, kg/m²)",
        min_value=10.0,
        max_value=50.0,
        value=float(clean_df["BMI"].mean()),
        step=0.1,
    )
    insulin = st.number_input(
        "공복 인슐린 (Insulin)",
        min_value=1.0,
        max_value=100.0,
        value=float(clean_df["Insulin"].mean()),
        step=0.5,
    )

st.markdown("---")

# 모델 예측 실행
input_data = pd.DataFrame(
    [[fasting_glu, a1c, bmi, insulin]], columns=feature_cols
)
pred_peak = lr_model.predict(input_data)[0]
pred_peak = round(pred_peak, 1)
glu_delta = round(pred_peak - fasting_glu, 1)

# 예측 결과 대시보드
st.subheader("📊 100% 데이터 기반 대사 예측 결과")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(
        "예상 식후 최고 혈당 (Peak)",
        f"{pred_peak} mg/dL",
        delta=f"+{glu_delta} mg/dL 상승",
        delta_color="inverse" if pred_peak >= 140 else "normal",
    )
with c2:
    st.metric(
        "모델 결정계수 ($R^2$ Score)",
        f"{r2_score:.3f}",
        help="모델의 설명력입니다. 1에 가까울수록 데이터 설명력이 높습니다.",
    )
with c3:
    st.metric("학습 임상 샘플 수", f"{len(clean_df)} 건")
with c4:
    st.metric(
        "데이터셋 평균 당화혈색소",
        f"{clean_df['A1c PDL (Lab)'].mean():.2f} %",
    )

st.markdown("---")

# 모델 회귀 계수 (가중치) 투명성 공개
st.subheader("🔍 머신러닝 학습 가중치 (Feature Coefficients)")
st.markdown(
    "데이터가 각 생체 지표를 어떻게 반영하여 혈당을 예측하는지 회귀 계수를 통해 확인할 수 있습니다."
)

coef_df = pd.DataFrame(
    {"임상 생체 변수": feature_cols, "회귀 가중치 (Coefficient)": lr_model.coef_}
)
st.dataframe(coef_df, use_container_width=True)
