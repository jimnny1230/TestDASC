import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# ==============================================================================
# 1. 페이지 기본 설정
# ==============================================================================
st.set_page_config(
    page_title="스마트 바이오푸드 & 혈당 시뮬레이터",
    page_icon="🥗",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/jimnny1230/TestDASC/main/bio.csv"
)


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
        "Body weight",
        "Height",
        "A1c PDL (Lab)",
        "Fasting GLU - PDL (Lab)",
        "Insulin",
        "Triglycerides",
        "Cholesterol",
        "HDL",
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

    clean_df = df.dropna(
        subset=["Fasting GLU - PDL (Lab)", "Observed_Peak_GLU"]
    ).copy()
    return clean_df


@st.cache_resource
def train_baseline_model(clean_df):
    X = clean_df[["Fasting GLU - PDL (Lab)"]]
    y = clean_df["Observed_Peak_GLU"]
    model = LinearRegression()
    model.fit(X, y)
    r2_score = model.score(X, y)
    return model, r2_score


def predict_postprandial_glucose(
    model, fasting_glu, carbs, sugars, fiber
):
    base_pred = model.predict(np.array([[fasting_glu]]))[0]
    fiber_ratio = (fiber / carbs * 100) if carbs > 0 else 0.0
    net_carbs = max(0, carbs - fiber)
    carb_impact = (net_carbs * 0.85) + (sugars * 0.45)
    attenuation_factor = max(0.65, 1.0 - (fiber_ratio * 0.015))

    adjusted_base = (base_pred * 0.4) + (fasting_glu * 0.6)
    predicted_peak = adjusted_base + (
        carb_impact * attenuation_factor
    )
    return round(predicted_peak, 1), round(fiber_ratio, 1)


# ==============================================================================
# 2. 메인 화면 UI 구성 (사이드바 없이 본문에서 바로 조작)
# ==============================================================================
st.title("🥗 스마트 바이오푸드 & 혈당 스파이크 예측 시뮬레이터")
st.markdown(
    "PhysioNet 임상 생체 데이터 기반 대사 웰니스 분석 및 AI 시뮬레이션 시스템입니다."
)
st.markdown("---")

clean_df = load_and_prepare_data()
lr_model, r2_score = train_baseline_model(clean_df)

# 메인 화면에 입력 섹션 배치 (2단 컬럼 구조)
st.subheader("🍽️ 식단 및 생체 지표 설정")
col_in1, col_in2 = st.columns(2)

with col_in1:
    carbs = st.slider(
        "총 탄수화물 섭취량 (g)",
        min_value=10,
        max_value=150,
        value=65,
        step=5,
    )
    sugars = st.slider(
        "당류 함량 (g)", min_value=0, max_value=100, value=20, step=2
    )

with col_in2:
    fiber = st.slider(
        "식이섬유 함량 (g)", min_value=0, max_value=40, value=6, step=1
    )
    fasting_glu = st.number_input(
        "현재 공복 혈당 (mg/dL)",
        min_value=60,
        max_value=250,
        value=95,
        step=1,
    )

if sugars > carbs:
    st.warning("⚠️ 당류는 총 탄수화물량을 초과할 수 없습니다.")
    sugars = carbs

st.markdown("---")

# 예측 실행
pred_peak, fiber_ratio = predict_postprandial_glucose(
    lr_model, fasting_glu, carbs, sugars, fiber
)
glu_delta = round(pred_peak - fasting_glu, 1)

# 메트릭 표시
st.subheader("📊 실시간 대사 예측 결과")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(
        "식이섬유 비율 (Fiber Ratio)",
        f"{fiber_ratio} %",
        delta="권장 ≥ 15%" if fiber_ratio >= 15 else "부족",
    )
with c2:
    st.metric(
        "예상 식후 최고 혈당",
        f"{pred_peak} mg/dL",
        delta=f"+{glu_delta} mg/dL",
        delta_color="inverse" if pred_peak >= 140 else "normal",
    )
with c3:
    st.metric("순수 혈당 부하 탄수화물", f"{max(0, carbs - fiber)} g")
with c4:
    st.metric(
        "임상 평균 공복혈당",
        f"{clean_df['Fasting GLU - PDL (Lab)'].mean():.1f} mg/dL",
    )

st.markdown("---")

# 리스크 진단
st.subheader("🩺 3단계 대사 건강 리스크 진단")
if pred_peak < 140:
    st.success(
        f"### 🟢 대사 안정형 (예상 피크: {pred_peak} mg/dL)\n"
        "수용성 식이섬유가 소장 내 점성 하이드로겔을 형성하여 포도당 흡수 속도를 완만하게 지연시켰습니다."
    )
elif 140 <= pred_peak < 180:
    st.warning(
        f"### 🟡 주의형 (예상 피크: {pred_peak} mg/dL)\n"
        "식이섬유 비율이 다소 부족하여 식후 혈당이 빠르게 상승하며 반응성 저혈당이 우려됩니다."
    )
else:
    st.error(
        f"### 🔴 고위험 경보 (예상 피크: {pred_peak} mg/dL)\n"
        "고탄수화물 및 낮은 식이섬유로 인해 혈당 스파이크가 발생하여 췌장 세포에 부담을 줍니다."
    )
