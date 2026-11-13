import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


@st.cache_resource
def train_multiple_regression_model(clean_df, feature_cols):
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
    "PhysioNet 임상 데이터(`bio.csv`)의 핵심 생체 지표(공복혈당, 당화혈색소, BMI, 인슐린)를 학습한 **다중선형회귀(Multiple Linear Regression)** 모델 기반 예측 시뮬레이터입니다."
)
st.markdown("---")

clean_df, feature_cols = load_and_prepare_data()
lr_model, r2_score = train_multiple_regression_model(clean_df, feature_cols)

st.subheader("📋 개인별 임상 검사 수치 입력 (Model Input)")
st.markdown(
    "본인의 건강 검진 또는 임상 검사 데이터를 입력하여 예상되는 식후 최고 혈당과 임상 진단 결과를 확인해 보세요."
)

col_in1, col_in2 = st.columns(2)
user_inputs = {}

with col_in1:
    if "Fasting GLU - PDL (Lab)" in clean_df.columns:
        user_inputs["Fasting GLU - PDL (Lab)"] = st.number_input(
            "공복 혈당 (Fasting GLU, mg/dL)",
            min_value=60.0,
            max_value=250.0,
            value=float(clean_df["Fasting GLU - PDL (Lab)"].mean()),
            step=1.0,
        )
    if "A1c PDL (Lab)" in clean_df.columns:
        user_inputs["A1c PDL (Lab)"] = st.number_input(
            "당화혈색소 (A1c, %)",
            min_value=4.0,
            max_value=15.0,
            value=float(clean_df["A1c PDL (Lab)"].mean()),
            step=0.1,
        )

with col_in2:
    if "BMI" in clean_df.columns:
        user_inputs["BMI"] = st.number_input(
            "체질량지수 (BMI, kg/m²)",
            min_value=10.0,
            max_value=50.0,
            value=float(clean_df["BMI"].mean()),
            step=0.1,
        )
    if "Insulin" in clean_df.columns:
        user_inputs["Insulin"] = st.number_input(
            "공복 인슐린 (Insulin)",
            min_value=1.0,
            max_value=100.0,
            value=float(clean_df["Insulin"].mean()),
            step=0.5,
        )

st.markdown("---")

# 모델 예측 실행
input_df = pd.DataFrame([user_inputs])
pred_peak = lr_model.predict(input_df)[0]
pred_peak = round(pred_peak, 1)

base_glu = user_inputs.get("Fasting GLU - PDL (Lab)", 100.0)
glu_delta = round(pred_peak - base_glu, 1)

# ==============================================================================
# 3. 임상 기준별 진단 로직 수행
# ==============================================================================
fasting_val = user_inputs.get("Fasting GLU - PDL (Lab)", 100.0)
a1c_val = user_inputs.get("A1c PDL (Lab)", 5.5)

if fasting_val >= 126:
    fasting_diagnosis = (
        "🔴 당뇨병 의심 (공복혈당 126 mg/dL 이상 기준 충족)"
    )
elif 100 <= fasting_val <= 125:
    fasting_diagnosis = (
        "🟡 당뇨병 전 단계 (공복혈당 100~125 mg/dL 기준 충족)"
    )
else:
    fasting_diagnosis = "🟢 정상 공복혈당 (100 mg/dL 미만)"

if a1c_val >= 6.5:
    a1c_diagnosis = "🔴 당뇨병 의심 (당화혈색소 6.5% 이상 기준 충족)"
else:
    a1c_diagnosis = "🟢 정상 범위 (당화혈색소 6.5% 미만)"

if glu_delta >= 50 or pred_peak >= 150:
    spike_diagnosis = (
        "⚠️ 혈당 스파이크 의심 (식후-공복 차이 ≥ 50 mg/dL 또는 식후 혈당 ≥ 150 mg/dL)"
    )
else:
    spike_diagnosis = "✅ 안정적인 혈당 변동 범위"


# ==============================================================================
# 4. 결과 대시보드 출력
# ==============================================================================
st.subheader("📊 100% 데이터 기반 대사 예측 및 임상 진단 결과")
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
    if "A1c PDL (Lab)" in clean_df.columns:
        st.metric(
            "데이터셋 평균 당화혈색소",
            f"{clean_df['A1c PDL (Lab)'].mean():.2f} %",
        )

st.markdown("---")

st.subheader("🩺 종합 임상 리스크 및 당뇨 진단 결과")
col_d1, col_d2, col_d3 = st.columns(3)
with col_d1:
    st.info(f"**[공복혈당 기반 진단]**\n\n{fasting_diagnosis}")
with col_d2:
    st.success(f"**[당화혈색소(A1c) 기반 진단]**\n\n{a1c_diagnosis}")
with col_d3:
    st.warning(f"**[혈당 스파이크 판정]**\n\n{spike_diagnosis}")

st.markdown("---")

st.subheader("📈 [식후 최고 혈당 기준 3단계 대사 리스크 진단]")
if pred_peak < 140:
    st.success(
        f"### 🟢 1. 대사 안정형 (Normal / < 140 mg/dL) — (현재 예측치: {pred_peak} mg/dL)\n"
        "- **기준**: 예상 식후 최고 혈당이 140 mg/dL 미만인 경우\n"
        "- **의미**: 탄수화물 섭취 후에도 인슐린 분비와 포도당 대사가 정상적으로 이루어져 혈당이 안정적으로 유지되는 상태입니다. 췌장에 부담이 적고 대사 건강이 양호함을 나타냅니다."
    )
elif 140 <= pred_peak < 180:
    st.warning(
        f"### 🟡 2. 주의형 (Pre-diabetic Range / 140 ~ 179 mg/dL) — (현재 예측치: {pred_peak} mg/dL)\n"
        "- **기준**: 예상 식후 최고 혈당이 140 mg/dL 이상 180 mg/dL 미만인 경우\n"
        "- **의미**: 내당능 장애(Impaired Glucose Tolerance) 초기 단계나 인슐린 저항성이 다소 진행된 상태를 시사합니다. 식후 혈당이 빠르게 상승하며, 향후 식후 반응성 저혈당이나 만성 대사 질환으로 발전할 수 있어 식단 및 운동 관리가 권장됩니다."
    )
else:
    st.error(
        f"### 🔴 3. 고위험 경보형 (High Risk / ≥ 180 mg/dL) — (현재 예측치: {pred_peak} mg/dL)\n"
        "- **기준**: 예상 식후 최고 혈당이 180 mg/dL 이상인 경우\n"
        "- **의미**: 인슐린 분비능이 저하되었거나 심각한 인슐린 저항성이 존재하여 뚜렷한 혈당 스파이크가 발생하는 고위험 상태입니다. 반복될 경우 혈관 내피세포 손상 및 췌장 베타세포에 과도한 부담을 주므로 전문적인 진료와 적극적인 생활습관 교정이 필요합니다."
    )

st.markdown("---")

# ==============================================================================
# 5. 360도 회전 가능한 인터랙티브 3D 회귀 평면 (Plotly 3D)
# ==============================================================================
st.subheader("🌐 [360° 인터랙티브 3D 모델] 다중선형회귀 초평면 시각화")
st.markdown(
    "마우스로 그래프를 **드래그하면 360도 자유롭게 회전**할 수 있으며, 스크롤로 확대/축소할 수 있습니다. "
    "**빨간색 다이아몬드 점**은 현재 사용자가 입력한 수치의 3차원 공간상 예측 위치를 나타냅니다."
)

axis_option = st.selectbox(
    "📊 3차원 공간 축 조합 선택 (Z축: 예상 최고 혈당)",
    [
        "1) 공복 혈당 (X) × 당화혈색소 (Y)",
        "2) 공복 혈당 (X) × 체질량지수 BMI (Y)",
        "3) 당화혈색소 (X) × 체질량지수 BMI (Y)",
    ],
)

if "1)" in axis_option:
    x_col, y_col = "Fasting GLU - PDL (Lab)", "A1c PDL (Lab)"
    x_label, y_label = "공복 혈당 (mg/dL)", "당화혈색소 (%)"
elif "2)" in axis_option:
    x_col, y_col = "Fasting GLU - PDL (Lab)", "BMI"
    x_label, y_label = "공복 혈당 (mg/dL)", "BMI (kg/m²)"
else:
    x_col, y_col = "A1c PDL (Lab)", "BMI"
    x_label, y_label = "당화혈색소 (%)", "BMI (kg/m²)"

if x_col in clean_df.columns and y_col in clean_df.columns:
    x_data = clean_df[x_col]
    y_data = clean_df[y_col]
    z_data = clean_df["Observed_Peak_GLU"]

    # 회귀 평면 격자 생성
    x_surf = np.linspace(x_data.min(), x_data.max(), 25)
    y_surf = np.linspace(y_data.min(), y_data.max(), 25)
    xx, yy = np.meshgrid(x_surf, y_surf)

    grid_dict = {}
    for col in feature_cols:
        if col == x_col:
            grid_dict[col] = xx.ravel()
        elif col == y_col:
            grid_dict[col] = yy.ravel()
        else:
            grid_dict[col] = [
                user_inputs.get(col, clean_df[col].mean())
            ] * xx.size

    grid_df = pd.DataFrame(grid_dict)[feature_cols]
    zz = lr_model.predict(grid_df).reshape(xx.shape)

    fig = go.Figure()

    # 1. 실제 임상 데이터 산점도 (3D Scatter)
    fig.add_trace(
        go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode="markers",
            name="실제 임상 데이터 (Samples)",
            marker=dict(
                size=5,
                color=z_data,
                colorscale="Viridis",
                opacity=0.8,
            ),
        )
    )

    # 2. 머신러닝 학습 회귀 평면 (3D Surface)
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            name="다중회귀 예측 평면",
            colorscale="Oranges",
            opacity=0.5,
            showscale=False,
        )
    )

    # 3. 현재 사용자 입력 위치 강조 표시 (빨간 다이아몬드)
    user_x = user_inputs.get(x_col, x_data.mean())
    user_y = user_inputs.get(y_col, y_data.mean())
    fig.add_trace(
        go.Scatter3d(
            x=[user_x],
            y=[user_y],
            z=[pred_peak],
            mode="markers+text",
            name="내 예측 위치 (Current User)",
            text=[f"예측: {pred_peak} mg/dL"],
            textposition="top center",
            marker=dict(
                size=10,
                color="red",
                symbol="diamond",
            ),
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title="예상 최고 혈당 (mg/dL)",
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05),
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 모델 회귀 계수 (가중치) 투명성 공개
st.subheader("🔍 머신러닝 학습 가중치 (Feature Coefficients)")
coef_df = pd.DataFrame(
    {"임상 생체 변수": feature_cols, "회귀 가중치 (Coefficient)": lr_model.coef_}
)
st.dataframe(coef_df, use_container_width=True)
