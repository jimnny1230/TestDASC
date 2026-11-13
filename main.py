from io import BytesIO
import urllib.request
from fpdf import FPDF
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

# ==============================================================================
# 2. 필수 의료 면책 동의 세션 및 팝업 대화상자 구현
# ==============================================================================
if "agreed_medical_disclaimer" not in st.session_state:
    st.session_state.agreed_medical_disclaimer = False


# '예'를 눌러야만 해제되는 모달 다이아몬드 창 정의
@st.dialog("⚠️ 필수 의료 법적 고지 및 면책 조항", width="large")
def show_medical_disclaimer_modal():
    st.markdown("### **반드시 읽고 동의하신 후 진행해 주세요.**")
    st.markdown(
        """
        <div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; border-left: 6px solid #ffc107; color: #856404;">
            <h3 style="margin-top:0; color: #856404;"><b>[면책 공고 및 이용 안내]</b></h3>
            <p style="font-size: 16px; font-weight: bold; line-height: 1.6;">
                본 프로그램은 PhysioNet 임상 데이터를 학습한 학술 연구 및 교육용 머신러닝 시뮬레이터입니다.<br>
                제공되는 예측 결과와 대사 리스크 진단은 통계적 추정치일 뿐이며, <b>어떠한 경우에도 의사의 전문적인 진단, 치료, 의료적 자문을 대체할 수 없습니다.</b>
            </p>
            <p style="font-size: 15px; margin-bottom: 0;">
                정확한 건강 상태 진단이나 의학적인 자문이 필요한 경우, 반드시 전문 의료기관을 방문하여 전문가와 상담하시기 바랍니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.write(
        "위 내용을 모두 숙지하였으며, 참고용 시뮬레이터로만 활용하는 것에 동의하십니까?"
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("✅ [예] 동의하고 프로그램 실행", use_container_width=True):
            st.session_state.agreed_medical_disclaimer = True
            st.rerun()
    with col_btn2:
        if st.button(
            "❌ [아오오/아니오] 거부 (프로그램 종료)", use_container_width=True
        ):
            st.error(
                "동의하지 않으시면 프로그램을 이용하실 수 없습니다. 창을 닫아주세요."
            )
            st.stop()


# 사용자가 동의하지 않았다면 무조건 모달 창을 띄우고 앱 실행 중단
if not st.session_state.agreed_medical_disclaimer:
    show_medical_disclaimer_modal()
    st.stop()  # '예'를 누르기 전에는 아래의 메인 코드가 절대 실행되지 않음

# ==============================================================================
# 3. 데이터 로드 및 전처리 함수
# ==============================================================================
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


def create_pdf_report(
    user_inputs,
    pred_peak,
    glu_delta,
    r2_score,
    fasting_diagnosis,
    a1c_diagnosis,
    spike_diagnosis,
    risk_tier_text,
    pred_peak_after,
    improvement_diff,
):
    pdf = FPDF()
    pdf.add_page()

    font_url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
    font_path = "/tmp/NanumGothic-Regular.ttf"
    try:
        urllib.request.urlretrieve(font_url, font_path)
        pdf.add_font("NanumGothic", "", font_path)
        pdf.set_font("NanumGothic", "", 12)
    except Exception:
        pdf.set_font("Helvetica", "", 12)

    pdf.set_font_size(15)
    pdf.cell(
        0, 10, "[ 임상 데이터 기반 대사 건강 예측 리포트 ]", ln=True, align="C"
    )
    pdf.ln(4)

    pdf.set_font_size(12)
    pdf.cell(0, 8, "1. 입력된 임상 검사 수치", ln=True)
    pdf.set_font_size(10)
    pdf.cell(
        0,
        6,
        f"- 공복 혈당 (Fasting GLU): {user_inputs.get('Fasting GLU - PDL (Lab)', 'N/A')} mg/dL",
        ln=True,
    )
    pdf.cell(
        0,
        6,
        f"- 당화혈색소 (A1c): {user_inputs.get('A1c PDL (Lab)', 'N/A')} %",
        ln=True,
    )
    pdf.cell(
        0,
        6,
        f"- 체질량지수 (BMI): {user_inputs.get('BMI', 'N/A')} kg/m2",
        ln=True,
    )
    pdf.cell(
        0,
        6,
        f"- 공복 인슐린 (Insulin): {user_inputs.get('Insulin', 'N/A')}",
        ln=True,
    )
    pdf.ln(3)

    pdf.set_font_size(12)
    pdf.cell(0, 8, "2. 예측 및 임상 진단 결과", ln=True)
    pdf.set_font_size(10)
    pdf.cell(
        0,
        6,
        f"- 예상 식후 최고 혈당 (Peak): {pred_peak} mg/dL (+{glu_delta} mg/dL 상승)",
        ln=True,
    )
    pdf.cell(0, 6, f"- 공복혈당 진단: {fasting_diagnosis}", ln=True)
    pdf.cell(0, 6, f"- 당화혈색소 진단: {a1c_diagnosis}", ln=True)
    pdf.cell(0, 6, f"- 혈당 스파이크 판정: {spike_diagnosis}", ln=True)
    pdf.cell(0, 6, f"- 3단계 대사 리스크 분류: {risk_tier_text}", ln=True)
    pdf.cell(
        0, 6, f"- 모델 설명력 (R2 Score): {r2_score:.3f} (Accuracy)", ln=True
    )
    pdf.ln(3)

    pdf.set_font_size(12)
    pdf.cell(0, 8, "3. What-if 개선 시뮬레이션 결과", ln=True)
    pdf.set_font_size(10)
    pdf.cell(
        0,
        6,
        f"- 목표 감량/개선 적용 시 예상 최고 혈당: {pred_peak_after} mg/dL (약 {improvement_diff} mg/dL 개선 효과)",
        ln=True,
    )
    pdf.ln(6)

    pdf.set_font_size(8)
    pdf.multi_cell(
        0,
        5,
        "안내: 본 리포트는 PhysioNet 임상 데이터(bio.csv)를 학습한 다중선형회귀 모델을 기반으로 자동 생성되었습니다. 참고용으로만 사용하시기 바랍니다.",
    )

    return bytes(pdf.output())


# ==============================================================================
# 4. 메인 화면 UI 및 시뮬레이터 구성
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

input_df = pd.DataFrame([user_inputs])
pred_peak = lr_model.predict(input_df)[0]
pred_peak = round(pred_peak, 1)

base_glu = user_inputs.get("Fasting GLU - PDL (Lab)", 100.0)
glu_delta = round(pred_peak - base_glu, 1)

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
    risk_tier_text = "1. 대사 안정형 (Normal / < 140 mg/dL)"
    st.success(
        f"### 🟢 1. 대사 안정형 (Normal / < 140 mg/dL) — (현재 예측치: {pred_peak} mg/dL)\n"
        "- **기준**: 예상 식후 최고 혈당이 140 mg/dL 미만인 경우\n"
        "- **의미**: 탄수화물 섭취 후에도 인슐린 분비와 포도당 대사가 정상적으로 이루어져 혈당이 안정적으로 유지되는 상태입니다."
    )
elif 140 <= pred_peak < 180:
    risk_tier_text = "2. 주의형 (Pre-diabetic Range / 140 ~ 179 mg/dL)"
    st.warning(
        f"### 🟡 2. 주의형 (Pre-diabetic Range / 140 ~ 179 mg/dL) — (현재 예측치: {pred_peak} mg/dL)\n"
        "- **기준**: 예상 식후 최고 혈당이 140 mg/dL 이상 180 mg/dL 미만인 경우\n"
        "- **의미**: 내당능 장애 초기 단계나 인슐린 저항성이 다소 진행된 상태를 시사합니다."
    )
else:
    risk_tier_text = "3. 고위험 경보형 (High Risk / ≥ 180 mg/dL)"
    st.error(
        f"### 🔴 3. 고위험 경보형 (High Risk / ≥ 180 mg/dL) — (현재 예측치: {pred_peak} mg/dL)\n"
        "- **기준**: 예상 식후 최고 혈당이 180 mg/dL 이상인 경우\n"
        "- **의미**: 뚜렷한 혈당 스파이크가 발생하는 고위험 상태입니다."
    )

st.markdown("---")
st.subheader("🔮 [What-if 시뮬레이터] 생활습관 개선에 따른 혈당 변화 예측")
col_w1, col_w2 = st.columns(2)
with col_w1:
    bmi_reduction = st.slider(
        "체질량지수(BMI) 감량 목표 (kg/m²)", 0.0, 10.0, 1.0, 0.5
    )
with col_w2:
    fasting_reduction = st.slider(
        "공복 혈당 개선 목표 (mg/dL)", 0.0, 50.0, 10.0, 1.0
    )

what_if_inputs = user_inputs.copy()
if "BMI" in what_if_inputs:
    what_if_inputs["BMI"] = max(10.0, what_if_inputs["BMI"] - bmi_reduction)
if "Fasting GLU - PDL (Lab)" in what_if_inputs:
    what_if_inputs["Fasting GLU - PDL (Lab)"] = max(
        60.0, what_if_inputs["Fasting GLU - PDL (Lab)"] - fasting_reduction
    )

what_if_df = pd.DataFrame([what_if_inputs])
pred_peak_after = round(lr_model.predict(what_if_df)[0], 1)
improvement_diff = round(pred_peak - pred_peak_after, 1)

wc1, wc2, wc3 = st.columns(3)
with wc1:
    st.metric("현재 예상 최고 혈당", f"{pred_peak} mg/dL")
with wc2:
    st.metric(
        "개선 후 예상 최고 혈당",
        f"{pred_peak_after} mg/dL",
        delta=f"-{improvement_diff} mg/dL 감소",
        delta_color="normal",
    )
with wc3:
    st.metric("개선 효과 진단", "개선 성공적 ✨" if improvement_diff > 0 else "유지 중")

st.markdown("---")
st.subheader("🌐 [360° 인터랙티브 3D 모델] 다중선형회귀 초평면 시각화")
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
    fig.add_trace(
        go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode="markers",
            name="실제 임상 데이터",
            marker=dict(size=5, color=z_data, colorscale="Viridis", opacity=0.8),
        )
    )
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
    user_x = user_inputs.get(x_col, x_data.mean())
    user_y = user_inputs.get(y_col, y_data.mean())
    fig.add_trace(
        go.Scatter3d(
            x=[user_x],
            y=[user_y],
            z=[pred_peak],
            mode="markers+text",
            name="내 예측 위치",
            text=[f"예측: {pred_peak} mg/dL"],
            textposition="top center",
            marker=dict(size=10, color="red", symbol="diamond"),
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
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("🔍 머신러닝 학습 가중치 (Feature Coefficients)")
coef_df = pd.DataFrame(
    {"임상 생체 변수": feature_cols, "회귀 가중치 (Coefficient)": lr_model.coef_}
)
st.dataframe(coef_df, use_container_width=True)

st.markdown("---")
st.subheader("📥 개인별 건강 진단 및 예측 리포트 다운로드")
st.markdown(
    "현재 입력한 검사 수치와 예측 결과를 바탕으로 작성된 리포트 파일을 **PDF 또는 TXT** 형식으로 다운로드할 수 있습니다."
)

txt_report_content = f"""============================================================
[임상 데이터 기반 대사 건강 예측 리포트]
============================================================

1. 입력된 임상 검사 수치
- 공복 혈당 (Fasting GLU): {user_inputs.get('Fasting GLU - PDL (Lab)', 'N/A')} mg/dL
- 당화혈색소 (A1c): {user_inputs.get('A1c PDL (Lab)', 'N/A')} %
- 체질량지수 (BMI): {user_inputs.get('BMI', 'N/A')} kg/m²
- 공복 인슐린 (Insulin): {user_inputs.get('Insulin', 'N/A')}

2. 예측 및 임상 진단 결과
- 예상 식후 최고 혈당 (Peak): {pred_peak} mg/dL (+{glu_delta} mg/dL 상승)
- 공복혈당 진단: {fasting_diagnosis}
- 당화혈색소 진단: {a1c_diagnosis}
- 혈당 스파이크 판정: {spike_diagnosis}
- 3단계 대사 리스크 분류: {risk_tier_text}
- 모델 설명력 (R2 Score): {r2_score:.3f}

3. What-if 개선 시뮬레이션 결과
- 목표 감량/개선 적용 시 예상 최고 혈당: {pred_peak_after} mg/dL (약 {improvement_diff} mg/dL 개선 효과)

------------------------------------------------------------
* 본 리포트는 참고용으로만 사용하시기 바랍니다. 
  의학적인 자문이나 진단이 필요한 경우 전문가에게 문의하세요.
============================================================
"""

pdf_bytes = create_pdf_report(
    user_inputs=user_inputs,
    pred_peak=pred_peak,
    glu_delta=glu_delta,
    r2_score=r2_score,
    fasting_diagnosis=fasting_diagnosis,
    a1c_diagnosis=a1c_diagnosis,
    spike_diagnosis=spike_diagnosis,
    risk_tier_text=risk_tier_text,
    pred_peak_after=pred_peak_after,
    improvement_diff=improvement_diff,
)

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    st.download_button(
        label="📕 PDF 리포트 다운로드 (.pdf)",
        data=pdf_bytes,
        file_name="health_prediction_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
with btn_col2:
    st.download_button(
        label="📄 텍스트 리포트 다운로드 (.txt)",
        data=txt_report_content,
        file_name="health_prediction_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
