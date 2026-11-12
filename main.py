import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# ==============================================================================
# 1. 페이지 기본 설정 (UI/UX 최적화)
# ==============================================================================
st.set_page_config(
    page_title="스마트 바이오푸드 & 혈당 스파이크 예측 시뮬레이터",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# 2. 임상 데이터 로드 및 전처리 엔진 (Engineering)
# ==============================================================================
DATA_URL = (
    "https://raw.githubusercontent.com/jimnny1230/TestDASC/main/bio.csv"
)


@st.cache_data
def load_and_prepare_data():
    """
    [공학적 의미] GitHub의 raw 데이터를 UTF-8로 불러오고,
    클리닝 및 파생 변수(실측 식후 최고 혈당)를 생성합니다.
    """
    try:
        df = pd.read_csv(DATA_URL, encoding="utf-8", sep="\t")
        if len(df.columns) <= 1:
            df = pd.read_csv(DATA_URL, encoding="utf-8")
    except Exception:
        df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 공복 혈당 및 1~3차 핑거스틱 혈당 수치 컬럼 추출
    glu_cols = [
        "Fasting GLU - PDL (Lab)",
        "#1 Contour Fingerstick GLU",
        "#2 Contour Fingerstick GLU",
        "#3 Contour Fingerstick GLU",
    ]
    existing_glu_cols = [c for c in glu_cols if c in df.columns]

    # 수치형 변환 및 결측치 처리
    for col in existing_glu_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "BMI" in df.columns:
        df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    if "A1c PDL (Lab)" in df.columns:
        df["A1c PDL (Lab)"] = pd.to_numeric(
            df["A1c PDL (Lab)"], errors="coerce"
        )

    # [생명과학적 의미] 개인별 측정된 핑거스틱 혈당 중 최댓값을 식후 최고 혈당(Peak Glucose) 대리 지표로 산출
    finger_cols = [
        c for c in existing_glu_cols if "Fingerstick" in c
    ]
    if finger_cols:
        df["Observed_Peak_GLU"] = df[finger_cols].max(axis=1)
    else:
        df["Observed_Peak_GLU"] = df["Fasting GLU - PDL (Lab)"] * 1.45

    return df


@st.cache_resource
def train_baseline_model(df):
    """
    [공학적 의미] 실제 임상 데이터(bio.csv)의 공복 혈당(X)과 최고 혈당(Y) 간의
    선형회귀(Linear Regression) 모델을 학습시키고, 모델 설명력(R2 Score)을 평가합니다.
    """
    clean_df = df.dropna(
        subset=["Fasting GLU - PDL (Lab)", "Observed_Peak_GLU"]
    )

    X = clean_df[["Fasting GLU - PDL (Lab)"]]
    y = clean_df["Observed_Peak_GLU"]

    model = LinearRegression()
    model.fit(X, y)
    r2_score = model.score(X, y)  # 결정계수(R-squared) 평가 지표 산출
    return model, r2_score, len(clean_df), clean_df


# ==============================================================================
# 3. 생리학적 혈당 피크 시뮬레이션 및 리스크 진단 엔진 (Bio-Simulation)
# ==============================================================================
def predict_postprandial_glucose(
    model, fasting_glu, carbs, sugars, fiber
):
    """
    [생명과학적 + 공학적 융합 모델]
    1) 선형회귀 모델로 공복 혈당 기반 기초 상승치를 예측합니다.
    2) 탄수화물과 당류 섭취량에 따른 혈당 상승 부하(Glycemic Load)를 반영합니다.
    3) 수용성 식이섬유 비율(Fiber Ratio)이 높을수록 소장 내 점성 하이드로겔을 형성하여
       포도당 확산 속도를 늦추고 혈당 피크를 감쇠시키는 생리학적 효과를 수식화합니다.
    """
    # 1. 기초 임상 회귀 예측치
    base_pred = model.predict(np.array([[fasting_glu]]))[0]

    # 2. 파생 변수: 식이섬유 비율 (%) 계산
    fiber_ratio = (fiber / carbs * 100) if carbs > 0 else 0.0

    # 3. 영양소 섭취에 따른 순수 혈당 상승폭 계산
    net_carbs = max(0, carbs - fiber)
    carb_impact = (net_carbs * 0.85) + (sugars * 0.45)

    # 4. 식이섬유 비율(Fiber Ratio)에 따른 혈당 피크 억제 계수 (점성 겔화 효과)
    attenuation_factor = max(0.65, 1.0 - (fiber_ratio * 0.015))

    # 5. 최종 예상 식후 최고 혈당 산출
    adjusted_base = (base_pred * 0.4) + (fasting_glu * 0.6)
    predicted_peak = adjusted_base + (
        carb_impact * attenuation_factor
    )

    return round(predicted_peak, 1), round(fiber_ratio, 1)


# ==============================================================================
# 4. 메인 애플리케이션 실행부
# ==============================================================================
st.title("🥗 스마트 바이오푸드 & 혈당 스파이크 예측 시뮬레이터")
st.markdown(
    "**식이섬유 비율(Fiber Ratio)**이 소장 내 포도당 흡수 속도와 식후 혈당 피크(Peak Glucose)에 미치는 영향을 분석하는 대사 웰니스 시뮬레이터입니다."
)
st.markdown("---")

# 데이터 로드 및 모델 학습 및 평가
try:
    df = load_and_prepare_data()
    lr_model, r2_score, sample_count, clean_df = train_baseline_model(df)
    is_ready = True
except Exception as e:
    st.error(f"데이터 로드 및 모델 학습 중 오류가 발생했습니다: {e}")
    is_ready = False

if is_ready:
    # --------------------------------------------------------------------------
    # [사이드바] 사용자 식단 및 기초 생체 지표 입력 패널
    # --------------------------------------------------------------------------
    st.sidebar.header("🍽️ 식단 및 생체 지표 입력")
    st.sidebar.markdown(
        "섭취할 식단의 영양성분과 현재 공복 혈당을 조절해 보세요."
    )

    with st.sidebar.form("simulation_form"):
        carbs = st.slider(
            "총 탄수화물 섭취량 (g)",
            min_value=10,
            max_value=150,
            value=65,
            step=5,
            help="식사에 포함된 전체 탄수화물의 양입니다.",
        )
        sugars = st.slider(
            "당류 함량 (g)",
            min_value=0,
            max_value=100,
            value=20,
            step=2,
            help="탄수화물 중 흡수가 빠른 단순당(설탕, 액상과당 등)의 양입니다.",
        )
        fiber = st.slider(
            "식이섬유 함량 (g)",
            min_value=0,
            max_value=40,
            value=6,
            step=1,
            help="수용성·불용성 식이섬유 섭취량입니다. 높을수록 혈당 상승을 억제합니다.",
        )
        fasting_glu = st.number_input(
            "현재 공복 혈당 (mg/dL)",
            min_value=60,
            max_value=250,
            value=95,
            step=1,
            help="식사 직전 공복 상태의 혈당 수치입니다.",
        )

        if sugars > carbs:
            st.warning(
                "⚠️ 당류는 총 탄수화물량을 초과할 수 없어 탄수화물량과 동일하게 계산됩니다."
            )
            sugars = carbs

        submitted = st.form_submit_button(
            "🔬 혈당 스파이크 시뮬레이션 실행",
            use_container_width=True,
        )

    # 예측 실행
    pred_peak, fiber_ratio = predict_postprandial_glucose(
        lr_model, fasting_glu, carbs, sugars, fiber
    )
    glu_delta = round(pred_peak - fasting_glu, 1)

    # --------------------------------------------------------------------------
    # [메인 대시보드 1] 핵심 지표 요약 (Metrics & Columns)
    # --------------------------------------------------------------------------
    st.subheader("📊 실시간 대사 예측 대시보드")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(
            label="식이섬유 비율 (Fiber Ratio)",
            value=f"{fiber_ratio} %",
            delta="권장 기준 ≥ 15%"
            if fiber_ratio >= 15
            else "식이섬유 부족",
            delta_color="normal"
            if fiber_ratio >= 15
            else "inverse",
        )
    with m_col2:
        st.metric(
            label="예상 식후 최고 혈당 (Peak)",
            value=f"{pred_peak} mg/dL",
            delta=f"+{glu_delta} mg/dL 상승",
            delta_color="inverse"
            if pred_peak >= 140
            else "normal",
        )
    with m_col3:
        st.metric(
            label="순수 혈당 부하 탄수화물",
            value=f"{max(0, carbs - fiber)} g",
            help="총 탄수화물에서 소화되지 않는 식이섬유를 제외한 순탄수화물(Net Carbs)입니다.",
        )
    with m_col4:
        st.metric(
            label="임상 데이터 기준 평균 공복혈당",
            value=f"{df['Fasting GLU - PDL (Lab)'].mean():.1f} mg/dL",
            help="PhysioNet bio.csv 참여자 전체의 평균 공복혈당입니다.",
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # [메인 대시보드 2] 3단계 맞춤형 건강 리스크 진단 시스템
    # --------------------------------------------------------------------------
    st.subheader("🩺 3단계 대사 건강 리스크 진단 결과")

    if pred_peak < 140:
        st.success(
            f"### 🟢 대사 안정형 (예상 피크: {pred_peak} mg/dL)\n"
            "**[진단 결과] 대사 안정 상태 — 인슐린 분비 부담 최소화**\n\n"
            "* **생리학적 분석**: 식단에 포함된 충분한 식이섬유가 소장 내 점성 하이드로겔 매트릭스를 형성하여 포도당의 확산 및 흡수 속도를 완만하게 지연시켰습니다.\n"
            "* **건강 기대 효과**: 급격한 혈당 상승이 억제되어 췌장 $\\beta$세포의 인슐린 과분비를 막고, 식후에도 안정적인 에너지 대사와 집중력이 유지됩니다."
        )
    elif 140 <= pred_peak < 180:
        st.warning(
            f"### 🟡 주의형 (예상 피크: {pred_peak} mg/dL)\n"
            "**[진단 결과] 혈당 상승 주의 — 반응성 저혈당 및 식후 나른함 우려**\n\n"
            "* **생리학적 분석**: 탄수화물 및 당류 함량 대비 식이섬유 비율({fiber_ratio}%)이 다소 부족하여 소장에서 포도당이 비교적 빠르게 흡수되고 있습니다.\n"
            "* **건강 우려 사항**: 급상승한 혈당을 낮추기 위해 인슐린이 다량 분비되면서, **식후 2~3시간 뒤 혈당이 급락하는 반응성 저혈당(식곤증, 무기력증, 가짜 허기)**이 유발될 수 있습니다. 식사 전 채소(식이섬유)를 먼저 섭취하는 '거꾸로 식사법'을 권장합니다."
        )
    else:
        st.error(
            f"### 🔴 고위험 경보 (예상 피크: {pred_peak} mg/dL)\n"
            "**[진단 결과] 혈당 스파이크 고위험 — 췌장 피로 및 인슐린 저항성 가중 우려**\n\n"
            "* **생리학적 분석**: 고탄수화물·고당류 식단과 낮은 식이섬유 비율({fiber_ratio}%)로 인해 소화 효소($\\alpha$-아밀라아제 등)가 탄수화물을 급속히 분해하여 혈중 포도당 농도가 폭발적으로 치솟는 **혈당 스파이크**가 예측됩니다.\n"
            "* **건강 우려 사항**: 췌장 랑게르한스섬 $\\beta$세포에 극심한 피로를 누적시키며, 이러한 식습관이 반복될 경우 세포의 인슐린 수용체 민감도가 떨어지는 **인슐린 저항성(Insulin Resistance) 및 제2형 당뇨병, 대사증후군** 발병 위험이 크게 증가합니다."
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # [메인 대시보드 3] 머신러닝 모델 성능 평가 및 선형회귀 그래프 시각화
    # --------------------------------------------------------------------------
    st.subheader(
        "📈 머신러닝 모델 성능 평가 및 선형회귀 분석 그래프"
    )
    st.markdown(
        "PhysioNet 임상 데이터의 **공복 혈당(X)**과 **실측 최고 혈당(Y)** 간의 상관관계를 선형회귀 모델 추세선으로 시각화합니다."
    )

    ev_col1, ev_col2, ev_col3 = st.columns(3)
    with ev_col1:
        st.metric(
            label="학습 모델 알고리즘",
            value="Linear Regression",
        )
    with ev_col2:
        st.metric(
            label="모델 결정계수 ($R^2$ Score)",
            value=f"{r2_score:.3f}",
        )
    with ev_col3:
        st.metric(
            label="유효 학습 임상 샘플 수",
            value=f"{sample_count} 건",
        )

    # Matplotlib 선형회귀 산점도 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 5))

    X_vals = clean_df["Fasting GLU - PDL (Lab)"]
    y_vals = clean_df["Observed_Peak_GLU"]

    # 실제 데이터 산점도
    ax.scatter(
        X_vals,
        y_vals,
        color="skyblue",
        alpha=0.7,
        edgecolors="k",
        label="Clinical Data (bio.csv)",
    )

    # 선형회귀 추세선
    X_line = np.linspace(X_vals.min(), X_vals.max(), 100).reshape(-1, 1)
    y_line = lr_model.predict(X_line)
    ax.plot(
        X_line,
        y_line,
        color="crimson",
        linewidth=2.5,
        label=f"Linear Fit (y = {lr_model.coef_[0]:.2f}x + {lr_model.intercept_:.2f})",
    )

    # 현재 사용자가 입력한 시뮬레이션 지점 강조 표시
    ax.scatter(
        [fasting_glu],
        [pred_peak],
        color="orange",
        s=150,
        zorder=5,
        marker="*",
        label=f"Current Simulation (Input: {fasting_glu}, Pred: {pred_peak})",
    )

    ax.set_title(
        "Fasting Glucose vs. Postprandial Peak Glucose Regression",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Fasting Glucose - PDL (Lab) [mg/dL]", fontsize=10)
    ax.set_ylabel("Observed Peak Glucose [mg/dL]", fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Streamlit에 맷플롯립 그래프 렌더링
    st.pyplot(fig)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # [하단 상세 정보] 생명과학적 원리 및 원본 임상 데이터 탐색 (Expander)
    # --------------------------------------------------------------------------
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        with st.expander(
            "🧬 생명과학적 원리: 식이섬유의 혈당 스파이크 억제 기전"
        ):
            st.markdown("""
            1. **점성 하이드로겔(Hydrogel) 형성**: 수용성 식이섬유(펙틴, 베타글루칸 등)는 소화관 내에서 수분을 흡수해 끈적한 젤(Gel) 형태의 매트릭스를 만듭니다.
            2. **물리적 확산 장벽**: 이 젤 구조가 소화 효소($\\alpha$-amylase)와 탄수화물의 접촉을 물리적으로 방해하고, 분해된 포도당이 소장 상피세포의 융모(SGLT1, GLUT2 수송체)로 확산되는 속도를 늦춥니다.
            3. **인크레틴(GLP-1) 호르몬 활성화**: 천천히 이동한 영양소가 소장 하부를 자극해 포만감과 적절한 인슐린 분비를 돕는 GLP-1 호르몬 분비를 촉진합니다.
            """)

    with col_exp2:
        with st.expander(
            "📂 학습에 활용된 PhysioNet 임상 데이터(bio.csv) 살펴보기"
        ):
            st.write(
                f"본 모델은 총 **{sample_count}명**의 임상 대상자 데이터(공복 혈당 및 핑거스틱 추적 혈당)를 기반으로 기초 대사 기울기(회귀 계수: `{lr_model.coef_[0]:.2f}`)를 학습했습니다."
            )
            display_cols = [
                c
                for c in [
                    "subject",
                    "Age",
                    "Gender",
                    "BMI",
                    "A1c PDL (Lab)",
                    "Fasting GLU - PDL (Lab)",
                    "Observed_Peak_GLU",
                ]
                if c in df.columns
            ]
            st.dataframe(
                df[display_cols].head(10), use_container_width=True
            )
