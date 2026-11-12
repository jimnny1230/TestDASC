import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
# 2. 임상 데이터 로드 및 전처리 엔진 (Engineering & Data Cleaning)
# ==============================================================================
DATA_URL = (
    "https://raw.githubusercontent.com/jimnny1230/TestDASC/main/bio.csv"
)


@st.cache_data
def load_and_prepare_data():
    """
    [공학적 의미] GitHub raw 데이터를 불러오고, 결측치 정제 및
    독립/종속변수 추출을 위한 전처리를 수행합니다.
    """
    try:
        df = pd.read_csv(DATA_URL, encoding="utf-8", sep="\t")
        if len(df.columns) <= 1:
            df = pd.read_csv(DATA_URL, encoding="utf-8")
    except Exception:
        df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 수치형 변환 대상 컬럼 지정
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
        "Non HDL",
        "LDL (Cal)",
        "VLDL (Cal)",
        "Cho/HDL Ratio",
        "#1 Contour Fingerstick GLU",
        "#2 Contour Fingerstick GLU",
        "#3 Contour Fingerstick GLU",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # [종속변수 생성] 핑거스틱 혈당 중 최댓값을 실측 식후 최고 혈당(Observed_Peak_GLU)으로 산출
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

    # [데이터 전처리: 결측치 제거 내역 추적]
    target_subset = ["Fasting GLU - PDL (Lab)", "Observed_Peak_GLU"]
    before_count = len(df)
    clean_df = df.dropna(subset=target_subset).copy()
    dropped_count = before_count - len(clean_df)

    return df, clean_df, dropped_count


@st.cache_resource
def train_baseline_model(clean_df):
    """
    [공학적 의미] 전처리가 완료된 데이터로 선형회귀 모델 학습
    """
    X = clean_df[["Fasting GLU - PDL (Lab)"]]
    y = clean_df["Observed_Peak_GLU"]

    model = LinearRegression()
    model.fit(X, y)
    r2_score = model.score(X, y)
    return model, r2_score


# ==============================================================================
# 3. 생리학적 혈당 피크 시뮬레이션 엔진
# ==============================================================================
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
# 4. 메인 애플리케이션 화면 구성 (탭 구조)
# ==============================================================================
st.title("🥗 스마트 바이오푸드 & 혈당 스파이크 예측 시뮬레이터")
st.markdown(
    "PhysioNet 임상 생체 데이터 기반 대사 웰니스 분석 및 AI 시뮬레이션 시스템"
)
st.markdown("---")

# 데이터 로드
try:
    raw_df, clean_df, dropped_count = load_and_prepare_data()
    lr_model, r2_score = train_baseline_model(clean_df)
    is_ready = True
except Exception as e:
    st.error(f"데이터 로드 및 전처리 중 오류가 발생했습니다: {e}")
    is_ready = False

if is_ready:
    # 탭 생성 (1번: 시뮬레이터, 2번: 데이터 전처리 및 상관관계 분석)
    tab1, tab2 = st.tabs(
        ["📊 시뮬레이터 및 모델 예측", "🧹 데이터 전처리 및 상관관계 분석"]
    )

    # --------------------------------------------------------------------------
    # [TAB 1] 시뮬레이터 메인 화면
    # --------------------------------------------------------------------------
    with tab1:
        st.sidebar.header("🍽️ 식단 및 생체 지표 입력")
        with st.sidebar.form("simulation_form"):
            carbs = st.slider("총 탄수화물 섭취량 (g)", 10, 150, 65, 5)
            sugars = st.slider("당류 함량 (g)", 0, 100, 20, 2)
            fiber = st.slider("식이섬유 함량 (g)", 0, 40, 6, 1)
            fasting_glu = st.number_input(
                "현재 공복 혈당 (mg/dL)", 60, 250, 95, 1
            )

            if sugars > carbs:
                sugars = carbs

            submitted = st.form_submit_button(
                "🔬 혈당 스파이크 시뮬레이션 실행",
                use_container_width=True,
            )

        pred_peak, fiber_ratio = predict_postprandial_glucose(
            lr_model, fasting_glu, carbs, sugars, fiber
        )
        glu_delta = round(pred_peak - fasting_glu, 1)

        # 핵심 지표 메트릭
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
            st.metric(
                "순수 혈당 부하 탄수화물", f"{max(0, carbs - fiber)} g"
            )
        with c4:
            st.metric(
                "임상 평균 공복혈당",
                f"{clean_df['Fasting GLU - PDL (Lab)'].mean():.1f} mg/dL",
            )

        st.markdown("---")

        # 리스크 진단
        st.subheader("🩺 3단계 대사 건강 리스크 진단 결과")
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

        st.markdown("---")

        # 모델 선형회귀 그래프
        st.subheader("📈 머신러닝 선형회귀 예측 모델 시각화")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        X_vals = clean_df["Fasting GLU - PDL (Lab)"]
        y_vals = clean_df["Observed_Peak_GLU"]

        ax.scatter(
            X_vals,
            y_vals,
            color="skyblue",
            alpha=0.7,
            edgecolors="k",
            label="Clinical Data",
        )
        X_line = np.linspace(
            X_vals.min(), X_vals.max(), 100
        ).reshape(-1, 1)
        y_line = lr_model.predict(X_line)
        ax.plot(
            X_line,
            y_line,
            color="crimson",
            linewidth=2.5,
            label=f"Linear Fit ($R^2$ = {r2_score:.3f})",
        )
        ax.scatter(
            [fasting_glu],
            [pred_peak],
            color="orange",
            s=150,
            zorder=5,
            marker="*",
            label="Current Input",
        )
        ax.set_xlabel("Fasting Glucose [mg/dL]")
        ax.set_ylabel("Observed Peak Glucose [mg/dL]")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

    # --------------------------------------------------------------------------
    # [TAB 2] 데이터 전처리 및 상관관계 분석 페이지
    # --------------------------------------------------------------------------
    with tab2:
        st.header("🧹 데이터 전처리 및 변수 간 상관관계 분석")
        st.markdown("""
        본 페이지에서는 머신러닝 모델 학습을 위해 원본 임상 데이터(`bio.csv`)를 정제한 과정과, 
        **1) 전체 속성 간 상관관계 히트맵** 및 **2) 독립변수와 종속변수 간의 상관관계 분석 결과**를 투명하게 공개합니다.
        """)
        st.markdown("---")

        # 1. 독립변수 및 종속변수 정의
        st.subheader("1️⃣ 독립변수(X) 및 종속변수(Y) 선정 내역")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.info("""
            **📌 독립변수 (Independent Variable, $X$)**
            * **선택 속성**: `Fasting GLU - PDL (Lab)` (공복 혈당)
            * **선정 이유**: 공복 상태의 기초 혈당 수치는 개인의 기저 대사 상태와 인슐린 저항성을 반영하며, 식후 혈당 피크에 직접적인 기저 영향을 미치므로 메인 독립변수로 선정했습니다.
            """)
        with col_v2:
            st.success("""
            **🎯 종속변수 (Dependent Variable, $Y$)**
            * **선택 속성**: `Observed_Peak_GLU` (실측 식후 최고 혈당)
            * **선정 이유**: 1~3차 핑거스틱(채혈식) 혈당 기록 중 최댓값을 추출하여, 식후에 도달하는 혈당의 최고점(Peak Glucose)을 모델이 예측해야 할 타겟으로 설정했습니다.
            """)

        st.markdown("---")

        # 2. 결측치 및 제거된 데이터 내역
        st.subheader("2️⃣ 데이터 정제 및 결측치 제거 내역")
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.metric("원본 데이터 총 행 수", f"{len(raw_df)} 건")
        with d_col2:
            st.metric(
                "결측치로 인해 제외된 행 수",
                f"{dropped_count} 건",
                delta="정제 완료",
                delta_color="off",
            )
        with d_col3:
            st.metric("최종 모델 학습용 샘플 수", f"{len(clean_df)} 건")

        with st.expander("🔍 결측치 정제 상세 레코드 보기"):
            st.dataframe(clean_df.head(10), use_container_width=True)

        st.markdown("---")

        # 3. 전체 속성 간 상관관계 히트맵
        st.subheader(
            "3️⃣ [분석 1] 전체 주요 임상 속성 간 상관관계 히트맵 (Heatmap)"
        )
        st.markdown(
            "나이, BMI, 공복혈당, 당화혈색소, 인슐린, 지질 수치 및 실측 최고혈당 등 주요 수치형 변수들 간의 다중 상관관계를 히트맵으로 시각화했습니다."
        )

        corr_cols = [
            "Age",
            "BMI",
            "A1c PDL (Lab)",
            "Fasting GLU - PDL (Lab)",
            "Insulin",
            "Triglycerides",
            "Cholesterol",
            "HDL",
            "Observed_Peak_GLU",
        ]
        available_corr_cols = [
            c for c in corr_cols if c in clean_df.columns
        ]

        if len(available_corr_cols) > 1:
            corr_matrix = clean_df[available_corr_cols].corr()

            fig_hm, ax_hm = plt.subplots(figsize=(10, 6.5))
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                ax=ax_hm,
                annot_kws={"size": 9},
            )
            ax_hm.set_title(
                "All Clinical Variables Correlation Heatmap",
                fontsize=12,
                fontweight="bold",
            )
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig_hm)

        st.markdown("---")

        # 4. 종속변수(Observed_Peak_GLU)와 독립변수들 간의 상관관계 집중 분석
        st.subheader(
            "4️⃣ [분석 2] 종속변수(최고혈당)와 독립변수들 간의 상관관계 비교"
        )
        st.markdown(
            "우리가 예측하려는 **종속변수(`Observed_Peak_GLU`)**와 모델에 활용된 **독립변수(`Fasting GLU`)** 및 기타 생체 속성들이 각각 어느 정도의 선형 상관관계를 가지는지 막대 그래프로 추출했습니다."
        )

        if "Observed_Peak_GLU" in clean_df.columns:
            # 종속변수와의 상관계수만 추출 후 내림차순 정렬
            target_corr = (
                clean_df[available_corr_cols]
                .corr()["Observed_Peak_GLU"]
                .drop("Observed_Peak_GLU")
                .sort_values()
            )

            fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
            bars = ax_bar.barh(
                target_corr.index,
                target_corr.values,
                color="teal",
                alpha=0.8,
            )
            ax_bar.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax_bar.set_title(
                "Correlation with Dependent Variable (Observed_Peak_GLU)",
                fontsize=12,
                fontweight="bold",
            )
            ax_bar.set_xlabel("Correlation Coefficient (r)", fontsize=10)
            ax_bar.grid(True, linestyle="--", alpha=0.5)

            # 막대 양옆에 수치 표시
            for bar in bars:
                width = bar.get_width()
                ax_bar.text(
                    width + (0.02 if width >= 0 else -0.07),
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.2f}",
                    va="center",
                    ha="left" if width >= 0 else "right",
                    fontsize=9,
                    fontweight="bold",
                )

            plt.tight_layout()
            st.pyplot(fig_bar)

            st.info("""
            💡 **[해석 및 인사이트]**
            * 공복 혈당(`Fasting GLU`)과 당화혈색소(`A1c`) 등은 종속변수인 실측 식후 최고 혈당(`Observed_Peak_GLU`)과 뚜렷한 양(+)의 상관관계를 보여줍니다. 
            * 이는 기초 대사 혈당이 높을 수록 식후 피크 혈당도 비례해서 상승한다는 것을 의미하며, 우리가 공복 혈당을 독립변수($X$)로 채택하여 선형회귀 모델을 구축한 강력한 통계적 근거가 됩니다.
            """)
