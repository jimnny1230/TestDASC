import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정 (페이지 제목, 아이콘, 레이아웃)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="스마트 바이오푸드 & 혈당 분석 시뮬레이터",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. 데이터 로드 함수 (캐싱 적용)
# -----------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/jimnny1230/TestDASC/main/bio.csv"
)


@st.cache_data
def load_data():
    try:
        # UTF-8 인코딩 및 탭(\t) 구분자 처리
        df = pd.read_csv(DATA_URL, encoding="utf-8", sep="\t")
    except Exception:
        # 혹시 일반 쉼표(,) 구분자인 경우에 대비한 2차 예외 처리
        df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


# -----------------------------------------------------------------------------
# 3. 메인 화면 - 헤더 & 데이터셋 소개
# -----------------------------------------------------------------------------
# 앱 상단 타이틀
st.title("🥗 스마트 바이오푸드 & 혈당 분석 시뮬레이터")
st.caption(
    "PhysioNet 임상 생체 데이터(Bio) 기반 대사 건강 및 혈당 피크 분석 시스템"
)

st.markdown("---")

# 데이터 로드
try:
    df = load_data()
    data_loaded = True
except Exception as e:
    st.error(
        f"데이터를 불러오는 중 오류가 발생했습니다. URL 및 인터넷 연결을 확인해 주세요. ({e})"
    )
    data_loaded = False

if data_loaded:
    # 섹션 1: 데이터셋 개요
    st.subheader("📌 데이터셋 소개")
    st.markdown("""
    본 애플리케이션은 **PhysioNet 임상 데이터셋** 기반의 생체(Bio) 및 혈당 측정 지표를 활용합니다.
    참가자들의 신체 측정 항목(BMI, 체중 등), 임상 혈액검사 결과(공복혈당, 당화혈혈소, 인슐린, 지질 수치), 
    그리고 **시간대별 핑거스틱(Fingerstick) 혈당 변화** 기록이 포함되어 있습니다.
    """)

    # 핵심 지표 요약 (Metrics)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 대상자 수", f"{len(df)} 명")
    with col2:
        st.metric(
            "평균 연령",
            f"{df['Age'].mean():.1f} 세"
            if "Age" in df.columns
            else "N/A",
        )
    with col3:
        st.metric(
            "평균 BMI",
            f"{df['BMI'].mean():.1f} kg/m²"
            if "BMI" in df.columns
            else "N/A",
        )
    with col4:
        st.metric(
            "평균 공복혈당",
            f"{df['Fasting GLU - PDL (Lab)'].mean():.1f} mg/dL"
            if "Fasting GLU - PDL (Lab)" in df.columns
            else "N/A",
        )

    st.markdown("---")

    # 섹션 2: 데이터 미리보기
    st.subheader("🔍 임상 데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True)

    # 섹션 3: 주요 수집 변수(컬럼) 설명
    with st.expander("📋 주요 데이터 컬럼(변수) 상세 보기"):
        st.markdown("""
        * **기본 인적사항**: `subject`(참가자 ID), `Age`(연령), `Gender`(성별), `BMI`, `Body weight`(체중), `Height`(신장)
        * **임상 혈액검사 (Lab)**:
          * `A1c PDL (Lab)`: 당화혈색소 수치 (%)
          * `Fasting GLU - PDL (Lab)`: 공복 혈당 수치 (mg/dL)
          * `Insulin`: 인슐린 수치
          * `Triglycerides`, `Cholesterol`, `HDL`, `LDL (Cal)`: 지질 프로필 수치
        * **실시간 혈당 추적 데이터**:
          * `#1 Contour Fingerstick GLU` / `Time (t)`: 1차 핑거스틱 혈당 및 측정 시간
          * `#2 Contour Fingerstick GLU` / `Time (t)`: 2차 핑거스틱 혈당 및 측정 시간
          * `#3 Contour Fingerstick GLU` / `Time (t)`: 3차 핑거스틱 혈당 및 측정 시간
        """)

    # 사이드바 안내
    st.sidebar.header("⚙️ 시뮬레이션 설정")
    st.sidebar.info(
        "사이드바에 식이섬유 비율 계산 및 혈당 피크 예측 머신러닝 모델을 연동할 예정입니다."
    )
