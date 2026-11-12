import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="데이터 전처리 및 상관관계 분석", page_icon="🧹", layout="wide"
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

    target_subset = ["Fasting GLU - PDL (Lab)", "Observed_Peak_GLU"]
    before_count = len(df)
    clean_df = df.dropna(subset=target_subset).copy()
    dropped_count = before_count - len(clean_df)

    return df, clean_df, dropped_count


raw_df, clean_df, dropped_count = load_and_prepare_data()

st.title("🧹 데이터 전처리 및 변수 간 상관관계 분석")
st.markdown("""
본 페이지는 머신러닝 모델 학습을 위해 원본 임상 데이터(`bio.csv`)를 정제한 과정과, 
**1) 전체 속성 간 상관관계 히트맵** 및 **2) 독립변수와 종속변수 간의 상관관계 분석 결과**를 독립된 페이지에서 제공합니다.
""")
st.markdown("---")

# 1. 독립/종속변수 정의
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

# 2. 결측치 정제 내역
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
st.subheader("3️⃣ [분석 1] 전체 주요 임상 속성 간 상관관계 히트맵")
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
available_corr_cols = [c for c in corr_cols if c in clean_df.columns]

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

# 4. 종속변수와 독립변수들 간의 상관관계 집중 분석
st.subheader("4️⃣ [분석 2] 종속변수(최고혈당)와 독립변수들 간의 상관관계 비교")
if "Observed_Peak_GLU" in clean_df.columns:
    target_corr = (
        clean_df[available_corr_cols]
        .corr()["Observed_Peak_GLU"]
        .drop("Observed_Peak_GLU")
        .sort_values()
    )

    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    bars = ax_bar.barh(
        target_corr.index, target_corr.values, color="teal", alpha=0.8
    )
    ax_bar.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax_bar.set_title(
        "Correlation with Dependent Variable (Observed_Peak_GLU)",
        fontsize=12,
        fontweight="bold",
    )
    ax_bar.set_xlabel("Correlation Coefficient (r)", fontsize=10)
    ax_bar.grid(True, linestyle="--", alpha=0.5)

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
