import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="탐색 | 뇌졸중 예측 실습실", page_icon="🔎", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/goodrin09-code/data/refs/heads/main/stroke.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🔎 데이터 탐색")
st.write(
    "나이, 평균 혈당, 질병 여부, 흡연 상태 등에 따라 데이터를 나누어 살펴봅니다. "
    "그래프와 표를 통해 변수들의 분포와 뇌졸중 발생 비율을 비교해 보세요."
)

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.caption(f"오류 내용: {error}")
    st.stop()

required_columns = [
    "age", "avg_glucose_level", "stroke", "hypertension",
    "heart_disease", "bmi", "smoking_status"
]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    st.error(f"분석에 필요한 열이 데이터에 없습니다: {', '.join(missing_columns)}")
    st.stop()

for column in ["age", "avg_glucose_level", "stroke", "hypertension", "heart_disease", "bmi"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

stroke_labels = {0: "뇌졸중 없음", 1: "뇌졸중 있음"}
df["stroke_group"] = df["stroke"].map(stroke_labels)

# 1. 나이와 평균 혈당 분포
st.header("1. 나이와 평균 혈당의 분포")
st.write("각 변수의 값이 어떤 구간에 많이 모여 있는지 히스토그램으로 확인합니다.")

hist_col1, hist_col2 = st.columns(2)

with hist_col1:
    fig_age_hist = px.histogram(
        df.dropna(subset=["age"]), x="age", nbins=30,
        title="나이 분포", labels={"age": "나이", "count": "사람 수"}
    )
    fig_age_hist.update_layout(yaxis_title="사람 수", xaxis_title="나이", bargap=0.08)
    st.plotly_chart(fig_age_hist, use_container_width=True)

with hist_col2:
    fig_glucose_hist = px.histogram(
        df.dropna(subset=["avg_glucose_level"]), x="avg_glucose_level", nbins=30,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"}
    )
    fig_glucose_hist.update_layout(
        yaxis_title="사람 수", xaxis_title="평균 혈당", bargap=0.08
    )
    st.plotly_chart(fig_glucose_hist, use_container_width=True)

# 2. 뇌졸중 유무에 따른 상자그림과 평균값
st.divider()
st.header("2. 뇌졸중 유무에 따른 비교")
st.write("두 그룹의 나이와 평균 혈당 분포를 상자그림으로 비교합니다.")

comparison_df = df.dropna(subset=["stroke_group"]).copy()
comparison_df["stroke_group"] = pd.Categorical(
    comparison_df["stroke_group"],
    categories=["뇌졸중 없음", "뇌졸중 있음"],
    ordered=True,
)

box_col1, box_col2 = st.columns(2)

with box_col1:
    fig_age_box = px.box(
        comparison_df.dropna(subset=["age"]),
        x="stroke_group", y="age", color="stroke_group",
        category_orders={"stroke_group": ["뇌졸중 없음", "뇌졸중 있음"]},
        title="뇌졸중 유무에 따른 나이",
        labels={"stroke_group": "그룹", "age": "나이", "color": "그룹"},
    )
    fig_age_box.update_layout(showlegend=False, xaxis_title="", yaxis_title="나이")
    st.plotly_chart(fig_age_box, use_container_width=True)

with box_col2:
    fig_glucose_box = px.box(
        comparison_df.dropna(subset=["avg_glucose_level"]),
        x="stroke_group", y="avg_glucose_level", color="stroke_group",
        category_orders={"stroke_group": ["뇌졸중 없음", "뇌졸중 있음"]},
        title="뇌졸중 유무에 따른 평균 혈당",
        labels={
            "stroke_group": "그룹",
            "avg_glucose_level": "평균 혈당",
            "color": "그룹",
        },
    )
    fig_glucose_box.update_layout(
        showlegend=False, xaxis_title="", yaxis_title="평균 혈당"
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)

st.subheader("그룹별 평균값")
mean_table = (
    comparison_df.groupby("stroke_group", observed=False)[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .reindex(["뇌졸중 없음", "뇌졸중 있음"])
    .rename(columns={"age": "평균 나이", "avg_glucose_level": "평균 혈당"})
    .round(2)
)
mean_table.index.name = "그룹"
st.dataframe(mean_table, use_container_width=True)

# 3. 고혈압 및 심장병 유무에 따른 뇌졸중 비율
st.divider()
st.header("3. 질병 유무에 따른 뇌졸중 비율")
st.write("각 그룹에서 stroke가 1인 사람의 비율을 계산합니다.")


def make_rate_table(data, group_column, labels):
    valid = data.dropna(subset=[group_column, "stroke"]).copy()
    valid = valid[valid[group_column].isin([0, 1])]
    result = (
        valid.groupby(group_column)["stroke"]
        .agg(전체_인원="count", 뇌졸중_인원="sum", 뇌졸중_비율="mean")
        .reindex([0, 1])
    )
    result.index = [labels[0], labels[1]]
    result["뇌졸중_비율(%)"] = (result["뇌졸중_비율"] * 100).round(2)
    return result.reset_index(names="구분")[
        ["구분", "전체_인원", "뇌졸중_인원", "뇌졸중_비율(%)"]
    ]


hypertension_table = make_rate_table(
    df, "hypertension", {0: "고혈압 없음", 1: "고혈압 있음"}
)
heart_table = make_rate_table(
    df, "heart_disease", {0: "심장병 없음", 1: "심장병 있음"}
)

disease_col1, disease_col2 = st.columns(2)

with disease_col1:
    fig_hypertension = px.bar(
        hypertension_table, x="구분", y="뇌졸중_비율(%)",
        text="뇌졸중_비율(%)", title="고혈압 유무에 따른 뇌졸중 비율",
        labels={"구분": "", "뇌졸중_비율(%)": "뇌졸중 비율(%)"},
    )
    fig_hypertension.update_traces(
        texttemplate="%{text:.2f}%", textposition="outside"
    )
    fig_hypertension.update_layout(yaxis_rangemode="tozero")
    st.plotly_chart(fig_hypertension, use_container_width=True)
    st.dataframe(hypertension_table, hide_index=True, use_container_width=True)

with disease_col2:
    fig_heart = px.bar(
        heart_table, x="구분", y="뇌졸중_비율(%)",
        text="뇌졸중_비율(%)", title="심장병 유무에 따른 뇌졸중 비율",
        labels={"구분": "", "뇌졸중_비율(%)": "뇌졸중 비율(%)"},
    )
    fig_heart.update_traces(
        texttemplate="%{text:.2f}%", textposition="outside"
    )
    fig_heart.update_layout(yaxis_rangemode="tozero")
    st.plotly_chart(fig_heart, use_container_width=True)
    st.dataframe(heart_table, hide_index=True, use_container_width=True)

# 4. BMI 결측자와 전체의 뇌졸중 비율
st.divider()
st.header("4. BMI가 비어 있는 사람들의 뇌졸중 비율")
st.write(
    "BMI 값이 없는 사람들만 따로 모아 뇌졸중 비율을 계산하고, "
    "전체 데이터의 비율과 비교합니다."
)

bmi_missing = df[df["bmi"].isna()]
valid_stroke_all = df["stroke"].dropna()
valid_stroke_missing_bmi = bmi_missing["stroke"].dropna()

all_stroke_rate = (
    (valid_stroke_all == 1).mean() * 100 if len(valid_stroke_all) else float("nan")
)
missing_bmi_stroke_rate = (
    (valid_stroke_missing_bmi == 1).mean() * 100
    if len(valid_stroke_missing_bmi) else float("nan")
)

bmi_compare = pd.DataFrame([
    {
        "대상": "BMI 결측자",
        "전체 인원": len(bmi_missing),
        "뇌졸중 인원": int((valid_stroke_missing_bmi == 1).sum()),
        "뇌졸중 비율(%)": round(missing_bmi_stroke_rate, 2),
    },
    {
        "대상": "전체 데이터",
        "전체 인원": len(df),
        "뇌졸중 인원": int((valid_stroke_all == 1).sum()),
        "뇌졸중 비율(%)": round(all_stroke_rate, 2),
    },
])
st.dataframe(bmi_compare, hide_index=True, use_container_width=True)

# 5. 흡연 상태별 사람 수
st.divider()
st.header("5. 흡연 상태별 사람 수")
st.write("흡연 상태의 각 값에 해당하는 사람 수를 셉니다. 빈 값도 별도로 표시합니다.")

smoking_counts = (
    df["smoking_status"]
    .fillna("결측값")
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking_counts, hide_index=True, use_container_width=True)
