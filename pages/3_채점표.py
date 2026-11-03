
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="채점표",
    page_icon="📊",
    layout="wide"
)

st.title("📊 채점표")
st.write(
    "분류 모델이 실제 뇌졸중 여부를 얼마나 정확하게 예측했는지 "
    "혼동행렬과 다양한 평가지표를 이용하여 확인합니다."
)

st.divider()


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "goodrin09-code/data/refs/heads/main/stroke.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 3. 입력 속성 선택
# --------------------------------------------------

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease"
]

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]

st.subheader("1. 분류에 사용할 속성 선택")

selected_labels = st.multiselect(
    "사용할 속성",
    options=[FEATURE_NAMES[f] for f in FEATURES],
    default=[FEATURE_NAMES[f] for f in DEFAULT_FEATURES]
)

selected_features = [
    f for f in FEATURES
    if FEATURE_NAMES[f] in selected_labels
]

if len(selected_features) < 2:
    st.warning(
        "분류 모델을 만들려면 속성을 두 개 이상 선택해야 합니다."
    )
    st.stop()

st.success(
    "선택한 속성: "
    + ", ".join(FEATURE_NAMES[f] for f in selected_features)
)


# --------------------------------------------------
# 4. 데이터 정렬 및 학습용 / 테스트용 분리
# --------------------------------------------------

st.subheader("2. 데이터 분리")

df = df.sort_values(
    "id",
    kind="mergesort"
).reset_index(drop=True)

df["stroke"] = pd.to_numeric(
    df["stroke"],
    errors="coerce"
)

df = df.dropna(
    subset=["stroke"]
).reset_index(drop=True)

df["stroke"] = df["stroke"].astype(int)

if not set(df["stroke"].unique()).issubset({0, 1}):
    st.error("stroke 열에는 0과 1만 있어야 합니다.")
    st.stop()

# 1 = 뇌졸중 (양성)
# 0 = 뇌졸중 아님 (음성)

# 10명씩 묶고 앞의 3명을 테스트용으로 지정
test_mask = np.arange(len(df)) % 10 < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].copy()
y_test = test_df["stroke"].copy()


# BMI를 선택한 경우에만 학습 데이터의 중앙값으로 결측치 처리
if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    st.info(
        f"체질량지수의 결측값은 학습 데이터의 중앙값 "
        f"{bmi_median:.2f}으로 채웠습니다."
    )


if X_train.isnull().any().any() or X_test.isnull().any().any():
    st.error(
        "선택한 속성에 결측값이 있습니다. "
        "데이터를 확인해 주세요."
    )
    st.stop()


# 데이터 개수 표시
total_col, train_col, test_col = st.columns(3)

total_col.metric("전체 데이터", f"{len(df):,}명")
train_col.metric("학습 데이터", f"{len(train_df):,}명")
test_col.metric("테스트 데이터", f"{len(test_df):,}명")

if len(test_df) != 1533:
    st.warning(
        f"테스트 데이터가 {len(test_df):,}명입니다. "
        "요청한 1,533명과 다른 경우 원본 데이터의 행 수를 확인하세요."
    )

st.caption(
    "번호 순으로 정렬한 뒤 10명씩 묶어 앞의 3명은 테스트용, "
    "나머지 7명은 학습용으로 분리했습니다."
)


# --------------------------------------------------
# 5. 모델 학습
# --------------------------------------------------

st.divider()
st.subheader("3. 분류 모델 학습")

# 로지스틱 회귀 (확률로 답하는 모델)
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

logistic_model.fit(
    X_train,
    y_train
)


# 의사결정트리 (질문으로 답하는 모델)
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 6. 테스트 데이터 예측
# --------------------------------------------------

logistic_pred = logistic_model.predict(X_test)
tree_pred = tree_model.predict(X_test)


# 훈련용에서 많은 쪽으로만 답하는 기준 모델
majority_class = int(
    y_train.value_counts().idxmax()
)

baseline_pred = np.full(
    len(y_test),
    majority_class
)


# --------------------------------------------------
# 7. 혼동행렬 계산 함수
# --------------------------------------------------

def calculate_confusion(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 양성 = 1 (뇌졸중)
    # 음성 = 0 (뇌졸중 아님)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN
    }


# --------------------------------------------------
# 8. 평가지표 계산 함수
# --------------------------------------------------

def calculate_metrics(cm):

    TP = cm["TP"]
    FP = cm["FP"]
    FN = cm["FN"]
    TN = cm["TN"]

    total = TP + FP + FN + TN

    # 정확도
    accuracy = (
        (TP + TN) / total
        if total > 0
        else None
    )

    # 재현율: 실제 뇌졸중인 사람 중 맞게 찾아낸 비율
    actual_positive = TP + FN

    recall = (
        TP / actual_positive
        if actual_positive > 0
        else None
    )

    # 정밀도: 뇌졸중이라고 예측한 사람 중 실제 환자의 비율
    predicted_positive = TP + FP

    if predicted_positive == 0:
        precision = None
        f1 = None

    else:
        precision = TP / predicted_positive

        # F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)
        if recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


def format_metric(value):

    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


# --------------------------------------------------
# 9. 모델별 혼동행렬 계산
# --------------------------------------------------

logistic_cm = calculate_confusion(
    y_test,
    logistic_pred
)

tree_cm = calculate_confusion(
    y_test,
    tree_pred
)

baseline_cm = calculate_confusion(
    y_test,
    baseline_pred
)


# --------------------------------------------------
# 10. 테스트 데이터 정보
# --------------------------------------------------

st.divider()
st.subheader("4. 테스트 데이터의 실제 결과")

actual_positive_count = int(
    (y_test == 1).sum()
)

actual_negative_count = int(
    (y_test == 0).sum()
)

st.markdown(
    f"""
    **테스트용 사람 수: {len(test_df):,}명**
    
    **그중 실제 뇌졸중인 사람 수: {actual_positive_count:,}명**
    
    뇌졸중이 아닌 사람 수: {actual_negative_count:,}명
    """
)


# --------------------------------------------------
# 11. 혼동행렬 표시 함수
# --------------------------------------------------

def show_confusion_matrix(title, cm):

    TP = cm["TP"]
    FP = cm["FP"]
    FN = cm["FN"]
    TN = cm["TN"]

    actual_negative = TN + FP
    actual_positive = FN + TP

    predicted_negative = TN + FN
    predicted_positive = FP + TP

    total = TN + FP + FN + TP

    st.markdown(f"#### {title}")

    matrix_df = pd.DataFrame(
        [
            [
                f"TN\n{TN:,}명",
                f"FP\n{FP:,}명",
                f"{actual_negative:,}명"
            ],
            [
                f"FN\n{FN:,}명",
                f"TP\n{TP:,}명",
                f"{actual_positive:,}명"
            ],
            [
                f"{predicted_negative:,}명",
                f"{predicted_positive:,}명",
                f"{total:,}명"
            ]
        ],
        index=[
            "실제 뇌졸중 아님 (0)",
            "실제 뇌졸중 (1)",
            "합계"
        ],
        columns=[
            "예측 뇌졸중 아님 (0)",
            "예측 뇌졸중 (1)",
            "합계"
        ]
    )

    st.dataframe(
        matrix_df,
        use_container_width=True
    )


# --------------------------------------------------
# 12. 두 모델의 혼동행렬
# --------------------------------------------------

st.divider()
st.subheader("5. 모델별 혼동행렬")

st.write(
    "행은 실제 결과, 열은 모델의 예측 결과입니다. "
    "뇌졸중을 양성(1)으로 설정했습니다."
)

st.caption(
    "TP: 실제 뇌졸중을 뇌졸중이라고 예측한 사람 / "
    "FP: 실제로는 아니지만 뇌졸중이라고 예측한 사람"
)

st.caption(
    "FN: 실제 뇌졸중인데 아니라고 예측한 사람 / "
    "TN: 실제로 뇌졸중이 아닌 사람을 아니라고 예측한 사람"
)

show_confusion_matrix(
    "로지스틱 회귀 (확률로 답하는 모델)",
    logistic_cm
)

st.divider()

show_confusion_matrix(
    "의사결정트리 (질문으로 답하는 모델)",
    tree_cm
)

st.divider()

show_confusion_matrix(
    "훈련용에서 많은 쪽으로만 답하는 모델",
    baseline_cm
)


# --------------------------------------------------
# 13. 네 가지 평가지표 비교
# --------------------------------------------------

st.divider()
st.subheader("6. 모델별 평가지표 비교")

st.write(
    "혼동행렬의 네 칸을 이용하여 정확도, 재현율, 정밀도, "
    "F1 점수를 계산했습니다."
)

logistic_metrics = calculate_metrics(logistic_cm)
tree_metrics = calculate_metrics(tree_cm)
baseline_metrics = calculate_metrics(baseline_cm)


comparison_df = pd.DataFrame(
    [
        {
            "모델": "로지스틱 회귀 (확률로 답하는 모델)",
            "정확도": format_metric(logistic_metrics["정확도"]),
            "재현율": format_metric(logistic_metrics["재현율"]),
            "정밀도": format_metric(logistic_metrics["정밀도"]),
            "F1": format_metric(logistic_metrics["F1"])
        },
        {
            "모델": "의사결정트리 (질문으로 답하는 모델)",
            "정확도": format_metric(tree_metrics["정확도"]),
            "재현율": format_metric(tree_metrics["재현율"]),
            "정밀도": format_metric(tree_metrics["정밀도"]),
            "F1": format_metric(tree_metrics["F1"])
        },
        {
            "모델": "훈련용에서 많은 쪽으로만 답하는 모델",
            "정확도": format_metric(baseline_metrics["정확도"]),
            "재현율": format_metric(baseline_metrics["재현율"]),
            "정밀도": format_metric(baseline_metrics["정밀도"]),
            "F1": format_metric(baseline_metrics["F1"])
        }
    ]
)

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "정확도 = (TP + TN) / 전체 인원"
)

st.caption(
    "재현율 = TP / (TP + FN) : 실제 뇌졸중인 사람 중 찾아낸 비율"
)

st.caption(
    "정밀도 = TP / (TP + FP) : 뇌졸중이라고 예측한 사람 중 실제 환자의 비율"
)

st.caption(
    "F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)"
)

st.info(
    "뇌졸중이라고 예측한 사람이 0명이면 정밀도와 F1은 "
    "'계산할 수 없음'으로 표시합니다. "
    "재현율의 분모인 실제 양성 인원이 0명인 경우에도 "
    "재현율과 F1은 계산할 수 없습니다."
)


# --------------------------------------------------
# 14. 놓친 환자와 헛짚은 사람 데이터 만들기
# --------------------------------------------------

st.divider()
st.subheader("7. 잘못 예측한 사람 확인")

st.write(
    "실제 뇌졸중 여부와 모델의 예측이 일치하지 않은 사람들을 확인합니다."
)

# 각 테스트 데이터의 실제 결과와 예측 결과를 정리
def make_error_dataframe(model_prediction):

    result = test_df.copy()

    result["실제 결과"] = y_test.to_numpy()
    result["예측 결과"] = np.asarray(model_prediction)

    # 놓친 환자: 실제 1, 예측 0
    missed = result[
        (result["실제 결과"] == 1)
        & (result["예측 결과"] == 0)
    ].copy()

    # 헛짚은 사람: 실제 0, 예측 1
    false_alarm = result[
        (result["실제 결과"] == 0)
        & (result["예측 결과"] == 1)
    ].copy()

    # 나이가 많은 순서로 정렬
    missed = missed.sort_values(
        "age",
        ascending=False
    )

    false_alarm = false_alarm.sort_values(
        "age",
        ascending=False
    )

    return missed, false_alarm


# --------------------------------------------------
# 15. 표시용 데이터 정리
# --------------------------------------------------

def prepare_display_data(data):

    display_df = pd.DataFrame({
        "번호": data["id"].values,
        "나이": data["age"].values,
        "평균 혈당": data["avg_glucose_level"].values,
        "고혈압 여부": data["hypertension"].values,
        "심장병 여부": data["heart_disease"].values
    })

    # 고혈압과 심장병 여부를 우리말로 표시
    display_df["고혈압 여부"] = display_df[
        "고혈압 여부"
    ].map({
        0: "없음",
        1: "있음"
    })

    display_df["심장병 여부"] = display_df[
        "심장병 여부"
    ].map({
        0: "없음",
        1: "있음"
    })

    display_df["나이"] = display_df["나이"].map(
        lambda x: f"{x:.1f}세"
    )

    display_df["평균 혈당"] = display_df[
        "평균 혈당"
    ].map(
        lambda x: f"{x:.2f}"
    )

    return display_df


# --------------------------------------------------
# 16. 모델 선택
# --------------------------------------------------

model_options = {
    "로지스틱 회귀 (확률로 답하는 모델)": logistic_pred,
    "의사결정트리 (질문으로 답하는 모델)": tree_pred
}

left_col, right_col = st.columns(2)

with left_col:

    missed_model_name = st.selectbox(
        "놓친 환자 목록의 모델 선택",
        options=list(model_options.keys()),
        key="missed_model"
    )

with right_col:

    false_alarm_model_name = st.selectbox(
        "헛짚은 사람 목록의 모델 선택",
        options=list(model_options.keys()),
        key="false_alarm_model"
    )


# --------------------------------------------------
# 17. 선택한 모델의 오류 목록 계산
# --------------------------------------------------

missed_df, false_alarm_df = make_error_dataframe(
    model_options[missed_model_name]
)

missed_df_2, false_alarm_df_2 = make_error_dataframe(
    model_options[false_alarm_model_name]
)


# --------------------------------------------------
# 18. 놓친 환자와 헛짚은 사람 목록 표시
# --------------------------------------------------

left_col, right_col = st.columns(2)

with left_col:

    st.markdown(
        f"#### 놓친 환자 ({len(missed_df):,}명)"
    )

    st.caption(
        f"선택 모델: {missed_model_name}"
    )

    st.write(
        "실제로 뇌졸중이지만 모델이 뇌졸중이 아니라고 예측한 사람입니다."
    )

    if len(missed_df) == 0:

        st.success("놓친 환자가 없습니다.")

    else:

        missed_display = prepare_display_data(
            missed_df.head(20)
        )

        st.dataframe(
            missed_display,
            use_container_width=True,
            hide_index=True
        )

        if len(missed_df) > 20:
            st.caption(
                f"전체 {len(missed_df):,}명 중 "
                "나이가 많은 순서로 앞의 20명만 표시했습니다."
            )


with right_col:

    st.markdown(
        f"#### 헛짚은 사람 ({len(false_alarm_df_2):,}명)"
    )

    st.caption(
        f"선택 모델: {false_alarm_model_name}"
    )

    st.write(
        "실제로는 뇌졸중이 아니지만 모델이 뇌졸중이라고 예측한 사람입니다."
    )

    if len(false_alarm_df_2) == 0:

        st.success("헛짚은 사람이 없습니다.")

    else:

        false_alarm_display = prepare_display_data(
            false_alarm_df_2.head(20)
        )

        st.dataframe(
            false_alarm_display,
            use_container_width=True,
            hide_index=True
        )

        if len(false_alarm_df_2) > 20:
            st.caption(
                f"전체 {len(false_alarm_df_2):,}명 중 "
                "나이가 많은 순서로 앞의 20명만 표시했습니다."
            )


# --------------------------------------------------
# 19. 혼동행렬과 목록의 인원수 비교
# --------------------------------------------------

st.divider()
st.subheader("8. 혼동행렬과 오류 목록 확인")

st.write(
    "아래 인원수는 앞에서 계산한 혼동행렬의 FN·FP 값과 "
    "오류 목록의 실제 인원수를 비교한 결과입니다."
)

# 선택한 모델의 혼동행렬
selected_missed_cm = calculate_confusion(
    y_test,
    model_options[missed_model_name]
)

selected_false_cm = calculate_confusion(
    y_test,
    model_options[false_alarm_model_name]
)

check_col1, check_col2 = st.columns(2)

with check_col1:

    st.markdown("**놓친 환자 (FN)**")

    st.write(
        f"혼동행렬 FN: {selected_missed_cm['FN']:,}명"
    )

    st.write(
        f"목록의 인원수: {len(missed_df):,}명"
    )

    if selected_missed_cm["FN"] == len(missed_df):
        st.success("혼동행렬의 FN과 목록의 인원수가 일치합니다.")
    else:
        st.error("인원수가 일치하지 않습니다.")


with check_col2:

    st.markdown("**헛짚은 사람 (FP)**")

    st.write(
        f"혼동행렬 FP: {selected_false_cm['FP']:,}명"
    )

    st.write(
        f"목록의 인원수: {len(false_alarm_df_2):,}명"
    )

    if selected_false_cm["FP"] == len(false_alarm_df_2):
        st.success("혼동행렬의 FP와 목록의 인원수가 일치합니다.")
    else:
        st.error("인원수가 일치하지 않습니다.")


# --------------------------------------------------
# 20. 마무리
# --------------------------------------------------

st.divider()

st.subheader("9. 결과 해석")

st.write(
    """
    - 정확도는 전체 테스트 데이터 중 올바르게 예측한 사람의 비율입니다.
    - 재현율은 실제 뇌졸중 환자 중 모델이 찾아낸 사람의 비율입니다.
    - 정밀도는 뇌졸중이라고 예측한 사람 중 실제 환자의 비율입니다.
    - F1 점수는 정밀도와 재현율을 함께 고려한 지표입니다.
    """
)

st.info(
    "뇌졸중 분류에서는 실제 환자를 놓치는 FN이 중요합니다. "
    "따라서 정확도뿐 아니라 재현율과 놓친 환자의 수도 함께 비교해야 합니다. "
    "이 페이지의 모델은 교육용 실습이며 실제 의료 진단에 사용할 수 없습니다."
)
