
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ==================================================
# 1. 페이지 설정
# ==================================================

st.set_page_config(
    page_title="성능 높이기",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ 성능 높이기")

st.write(
    "뇌졸중 환자와 비환자의 데이터 수 차이를 고려하여 "
    "클래스 가중치를 적용하기 전과 후의 분류 성능을 비교합니다."
)

st.divider()


# ==================================================
# 2. 입력 속성 선택
# ==================================================

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


# ==================================================
# 3. 데이터 불러오기
# ==================================================

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


# ==================================================
# 4. 데이터 정렬 및 분리
# ==================================================

st.subheader("2. 학습 데이터와 테스트 데이터 분리")

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

# 뇌졸중 = 1 (양성)
# 뇌졸중 아님 = 0 (음성)

# 10명씩 묶어 앞의 3명을 테스트용으로 지정
test_mask = np.arange(len(df)) % 10 < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].copy()
y_test = test_df["stroke"].copy()


# BMI를 선택한 경우에만 학습용 중앙값으로 결측치 처리
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
        "선택한 속성에 결측값이 있습니다. 데이터를 확인해 주세요."
    )
    st.stop()


# 데이터 정보
col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(df):,}명")
col2.metric("학습 데이터", f"{len(train_df):,}명")
col3.metric("테스트 데이터", f"{len(test_df):,}명")

if len(test_df) != 1533:
    st.warning(
        f"테스트 데이터가 {len(test_df):,}명입니다. "
        "요청한 1,533명과 다르면 원본 데이터의 행 수를 확인하세요."
    )

# 학습용 클래스 분포
train_positive = int((y_train == 1).sum())
train_negative = int((y_train == 0).sum())

st.write(
    f"학습 데이터의 뇌졸중 환자: **{train_positive:,}명** / "
    f"비환자: **{train_negative:,}명**"
)

st.caption(
    "번호 순으로 정렬한 뒤 10명씩 묶어 앞의 3명은 테스트용, "
    "나머지 7명은 학습용으로 분리했습니다."
)


# ==================================================
# 5. 클래스 가중치 설명
# ==================================================

st.divider()
st.subheader("3. 클래스 가중치란?")

st.write(
    """
    뇌졸중 데이터는 뇌졸중을 겪은 사람보다 겪지 않은 사람이 훨씬 많습니다.
    이때 모델이 많은 쪽인 비환자만 잘 맞히려고 하면
    실제 환자를 놓치는 문제가 발생할 수 있습니다.

    클래스 가중치란 학습 과정에서 적은 수의 데이터에 더 큰 중요도를
    부여하여 두 집단의 영향력을 조정하는 방법입니다.
    """
)

st.markdown(
    """
    - **가중치 적용 전:** 모든 클래스에 동일한 기본 가중치를 적용합니다.
    - **가중치 적용 후:** `class_weight="balanced"`를 사용하여
      학습 데이터의 클래스별 빈도에 반비례하는 가중치를 적용합니다.
    """
)

st.info(
    "가중치는 학습 데이터에만 적용합니다. "
    "테스트 데이터는 두 모델의 성능을 비교하기 위해 동일하게 유지합니다."
)


# ==================================================
# 6. 로지스틱 회귀 모델
# ==================================================

st.divider()
st.subheader("4. 로지스틱 회귀 (확률로 답하는 모델)")

# 가중치 적용 전
logistic_before = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

logistic_before.fit(
    X_train,
    y_train
)

# 가중치 적용 후
logistic_after = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced"
    ))
])

logistic_after.fit(
    X_train,
    y_train
)


# ==================================================
# 7. 의사결정트리 모델
# ==================================================

st.divider()
st.subheader("5. 의사결정트리 (질문으로 답하는 모델)")

# 가중치 적용 전
tree_before = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_before.fit(
    X_train,
    y_train
)

# 가중치 적용 후
tree_after = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)

tree_after.fit(
    X_train,
    y_train
)


# ==================================================
# 8. 테스트 데이터 예측
# ==================================================

predictions = {
    "logistic_before": logistic_before.predict(X_test),
    "logistic_after": logistic_after.predict(X_test),
    "tree_before": tree_before.predict(X_test),
    "tree_after": tree_after.predict(X_test)
}


# ==================================================
# 9. 혼동행렬 계산
# ==================================================

def calculate_confusion(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

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


# ==================================================
# 10. 평가지표 계산
# ==================================================

def calculate_metrics(cm):

    TP = cm["TP"]
    FP = cm["FP"]
    FN = cm["FN"]
    TN = cm["TN"]

    total = TP + FP + FN + TN

    accuracy = (
        (TP + TN) / total
        if total > 0
        else None
    )

    # 실제 뇌졸중인 사람 중 찾아낸 비율
    recall = (
        TP / (TP + FN)
        if TP + FN > 0
        else None
    )

    # 뇌졸중이라고 예측한 사람 중 실제 환자의 비율
    if TP + FP == 0:
        precision = None
        f1 = None

    else:
        precision = TP / (TP + FP)

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


def fmt(value):

    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


# ==================================================
# 11. 네 가지 결과 계산
# ==================================================

results = {}

for key, pred in predictions.items():

    cm = calculate_confusion(
        y_test,
        pred
    )

    metrics = calculate_metrics(cm)

    results[key] = {
        "cm": cm,
        "metrics": metrics
    }


# ==================================================
# 12. 혼동행렬 표시 함수
# ==================================================

def make_confusion_table(cm):

    TP = cm["TP"]
    FP = cm["FP"]
    FN = cm["FN"]
    TN = cm["TN"]

    actual_negative = TN + FP
    actual_positive = FN + TP

    predicted_negative = TN + FN
    predicted_positive = FP + TP

    total = TP + FP + FN + TN

    return pd.DataFrame(
        [
            [
                f"TN: {TN:,}명",
                f"FP: {FP:,}명",
                f"{actual_negative:,}명"
            ],
            [
                f"FN: {FN:,}명",
                f"TP: {TP:,}명",
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


def make_metric_table(before, after):

    return pd.DataFrame(
        {
            "가중치 적용 전": [
                fmt(before["정확도"]),
                fmt(before["재현율"]),
                fmt(before["정밀도"]),
                fmt(before["F1"])
            ],
            "가중치 적용 후": [
                fmt(after["정확도"]),
                fmt(after["재현율"]),
                fmt(after["정밀도"]),
                fmt(after["F1"])
            ]
        },
        index=[
            "정확도",
            "재현율",
            "정밀도",
            "F1"
        ]
    )


# ==================================================
# 13. 로지스틱 회귀 전후 비교
# ==================================================

st.divider()
st.subheader("6. 로지스틱 회귀 성능 비교")

st.write(
    "클래스 가중치를 적용하기 전과 후의 혼동행렬 및 평가지표입니다."
)

before_col, after_col = st.columns(2)

with before_col:

    st.markdown("#### 가중치 적용 전")

    st.dataframe(
        make_confusion_table(
            results["logistic_before"]["cm"]
        ),
        use_container_width=True
    )

with after_col:

    st.markdown("#### 가중치 적용 후")

    st.dataframe(
        make_confusion_table(
            results["logistic_after"]["cm"]
        ),
        use_container_width=True
    )


st.markdown("#### 평가지표 비교")

st.dataframe(
    make_metric_table(
        results["logistic_before"]["metrics"],
        results["logistic_after"]["metrics"]
    ),
    use_container_width=True
)


# ==================================================
# 14. 의사결정트리 전후 비교
# ==================================================

st.divider()
st.subheader("7. 의사결정트리 성능 비교")

before_col, after_col = st.columns(2)

with before_col:

    st.markdown("#### 가중치 적용 전")

    st.dataframe(
        make_confusion_table(
            results["tree_before"]["cm"]
        ),
        use_container_width=True
    )

with after_col:

    st.markdown("#### 가중치 적용 후")

    st.dataframe(
        make_confusion_table(
            results["tree_after"]["cm"]
        ),
        use_container_width=True
    )


st.markdown("#### 평가지표 비교")

st.dataframe(
    make_metric_table(
        results["tree_before"]["metrics"],
        results["tree_after"]["metrics"]
    ),
    use_container_width=True
)


# ==================================================
# 15. 환자 발견 및 오류 인원 변화
# ==================================================

st.divider()
st.subheader("8. 실제 환자 발견 결과 변화")

st.write(
    "가중치를 적용한 뒤 실제 뇌졸중 환자를 얼마나 더 찾아냈는지 "
    "확인합니다."
)


def make_change_text(before, after):

    TP_before = before["TP"]
    TP_after = after["TP"]

    FN_before = before["FN"]
    FN_after = after["FN"]

    FP_before = before["FP"]
    FP_after = after["FP"]

    return (
        f"찾아낸 환자(TP): {TP_before:,}명 → {TP_after:,}명  |  "
        f"놓친 환자(FN): {FN_before:,}명 → {FN_after:,}명  |  "
        f"헛짚은 사람(FP): {FP_before:,}명 → {FP_after:,}명"
    )


st.markdown("**로지스틱 회귀 (확률로 답하는 모델)**")

st.write(
    make_change_text(
        results["logistic_before"]["cm"],
        results["logistic_after"]["cm"]
    )
)

st.divider()

st.markdown("**의사결정트리 (질문으로 답하는 모델)**")

st.write(
    make_change_text(
        results["tree_before"]["cm"],
        results["tree_after"]["cm"]
    )
)


# ==================================================
# 16. 시각화에 사용할 모델 선택
# ==================================================

st.divider()
st.subheader("9. 가중치 적용 전후 분류 경계 비교")

st.write(
    "선택한 모델이 실제 데이터를 어떻게 분류하는지 "
    "가중치 적용 전과 후를 시각적으로 비교합니다."
)

visual_model_name = st.selectbox(
    "시각화할 모델 선택",
    [
        "로지스틱 회귀 (확률로 답하는 모델)",
        "의사결정트리 (질문으로 답하는 모델)"
    ]
)

if visual_model_name.startswith("로지스틱"):

    visual_before = logistic_before
    visual_after = logistic_after

else:

    visual_before = tree_before
    visual_after = tree_after


# ==================================================
# 17. 축 선택
# ==================================================

if len(selected_features) == 2:

    st.markdown("#### 2차원 산점도")

    axis_col1, axis_col2 = st.columns(2)

    with axis_col1:

        x_feature = st.selectbox(
            "가로축",
            selected_features,
            format_func=lambda x: FEATURE_NAMES[x],
            index=0
        )

    y_options = [
        f for f in selected_features
        if f != x_feature
    ]

    with axis_col2:

        y_feature = st.selectbox(
            "세로축",
            y_options,
            format_func=lambda x: FEATURE_NAMES[x]
        )

else:

    st.markdown("#### 3차원 산점도")

    st.write(
        "선택한 속성 중 세 가지를 골라 세 축으로 사용합니다. "
        "나머지 속성은 테스트 데이터의 중앙값으로 고정합니다."
    )

    axis_col1, axis_col2, axis_col3 = st.columns(3)

    with axis_col1:

        x_feature = st.selectbox(
            "X축",
            selected_features,
            format_func=lambda x: FEATURE_NAMES[x],
            index=0
        )

    remaining_y = [
        f for f in selected_features
        if f != x_feature
    ]

    with axis_col2:

        y_feature = st.selectbox(
            "Y축",
            remaining_y,
            format_func=lambda x: FEATURE_NAMES[x],
            index=0
        )

    remaining_z = [
        f for f in selected_features
        if f not in [x_feature, y_feature]
    ]

    with axis_col3:

        z_feature = st.selectbox(
            "Z축",
            remaining_z,
            format_func=lambda x: FEATURE_NAMES[x],
            index=0
        )


# ==================================================
# 18. 나머지 속성의 중앙값 고정
# ==================================================

if len(selected_features) >= 3:

    fixed_values = {}

    for feature in selected_features:

        if feature not in [
            x_feature,
            y_feature,
            z_feature
        ]:

            fixed_values[feature] = float(
                X_test[feature].median()
            )

    if fixed_values:

        fixed_text = ", ".join(
            f"{FEATURE_NAMES[k]}: {v:.2f}"
            for k, v in fixed_values.items()
        )

        st.info(
            "세 축 이외의 속성은 테스트 데이터의 중앙값으로 고정합니다.\n\n"
            + fixed_text
        )

    else:

        st.info("선택한 세 속성 모두 축으로 사용합니다.")

else:

    fixed_values = {}


# ==================================================
# 19. 2차원 산점도
# ==================================================

if len(selected_features) == 2:

    x_min = float(X_test[x_feature].min())
    x_max = float(X_test[x_feature].max())

    y_min = float(X_test[y_feature].min())
    y_max = float(X_test[y_feature].max())

    x_pad = max((x_max - x_min) * 0.08, 0.1)
    y_pad = max((y_max - y_min) * 0.08, 0.1)

    x_range = np.linspace(
        x_min - x_pad,
        x_max + x_pad,
        180
    )

    y_range = np.linspace(
        y_min - y_pad,
        y_max + y_pad,
        180
    )

    xx, yy = np.meshgrid(
        x_range,
        y_range
    )

    grid_data = pd.DataFrame(
        {
            feature: np.zeros(xx.size)
            for feature in selected_features
        }
    )

    grid_data[x_feature] = xx.ravel()
    grid_data[y_feature] = yy.ravel()

    # 가중치 적용 전 확률
    before_prob = visual_before.predict_proba(
        grid_data
    )[:, 1].reshape(xx.shape)

    # 가중치 적용 후 확률
    after_prob = visual_after.predict_proba(
        grid_data
    )[:, 1].reshape(xx.shape)

    fig = go.Figure()

    # 실제 뇌졸중 여부에 따른 산점도
    for label, color, name in [
        (0, "#2878B5", "뇌졸중 아님 (0)"),
        (1, "#E64B35", "뇌졸중 (1)")
    ]:

        mask = y_test.to_numpy() == label

        fig.add_trace(
            go.Scatter(
                x=X_test.loc[mask, x_feature],
                y=X_test.loc[mask, y_feature],
                mode="markers",
                name=name,
                marker=dict(
                    color=color,
                    size=7,
                    opacity=0.65,
                    line=dict(
                        color="white",
                        width=0.5
                    )
                ),
                hovertemplate=(
                    f"{FEATURE_NAMES[x_feature]}: %{{x:.2f}}<br>"
                    f"{FEATURE_NAMES[y_feature]}: %{{y:.2f}}<br>"
                    f"실제 결과: {name}<extra></extra>"
                )
            )
        )

    # 가중치 적용 전 경계선
    fig.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=before_prob,
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring="none"
            ),
            line=dict(
                color="#00A878",
                width=3
            ),
            showscale=False,
            name="가중치 적용 전 0.5 경계선",
            hoverinfo="skip"
        )
    )

    # 가중치 적용 후 경계선
    fig.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=after_prob,
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring="none"
            ),
            line=dict(
                color="#9B59B6",
                width=3,
                dash="dash"
            ),
            showscale=False,
            name="가중치 적용 후 0.5 경계선",
            hoverinfo="skip"
        )
    )

    fig.update_layout(
        title=(
            f"{FEATURE_NAMES[x_feature]}과 "
            f"{FEATURE_NAMES[y_feature]}에 따른 분류 경계"
        ),
        xaxis_title=FEATURE_NAMES[x_feature],
        yaxis_title=FEATURE_NAMES[y_feature],
        height=650,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # 경계선이 그림 범위 안에 있는지 확인
    before_inside = (
        before_prob.min() <= 0.5 <= before_prob.max()
    )

    after_inside = (
        after_prob.min() <= 0.5 <= after_prob.max()
    )

    if before_inside:

        st.success(
            "가중치 적용 전의 0.5 경계선이 그림 범위 안에 있습니다."
        )

    else:

        st.warning(
            "가중치 적용 전의 0.5 경계선은 그림 범위 밖에 있습니다."
        )

    if after_inside:

        st.success(
            "가중치 적용 후의 0.5 경계선이 그림 범위 안에 있습니다."
        )

    else:

        st.warning(
            "가중치 적용 후의 0.5 경계선은 그림 범위 밖에 있습니다."
        )


# ==================================================
# 20. 3차원 산점도 및 결정 경계
# ==================================================

else:

    x_min = float(X_test[x_feature].min())
    x_max = float(X_test[x_feature].max())

    y_min = float(X_test[y_feature].min())
    y_max = float(X_test[y_feature].max())

    z_min = float(X_test[z_feature].min())
    z_max = float(X_test[z_feature].max())

    x_pad = max((x_max - x_min) * 0.05, 0.1)
    y_pad = max((y_max - y_min) * 0.05, 0.1)
    z_pad = max((z_max - z_min) * 0.05, 0.1)

    x_range = np.linspace(
        x_min - x_pad,
        x_max + x_pad,
        28
    )

    y_range = np.linspace(
        y_min - y_pad,
        y_max + y_pad,
        28
    )

    z_range = np.linspace(
        z_min - z_pad,
        z_max + z_pad,
        28
    )

    xx, yy, zz = np.meshgrid(
        x_range,
        y_range,
        z_range,
        indexing="ij"
    )

    grid_data = pd.DataFrame(
        {
            feature: np.full(
                xx.size,
                fixed_values[feature]
                if feature in fixed_values
                else 0.0
            )
            for feature in selected_features
        }
    )

    grid_data[x_feature] = xx.ravel()
    grid_data[y_feature] = yy.ravel()
    grid_data[z_feature] = zz.ravel()

    # 두 모델의 3차원 확률 계산
    before_prob = visual_before.predict_proba(
        grid_data
    )[:, 1]

    after_prob = visual_after.predict_proba(
        grid_data
    )[:, 1]

    # 격자점에서 0.5에 얼마나 가까운지 확인
    before_min = float(
        np.min(np.abs(before_prob - 0.5))
    )

    after_min = float(
        np.min(np.abs(after_prob - 0.5))
    )

    fig = go.Figure()

    # 실제 뇌졸중 여부에 따른 점
    for label, color, name in [
        (0, "#2878B5", "뇌졸중 아님 (0)"),
        (1, "#E64B35", "뇌졸중 (1)")
    ]:

        mask = y_test.to_numpy() == label

        fig.add_trace(
            go.Scatter3d(
                x=X_test.loc[mask, x_feature],
                y=X_test.loc[mask, y_feature],
                z=X_test.loc[mask, z_feature],
                mode="markers",
                name=name,
                marker=dict(
                    color=color,
                    size=4,
                    opacity=0.7
                ),
                hovertemplate=(
                    f"{FEATURE_NAMES[x_feature]}: %{{x:.2f}}<br>"
                    f"{FEATURE_NAMES[y_feature]}: %{{y:.2f}}<br>"
                    f"{FEATURE_NAMES[z_feature]}: %{{z:.2f}}<br>"
                    f"실제 결과: {name}<extra></extra>"
                )
            )
        )

    # 가중치 적용 전 0.5 평면
    if before_prob.min() <= 0.5 <= before_prob.max():

        fig.add_trace(
            go.Isosurface(
                x=xx.ravel(),
                y=yy.ravel(),
                z=zz.ravel(),
                value=before_prob,
                isomin=0.499,
                isomax=0.501,
                surface_count=1,
                opacity=0.28,
                colorscale=[
                    [0, "#00A878"],
                    [1, "#00A878"]
                ],
                showscale=False,
                caps=dict(
                    x_show=False,
                    y_show=False,
                    z_show=False
                ),
                name="가중치 적용 전 0.5 평면",
                hoverinfo="skip"
            )
        )

    # 가중치 적용 후 0.5 평면
    if after_prob.min() <= 0.5 <= after_prob.max():

        fig.add_trace(
            go.Isosurface(
                x=xx.ravel(),
                y=yy.ravel(),
                z=zz.ravel(),
                value=after_prob,
                isomin=0.499,
                isomax=0.501,
                surface_count=1,
                opacity=0.28,
                colorscale=[
                    [0, "#9B59B6"],
                    [1, "#9B59B6"]
                ],
                showscale=False,
                caps=dict(
                    x_show=False,
                    y_show=False,
                    z_show=False
                ),
                name="가중치 적용 후 0.5 평면",
                hoverinfo="skip"
            )
        )

    fig.update_layout(
        title="가중치 적용 전후 3차원 분류 경계",
        scene=dict(
            xaxis_title=FEATURE_NAMES[x_feature],
            yaxis_title=FEATURE_NAMES[y_feature],
            zaxis_title=FEATURE_NAMES[z_feature],
            aspectmode="auto"
        ),
        height=750,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # 평면 표시 여부
    if before_prob.min() <= 0.5 <= before_prob.max():

        st.success(
            "가중치 적용 전의 0.5 결정 경계가 3차원 그림 범위 안에 있습니다."
        )

    else:

        st.warning(
            "가중치 적용 전의 0.5 결정 경계는 "
            "현재 3차원 그림 범위 밖에 있습니다."
        )

    if after_prob.min() <= 0.5 <= after_prob.max():

        st.success(
            "가중치 적용 후의 0.5 결정 경계가 3차원 그림 범위 안에 있습니다."
        )

    else:

        st.warning(
            "가중치 적용 후의 0.5 결정 경계는 "
            "현재 3차원 그림 범위 밖에 있습니다."
        )

    st.caption(
        "3차원 그림은 마우스로 회전하거나 확대할 수 있습니다. "
        "두 경계는 선택한 모델의 뇌졸중 예측 확률이 0.5인 지점입니다. "
        "격자 기반으로 계산하므로 경계 표현은 근사적입니다."
    )


# ==================================================
# 21. 마무리
# ==================================================

st.divider()
st.subheader("10. 결과 해석")

st.write(
    """
    클래스 가중치를 적용하면 소수 클래스인 뇌졸중 환자에게
    더 큰 중요도를 부여하여 학습할 수 있습니다.

    일반적으로 가중치를 적용하면 실제 환자를 찾아내는 재현율이
    높아질 수 있지만, 뇌졸중이 아닌 사람을 환자로 잘못 판단하는
    경우도 증가할 수 있습니다.

    따라서 정확도 하나만으로 모델을 평가하기보다는
    재현율, 정밀도, F1, 혼동행렬을 함께 비교하는 것이 중요합니다.
    """
)

st.info(
    "가중치 적용 후 반드시 모든 평가지표가 좋아지는 것은 아닙니다. "
    "데이터와 모델에 따라 결과가 달라지므로 실제 실행 결과를 "
    "비교하여 어떤 모델이 더 적절한지 판단해야 합니다. "
    "이 모델은 교육용 실습이며 실제 의료 진단에 사용할 수 없습니다."
)
