
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="분류 모델",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 분류 모델")
st.write(
    "선택한 건강 정보를 바탕으로 뇌졸중 여부를 분류하는 "
    "인공지능 모델을 만들고 성능을 비교합니다."
)

st.divider()


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "goodrin09-code/data/refs/heads/main/stroke.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 열 이름 설정
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

# 처음 선택되어 있는 속성
DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]


# --------------------------------------------------
# 입력 속성 선택
# --------------------------------------------------

st.subheader("1. 분류에 사용할 속성 선택")

st.write(
    "뇌졸중 여부를 예측하는 데 사용할 건강 정보를 선택하세요. "
    "두 개 이상의 속성을 선택해야 합니다."
)

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
    st.warning("분류 모델을 만들려면 속성을 두 개 이상 선택해야 합니다.")
    st.stop()

st.success(
    "선택한 속성: "
    + ", ".join(FEATURE_NAMES[f] for f in selected_features)
)


# --------------------------------------------------
# 데이터 정렬 및 학습용 / 테스트용 분리
# --------------------------------------------------

st.subheader("2. 학습 데이터와 테스트 데이터 분리")

# 번호 순으로 정렬
df = df.sort_values("id", kind="mergesort").reset_index(drop=True)

# 뇌졸중 여부
# 1 = 뇌졸중 (양성)
# 0 = 뇌졸중 아님 (음성)

df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")
df = df.dropna(subset=["stroke"]).reset_index(drop=True)
df["stroke"] = df["stroke"].astype(int)

if not set(df["stroke"].unique()).issubset({0, 1}):
    st.error("stroke 열에는 0과 1만 있어야 합니다.")
    st.stop()

# 10명씩 묶어 앞의 3명을 테스트 데이터로 지정
test_mask = np.arange(len(df)) % 10 < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

# 입력 데이터와 정답 분리
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

# 다른 선택 속성에 결측치가 있는지 확인
if X_train.isnull().any().any() or X_test.isnull().any().any():
    st.error(
        "선택한 속성에 BMI 이외의 결측값이 있습니다. "
        "데이터를 확인해 주세요."
    )
    st.stop()

# 데이터 분리 결과
col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(df):,}명")
col2.metric("학습 데이터", f"{len(train_df):,}명")
col3.metric("테스트 데이터", f"{len(test_df):,}명")

if len(test_df) != 1533:
    st.warning(
        f"현재 테스트 데이터는 {len(test_df):,}명입니다. "
        "요청한 1,533명과 다르므로 원본 데이터의 행 수를 확인해 주세요."
    )

st.caption(
    "번호 순으로 정렬한 뒤 10명씩 묶어 앞의 3명은 테스트용, "
    "나머지 7명은 학습용으로 고정했습니다."
)


# --------------------------------------------------
# 3. 모델 생성 및 학습
# --------------------------------------------------

st.divider()
st.subheader("3. 분류 모델 학습")

# 로지스틱 회귀
# 표준화는 학습 데이터에만 맞추고 테스트 데이터에 적용
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

logistic_model.fit(X_train, y_train)


# 의사결정트리
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(X_train, y_train)


# --------------------------------------------------
# 4. 입력을 보지 않는 기준 모델
# --------------------------------------------------

# 학습 데이터에서 더 많은 쪽의 클래스를 항상 예측
majority_class = int(y_train.value_counts().idxmax())

baseline_train_pred = np.full(
    len(y_train), majority_class
)

baseline_test_pred = np.full(
    len(y_test), majority_class
)


# --------------------------------------------------
# 5. 정확도 계산
# --------------------------------------------------

# 로지스틱 회귀
log_train_acc = logistic_model.score(X_train, y_train)
log_test_acc = logistic_model.score(X_test, y_test)

# 의사결정트리
tree_train_acc = tree_model.score(X_train, y_train)
tree_test_acc = tree_model.score(X_test, y_test)

# 기준 모델
baseline_train_acc = np.mean(baseline_train_pred == y_train)
baseline_test_acc = np.mean(baseline_test_pred == y_test)


# --------------------------------------------------
# 6. 정확도 카드
# --------------------------------------------------

st.subheader("4. 모델별 정확도 비교")

def accuracy_card(title, train_acc, test_acc):
    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(128,128,128,0.35);
            border-radius: 12px;
            padding: 18px;
            min-height: 165px;
        ">
            <div style="
                font-size: 17px;
                font-weight: 650;
                margin-bottom: 12px;
            ">{title}</div>

            <div style="
                font-size: 38px;
                font-weight: 750;
                margin-bottom: 12px;
            ">{test_acc * 100:.2f}%</div>

            <div style="
                font-size: 13px;
                color: gray;
            ">
                학습 데이터 정확도: {train_acc * 100:.2f}%<br>
                테스트 데이터 정확도: {test_acc * 100:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


col1, col2, col3 = st.columns(3)

with col1:
    accuracy_card(
        "로지스틱 회귀 (확률로 답하는 모델)",
        log_train_acc,
        log_test_acc
    )

with col2:
    accuracy_card(
        "의사결정트리 (질문으로 답하는 모델)",
        tree_train_acc,
        tree_test_acc
    )

with col3:
    accuracy_card(
        "입력을 보지 않는 기준 모델",
        baseline_train_acc,
        baseline_test_acc
    )


st.caption(
    "정확도는 전체 예측 중 실제 정답과 일치한 비율입니다. "
    "큰 숫자는 테스트 데이터의 정확도입니다."
)


# --------------------------------------------------
# 7. 산점도 축 선택
# --------------------------------------------------

st.divider()
st.subheader("5. 두 속성에 따른 분류 결과 시각화")

st.write(
    "선택한 속성 중 두 가지를 골라 가로축과 세로축으로 사용합니다. "
    "점의 색은 실제 뇌졸중 여부를 나타냅니다."
)

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축",
        options=selected_features,
        format_func=lambda x: FEATURE_NAMES[x],
        index=0
    )

with axis_col2:
    y_options = [
        f for f in selected_features
        if f != x_feature
    ]

    y_feature = st.selectbox(
        "세로축",
        options=y_options,
        format_func=lambda x: FEATURE_NAMES[x],
        index=0
    )


# --------------------------------------------------
# 8. 나머지 속성의 값 고정
# --------------------------------------------------

# 두 축 이외의 속성은 테스트 데이터의 중앙값으로 고정
fixed_values = {}

for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        fixed_values[feature] = float(X_test[feature].median())

if fixed_values:
    fixed_text = ", ".join(
        f"{FEATURE_NAMES[k]}: {v:.2f}"
        for k, v in fixed_values.items()
    )

    st.info(
        "그림을 그릴 때 두 축 이외의 속성은 "
        "테스트 데이터의 중앙값으로 고정했습니다.\n\n"
        + fixed_text
    )
else:
    st.info("선택한 두 속성이 모든 입력 속성이므로 고정할 속성이 없습니다.")


# --------------------------------------------------
# 9. 예측 영역 계산
# --------------------------------------------------

# 그림의 범위 설정
x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

x_padding = max((x_max - x_min) * 0.08, 0.1)
y_padding = max((y_max - y_min) * 0.08, 0.1)

x_range = np.linspace(
    x_min - x_padding,
    x_max + x_padding,
    150
)

y_range = np.linspace(
    y_min - y_padding,
    y_max + y_padding,
    150
)

xx, yy = np.meshgrid(x_range, y_range)

# 다른 속성은 테스트 데이터 중앙값으로 고정
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

# 로지스틱 회귀의 뇌졸중 확률
grid_probability = logistic_model.predict_proba(
    grid_data
)[:, 1].reshape(xx.shape)

# 의사결정트리의 예측 영역
grid_tree_prediction = tree_model.predict(
    grid_data
).reshape(xx.shape)


# --------------------------------------------------
# 10. Plotly 산점도
# --------------------------------------------------

fig = go.Figure()

# 의사결정트리의 분류 영역
fig.add_trace(
    go.Heatmap(
        x=x_range,
        y=y_range,
        z=grid_tree_prediction,
        colorscale=[
            [0, "rgba(255,180,180,0.32)"],
            [1, "rgba(160,210,255,0.32)"]
        ],
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 분류 영역"
    )
)

# 로지스틱 회귀의 0.5 경계선
fig.add_trace(
    go.Contour(
        x=x_range,
        y=y_range,
        z=grid_probability,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="none",
            showlabels=True
        ),
        line=dict(
            color="black",
            width=3
        ),
        showscale=False,
        hoverinfo="skip",
        name="로지스틱 회귀 0.5 경계선"
    )
)


# 실제 뇌졸중 여부에 따른 점 표시
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
                opacity=0.7,
                line=dict(
                    width=0.5,
                    color="white"
                )
            ),
            hovertemplate=(
                f"{FEATURE_NAMES[x_feature]}: %{{x:.2f}}<br>"
                f"{FEATURE_NAMES[y_feature]}: %{{y:.2f}}<br>"
                f"실제 결과: {name}<extra></extra>"
            )
        )
    )


fig.update_layout(
    title="테스트 데이터의 실제 결과와 모델 분류 영역",
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
if grid_probability.min() <= 0.5 <= grid_probability.max():
    st.success(
        "로지스틱 회귀의 0.5 확률 경계선이 그림 범위 안에 있습니다."
    )
else:
    st.warning(
        "로지스틱 회귀의 0.5 확률 경계선은 "
        "현재 그림 범위 안에 나타나지 않습니다."
    )

st.caption(
    "검은 선은 로지스틱 회귀가 뇌졸중 확률 0.5를 기준으로 "
    "분류하는 경계입니다. 배경색은 의사결정트리의 분류 영역입니다. "
    "배경색과 점의 색은 서로 다른 정보를 나타냅니다."
)


# --------------------------------------------------
# 11. 의사결정트리 가지 그림
# --------------------------------------------------

st.divider()
st.subheader("6. 의사결정트리의 질문 구조")

st.write(
    "의사결정트리는 선택한 속성에 관한 질문을 최대 세 번까지 하며 "
    "뇌졸중 여부를 분류합니다."
)

tree = tree_model.tree_

children_left = tree.children_left
children_right = tree.children_right
features = tree.feature
thresholds = tree.threshold

# 각 노드에 도달한 학습 데이터의 사람 수 계산
decision_path = tree_model.decision_path(X_train)

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, nodesep=0.45, ranksep=0.65];',
    'node [shape=box, style="rounded,filled", '
    'fontname="Malgun Gothic", fontsize=10, margin=0.15];',
    'edge [fontname="Malgun Gothic", fontsize=10];'
]

leaf_count = 0
negative_leaf_count = 0
positive_leaf_count = 0

used_features = set()


# --------------------------------------------------
# 12. 노드별 정보 계산
# --------------------------------------------------

for node_id in range(tree.node_count):

    # 해당 노드에 도달한 학습 데이터 인덱스
    indices = decision_path.getcol(node_id).indices

    node_total = len(indices)

    # 해당 노드의 실제 뇌졸중 환자 수
    node_positive = int(y_train.iloc[indices].sum())

    node_ratio = (
        node_positive / node_total * 100
        if node_total > 0
        else 0
    )

    # 말단 노드인지 확인
    is_leaf = children_left[node_id] == children_right[node_id]

    if is_leaf:

        leaf_count += 1

        # 해당 노드의 예측 결과
        predicted_class = int(
            tree_model.classes_[
                np.argmax(tree.value[node_id][0])
            ]
        )

        if predicted_class == 0:
            negative_leaf_count += 1
            fill_color = "lightblue"
            answer = "뇌졸중 아님"
        else:
            positive_leaf_count += 1
            fill_color = "mistyrose"
            answer = "뇌졸중"

        label = (
            f"답: {answer}\\n"
            f"전체: {node_total}명\\n"
            f"뇌졸중: {node_positive}명\\n"
            f"비율: {node_ratio:.1f}%"
        )

    else:

        feature_name = FEATURES[features[node_id]]
        used_features.add(feature_name)

        korean_name = FEATURE_NAMES[feature_name]
        threshold = thresholds[node_id]

        # 질문 노드
        label = (
            f"{korean_name} ≤ {threshold:.2f}?\\n"
            f"전체: {node_total}명\\n"
            f"뇌졸중: {node_positive}명\\n"
            f"비율: {node_ratio:.1f}%"
        )

        fill_color = "lightyellow"

    dot_lines.append(
        f'n{node_id} [label={json.dumps(label, ensure_ascii=False)}, '
        f'fillcolor="{fill_color}"];'
    )


# --------------------------------------------------
# 13. 가지 연결
# --------------------------------------------------

for node_id in range(tree.node_count):

    if children_left[node_id] == children_right[node_id]:
        continue

    left_id = children_left[node_id]
    right_id = children_right[node_id]

    # 왼쪽 가지: 조건을 만족하는 경우
    dot_lines.append(
        f'n{node_id} -> n{left_id} [label="예"];'
    )

    # 오른쪽 가지: 조건을 만족하지 않는 경우
    dot_lines.append(
        f'n{node_id} -> n{right_id} [label="아니요"];'
    )


dot_lines.append("}")

dot_string = "\n".join(dot_lines)

# Streamlit 내장 Graphviz 기능 사용
st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# --------------------------------------------------
# 14. 의사결정트리 분석
# --------------------------------------------------

st.subheader("의사결정트리 구조 분석")

col1, col2, col3 = st.columns(3)

col1.metric(
    "전체 답변 마디 수",
    f"{leaf_count}개"
)

col2.metric(
    "뇌졸중 아님으로 답하는 마디",
    f"{negative_leaf_count}개"
)

col3.metric(
    "뇌졸중으로 답하는 마디",
    f"{positive_leaf_count}개"
)

used_feature_names = [
    FEATURE_NAMES[f]
    for f in selected_features
    if f in used_features
]

if used_feature_names:
    used_text = ", ".join(used_feature_names)
else:
    used_text = "없음"

st.write(
    f"**이 나무가 실제로 질문한 속성:** {used_text}"
)

st.caption(
    "각 마디의 전체 인원과 뇌졸중 인원은 학습 데이터 기준입니다. "
    "뇌졸중 비율은 해당 마디에 도달한 사람 중 실제 뇌졸중인 사람의 비율입니다."
)

st.caption(
    "파란색 마디는 뇌졸중이 아니라고 답하고, "
    "분홍색 마디는 뇌졸중이라고 답합니다. "
    "노란색 마디는 다음 질문으로 이어지는 마디입니다."
)


# --------------------------------------------------
# 15. 모델 비교 및 해석
# --------------------------------------------------

st.divider()
st.subheader("7. 모델 비교 결과")

comparison_df = pd.DataFrame({
    "모델": [
        "로지스틱 회귀",
        "의사결정트리",
        "입력을 보지 않는 기준 모델"
    ],
    "학습 데이터 정확도": [
        log_train_acc,
        tree_train_acc,
        baseline_train_acc
    ],
    "테스트 데이터 정확도": [
        log_test_acc,
        tree_test_acc,
        baseline_test_acc
    ]
})

comparison_display = comparison_df.copy()

comparison_display["학습 데이터 정확도"] = (
    comparison_display["학습 데이터 정확도"] * 100
).map(lambda x: f"{x:.2f}%")

comparison_display["테스트 데이터 정확도"] = (
    comparison_display["테스트 데이터 정확도"] * 100
).map(lambda x: f"{x:.2f}%")

st.dataframe(
    comparison_display,
    use_container_width=True,
    hide_index=True
)

best_model = max(
    [
        ("로지스틱 회귀", log_test_acc),
        ("의사결정트리", tree_test_acc),
        ("입력을 보지 않는 기준 모델", baseline_test_acc)
    ],
    key=lambda x: x[1]
)

st.write(
    f"테스트 데이터에서 가장 높은 정확도를 보인 모델은 "
    f"**{best_model[0]}**이며, 정확도는 "
    f"**{best_model[1] * 100:.2f}%**입니다."
)

st.info(
    "정확도만으로 뇌졸중 예측 모델의 성능을 완전히 판단할 수는 없습니다. "
    "뇌졸중 데이터처럼 양성 데이터와 음성 데이터의 수에 차이가 있는 경우 "
    "실제 뇌졸중 환자를 얼마나 잘 찾아내는지도 함께 살펴볼 필요가 있습니다. "
    "이 모델은 교육용 분류 실습이며 실제 의료 진단에 사용할 수 없습니다."
)
