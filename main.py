import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/goodrin09-code/data/refs/heads/main/stroke.csv"
RECORD_DIR = Path(__file__).parent / "내_기록"
RECORD_FILE = RECORD_DIR / "stroke_records.json"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
)

EXPECTED_COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke"
]


# ------------------------------------------------------------
# 기록 불러오기 / 저장하기
# ------------------------------------------------------------
def load_records():
    """저장된 우리말 뜻과 데이터 출처를 불러온다."""
    if not RECORD_FILE.exists():
        return {}, ""

    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            saved = json.load(file)
        return saved.get("meanings", {}), saved.get("source", "")
    except (OSError, json.JSONDecodeError):
        return {}, ""


def save_records(meanings, source):
    """우리말 뜻과 데이터 출처를 JSON 파일로 저장한다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "meanings": meanings,
        "source": source,
    }
    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)


# ------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    return data


try:
    df = load_data()
except Exception as error:
    st.title("🧠 뇌졸중 예측 실습실")
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.caption(f"오류 내용: {error}")
    st.stop()

# 열 이름이 예상과 다를 경우에도 화면이 중단되지 않도록 안내
missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
if missing_columns:
    st.warning(f"예상한 데이터 열이 일부 없습니다: {', '.join(missing_columns)}")


# ------------------------------------------------------------
# 저장된 사용자 기록 불러오기
# ------------------------------------------------------------
if "saved_meanings" not in st.session_state or "saved_source" not in st.session_state:
    meanings, source = load_records()
    st.session_state.saved_meanings = meanings
    st.session_state.saved_source = source


# ------------------------------------------------------------
# 화면 제목 및 데이터 소개
# ------------------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.subheader("데이터 살펴보기")
st.write(
    "이 데이터는 여러 사람의 기본 정보와 건강 관련 특성, "
    "그리고 뇌졸중 발생 여부를 담고 있습니다. "
    "각 열의 의미와 값의 특징을 확인하며 예측 실습에 사용할 데이터를 살펴봅니다."
)

st.info(
    "안내: 이 앱은 데이터 분석과 프로그래밍 학습을 위한 실습용입니다. "
    "실제 개인의 뇌졸중 발생 가능성을 진단하거나 의료적 판단을 내리는 용도로 사용할 수 없습니다."
)


# ------------------------------------------------------------
# 요약 카드 4개
# ------------------------------------------------------------
total_people = len(df)
column_count = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum()) if "stroke" in df.columns else 0
stroke_ratio = (stroke_count / total_people * 100) if total_people else 0

card1, card2, card3, card4 = st.columns(4)
card1.metric("전체 사람 수", f"{total_people:,}명")
card2.metric("열 개수", f"{column_count:,}개")
card3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
card4.metric("stroke가 1인 비율", f"{stroke_ratio:.2f}%")


# ------------------------------------------------------------
# 열 정보 표: 우리말 뜻은 사용자가 직접 입력
# ------------------------------------------------------------
st.divider()
st.subheader("열 이름과 데이터 구조")
st.write("교재를 참고해 **우리말 뜻** 칸을 직접 채운 뒤 저장하세요.")

saved_meanings = st.session_state.saved_meanings
column_info = []

for column in df.columns:
    non_null = df[column].dropna()
    unique_count = int(non_null.nunique())

    if pd.api.types.is_numeric_dtype(df[column]):
        value_kind = f"수치형 · 고유값 {unique_count:,}개"
    else:
        value_kind = f"범주형 · 고유값 {unique_count:,}개"

    column_info.append({
        "열 이름": column,
        "우리말 뜻": saved_meanings.get(column, ""),
        "값의 종류": value_kind,
        "빈 값 개수": int(df[column].isna().sum()),
    })

info_df = pd.DataFrame(
    column_info,
    columns=["열 이름", "우리말 뜻", "값의 종류", "빈 값 개수"]
)

edited_info = st.data_editor(
    info_df,
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름", width="medium"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 보고 각 열의 뜻을 직접 입력하세요.",
            width="large",
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
    },
    key="column_meanings_editor",
)

# ------------------------------------------------------------
# 데이터 미리보기
# ------------------------------------------------------------
st.divider()
st.subheader("데이터 미리보기")
st.caption("원본 데이터의 처음 다섯 행입니다.")
st.dataframe(df.head(5), hide_index=True, use_container_width=True)


# ------------------------------------------------------------
# 데이터 출처 입력 및 저장
# ------------------------------------------------------------
st.divider()
st.subheader("데이터 출처")
source_text = st.text_area(
    "교재에 적힌 데이터 출처를 입력하세요.",
    value=st.session_state.saved_source,
    placeholder="교재에 제시된 데이터 출처를 여기에 입력하세요.",
    height=100,
    key="source_text_area",
)

if st.button("우리말 뜻과 출처 저장", type="primary", use_container_width=True):
    updated_meanings = {}
    for _, row in edited_info.iterrows():
        updated_meanings[str(row["열 이름"])] = str(row["우리말 뜻"]) if pd.notna(row["우리말 뜻"]) else ""

    try:
        save_records(updated_meanings, source_text)
        st.session_state.saved_meanings = updated_meanings
        st.session_state.saved_source = source_text
        st.success("입력한 내용이 내_기록 폴더에 저장되었습니다.")
    except OSError as error:
        st.error("기록 파일을 저장하지 못했습니다. 폴더 쓰기 권한을 확인해 주세요.")
        st.caption(f"오류 내용: {error}")

st.caption("저장 파일 위치: 내_기록/stroke_records.json")
