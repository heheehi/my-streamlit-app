import base64
import hashlib
import html
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from uuid import uuid4

import streamlit as st
from rag_core import rag_pipeline, source_label, user_error, check_training_card, CARD_RULE_URL, CARD_RULE_DATE

LOGO_URL = "https://raw.githubusercontent.com/Jangrae/img/master/aivle.png"
INDEX_VERSION = "1200-200-body-v4-mate"
st.set_page_config(page_title="AIVLE MATE · 나의 학습 캠퍼스", page_icon="🌿", layout="wide")

FAQ = {
    "입교 준비": ["입교 전에 무엇을 준비하면 좋을까요?", "비전공자가 미리 공부할 내용을 알려 주세요.", "교육 과정과 하루 일과는 어떻게 되나요?"],
    "공부 고민": ["수업 진도를 따라가기 어려울 때 어떻게 공부해야 하나요?", "백서에서 추천하는 복습 방법을 알려 주세요.", "백서의 조언을 바탕으로 일주일 학습 계획을 제안해 주세요."],
    "팀 프로젝트": ["미니프로젝트는 어떻게 준비하면 좋을까요?", "팀의 역할 분담과 협업 방법을 알려 주세요.", "프로젝트에서 나의 기여를 어떻게 정리하면 좋을까요?"],
    "취업 준비": ["관심 트랙의 취업 사례와 준비 방법을 알려 주세요.", "포트폴리오에 어떤 내용을 담으면 좋을까요?", "프로젝트 경험을 자기소개서와 면접에 어떻게 활용하나요?"],
}
st.markdown("""<style>
.stApp{background:linear-gradient(140deg,#f5faf8,#fffaf2 48%,#f5f2ff);color:#18213d;}
.block-container{max-width:1350px;padding-top:5.5rem;padding-bottom:3rem;}
h1,h2,h3{letter-spacing:-1px;color:#17203c;}h1{font-size:2.2rem!important;}
[data-testid="stSidebar"]{background:#f4f9f7;}
[data-testid="stVerticalBlockBorderWrapper"]{background:#ffffffbd;border-radius:20px;}
.stButton button{border-radius:12px;border-color:#d5e8e2;min-height:42px;}
.stButton button:hover{border-color:#168c86;color:#0d7875;}
.stButton button[kind="primary"]{background:#087e7c;border-color:#087e7c;}
[data-testid="stChatMessage"]{background:#ffffffd9;border:1px solid #e4ece8;border-radius:18px;margin-bottom:12px;padding:1.2rem 1.5rem!important;box-sizing:border-box;}
[data-testid="stChatMessageContent"]{min-width:0;overflow-wrap:anywhere;}
.st-key-campus_map{position:relative;isolation:isolate;}
.st-key-campus_map [class*="st-key-map_"]{position:absolute!important;width:31%!important;z-index:20;}
.st-key-campus_map [class*="st-key-map_"] button{width:100%;border-radius:30px;background:#fffef4;box-shadow:0 5px 16px #43694c20;color:#205953;font-weight:600;}
.st-key-map_admission{left:3%;top:64%;}.st-key-map_library{left:29%;top:30%;}
.st-key-map_project{left:67%;top:46%;}.st-key-map_career{left:51%;top:77%;}
@media(max-width:640px){.block-container{padding-top:5rem;padding-left:1rem;padding-right:1rem;}.st-key-campus_map button p{font-size:12px;}.st-key-campus_map [class*="st-key-map_"] button{min-height:34px;padding:2px;}}
[data-testid="stChatInput"]{border-radius:16px;border-color:#a8d4cb;}
.hero{padding:26px 30px;border-radius:24px;background:linear-gradient(120deg,#dff4e9,#fff1cf);margin-bottom:20px;}
.hero h1{margin:5px 0}.hero p{color:#52645e;margin:0;}
.eyebrow{font-size:13px;color:#07827c;font-weight:700;letter-spacing:1px;}
.st-key-page label[data-baseweb="radio"]{background:#fff9;padding:8px 15px;border-radius:12px;}
.st-key-page label[data-baseweb="radio"]:has(input:checked){background:#c9f1e5;color:#076b65;}
.st-key-page label[data-baseweb="radio"]>div:first-child{display:none;}
</style>""", unsafe_allow_html=True)

DEFAULTS = {"pipeline": None, "messages": [], "signature": None, "suggestions": [],
            "suggestion_error": None, "document_error": None, "active_file": None,
            "pending": None, "failed_question": None, "suggestion_round": 0,
            "tasks": [], "memo": "", "assessment": None, "assessment_profile": None,
            "page": "캠퍼스", "topic": "전체", "upload_epoch": 0, "suggestion_context": None,
            "stage": "교육 중", "track": "공통", "bookmarks": [], "faq_topic": "공부 고민",
            "pending_sources": [], "card_result": None}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
s = st.session_state


def learner_profile():
    return {"나의 단계": s.stage, "관심 트랙": s.track}


def now_text():
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M")


def record_data(entry):
    # JSON 내보내기에서는 Document 객체를 텍스트와 메타데이터로 바꿉니다.
    return {key: ([{"text": d.page_content, "metadata": d.metadata} for d in value]
                  if key == "sources" else value) for key, value in entry.items()}


def store_bookmark(entry):
    if not any(item["id"] == entry["id"] for item in s.bookmarks):
        s.bookmarks.append(dict(entry))


def remove_bookmark(entry_id):
    s.bookmarks = [item for item in s.bookmarks if item["id"] != entry_id]


def bookmark_button(message, question, prefix):
    # 저장 시점의 문서·쪽수·사용자 설정을 함께 보관합니다.
    message.setdefault("id", uuid4().hex)
    message.setdefault("time", now_text())
    message.setdefault("doc_name", s.active_file["name"] if s.active_file else "")
    message.setdefault("profile", learner_profile())
    entry = {"id": message["id"], "question": question, "content": message.get("content", message.get("answer", "")),
             "sources": message.get("sources", []), "time": message["time"],
             "doc_name": message["doc_name"], "profile": message["profile"]}
    saved = any(item["id"] == entry["id"] for item in s.bookmarks)
    st.button("✓ 저장한 답변" if saved else "☆ 답변 저장", key=prefix + entry["id"], disabled=saved,
              on_click=store_bookmark, args=(entry,))


def setting(name):
    try:
        return str(st.secrets.get(name, "")).strip()
    except FileNotFoundError:
        return ""


def reset_document(signature):
    if s.pipeline is not None:
        s.pipeline.close()
    s.pipeline, s.signature = None, signature
    s.messages, s.suggestions = [], []
    s.document_error = s.suggestion_error = s.pending = s.assessment = None
    s.failed_question = s.suggestion_context = None
    s.pending_sources, s.card_result = [], None
    s.suggestion_round = 0


def use_file(upload):
    if upload is None:
        return
    data = upload.getvalue()
    signature = hashlib.sha256(INDEX_VERSION.encode() + upload.name.encode() + data).hexdigest()
    if signature != s.signature:
        reset_document(signature)
    s.active_file = {"name": upload.name, "data": data}


def navigate(page, topic=None, question=None):
    # 버튼으로 이동한 페이지는 다음 화면 실행 전에 반영합니다.
    s.go_page = page
    if topic:
        s.go_topic = topic
    if question:
        s.pending = question
        s.pending_sources = []
    st.rerun()


def start_topic(topic):
    starter = s.pipeline.topic_starter(topic) if s.pipeline else None
    if starter:
        s.pending, s.pending_sources = starter["question"], starter["sources"]
        navigate("질문하기", topic)
    else:
        st.info("이 주제의 본문을 찾지 못했어요. 백서를 올리거나 다른 주제를 선택해 주세요.")


def add_task(text, source="직접 작성"):
    text = text.strip()
    if text and not any(t["text"] == text for t in s.tasks):
        s.tasks.append({"id": uuid4().hex, "text": text, "done": False, "source": source})


def remove_task(task_id):
    s.tasks = [t for t in s.tasks if t["id"] != task_id]
    s.pop("task_" + task_id, None)


def save_memo():
    # 페이지 이동 시 Streamlit이 입력 위젯을 정리해도 메모 원본은 남깁니다.
    s.memo = s.memo_input


def save_topic():
    s.topic = s.topic_widget


def sources_panel(docs):
    if docs:
        with st.expander("백서 근거 보기"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**[근거 {i}] {html.escape(source_label(doc))}**")
                st.text(doc.page_content)


def refresh_suggestions():
    latest = next((m for m in reversed(s.messages) if m["role"] == "assistant"), {})
    s.suggestions = s.pipeline.make_suggestions(history=[{"role": m["role"], "content": m["content"]}
        for m in s.messages[-4:]], sources=latest.get("sources"), offset=s.suggestion_round * 3,
        previous=s.suggestions, topic=s.topic, profile=learner_profile())
    s.suggestion_round += 1
    s.suggestion_context = (len(s.messages), s.topic, s.stage, s.track)
    s.suggestion_error = None


def hero(kicker, title, subtitle):
    st.markdown(f'<div class="hero"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def campus_art():
    # 확대해도 선명한 벡터 캠퍼스: 외부 이미지 생성 API가 필요 없습니다.
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 360"><rect width="820" height="360" rx="24" fill="#e8f3cf"/><circle cx="700" cy="52" r="32" fill="#ffdc78"/><path d="M0 315Q180 100 390 245T820 170" stroke="#fff2bd" stroke-width="65" fill="none"/><path d="M400 360L380 200L510 80" stroke="#fff2bd" stroke-width="45" fill="none"/>'
    for x, y, color, label in [(110, 235, "#e7be7a", "입교 안내소"), (350, 130, "#d3b78e", "학습 도서관"), (635, 165, "#69bfb8", "프로젝트 랩"), (490, 280, "#bda8dc", "커리어 센터")]:
        svg += f'<g transform="translate({x-65},{y-80})"><path d="M0 20L65 0L145 25L80 48Z" fill="{color}"/><path d="M0 20V90L80 120V48Z" fill="#fff6da"/><path d="M80 48V120L145 92V25Z" fill="{color}"/>'
        for a in (15, 42):
            svg += f'<path d="M{a} 45v25l17 6V51Z" fill="#7ec4c4"/>'
        svg += '</g>'
    for x, y in [(35,170),(220,60),(270,230),(470,70),(760,250),(730,115),(60,320),(490,295)]:
        svg += f'<path d="M{x} {y}v40" stroke="#8f9a56" stroke-width="6"/><circle cx="{x}" cy="{y}" r="22" fill="#8ebf68"/><circle cx="{x-10}" cy="{y+5}" r="15" fill="#a9d27c"/>'
    svg += '</svg>'
    encoded = base64.b64encode(svg.encode()).decode()
    st.markdown(f'<img alt="네 개의 학습 공간이 있는 캠퍼스 지도" style="width:100%;border-radius:24px" src="data:image/svg+xml;base64,{encoded}">', unsafe_allow_html=True)


if s.pipeline is not None and s.get("pipeline_version") != INDEX_VERSION:
    reset_document(s.signature)

# 업로더의 이전 값이 채팅에서 교체한 파일을 덮어쓰지 않도록 변경을 구분합니다.
with st.sidebar:
    st.image(LOGO_URL, width=165)
    st.selectbox("나의 단계", ["입교 전", "교육 중", "수료 후"], key="stage")
    st.selectbox("관심 트랙", ["공통", "AI 개발자", "DX 컨설턴트"], key="track")
    uploaded = st.file_uploader("백서 올리기", type=["pdf", "txt"], key=f"upload_{s.upload_epoch}")
    if uploaded is not None:
        widget_id = hashlib.sha256(uploaded.name.encode() + uploaded.getvalue()).hexdigest()
        if s.get("last_widget_file") != widget_id:
            use_file(uploaded)
            s.last_widget_file = widget_id
            s.file_origin = "sidebar"
    elif s.get("last_widget_file") is not None:
        s.last_widget_file = None
        if s.get("file_origin") == "sidebar":
            reset_document(None)
            s.active_file = None
    if s.active_file:
        st.caption(s.active_file["name"])
        st.download_button("백서 다운로드", s.active_file["data"], s.active_file["name"],
                           "application/pdf" if s.active_file["name"].lower().endswith(".pdf") else "text/plain")
        if st.button("문서 제거"):
            reset_document(None)
            s.active_file, s.last_widget_file = None, None
            s.upload_epoch += 1
            st.rerun()
    with st.expander("연결 · 이용 안내"):
        st.link_button("AIVLE 공식 홈페이지 ↗", "https://aivle.kt.co.kr/")
        st.link_button("AIVLE 인스타그램 ↗", "https://www.instagram.com/aivlestory/")
        st.link_button("잡다 역량검사 ↗", "https://www.jobda.im/acca/introduce")
        st.link_button("잡다 역량검사 연습 ↗", "https://www.jobda.im/acc/tutorial")
        st.link_button("고용24 ↗", "https://www.work24.go.kr/")

if s.active_file and s.pipeline is None:
    if not setting("OPENAI_API_KEY"):
        st.warning("앱 Secrets에 OPENAI_API_KEY를 설정해 주세요.")
    elif not s.document_error or st.sidebar.button("문서 준비 다시 시도"):
        try:
            with st.spinner("백서를 펼치고 있어요…"):
                s.pipeline = rag_pipeline(s.active_file["data"], s.active_file["name"], api_key=setting("OPENAI_API_KEY"),
                    base_url=setting("OPENAI_BASE_URL") or setting("OPENAI_API_BASE") or None,
                    organization=setting("OPENAI_ORGANIZATION") or None)
                s.pipeline_version = INDEX_VERSION
            s.document_error = None
        except Exception as error:
            s.document_error = user_error(error)
    if s.document_error:
        st.error(s.document_error)

if "go_page" in s:
    s.page = s.pop("go_page")
if "go_topic" in s:
    s.topic = s.pop("go_topic")
    s.pop("topic_widget", None)
if "import_memo" in s:
    s.memo = s.pop("import_memo")
    s.pop("memo_input", None)
brand, navigation = st.columns([1, 4])
with brand:
    st.image(LOGO_URL, width=155)
with navigation:
    st.radio("공간 이동", ["캠퍼스", "질문하기", "주제별 FAQ", "백서 살펴보기", "나의 기록", "지원 체크리스트"], key="page", horizontal=True, label_visibility="collapsed")
st.divider()

if s.page == "캠퍼스":
    left, right = st.columns([2.3, 1], gap="large")
    with left:
        hero("AIVLE MATE", "어디부터 둘러볼까요?", "나의 속도로 탐험하는 배움의 캠퍼스")
        with st.container(key="campus_map"):
            campus_art()
            for key, label, topic in [("admission", "🏫 입교 안내소", "입교 준비"), ("library", "📚 학습 도서관", "공부 고민"), ("project", "💻 프로젝트 랩", "팀 프로젝트"), ("career", "💼 커리어 센터", "취업 준비")]:
                with st.container(key="map_" + key):
                    if st.button(label + " →", use_container_width=True):
                        navigate("질문하기", topic)
    with right:
        with st.container(border=True):
            st.subheader("✧ 선배에게 물어보기")
            st.write("막막한 곳부터 함께 찾아봐요.")
            for q in ["어떤 것부터 준비할까요?", "공부 방법을 추천해 주세요.", "프로젝트는 어떻게 진행하나요?"]:
                if st.button(q, use_container_width=True):
                    navigate("질문하기", question=q)
            st.link_button("공식 모집 안내 ↗", "https://aivle.kt.co.kr/", use_container_width=True)
        with st.container(border=True):
            st.subheader("오늘의 작은 목표")
            st.write(f"{sum(t['done'] for t in s.tasks)} / {len(s.tasks)} 완료")
            if st.button("나의 학습 데스크 →"):
                navigate("나의 기록")

elif s.page == "주제별 FAQ":
    hero("AIVLE MATE · FAQ", "궁금한 주제부터 골라요.", "질문을 선택하면 지금 올린 백서에서 답을 찾습니다")
    topic = st.selectbox("FAQ 주제", list(FAQ), key="faq_topic")
    for i, question in enumerate(FAQ[topic]):
        if st.button(question, key=f"faq_{i}", disabled=s.pipeline is None, use_container_width=True):
            navigate("질문하기", topic, question)
    if s.pipeline is None:
        st.info("백서를 올리면 주제별 질문을 시작할 수 있어요.")

elif s.page == "백서 살펴보기":
    hero("WHITEPAPER", "백서를 직접 펼쳐보세요.", "전체 검색과 페이지별 원문 열람")
    if s.pipeline is None:
        st.info("먼저 PDF 또는 TXT를 올려 주세요.")
    else:
        pages = s.pipeline.pages
        keyword = st.text_input("백서 전체에서 검색", key="reader_search").strip()
        matches = [p for p in pages if not keyword or keyword.casefold() in p["text"].casefold()]
        is_pdf = s.active_file["name"].lower().endswith(".pdf")
        if not matches:
            st.info("검색어가 포함된 원문이 없습니다.")
        else:
            numbers = [p["page"] for p in matches]
            if s.get("reader_page") not in numbers:
                s.reader_page = numbers[0]
            number = st.selectbox("페이지 선택" if is_pdf else "문서 구간", numbers,
                format_func=lambda n: f"PDF {n}쪽" if is_pdf else "TXT 전체", key="reader_page")
            st.caption(f"{len(matches)} / {len(pages)}쪽 · 실제 PDF 순서" if is_pdf else "TXT 전체 원문")
            text = next(p["text"] for p in matches if p["page"] == number)
            if text.strip():
                st.text(text)
            else:
                st.info("이 페이지에는 추출된 텍스트가 없습니다.")

elif s.page == "나의 기록":
    hero("MY LEARNING DESK", "오늘은 한 걸음만 더.", "작은 실천을 모아 나의 학습 기록으로")
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("☑ 오늘의 작은 목표")
            with st.form("new_task", clear_on_submit=True):
                text = st.text_input("새 활동", placeholder="오늘 실천할 일을 적어 주세요")
                if st.form_submit_button("활동 추가"):
                    add_task(text)
                    st.rerun()
            for task in list(s.tasks):
                check, delete = st.columns([6, 1])
                task["done"] = check.checkbox(task["text"], value=task["done"], key="task_" + task["id"])
                delete.button("삭제", key="delete_task_" + task["id"], on_click=remove_task, args=(task["id"],))
            if s.tasks:
                st.progress(sum(t["done"] for t in s.tasks) / len(s.tasks))
        st.subheader("백서 더 알아보기")
        a, b = st.columns(2)
        if a.button("📖 비전공자 가이드", use_container_width=True):
            start_topic("공부 고민")
        if b.button("♧ 팀 프로젝트", use_container_width=True):
            start_topic("팀 프로젝트")
    with right:
        with st.container(border=True):
            st.subheader("✎ 나의 메모")
            st.text_area("학습 메모", value=s.memo, key="memo_input", on_change=save_memo,
                         height=220, label_visibility="collapsed")
        st.download_button("학습 기록 저장", json.dumps({"tasks": s.tasks, "memo": s.memo}, ensure_ascii=False, indent=2), "aivle_learning.json", "application/json")
        with st.expander("저장한 기록 불러오기"):
            imported = st.file_uploader("학습 기록 JSON", type=["json"], key="import_records")
            if st.button("기록 불러오기", disabled=imported is None):
                try:
                    data = json.loads(imported.getvalue())
                    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list) or not isinstance(data.get("memo", ""), str):
                        raise ValueError()
                    clean = []
                    for item in data["tasks"][:200]:
                        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not isinstance(item.get("done"), bool):
                            raise ValueError()
                        clean.append({"id": uuid4().hex, "text": item["text"][:500], "done": item["done"], "source": "불러온 기록"})
                    s.tasks = clean
                    s.import_memo = data.get("memo", "")[:20000]
                    st.rerun()
                except (ValueError, TypeError, KeyError):
                    st.error("올바른 학습 기록 파일을 선택해 주세요.")
    st.divider()
    st.subheader("☆ 저장한 답변")
    st.download_button("저장한 답변 다운로드", json.dumps([record_data(item) for item in s.bookmarks],
        ensure_ascii=False, indent=2).encode("utf-8"), "aivle_saved_answers.json", "application/json", disabled=not s.bookmarks)
    if not s.bookmarks:
        st.caption("대화나 체크리스트 결과에서 ☆ 답변 저장을 눌러 보세요.")
    for item in list(s.bookmarks):
        with st.expander(item["question"], expanded=True):
            st.caption(f"{item['doc_name']} · {item['time']} · {item['profile']['나의 단계']} · {item['profile']['관심 트랙']}")
            st.markdown(item["content"])
            sources_panel(item["sources"])
            st.button("저장 해제", key="remove_" + item["id"], on_click=remove_bookmark, args=(item["id"],))

elif s.page == "지원 체크리스트":
    hero("BEFORE YOU APPLY", "나의 상황으로 확인해요.", "내일배움카드 신청 가능성과 백서의 지원 기준 살펴보기")
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        with st.form("eligibility"):
            cohort = st.text_input("확인하려는 기수 / 공고 연도", placeholder="예: 백서에 소개된 10기")
            profile = {}
            st.subheader("현재 나의 상황")
            profile["만 나이"] = st.text_input("만 나이", placeholder="예: 25 · 모르면 비워 두세요")
            profile["국적"] = st.selectbox("국적", ["미확인", "대한민국", "외국인·체류자격 확인 필요"])
            profile["현재 일하는 형태"] = st.selectbox("현재 일하는 형태", ["미확인", "미취업·사업자등록 없음", "임금근로자", "공무원", "사립학교 교직원", "특수형태근로·프리랜서", "개인사업자", "법인대표", "비영리단체 대표", "휴·폐업 등 별도 상황"])
            profile["현재 학적"] = st.selectbox("현재 학적", ["미확인", "비재학·졸업", "고등학교 1~2학년", "고등학교 졸업예정학년", "대학·대학원 재학", "휴학·원격대학 등 별도 상황"])
            profile["졸업까지 남은 수업연한 (년)"] = st.text_input("졸업까지 남은 수업연한 (년)", placeholder="대학·대학원 재학생만 · 예: 1.5")
            profile["학력 / 졸업 예정 시기"] = st.text_input("최종 학력 / 졸업 예정 시기", placeholder="예: 4년제 대학, 2027년 2월 졸업 예정")
            profile["참여 가능한 시간"] = st.text_input("참여 가능한 시간", placeholder="예: 평일 전일 / 저녁만 가능")
            with st.expander("재직·사업 중이라면 입력해 주세요"):
                profile["회사 규모"] = st.selectbox("회사 규모", ["미확인", "대규모 기업", "중소·중견기업"])
                profile["근로 계약 형태"] = st.selectbox("근로 계약 형태", ["미확인", "일반 상용근로", "기간제·파견·단시간·일용 등"])
                for label in ["월 임금·소득 (만원)", "사업 기간 (개월)", "연 매출 (만원)"]:
                    profile[label] = st.text_input(label, placeholder="해당할 때 숫자만 입력 · 모르면 비워 두세요")
            st.subheader("현재 받고 있는 지원")
            profile["생계급여"] = st.selectbox("생계급여 수급 상태", ["미확인", "받지 않음", "일반 수급", "조건부 수급·조건부과 유예"])
            profile["다른 부처·지자체 훈련비"] = st.selectbox("다른 부처·지자체 교육훈련비", ["미확인", "받지 않음", "지원받는 중"])
            profile["부정수급 관련 상태"] = st.selectbox("훈련 지원 제한·미반환금", ["미확인", "해당 없음", "제한 기간 또는 미반환금 있음"])
            profile["내일배움카드 발급 상태"] = st.selectbox("현재 카드 상태", ["잘 모르겠음", "아직 신청하지 않음", "신청 중", "유효한 카드 보유", "만료·잔액 등 확인 필요"])
            profile["다른 정부지원 훈련 수강 이력"] = st.text_input("다른 정부지원 훈련 수강 이력", placeholder="예: 없음 / K-DT 수강한 과정과 시기")
            st.subheader("자비부담금 · 국민취업지원제도")
            profile["국민취업지원제도"] = st.selectbox("국취제 참여 유형", ["미확인", "미참여", "1유형", "2유형 (특정계층)", "2유형 (그 외)"])
            profile["면제 증빙 확인"] = st.checkbox("해당 과정의 면제 대상 증빙을 고용센터에서 확인했어요")
            profile["다른 면제 사유"] = st.text_input("다른 면제 사유 (선택)", placeholder="공식 확인한 사항만 입력")
            profile["기수·공고"] = cohort or "미확인"
            profile["학습 설정"] = learner_profile()
            profile["추가 상황 / 궁금한 점"] = st.text_area("추가 상황 / 궁금한 점").strip() or "미입력"
            profile["공식 과정 상세의 본인 결제금액"] = st.text_input("고용24 과정 상세의 본인 결제금액 (원, 선택)", placeholder="미확인이라면 비워 두세요").strip() or "미확인"
            submitted = st.form_submit_button("나의 상황 확인하기", type="primary")
        if submitted:
            # 실패한 재요청에서 이전 결과가 새 결과처럼 남지 않도록 먼저 비웁니다.
            s.assessment = s.assessment_profile = s.card_result = None
            try:
                s.card_result = check_training_card(profile)
                s.assessment_profile = dict(profile)
                if s.pipeline:
                    with st.spinner("백서의 지원 조건과 대조하고 있어요…"):
                        s.assessment = s.pipeline.check_application(profile)
            except ValueError as error:
                st.error(str(error))
            except Exception as error:
                st.error(user_error(error))
        if s.card_result:
            with st.container(border=True):
                st.subheader("국민내일배움카드 사전 확인")
                st.write("**" + s.card_result["status"] + "**")
                for reason in s.card_result["excluded"] + s.card_result["review"]:
                    st.write("• " + reason)
                st.write(s.card_result["next"])
                st.caption("입력 상황과 주요 제외 조건을 대조한 결과이며 최종 발급은 고용센터 심사로 결정됩니다. 카드 발급과 AIVLE 선발·K-DT 수강·결제금액은 별도입니다.")
                st.markdown(f"[고용24 지원 대상·발급 안내]({CARD_RULE_URL}) · 기준 확인 {CARD_RULE_DATE}")
                if not s.pipeline:
                    st.info("백서를 올리면 AIVLE 지원 요건과 자비부담금도 함께 대조할 수 있어요.")
        if s.assessment:
            with st.container(border=True):
                st.subheader("백서 기반 점검 결과")
                st.caption("마지막 제출 기준입니다. 값을 바꾸면 다시 확인해 주세요.")
                submitted_profile = s.assessment_profile or {}
                entered_fee = submitted_profile.get("공식 과정 상세의 본인 결제금액", "미확인")
                st.write("본인 입력 결제금액: " + (entered_fee + "원 (미검증)" if entered_fee != "미확인" else "미확인 · 고용24 과정 상세에서 확인"))
                st.markdown(s.assessment["answer"])
                sources_panel(s.assessment["sources"])
                bookmark_button(s.assessment, "지원 체크리스트 검토", "check_")
    with right:
        with st.container(border=True):
            st.subheader("확인부터 시작해요")
            st.write("최종 지원 자격과 본인 결제금액은 해당 기수 공고·고용24·고용센터에서 확인해 주세요.")
            st.link_button("AIVLE 공식 공고 ↗", "https://aivle.kt.co.kr/", use_container_width=True)
            st.link_button("고용24 과정·지원금 확인 ↗", "https://www.work24.go.kr/", use_container_width=True)
            if not s.pipeline:
                st.info("백서를 올리면 조건을 함께 확인할 수 있어요.")

else:
    topics, conversation, references = st.columns([0.75, 2.25, 1.1], gap="large")
    with topics:
        st.subheader("대화 주제")
        topic_options = ["전체", "입교 준비", "공부 고민", "팀 프로젝트", "취업 준비"]
        st.radio("주제 선택", topic_options, index=topic_options.index(s.topic), key="topic_widget",
                 on_change=save_topic, label_visibility="collapsed")
        if st.button("대화 삭제", use_container_width=True):
            s.messages, s.suggestions, s.pending = [], [], None
            s.suggestion_context = s.failed_question = None
            st.rerun()
        transcript = "\n\n".join(("나" if m["role"] == "user" else "학습 도우미") + ": " + m["content"] + "\n" +
            "\n".join(f"[근거 {i}] {source_label(d)}\n{d.page_content}" for i, d in enumerate(m.get("sources", []), 1)) for m in s.messages)
        st.download_button("대화 저장", transcript.encode("utf-8-sig"), "aivle_chat.txt", disabled=not transcript, use_container_width=True)
    with conversation:
        st.subheader("공부 고민, 편하게 물어봐요.")
        for i, message in enumerate(s.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    sources_panel(message["sources"])
                    previous_question = next((m["content"] for m in reversed(s.messages[:i]) if m["role"] == "user"), "저장한 답변")
                    bookmark_button(message, previous_question, "chat_")
                    if "actions" not in message and st.button("이 답변에서 실천할 활동 찾기", key=f"find_{i}"):
                        try:
                            message["actions"] = s.pipeline.make_actions(message["content"])
                            st.rerun()
                        except Exception as error:
                            st.error(user_error(error))
                    for j, action in enumerate(message.get("actions", [])):
                        saved = any(t["text"] == action for t in s.tasks)
                        if st.button(("✓ 저장됨 · " if saved else "+ 활동에 추가 · ") + action, key=f"action_{i}_{j}", disabled=saved):
                            add_task(action, message["content"])
                            st.rerun()
        if s.pending and s.pipeline:
            question, s.pending = s.pending, None
            # 먼저 질문만 기록하여 화면에 표시하고, 다음 실행에서 답을 생성합니다.
            if not s.messages or s.messages[-1]["role"] != "user" or s.messages[-1]["content"] != question:
                s.messages.append({"role": "user", "content": question})
                s.pending = question
                st.rerun()
            try:
                with st.chat_message("assistant"):
                    with st.spinner("백서에서 함께 찾아보고 있어요…"):
                        result = s.pipeline.invoke({"input": question, "history": s.messages[:-1], "profile": learner_profile(), "carry_sources": s.pending_sources})
                    st.markdown(result["answer"])
                s.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
                s.failed_question = s.suggestion_context = None
                s.pending_sources = []
                st.rerun()
            except Exception as error:
                s.failed_question = question
                st.error(user_error(error))
        if s.failed_question and st.button("답변 다시 시도"):
            s.pending, s.failed_question = s.failed_question, None
            st.rerun()
        if s.pipeline:
            if s.suggestion_context != (len(s.messages), s.topic, s.stage, s.track) and not s.suggestion_error:
                try:
                    refresh_suggestions()
                except Exception as error:
                    s.suggestion_error = user_error(error)
            for i, question in enumerate(s.suggestions):
                if st.button(question, key=f"suggestion_{i}", use_container_width=True):
                    s.pending = question
                    s.pending_sources = []
                    st.rerun()
            if st.button("추천 질문 다시 만들기"):
                try:
                    with st.spinner("새로운 질문을 찾고 있어요…"):
                        refresh_suggestions()
                    st.rerun()
                except Exception as error:
                    s.suggestion_error = user_error(error)
            if s.suggestion_error:
                st.warning(s.suggestion_error)
        # 입력창에서도 PDF/TXT 파일을 드래그 앤 드롭할 수 있습니다.
        incoming = st.chat_input("질문하거나 PDF · TXT를 놓아 주세요", accept_file=True, file_type=["pdf", "txt"])
        if incoming:
            text, files = (incoming, []) if isinstance(incoming, str) else (incoming.text, incoming.files)
            if files:
                use_file(files[0])
                s.file_origin = "chat"
            if text:
                s.pending = text
                s.pending_sources = []
            st.rerun()
        if not s.pipeline:
            st.info("백서를 첨부하면 대화를 시작할 수 있어요.")
    with references:
        with st.container(border=True):
            st.subheader("📖 함께 보는 백서")
            latest = next((m for m in reversed(s.messages) if m.get("sources")), None)
            if latest:
                st.caption(source_label(latest["sources"][0]))
                st.text(latest["sources"][0].page_content[:1800])
            else:
                st.write("대화를 시작하면 관련 백서가 이곳에 펼쳐져요.")
        with st.container(border=True):
            st.subheader("나의 작은 실천")
            st.write(f"{sum(t['done'] for t in s.tasks)} / {len(s.tasks)} 완료")
            if st.button("학습 데스크 열기 →"):
                navigate("나의 기록")
