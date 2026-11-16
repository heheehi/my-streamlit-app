import base64
import hashlib
import html
import json
from uuid import uuid4

import streamlit as st
from rag_core import rag_pipeline, source_label, user_error

LOGO_URL = "https://raw.githubusercontent.com/Jangrae/img/master/aivle.png"
INDEX_VERSION = "1200-200-body-v3-desk"
st.set_page_config(page_title="AIVLE · 나의 학습 캠퍼스", page_icon="🌿", layout="wide")
st.markdown("""<style>
.stApp{background:linear-gradient(140deg,#f5faf8,#fffaf2 48%,#f5f2ff);color:#18213d;}
.block-container{max-width:1350px;padding-top:1.8rem;padding-bottom:3rem;}
h1,h2,h3{letter-spacing:-1px;color:#17203c;}h1{font-size:2.2rem!important;}
[data-testid="stSidebar"]{background:#f4f9f7;}
[data-testid="stVerticalBlockBorderWrapper"]{background:#ffffffbd;border-radius:20px;}
.stButton button{border-radius:12px;border-color:#d5e8e2;min-height:42px;}
.stButton button:hover{border-color:#168c86;color:#0d7875;}
.stButton button[kind="primary"]{background:#087e7c;border-color:#087e7c;}
[data-testid="stChatMessage"]{background:#ffffffd9;border:1px solid #e4ece8;border-radius:18px;margin-bottom:12px;}
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
            "page": "캠퍼스", "topic": "전체", "upload_epoch": 0, "suggestion_context": None}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
s = st.session_state


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
    st.rerun()


def add_task(text, source="직접 작성"):
    text = text.strip()
    if text and not any(t["text"] == text for t in s.tasks):
        s.tasks.append({"id": uuid4().hex, "text": text, "done": False, "source": source})


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
        previous=s.suggestions, topic=s.topic)
    s.suggestion_round += 1
    s.suggestion_context = (len(s.messages), s.topic)
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
        svg += f'<rect x="25" y="123" width="115" height="26" rx="13" fill="white"/><text x="82" y="141" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#24534b">{label}</text></g>'
    for x, y in [(35,170),(220,60),(270,230),(470,70),(760,250),(730,115),(60,320),(490,295)]:
        svg += f'<path d="M{x} {y}v40" stroke="#8f9a56" stroke-width="6"/><circle cx="{x}" cy="{y}" r="22" fill="#8ebf68"/><circle cx="{x-10}" cy="{y+5}" r="15" fill="#a9d27c"/>'
    svg += '</svg>'
    encoded = base64.b64encode(svg.encode()).decode()
    st.markdown(f'<img alt="네 개의 학습 공간이 있는 캠퍼스 지도" style="width:100%;border-radius:24px" src="data:image/svg+xml;base64,{encoded}">', unsafe_allow_html=True)


# 업로더의 이전 값이 채팅에서 교체한 파일을 덮어쓰지 않도록 변경을 구분합니다.
with st.sidebar:
    st.image(LOGO_URL, width=165)
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
        st.caption("문서와 질문은 설정된 API로 전송됩니다. 최근 6턴을 대화에 반영합니다. 기록은 현재 접속 세션에 보관됩니다. PDF 쪽수는 파일 순서입니다.")

if s.active_file and s.pipeline is None:
    if not setting("OPENAI_API_KEY"):
        st.warning("앱 Secrets에 OPENAI_API_KEY를 설정해 주세요.")
    elif not s.document_error or st.sidebar.button("문서 준비 다시 시도"):
        try:
            with st.spinner("백서를 펼치고 있어요…"):
                s.pipeline = rag_pipeline(s.active_file["data"], s.active_file["name"], api_key=setting("OPENAI_API_KEY"),
                    base_url=setting("OPENAI_BASE_URL") or setting("OPENAI_API_BASE") or None,
                    organization=setting("OPENAI_ORGANIZATION") or None)
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
    st.radio("공간 이동", ["캠퍼스", "질문하기", "나의 기록", "지원 체크리스트"], key="page", horizontal=True, label_visibility="collapsed")
st.divider()

if s.page == "캠퍼스":
    left, right = st.columns([2.3, 1], gap="large")
    with left:
        hero("AIVLE SCHOOL 학습도우미", "어디부터 둘러볼까요?", "나의 속도로 탐험하는 배움의 캠퍼스")
        campus_art()
        for col, (label, topic) in zip(st.columns(4), [("입교 안내소", "입교 준비"), ("학습 도서관", "공부 고민"), ("프로젝트 랩", "팀 프로젝트"), ("커리어 센터", "취업 준비")]):
            if col.button(label + " →", use_container_width=True):
                navigate("지원 체크리스트" if topic == "입교 준비" else "질문하기", topic)
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
            for task in s.tasks:
                task["done"] = st.checkbox(task["text"], value=task["done"], key="task_" + task["id"])
            if s.tasks:
                st.progress(sum(t["done"] for t in s.tasks) / len(s.tasks))
        st.subheader("백서 더 알아보기")
        a, b = st.columns(2)
        if a.button("📖 비전공자 가이드", use_container_width=True):
            navigate("질문하기", "공부 고민", "비전공자가 학습을 시작할 때 어떤 순서로 준비하면 좋을까요?")
        if b.button("♧ 팀 프로젝트", use_container_width=True):
            navigate("질문하기", "팀 프로젝트", "팀 프로젝트에서 협업과 결과 정리를 어떻게 하면 좋을까요?")
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

elif s.page == "지원 체크리스트":
    hero("BEFORE YOU APPLY", "지원 전, 하나씩 확인해요.", "백서의 기준과 나의 준비 상태를 함께 살펴봐요")
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        with st.form("eligibility"):
            cohort = st.text_input("확인하려는 기수 / 공고 연도", placeholder="예: 백서에 소개된 10기")
            profile = {}
            for label in ["해당 공고의 연령 기준", "해당 공고의 학력 기준", "교육 시작일 기준 취업 상태 요건", "교육 시간 전체 참여", "내일배움카드 보유·발급 가능", "K-DT 수강 가능 여부"]:
                profile[label] = st.radio(label, ["미확인", "충족", "미충족"], horizontal=True, key="elig_" + label)
            st.subheader("자비부담금 · 국민취업지원제도")
            profile["국민취업지원제도"] = st.selectbox("국취제 참여 유형", ["미확인", "미참여", "1유형", "2유형 (특정계층)", "2유형 (그 외)"])
            profile["면제 증빙 확인"] = st.checkbox("해당 과정의 면제 대상 증빙을 고용센터에서 확인했어요")
            profile["다른 면제 사유"] = st.text_input("다른 면제 사유 (선택)", placeholder="공식 확인한 사항만 입력")
            profile["기수·공고"] = cohort or "미확인"
            profile["공식 과정 상세의 본인 결제금액"] = st.text_input("고용24 과정 상세의 본인 결제금액 (원, 선택)", placeholder="미확인이라면 비워 두세요").strip() or "미확인"
            submitted = st.form_submit_button("백서로 체크리스트 확인", type="primary", disabled=s.pipeline is None)
        if submitted:
            try:
                with st.spinner("백서의 조건과 대조하고 있어요…"):
                    s.assessment = s.pipeline.check_application(profile)
                    s.assessment_profile = dict(profile)
            except Exception as error:
                st.error(user_error(error))
        if s.assessment:
            with st.container(border=True):
                st.subheader("백서 기반 점검 결과")
                st.caption("마지막 제출 기준입니다. 값을 바꾸면 다시 확인해 주세요.")
                submitted_profile = s.assessment_profile or {}
                status_names = {"충족": "충족 (자가입력)", "미충족": "미충족 (자가입력)", "미확인": "추가 확인"}
                st.table([{"점검 항목": key, "자기점검 상태": status_names[value]}
                          for key, value in submitted_profile.items() if value in status_names])
                entered_fee = submitted_profile.get("공식 과정 상세의 본인 결제금액", "미확인")
                st.write("본인 입력 결제금액: " + (entered_fee + "원 (미검증)" if entered_fee != "미확인" else "미확인 · 고용24 과정 상세에서 확인"))
                st.markdown(s.assessment["answer"])
                sources_panel(s.assessment["sources"])
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
                        result = s.pipeline.invoke({"input": question, "history": s.messages[:-1]})
                    st.markdown(result["answer"])
                s.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
                s.failed_question = s.suggestion_context = None
                st.rerun()
            except Exception as error:
                s.failed_question = question
                st.error(user_error(error))
        if s.failed_question and st.button("답변 다시 시도"):
            s.pending, s.failed_question = s.failed_question, None
            st.rerun()
        if s.pipeline:
            if s.suggestion_context != (len(s.messages), s.topic) and not s.suggestion_error:
                try:
                    refresh_suggestions()
                except Exception as error:
                    s.suggestion_error = user_error(error)
            for i, question in enumerate(s.suggestions):
                if st.button(question, key=f"suggestion_{i}", use_container_width=True):
                    s.pending = question
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
