import hashlib

import streamlit as st
from rag_core import rag_pipeline, source_label, user_error

LOGO_URL = "https://raw.githubusercontent.com/Jangrae/img/master/aivle.png"
# 검색 방식이 바뀐 배포에서는 이전 세션의 검색 자료를 다시 준비합니다.
INDEX_VERSION = "1200-200-body-v2"


st.set_page_config(page_title="AIVLE School 학습 도우미", page_icon="📖", layout="centered")
st.markdown("""<style>
.block-container{max-width:860px;padding-top:2.8rem;}
[data-testid="stSidebar"]{background:#f7f7f5;}
h1{letter-spacing:-1.3px;} .stCaption{color:#787774;}
</style>""", unsafe_allow_html=True)

for key, value in {"pipeline": None, "messages": [], "signature": None,
                   "suggestions": [], "suggestion_error": None,
                   "document_error": None}.items():
    st.session_state.setdefault(key, value)
s = st.session_state


def reset_document(signature):
    previous = s.pipeline
    s.pipeline = None
    s.messages = []
    s.suggestions = []
    s.suggestion_error = None
    s.document_error = None
    s.signature = signature
    if previous is not None:
        previous.close()


def get_setting(name):
    # 환경 전체를 바꾸지 않고 이 사용자의 모델에만 Secrets를 전달합니다.
    try:
        return str(st.secrets.get(name, "")).strip()
    except FileNotFoundError:
        return ""


st.image(LOGO_URL, width=220)
st.caption("AIVLE SCHOOL · LEARNING COMPANION")
st.title("문서에서 찾고, 배움으로 이어가요")
st.write("백서와 학습 자료를 바탕으로 궁금한 내용을 함께 살펴보세요.")
st.caption("대화 기록은 현재 세션의 화면에만 저장됩니다. 모델에는 현재 질문과 검색 자료만 전달하며, "
           "이전 대화를 기억하지 않습니다. 필요한 조건을 질문에 함께 적어 주세요.")

with st.sidebar:
    st.image(LOGO_URL, width=180)
    st.markdown("### 문서 설정")
    uploaded = st.file_uploader("PDF 또는 TXT 업로드", type=["pdf", "txt"])
    st.caption("텍스트가 있는 PDF와 UTF-8·CP949 TXT를 지원합니다. "
               "문서가 바뀌면 대화와 추천 질문이 초기화됩니다.")

data = uploaded.getvalue() if uploaded is not None else None
signature = hashlib.sha256(INDEX_VERSION.encode() + uploaded.name.encode("utf-8") + b"\0" + data).hexdigest() if uploaded else None
if signature != s.signature:
    reset_document(signature)

api_key = get_setting("OPENAI_API_KEY")
base_url = get_setting("OPENAI_BASE_URL") or get_setting("OPENAI_API_BASE") or None
organization = get_setting("OPENAI_ORGANIZATION") or None

if uploaded is not None and s.pipeline is None:
    retry = st.sidebar.button("문서 준비 다시 시도", disabled=not api_key) if s.document_error else False
    if not api_key:
        st.warning("Streamlit Secrets에 OPENAI_API_KEY를 등록해 주세요. "
                   "회사 API는 OPENAI_BASE_URL 또는 OPENAI_API_BASE도 설정할 수 있습니다.")
    elif s.document_error is None or retry:
        try:
            with st.spinner("문서를 읽고 검색 자료를 준비하고 있어요…"):
                s.pipeline = rag_pipeline(data, uploaded.name, api_key=api_key,
                                          base_url=base_url, organization=organization)
            s.document_error = None
        except Exception as error:
            s.document_error = user_error(error)
    if s.document_error:
        st.error(s.document_error)

if s.pipeline is not None:
    st.sidebar.success(f"검색 준비 완료 · {len(s.pipeline.documents)}개 문서 조각")
    st.sidebar.caption("같은 문서는 현재 세션의 화면 재실행 때 다시 임베딩하지 않습니다. "
                       "새로 접속하거나 문서를 교체하면 다시 준비합니다.")
    retry_suggestions = st.sidebar.button("추천 질문 다시 만들기")
    if retry_suggestions or (not s.suggestions and not s.suggestion_error):
        try:
            with st.spinner("문서에서 추천 질문 3개를 만들고 있어요…"):
                s.suggestions = s.pipeline.make_suggestions()
            s.suggestion_error = None
        except Exception as error:
            s.suggestion_error = user_error(error)
    if s.suggestion_error:
        st.sidebar.warning(s.suggestion_error)

with st.sidebar:
    st.divider()
    if st.button("대화 삭제", use_container_width=True):
        s.messages = []
        st.rerun()
    transcript = []
    for message in s.messages:
        role = "나" if message["role"] == "user" else "학습 도우미"
        transcript.append(f"{role}: {message['content']}")
        for i, doc in enumerate(message.get("sources", []), 1):
            transcript.append(f"[근거 {i}] {source_label(doc)}\n{doc.page_content}")
    st.download_button("대화 다운로드", "\n\n".join(transcript).encode("utf-8-sig"),
                       "aivle_chat.txt", "text/plain", disabled=not transcript,
                       use_container_width=True)
    st.caption("업로드한 텍스트는 임베딩·답변·추천 질문 생성을 위해 설정한 API로 전송됩니다. "
               "PDF 쪽수는 인쇄된 번호가 아닌 파일의 실제 페이지 순서입니다. "
               "이미지뿐인 페이지는 검색에서 제외됩니다.")

if uploaded is None:
    st.info("왼쪽에서 PDF 또는 TXT를 올리면 문서 기반 학습을 시작할 수 있어요.")

for message in s.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("검색된 원문과 출처 확인"):
                st.caption("이번 답변에 전달한 검색 결과입니다. 번호는 답변의 [근거 n]과 같습니다. "
                           "검색 결과가 질문과 관련 있는지도 함께 확인해 주세요.")
                for i, doc in enumerate(message["sources"], 1):
                    st.write(f"[근거 {i}] {source_label(doc)}")
                    st.text(doc.page_content)

selected = None
if s.suggestions:
    st.caption("문서에서 시작하는 질문")
    for i, suggestion in enumerate(s.suggestions):
        if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
            selected = suggestion
typed = st.chat_input("문서에 대해 질문해 주세요", disabled=s.pipeline is None)
question = selected or typed
if question:
    try:
        with st.spinner("문서에서 답을 찾고 있어요…"):
            # 질문과 검색 자료만 사용합니다. 화면의 대화 목록은 모델에 보내지 않습니다.
            result = s.pipeline.invoke({"input": question})
        s.messages.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        ])
        st.rerun()
    except Exception as error:
        st.error(user_error(error))
