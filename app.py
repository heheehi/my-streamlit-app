
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 페이지 설정 (가장 먼저 실행)
st.set_page_config(page_title="FAQ 챗봇", page_icon="💬")
st.title("💬 EduVerse Academy FAQ 챗봇")

# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 벡터 저장소 캐싱
@st.cache_resource
def get_vectorstore():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name="eduverse_faq",
        embedding_function=embedding,
        persist_directory="chroma_db"
    )

# 필터 선택지: 저장된 메타데이터에서 값 목록 추출
@st.cache_resource
def get_options():
    metas = get_vectorstore().get(include=["metadatas"])["metadatas"]
    keys = ["category", "user_type", "plan", "channel"]
    return {k: sorted({m[k] for m in metas if m.get(k)}) for k in keys}

# 체인 캐싱 (검색은 체인 밖에서 수행하고 context로 전달)
@st.cache_resource
def get_chain():
    s_msg = """
    1. 반드시 제공된 문서 내용(context)에 근거하여 답변하세요.
    2. 문서에 없는 내용은 추측하지 말고 "업로드된 문서에서 확인되지 않습니다."라고 답하세요.
    3. 답변은 간결하고 명확하게 작성하세요.
    [문서]
    {context}"""
    prompt = ChatPromptTemplate.from_messages([("system", s_msg), ("human", "{input}")])
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0)
    return prompt | llm | StrOutputParser()

vectorstore = get_vectorstore()
options = get_options()
chain = get_chain()

# 사이드바: 메타데이터 필터
with st.sidebar:
    st.header("🔎 검색 필터")
    categories = st.multiselect("카테고리", options["category"],
                                placeholder="전체")
    user_type = st.selectbox("사용자 유형", ["전체"] + options["user_type"])
    plan = st.selectbox("요금제", ["전체"] + [p for p in options["plan"] if p != "All"])
    channel = st.selectbox("접속 채널", ["전체"] + [c for c in options["channel"] if c != "All"])
    k = st.slider("검색 문서 수(k)", 1, 10, 4)

# 선택값 → Chroma 필터 변환
def build_filter():
    conditions = []
    if categories:
        conditions.append({"category": {"$in": categories}})
    if user_type != "전체":
        conditions.append({"user_type": user_type})
    if plan != "전체":                      # 'All' 행은 모든 요금제에 해당하므로 함께 포함
        conditions.append({"plan": {"$in": [plan, "All"]}})
    if channel != "전체":
        conditions.append({"channel": {"$in": [channel, "All"]}})

    if not conditions:
        return None
    if len(conditions) == 1:                # Chroma의 $and는 조건이 2개 이상이어야 함
        return conditions[0]
    return {"$and": conditions}

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 내용 출력
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# 사용자 입력 처리
user_input = st.chat_input("메시지를 입력하세요.")
if user_input:
    st.chat_message("user").markdown(user_input)

    # 필터 적용 검색
    docs = vectorstore.similarity_search(user_input, k=k, filter=build_filter())

    with st.chat_message("assistant"):
        response_stream = chain.stream({"context": format_docs(docs), "input": user_input})
        full_response = st.write_stream(response_stream)

        # 참고한 FAQ 표시
        with st.expander(f"참고한 FAQ ({len(docs)}건)"):
            for doc in docs:
                m = doc.metadata
                st.markdown(f"**[{m.get('category')} · {m.get('user_type')} · "
                            f"{m.get('plan')} · {m.get('channel')}]**")
                st.text(doc.page_content)

    # 대화 기록 저장
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.messages.append(AIMessage(content=full_response))
