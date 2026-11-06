
import time, base64, mimetypes
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 이미지 변환 함수
@st.cache_resource
def bytes_to_data_url(file_bytes: bytes, filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    if mime is None:
        mime = "image/jpeg"
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# 체인 생성 캐싱
@st.cache_resource
def get_chain():
    s_msg = "당신은 이미지 내용을 정확히 묘사하고 요약하는 전문가입니다."
    h_msg = [
        {"type": "text", "text": "{question}"},
        {"type": "image_url", "image_url": {"url": "{image_url}"}},
    ]
    messages = [("system", s_msg), ("human", h_msg)]
    prompt = ChatPromptTemplate.from_messages(messages)
    llm = ChatOpenAI(model="gpt-4o-mini")
    return prompt | llm

# Streamlit UI
st.set_page_config(page_title="🖼️ 이미지 분석")
st.title("🖼️ 이미지 업로드 분석기")

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    user_prompt = st.text_area(
        "분석 요청 사항",
        value="이미지를 3문장으로 설명하고, 마지막에 한 줄 캡션을 써주세요.",
        height=150
    )

# 이미지 업로드
uploaded = st.file_uploader("이미지 선택", type=["jpg", "jpeg", "png"])

if uploaded:
    img_bytes = uploaded.read()
    # 이미지 표시
    st.image(img_bytes, caption="분석 대상 이미지", width=600)

    if st.button("이미지 분석 시작", type="primary"):
        if not user_prompt.strip():
            st.warning("요청 사항을 입력해주세요.")
            st.stop()

        # 이미지 경로
        data_url = bytes_to_data_url(img_bytes, uploaded.name)

        # 체인 구성
        chain = get_chain()

        # 체인 실행
        with st.spinner(f"🔍이미지를 분석하는 중..."):
            response = chain.invoke(
                {
                    "question": user_prompt.strip(),
                    "image_url": data_url,
                }
            )

        # 결과 출력
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            for char in response.content:
                full_response += char
                response_container.markdown(full_response + "▌")
                time.sleep(0.01)
            response_container.markdown(full_response)
else:
    st.info("이미지를 업로드해 주세요.")
