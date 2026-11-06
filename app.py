
import time
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 체인 생성 캐싱
@st.cache_resource
def create_movie_chain():
    s_msg = "당신은 명작 영화를 추천하는 영화 전문가입니다."
    h_msg = "{genre} 영화 중 명작을 추천하고, 영화 제목, 개봉 연도, 추천 이유를 설명하세요."
    messages = [("system", s_msg), ("human", h_msg)]
    prompt = ChatPromptTemplate.from_messages(messages)

    llm = ChatOpenAI(model="gpt-6-luna")
    return prompt | llm

# Streamlit UI
st.set_page_config(page_title="🎬 AI Movie Rec", page_icon="🍿")
st.title("🎬 영화 추천")

genre = st.selectbox("장르 선택", ["공상과학", "액션", "드라마", "코미디", "스릴러", "공포", "로맨스"])

# 버튼 클릭 시 추천
if st.button("영화 추천 받기", type="primary"):
    with st.spinner(f"📡 {genre} 영화 정보를 가져오는 중..."):
        chain = create_movie_chain()
        response = chain.invoke({"genre": genre})

    # 타이핑 효과로 출력
    st.subheader("추천 결과")
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        for char in response.content:
            full_response += char
            response_container.markdown(full_response + "▌")
            time.sleep(0.01)

        response_container.markdown(full_response)
