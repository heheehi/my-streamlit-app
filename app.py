
import os
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(page_title="OpenAI 챗봇", page_icon="🤖")
st.title("🤖 OpenAI 번역 챗봇")
st.write("대화를 기억하지 않는 번역 챗봇입니다.")

# 사용자 입력(텍스트)
korean_text = st.text_area(
    "번역할 한글 문장을 입력하세요.",
    height=150,
    placeholder="예: 인공지능은 우리의 일상과 업무 방식을 빠르게 변화시키고 있습니다."
)

# 번역 버튼
if st.button("번역하기"):

    if not korean_text.strip():
        st.warning("번역할 한글 문장을 입력하세요.")
    else:
        with st.spinner("번역 중입니다..."):

            # 시스템 역할
            sys_role = """
            당신은 전문 한영 번역가입니다.
            주어진 문장을 영어로 번역하세요.
            쉬운 문장으로 번역하세요.
            """

            # 프롬프트 구성
            prompt = f"""
            다음 문장을 영어로 번역하세요.
            TEXT:
            {korean_text}
            """

            # GPT 호출
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content":sys_role},
                    {"role": "user", "content":prompt}
                ]
            )

            # 결과 출력
            translated_text = response.output_text

            st.subheader("AI 번역")
            st.success(translated_text)
