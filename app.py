
import os
import base64
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(page_title="OpenAI 음성 변환 챗봇", page_icon="🤖")
st.title("🤖 음성 변환 챗봇")

# 텍스트 입력
text = st.text_area(
    "읽을 텍스트를 입력하세요",
    "안녕하세요. OpenAI 텍스트 음성 변환 실습입니다."
)

# 버튼
if st.button("변환하기"):
    if text.strip() == "":
        st.warning("텍스트를 입력해 주세요.")
    else:
        # TTS 생성
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="nova",
            input=text
        )
        audio_bytes = speech.content

        # 음성 출력
        st.markdown("### 🔊 음성 재생")
        st.audio(speech.content, format="audio/mp3")
