
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(page_title="음성 AI 서비스", page_icon="🤖", layout="wide")
st.title("음성 인식 AI 서비스")
st.write("음성으로 질문하시면 답변해 드립니다.")

# 역할 지정
sys_role = """
당신은 다양한 업무를 처리하는 AI 서비스입니다.
개인 비서처럼 친절하게 응대합니다.
"""

# 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 지시 영역
with st.form("task_form"):
    uploaded_file = st.file_uploader("텍스트 파일 업로드", type=["txt"])
    audio_value = st.audio_input("무엇을 도와드릴까요?")
    submit = st.form_submit_button("실행")

# 버튼을 눌렀을 때만 실행
if submit:
    # 업로드된 파일이 있으면 디코딩
    text = ""
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")

    # 녹음이 없으면 녹음 요청
    if audio_value is None:
        st.warning("음성을 녹음해주세요.")
        st.stop()

    # 시스템 메시지
    system_content = sys_role
    if text:
        system_content += f"\n\n다음 내용을 참고하세요:\n{text}"

    # STT: 음성 → 텍스트
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", audio_value, "audio/wav"),
    )

    # 음성 인식 결과 출력
    st.subheader("🎤 인식된 질문")
    st.write(transcript.text)

    # 사용자 질문 저장
    st.session_state.messages.append({
        "role": "user",
        "content": transcript.text,
    })

    # 시스템 메시지 + 이전 대화 전달
    input_messages = [
        {"role": "system", "content": system_content}
    ] + st.session_state.messages

    # GPT 답변 생성
    with st.spinner("답변을 준비하고 있습니다..."):
        response = client.responses.create(
            model="gpt-4o-mini",
            input=input_messages,
        )

        answer = response.output_text

        # AI 답변 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })

        # 최근 10번의 질문과 답변만 유지
        st.session_state.messages = st.session_state.messages[-20:]

        st.subheader("🤖 AI 답변")
        st.write(answer)

# 대화 기록 표시 (if submit 바깥쪽)
st.divider()

with st.expander("📜 최근 대화 기록"):
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
