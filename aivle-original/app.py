import hashlib
import tempfile
from pathlib import Path
import streamlit as st
from rag_core import rag_pipeline

import os
for _key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_API_BASE', 'OPENAI_ORGANIZATION'):
    try:
        _value = st.secrets.get(_key)
    except Exception:
        _value = None
    if _value:
        os.environ[_key] = str(_value)


st.set_page_config(page_title="AIVLE School �н� �����", page_icon="??", layout="centered")
st.markdown('''<style>.block-container{max-width:900px;padding-top:2.5rem;}
[data-testid="stSidebar"]{background:#f7f7f5;}h1{letter-spacing:-1px;}</style>''', unsafe_allow_html=True)
for key, value in {"chain": None, "messages": [], "signature": None, "name": ""}.items():
    st.session_state.setdefault(key, value)
s = st.session_state
with st.sidebar:
    st.markdown("### AIVLE SCHOOL")
    st.caption("�鼭���� ã�� ����� ����")
    uploaded = st.file_uploader("�鼭 PDF ���ε�", type=["pdf"])
    st.caption("PDF�� �ٲ� �ڿ��� �ڷ� �غ��ϱ⸦ �ٽ� ���� �ּ���.")
    if st.button("�ڷ� �غ��ϱ�", type="primary", use_container_width=True):
        if uploaded is None:
            st.warning("�鼭 PDF�� ���� ������ �ּ���.")
        else:
            data = uploaded.getvalue()
            signature = hashlib.sha256(data).hexdigest()
            if s.signature == signature:
                st.info("�̹� �غ��� �鼭�Դϴ�.")
            else:
                try:
                    # ���� reset_collection�� �ٸ� ������ DB�� ������ �ʵ��� ������ �и��մϴ�.
                    folder = Path(tempfile.mkdtemp(prefix="aivle_"))
                    pdf_path = folder / "whitepaper.pdf"
                    pdf_path.write_bytes(data)
                    with st.spinner("���� RAG �������������� �鼭�� �غ��ϰ� �־�䡦"):
                        chain = rag_pipeline(str(pdf_path), str(folder))
                    s.chain, s.signature, s.name = chain, signature, uploaded.name
                    s.messages = []
                    st.success("�ڷ� �غ� �Ϸ�")
                except Exception:
                    st.error("�ڷḦ �غ����� ���߾��. PDF ��ȣ ���ο� API ������ Ȯ���� �ּ���. �� �������� ���� PDF �δ��� ����մϴ�.")
    if s.chain is not None:
        st.caption("���� �鼭: " + s.name)
    st.divider()
    if st.button("��ȭ �����", use_container_width=True):
        s.messages = []
        st.rerun()
    transcript = "\n\n".join(("��" if m["role"] == "user" else "�н� �����") + ": " + m["content"] for m in s.messages)
    st.download_button("��ȭ �����ޱ�", transcript, "aivle_chat.txt", "text/plain", disabled=not transcript, use_container_width=True)

st.caption("AIVLE SCHOOL �� LEARNING COMPANION")
st.title("����� ���� ����, �Բ� ã�ƿ�")
st.write("���� �������� ���� �������, �鼭�� �������� ���� ������ �����.")
st.caption("���� �ڵ�� �����ϰ� �������� �˻��մϴ�. ���� ��ȭ�� ȭ�鿡�� ǥ�õǹǷ� ������ �ʿ��� ������ �Բ� ���� �ּ���.")
if not s.messages:
    st.info("���ʿ��� �鼭�� �÷� �ڷḦ �غ��� �� ������ ������." if s.chain is None else "�鼭�� �غ�ƾ��. �ñ��� ������ ������ ������.")
selected = None
examples = [("���� ����", "KT AIVLE School�� ���� ������ �ֿ� �н� ������ �����ΰ���?"),
            ("���� ����", "�鼭���� ��õ�ϴ� �������� �н� ������ �����ΰ���?"),
            ("��� �غ�", "������ �м� ������ ��ǥ�� �ϸ� ���̺����𿡼� �� ���� �� ������ ��Ʈ�������� ������ �� �ֳ���?")]
for col, (label, question) in zip(st.columns(3), examples):
    if col.button(label, use_container_width=True, disabled=s.chain is None):
        selected = question
for message in s.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
question = st.chat_input("�ñ��� ������ ��ü������ ���� �ּ���", disabled=s.chain is None) or selected
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("�鼭���� ���� ã�� �־�䡦"):
                # ���� ��Ʈ�Ͽ��� ����� ȣ�� ��� �״�� �����մϴ�.
                answer = s.chain.invoke({"input": question})
            st.markdown(answer)
            s.messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        except Exception:
            st.error("������ ���� ���߾��. API Ű���ּҿ� �� ��� ������ Ȯ���ϰ� �ٽ� ������ �ּ���.")

