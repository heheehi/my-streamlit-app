# AIVLE School 최종 백서 RAG 학습 도우미

최종 백서(71페이지)의 **페이지별 추출 텍스트** `whitepaper_pages.txt`를 기본 검색 문서로 사용하는 Streamlit 앱입니다. 기존 저장소 루트의 앱과 별개로 배포합니다.

## Streamlit Community Cloud 설정

- Repository: `heheehi/my-streamlit-app`
- Branch: `main`
- Main file path: `aivle-rag/app.py`

Streamlit 앱 Settings → Secrets:

```toml
OPENAI_API_KEY = "YOUR_API_KEY"
CHAT_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"
ADMIN_PASSWORD = "CHOOSE_A_STRONG_PASSWORD"
```

선택적으로 `APP_PASSWORD`를 설정하면 입장 비밀번호가 필요합니다. 회사 제공 API 호환 서버를 쓰는 경우 `OPENAI_BASE_URL`을 추가할 수 있으나, 임베딩 API도 지원해야 합니다.

## 주요 기능

- 백서 페이지 텍스트 → 청크 분할 → OpenAI 임베딩 → 코사인 유사도 검색 → 답변 생성
- 근거 문단의 실제 PDF 페이지 번호 안내
- 기본 문서 인덱스 캐시로 Streamlit 재실행시 중복 벡터화 최소화
- 추천 질문, 대화 이력, 대화 초기화
- 관리자 비밀번호를 입력한 경우 추가 PDF 최대 3개 업로드(현재 세션에서만 유지)

## 원본 PDF에 관하여

현재 GitHub에는 **PDF 파일 자체가 아닌 해당 PDF에서 추출한 검색용 텍스트**가 들어 있습니다. 이 방식으로도 문서 기반 RAG 질문·답변을 할 수 있습니다. 같은 폴더에 `AIVLE_School_백서 최종.pdf`를 추가하면 앱은 PDF를 우선 읽고 PDF 다운로드 버튼도 보여줍니다.

모집·비용·운영 조건은 백서 작성 시점의 정보이며, 최신 기준은 공식 공지를 확인하세요. 공개 저장소에 올린 내용은 누구나 볼 수 있습니다.

## 비용 및 주의사항

- ChatGPT 구독과 OpenAI API 과금은 별개입니다. 첫 임베딩과 질문·답변 호출은 API 요금이 발생합니다.
- API 키 및 관리자 비밀번호를 GitHub에 직접 올리지 마세요.
- 본 앱은 아직 실서비스 배포 및 API 동작 검증 전입니다.
