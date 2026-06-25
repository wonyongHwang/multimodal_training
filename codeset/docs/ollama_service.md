# 모델 정의서: ollama_service.py

## 개요

| 항목 | 내용 |
|------|------|
| 파일명 | `SimulApp/ollama_service.py` |
| 목적 | GGUF 모델 Ollama 등록·관리 및 멀티모달 채팅 테스트 |
| 실행 환경 | `.venvgguf` (Python 3.12.3) |
| 포트 | 8001 |
| 의존 서비스 | Ollama (포트 11434) |

---

## 실행 방법

server.js 기동 시 자동 실행됩니다. 수동 실행:

```bash
/home/wyhwang/multimodal_training/.venvgguf/bin/python SimulApp/ollama_service.py
```

설치 필요 패키지 (`.venvgguf` 기준):

```bash
pip install fastapi uvicorn python-multipart
```

---

## 함수 구조

| 함수 / 엔드포인트 | 역할 |
|-----------------|------|
| `readGgufMeta(filePath)` | GGUF 헤더에서 `general.architecture`, `general.name` 추출 |
| `buildModelfile(ggufPath, arch)` | 아키텍처별 Modelfile 템플릿 생성 |
| `isExcluded(filePath)` | llama.cpp 내부 파일 등 제외 여부 판별 |
| `GET /gguf/list` | 프로젝트 전체 .gguf 파일 스캔 (10MB 이상만) |
| `POST /gguf/metadata` | GGUF 메타데이터 읽기 + Modelfile 초안 반환 |
| `GET /ollama/models` | 등록된 Ollama 모델 목록 |
| `POST /ollama/register` | Modelfile 저장 → `ollama create` 실행 |
| `DELETE /ollama/model/{name}` | `ollama rm` 실행 |
| `POST /ollama/chat` | 이미지 포함 채팅 → Ollama API 전달 |

---

## 아키텍처별 Modelfile 템플릿

| `general.architecture` | 모델 계열 | stop 토큰 |
|----------------------|---------|---------|
| `gemma3` | Gemma 3 | `<end_of_turn>` |
| `gemma` | Gemma 2 | `<end_of_turn>` |
| `gemma4` | Gemma 4 | `<end_of_turn>` |
| `llama` | Llama 3.x | `<\|eot_id\|>` |
| `qwen2` | Qwen 2 / 2.5 / 3 | `<\|im_end\|>` |
| `phi3` | Phi 3 / 4 | `<\|end\|>` |
| `mistral` | Mistral / Mixtral | `[/INST]` |
| `command-r` | Cohere Command R | `<\|END_OF_TURN_TOKEN\|>` |

감지 실패 시 사용자가 직접 Modelfile 텍스트박스에서 편집 가능.

---

## 이미지 처리 흐름 (멀티모달)

1. 브라우저에서 이미지 파일 선택 → `multipart/form-data` 전송
2. `POST /ollama/chat` 수신 → `PIL.Image` 로드
3. 최대 1024px 리사이즈 (LANCZOS)
4. PNG → base64 인코딩
5. Ollama `POST /api/chat` 에 `images: [base64_str]` 포함 전송

---

## GGUF 스캔 규칙

- 스캔 루트: `multimodal_training/` (프로젝트 최상위)
- 제외 디렉터리: `llama.cpp`, `.venvg3`, `.venvgguf`, `__pycache__`
- 10 MB 미만 파일 제외 (vocab 파일 필터링)

---

## Modelfile 저장 경로

등록 시 생성된 Modelfile:

```
SimulApp/modelfiles/{modelName}.Modelfile
```

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-06-25 | 초기 생성 — GGUF 메타데이터 읽기, Ollama 모델 등록/삭제, 멀티모달 채팅 |
