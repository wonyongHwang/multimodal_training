## 시스템 아키텍처
모든작업은 codeset 폴더 내가 기준입니다!

```
MySQL → Vector DB (ChromaDB) → FastAPI → Streamlit UI
```

## 데이터셋 압축해제
codeset/dataset 폴더 내 01. 필수의학.zip 압축해제
dataset 폴더 내 01. 필수의학 폴더가 존재하고 하위에 TL_내과 등이 존재해야함

## 필수 패키지 설치

설치전 torch 2.7.0 이상 설치 (가상상)
<!-- pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118 -->
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 
## 패키지 설치 후 아래 명령어 파이썬 쉘에서 실행 후 정상 구동여부 확인
import torch

print(torch.__version__)          # 설치된 버전 출력
print(torch.cuda.is_available()

## 추가 패키지 설치

```bash
pip install -r requirements.txt
```

이후 패키지 추가 설치
```bash
pip install -r requirements_finetune.txt
```

########## 중요 #############
프로젝트 루트폴더(codeset 폴더 내) `.env` 파일을 생성하고 다음 내용을 추가하세요
#(힘들면 env.example 파일을 .env 파일로 저장 후 codeset 폴더 내 저장)

```env
# MySQL 설정

```

## 스크립트 실행 순서 파트1 LLM 모델 파인튜닝
## 파인튜닝 코드는 jupyter lab으로 실행 추천

### 1. 데이터 전처리 (01. data_preprocessing.ipynb)
```bash
python "01. data_preprocessing.ipynb"
```
- TL_예방의학, TL_의료법규 폴더의 JSON 파일들을 통합
- Huggingface Datasets 포맷으로 변환
- 허깅페이스 허브에 업로드

### 2. 모델 파인튜닝 (02. Fine_tune.ipynb)
```bash
python "02. Fine_tune.ipynb"
```
- 사전 훈련된 모델을 의료 데이터로 파인튜닝
- LoRA 기반 효율적 파인튜닝 수행
- 모델 체크포인트 저장

### 3. 모델 로드 및 저장 (03. Load_And_Save.ipynb)
```bash
python "03. Load_And_Save.ipynb"
```
- 파인튜닝된 모델 로드
- 모델 성능 평가
- 최종 모델 저장

## 스크립트 실행 순서 파트2 LLM RAG 및 AGENT
## 파인튜닝 코드는 jupyter lab으로 실행 추천

### 4. 벡터 데이터베이스 생성 (04. Gen_vertordb.py)
```bash
python "04. Gen_vertordb.py"
```
- MySQL에서 건강 데이터 테이블들을 가져옴
  - blood_glucose (혈당)
  - spo2 (산소포화도)
  - blood_pressure (혈압)
  - heart_rate (심박수)
  - hdl_cholesterol (HDL 콜레스테롤)
  - ldl_cholesterol (LDL 콜레스테롤)
- 데이터를 임베딩으로 변환하여 ChromaDB에 저장
- 증분 로딩 및 전체 재로딩 모드 지원
- 개인정보 보호를 위한 메타데이터 관리

### 5. FastAPI 백엔드 서버 (05. Chatbot_with_fastapi.py)
```bash
python "05. Chatbot_with_fastapi.py"
```
- FastAPI 기반 REST API 서버 실행
- `/generate_report` 엔드포인트 제공
- member_id 기반 건강 데이터 조회
- OpenAI GPT 또는 Ollama LLM을 통한 건강 진단 리포트 생성
- 프롬프트 엔지니어링을 통한 상세한 건강 분석

### 6. 벡터 데이터베이스 확인 (06. VectordbConfirm.py)
```bash
python "06. VectordbConfirm.py"
```
- ChromaDB 컬렉션 목록 확인
- 저장된 데이터 샘플 및 메타데이터 검증
- 벡터 DB 상태 진단

### 7. Streamlit 챗봇 UI (07. chatbot_ui.py)
```bash
streamlit run "07. chatbot_ui.py"
```
- 사용자 친화적인 웹 인터페이스
- member_id 입력 및 LLM 모델 선택
- FastAPI 서버와 연동하여 건강 진단 리포트 표시
- 기본 member_id: "52a62c11-7c4f-4912-91c9-ff5c145328cd"

## 주요 기능

### 건강 데이터 관리
- **다중 테이블 지원**: 혈당, 산소포화도, 혈압, 심박수, 콜레스테롤 등
- **증분 업데이트**: 새로운 데이터만 벡터 DB에 추가
- **개인정보 보호**: 메타데이터에 프라이버시 플래그 포함

### RAG 기반 건강 진단
- **컨텍스트 기반 분석**: 벡터 DB에서 관련 건강 데이터 검색
- **프롬프트 엔지니어링**: 정상/경계/위험 수준 해석 및 개인화된 조언
- **다중 LLM 지원**: OpenAI GPT, Ollama 모델 선택 가능

### 사용자 인터페이스
- **간단한 웹 UI**: Streamlit 기반 직관적 인터페이스
- **실시간 응답**: FastAPI를 통한 빠른 건강 진단 리포트 생성
- **모바일 친화적**: 반응형 디자인

## 시스템 메시지

챗봇은 다음 시스템 메시지로 식별됩니다:
```
"파밀리케어 담당 건강관리 챗봇입니다. 건강 데이터를 분석하여 개인화된 건강 진단과 조언을 제공합니다."
```

## 문제 해결

### 일반적인 오류
1. **ChromaDB 연결 오류**: VECTOR_DB_PATH 환경변수 확인
2. **MySQL 연결 실패**: 데이터베이스 접속 정보 확인
3. **OpenAI API 오류**: API 키 설정 확인
4. **모듈 import 오류**: requirements_rag.txt 설치 확인

### 디버깅 도구
- `06. VectordbConfirm.py`: 벡터 DB 상태 확인
- 로그 출력: 각 스크립트의 상세 실행 로그 확인

## 기타 프로그램 설치

- ollama 설치 ollama run gemma3:4b