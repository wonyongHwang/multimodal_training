# Gemma3 멀티모달 파인튜닝 & 시뮬레이션 (Gemma1~3가능)

Gemma3 모델을 멀티모달(이미지+텍스트) 데이터로 파인튜닝하고,  
GGUF 변환까지 웹 UI에서 한 번에 관리가능
CUDA12.6 설치된 환경기준으로 README 작성됨
- MYSQL DB, NODE 설치 필요

---

## 전체 흐름 요약

```
GitHub Clone
    ↓
GPU 환경 설정 (CUDA 12.6)
    ↓
가상환경 2개 생성
    ├── .venvg3   ← 파인튜닝 / 병합 전용
    └── .venvgguf ← GGUF 변환 전용
    ↓
환경 변수 설정 (.env)
    ↓
MySQL DB 테이블 생성
    ↓
웹 서버 구동 (Node.js)
    ↓
브라우저에서 파인튜닝 → 병합 → GGUF 변환 실행
```
---

## 사전 요구사항

| 항목 | 버전 |
|------|------|
| Python | **3.12.10** 필수 |
| Node.js | 18 이상 |
| CUDA | 12.6 (NVIDIA GPU 필수) |
| MySQL | 8.0 이상 |
| Git | 최신 권장 |

---

## STEP 1 — 프로젝트 다운로드

```bash
git clone https://github.com/hyokwan/multimodal_training.git
cd multimodal_training
```

---

## STEP 2 — 가상환경 2개 생성

> **중요:** 웹 서버가 가상환경 이름(`.venvg3`, `.venvgguf`)을 기준으로 Python 경로를 자동 탐색합니다.  
> 이름이 다르면 웹 UI에서 실행이 되지 않으니 반드시 아래 이름 그대로 생성하세요.

### 가상환경 1: 파인튜닝 / 병합 전용 (`.venvg3`)

```bash
# 프로젝트 루트(README.md가 있는 폴더)에서 실행
# alias 미 적용상태인 경우 python3 -m venv .venvg3 명령어 실행
python -m venv .venvg3
```

**Windows 활성화:**
```bash
.venvg3\Scripts\activate
```

**Mac/Linux 활성화:**
```bash
source .venvg3/bin/activate
```

#### PyTorch GPU 버전 설치 (CUDA 12.6)
브랙웰 사용이유 : https://docs.vllm.ai/en/v0.11.1/getting_started/installation/gpu/#create-a-new-python-environment


```bash
# B200 등 블랙웰 최신 버전인 경우 아래 명령어 실행
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

```bash
# 일반 GPU 경우 경우 아래 명령어 실행
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

GPU 인식 확인:
```bash
python -c "import torch; print('GPU 사용 가능:', torch.cuda.is_available())"
# GPU 사용 가능: True  가 출력되어야 정상
```

#### 파인튜닝 패키지 설치

```bash
pip install -r codeset/requirements_finetuneg3.txt
```

설치되는 주요 패키지:

| 패키지 | 버전 | 역할 |
|--------|------|------|
| transformers | 4.51.3 | 모델 로드/추론 |
| peft | 0.15.2 | LoRA 파인튜닝 |
| trl | 0.16.1 | SFT 학습 루프 |
| accelerate | 1.6.0 | GPU 가속 |
| bitsandbytes | 0.45.5 | 4bit 양자화 |
| datasets | 4.8.5 | 데이터 로드 |
| pillow | 12.2.0 | 이미지 처리 |
| mysql-connector-python | 9.3.0 | DB 연동 |
| matplotlib | 3.10.9 | 학습 곡선 시각화 |

가상환경 비활성화:
```bash
deactivate
```

---

### 가상환경 2: GGUF 변환 전용 (`.venvgguf`)

```bash
# 프로젝트 루트에서 실행
# alias 미 적용상태인 경우 python3 -m venv .venvgguf 명령어 실행
python -m venv .venvgguf
```

**Windows 활성화:**
```bash
.venvgguf\Scripts\activate
```

**Mac/Linux 활성화:**
```bash
source .venvgguf/bin/activate
```

#### GGUF 전용 패키지 설치

```bash
pip install -r codeset/requirements_gguf.txt
```

```bash 향후 GGUF 파일 변환을 위해 필수 주의: .venvgguf 가상환경에 있어야함!
# 1. Navigate to the working directory
cd codeset

# 2. Clone the llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp

# 3. Install the required Python dependencies
pip install -r llama.cpp/requirements.txt

가상환경 비활성화 및 multimodal_training 폴더로 다시 이동:
```bash
deactivate
cd ..
```

> **GGUF 별도 디버깅 시:** 반드시 `.venvgguf` 환경을 활성화한 상태에서 작업하세요.

---

## STEP 3 — 환경 변수 설정

> **사전 조건:** MySQL DB가 먼저 구축되어 있어야 합니다.

`codeset/.env.example`을 복사하여 `codeset/.env`를 생성합니다.

**Windows:**
```bash
copy codeset\.env.example codeset\.env
```

**Mac/Linux:**
```bash
cp codeset/.env.example codeset/.env
```

`codeset/.env`를 텍스트 편집기로 열고 **아래 항목을 반드시 본인 환경에 맞게 수정**합니다.

> ⚠️ **수정 필수 항목** (그대로 두면 서버가 DB에 연결되지 않습니다)

```env
# ┌─────────────────────────────────────────────────────────┐
# │  ★ 수정 필수 — DB 접속 정보                              │
# └─────────────────────────────────────────────────────────┘
DB_HOST=yourdbhost        # ← 실제 DB 호스트 주소로 변경
DB_PORT=3306              # ← 실제 포트번호로 변경 (기본 3306)
DB_NAME=yourdbname        # ← 실제 DB명으로 변경
DB_USER=yourdbuser        # ← 실제 DB 계정으로 변경
DB_PASSWORD=yourdbpw      # ← 실제 비밀번호로 변경

# ┌─────────────────────────────────────────────────────────┐
# │  ★ 수정 필수 — Hugging Face 토큰                         │
# └─────────────────────────────────────────────────────────┘
# 발급 주소: https://huggingface.co/settings/tokens
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx   # ← 본인 HF 토큰으로 변경

# ┌─────────────────────────────────────────────────────────┐
# │  선택 수정 — 기본 모델 (기본값: gemma3 4B)                │
# └─────────────────────────────────────────────────────────┘
BASE_MODEL=google/gemma-3-4b-it   # ← 다른 모델 사용 시 변경
OUTPUT_BASE_DIR=./models
```

> 이 외 나머지 하이퍼파라미터(학습률, LoRA r 등)는 **웹 UI 학습 설정 화면에서 실시간으로 변경 가능**합니다.

---

## STEP 4 — MySQL DB 테이블 생성

MySQL에 접속해서 아래 명령을 실행합니다.

```bash
mysql -u yourdbuser -p yourdbname < SimulApp/db_create.sql
```

> MySQL Workbench / DBeaver 등 GUI 툴에서 `SimulApp/db_create.sql` 파일을 열어 직접 실행해도 됩니다.

생성되는 테이블:

| 테이블 | 설명 |
|--------|------|
| `simulation_runs` | 파인튜닝 실행 이력 (설정값 스냅샷 + 결과) |
| `simulation_logs` | 스텝별 학습 손실 로그 |

---

## STEP 5 — 웹 서버 설치 및 구동 (서버 내 NodeJS 설치 필수) 3000번 포트 웹에서 활용

> 구글에서 NVM GITHUB 검색 후 설치 링크 확인 참고요 URL https://github.com/nvm-sh/nvm
> 다운로드 및 설치 (저장소에 등록)
```bash
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
```

환경설정 적용
```bash
source ~/.bashrc
```
nvm 버전확인

```bash
nvm --version
```

node 설치 가능버전 확인

```bash
nvm ls-remote
```

원하는 node 버전 설치
```bash
nvm install 24.16.0
```

```bash
cd SimulApp
npm install
npm install -g pm2
pm2 start server.js
```
종료 시 pm2 stop server.js
또는:

```bash
cd SimulApp
npm install
node server.js
```

서버가 정상 구동되면 터미널에 아래와 같이 출력됩니다:

```
SimulApp 서버 실행 중: http://localhost:3000
파인튜닝/병합 Python 경로: ../.venvg3/Scripts/python.exe
GGUF Python 경로: ../.venvgguf/Scripts/python.exe
[DB] 연결 성공: localhost/hkcodedb
```

브라우저에서 **http://localhost:3000** 접속

---

## STEP 6 — 웹 UI 사용 순서

### 1단계: 데이터 준비 (터미널에서 직접 실행)

`.venvg3`를 활성화한 상태에서 실행합니다.

```bash
# Windows
.venvg3\Scripts\activate

# 스포츠 이미지 데이터 준비
python "codeset/★01. 데이터 준비_멀티모달데이터_Gemma3_기본.py"

# 또는 OCR 데이터 준비
python "codeset/★01. 데이터 준비_멀티모달데이터_Gemma3_OCR.py"
```

### 2단계: 파인튜닝 (웹 UI)

1. 브라우저에서 **파인튜닝 탭** 이동
2. **학습 설정** 화면에서 아래 항목을 확인·수정합니다

| 설정 항목 | 기본값 | 처리 방법 |
|-----------|--------|-----------|
| `DATASET_REPO` | 기본 샘플 데이터 | **그대로 두세요** — 샘플 데이터로 테스트 가능 |
| `BASE_MODEL` | `google/gemma-3-4b-it` | **변경 가능** — 다른 모델로 교체 시 수정 |
| `MAX_TRAIN_SAMPLES` | (전체 사용) | **`20` 으로 입력 권장** — 초기 빠른 테스트용, 20건만 학습 |

> **처음 실행하는 경우:** `MAX_TRAIN_SAMPLES` 를 **`20`** 으로 설정하면 전체 데이터 대신 20건만 학습해서 빠르게 동작 확인이 가능합니다. 이상 없으면 비워두고 재실행하면 전체 데이터로 본 학습이 진행됩니다.

3. **설정 저장** 버튼 클릭 후 **학습 시작** 버튼 클릭
4. 실시간 로그로 진행상황 확인
5. 학습 완료 후 **시뮬레이션 결과 탭**에서 학습 이력·손실 그래프 확인
6. 완료 시 `codeset/models/` 아래에 어댑터 저장

### 3단계: 모델 병합 (웹 UI)

1. 브라우저에서 **병합 탭** 이동
2. 병합 설정 화면에서 아래 항목을 확인·수정합니다

| 설정 항목 | 처리 방법 |
|-----------|-----------|
| `ADAPTER_PATH` | **★ 필수 수정** — 파인튜닝 완료 후 생성된 어댑터 폴더를 선택 (예: `./models/gemma3_multimodal_lora_output_날짜`) |
| `MERGED_MODEL_REPO` | **★ 필수 수정** — 본인의 HuggingFace Hub 레포 주소 입력 (예: `yourHFID/my-gemma3-merged`) |
| `MERGED_LOCAL_DIR` | 변경 불필요 — 병합 결과가 로컬에 저장되는 경로, 기본값 그대로 사용 가능 |
| `BASE_MODEL` | 변경 불필요 — 파인튜닝과 동일한 베이스 모델 사용 |

> ⚠️ `ADAPTER_PATH` 를 지정하지 않으면 병합이 실패합니다.  
> 파인튜닝 완료 후 `codeset/models/` 폴더에 생성된 폴더명을 그대로 입력하세요.

3. **설정 저장** 버튼 클릭 후 **병합 시작** 버튼 클릭
4. 로그에서 진행상황 확인
5. 완료 시 지정한 HuggingFace Hub 레포에 병합 모델 자동 업로드

### 4단계: GGUF 변환 (웹 UI)

1. **GGUF 탭** 이동
2. HuggingFace 소스 레포, 출력 파일명, 정밀도(f16 / bf16 / q8_0 등) 설정
3. **변환 시작** 버튼 클릭
4. 변환 완료 후 HuggingFace Hub에 자동 업로드

---

## GGUF 변환 사전 준비 (llama.cpp)

GGUF 변환은 `llama.cpp`의 변환 스크립트를 사용합니다.  
최초 1회 아래 절차로 설치합니다.

```bash
# codeset 폴더 안에 clone
cd codeset
git clone https://github.com/ggerganov/llama.cpp.git

# .venvgguf 활성화 후 의존성 설치
cd ..
.venvgguf\Scripts\activate
pip install -r codeset/llama.cpp/requirements.txt
```

`codeset/.env`의 `GGUF_LLAMACPP_DIR` 경로를 확인합니다:

```env
GGUF_LLAMACPP_DIR=./llama.cpp
```

---

## 디렉터리 구조

```
llm_multimodal/
├── .venvg3/                         ← 파인튜닝/병합 가상환경 (직접 생성)
├── .venvgguf/                       ← GGUF 변환 가상환경 (직접 생성)
├── codeset/
│   ├── .env                         ← 환경 변수 (직접 생성, git 제외)
│   ├── .env.example                 ← 환경 변수 템플릿
│   ├── requirements_finetuneg3.txt  ← 파인튜닝 패키지 목록
│   ├── requirements_gguf.txt        ← GGUF 패키지 목록
│   ├── ★01. 데이터 준비_멀티모달데이터_Gemma3_기본.py
│   ├── ★01. 데이터 준비_멀티모달데이터_Gemma3_OCR.py
│   ├── ★02. 파인튜닝_멀티모달데이터_Gemma3.py
│   ├── ★03. 데이터병합 및 저장_멀티모달데이터_Gemma3.py
│   ├── ★04. GGUF 모델 변환.py
│   ├── dataset/                     ← 학습 데이터
│   ├── models/                      ← 학습된 모델 저장
│   ├── llama.cpp/                   ← GGUF 변환 도구 (별도 clone 필요)
│   └── docs/                        ← 모델 정의서 (각 파일별 .md)
├── SimulApp/
│   ├── server.js                    ← Node.js Express 서버
│   ├── package.json
│   ├── db_create.sql                ← DB 테이블 생성 스크립트
│   └── public/                      ← 웹 UI (HTML/CSS/JS)
├── docs/
│   └── rev_log.md                   ← 수정 이력
└── README.md
```

---

## 자주 발생하는 문제

### DB 연결 실패
- `codeset/.env`의 `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`를 확인하세요.
- MySQL 서비스가 실행 중인지 확인하세요.
- 웹 UI의 **[DB 재연결]** 버튼으로 재시도 가능합니다.
- DB 연결 없이도 파인튜닝·병합·GGUF 변환은 동작하나, 시뮬레이션 이력 탭은 비활성화됩니다.

### GPU 인식 안 됨
```bash
# .venvg3 활성화 후 확인
python -c "import torch; print(torch.cuda.is_available())"
# True 가 출력되어야 정상
```
`False`면 PyTorch CUDA 버전 재설치:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 웹에서 Python을 못 찾는 경우
- 가상환경 이름이 정확히 `.venvg3` / `.venvgguf` 인지 확인하세요.
- 프로젝트 루트(`.git` 폴더가 있는 위치)에 생성되어야 합니다.
- 서버 터미널에서 `파인튜닝/병합 Python 경로`, `GGUF Python 경로` 로그를 확인하세요.

### GGUF 변환 오류
- `.venvgguf` 가 별도로 구성되어 있는지 확인하세요.
- `codeset/llama.cpp` 폴더가 존재하는지 확인하세요.
- `GGUF_LLAMACPP_DIR` 경로가 `.env`에 올바르게 설정되어 있는지 확인하세요.

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| AI 프레임워크 | PyTorch, Transformers, PEFT (LoRA), TRL |
| 기반 모델 | Google Gemma3 4B Instruct |
| 데이터 관리 | HuggingFace Datasets / Hub |
| GGUF 변환 | llama.cpp |
| 웹 서버 | Node.js + Express |
| DB | MySQL 8.0 |
| 환경 관리 | Python venv (2개: `.venvg3` / `.venvgguf`) |
