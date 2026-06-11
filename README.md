# LLM Multimodal — Gemma3 파인튜닝 프로젝트

모든 작업 파일은 `codeset/` 폴더 기준입니다.

---

## 1. 환경 설정

### Python 가상환경 생성 (권장)
```bash
python -m venv .venv
.venv\Scripts\activate
```

### PyTorch 설치 (CUDA 12.6)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

설치 확인:
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # True 여야 함
```

### 패키지 설치
```bash
pip install -r codeset/requirements.txt
```

---

## 2. 환경변수 설정

`codeset/.env.example` 파일을 복사해서 `codeset/.env`로 저장 후 값을 채워주세요.

```bash
copy codeset\.env.example codeset\.env
```

필수 입력 항목:

```ini
HF_TOKEN=your_huggingface_token
DB_HOST=your_db_host
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
```

---

## 3. 실행 순서

> `codeset/` 폴더 안 **★** 가 붙은 파일만 실행합니다.

### STEP 1 — 데이터 준비

스포츠(필라테스) 이미지 + 텍스트 데이터를 HuggingFace Dataset으로 수집·업로드합니다.

```bash
# Jupyter 노트북으로 실행
codeset/★01. 데이터 준비_멀티모달데이터_Gemma3_스포츠.ipynb

# 또는 스크립트로 실행
python "codeset/★01. 데이터 준비_멀티모달데이터_Gemma3_스포츠.py"
```

### STEP 2 — 파인튜닝

**SimulApp 웹 화면**에서 하이퍼파라미터를 설정하고 학습을 실행합니다.

```bash
cd SimulApp
npm install        # 최초 1회
node server.js
```

브라우저에서 `http://localhost:3000` 접속 →  
**⚙ 학습 설정** 탭에서 파라미터 저장 후 **▶ 학습 시작** 클릭

> 직접 실행도 가능합니다:
> ```bash
> python "codeset/★02. 파인튜닝_멀티모달데이터_Gemma3.py"
> ```

### STEP 3 — 모델 병합 및 업로드

학습된 LoRA 어댑터를 베이스 모델에 병합하고 HuggingFace Hub에 업로드합니다.

```bash
# Jupyter 노트북으로 실행
codeset/★03_데이터병합 및 저장_멀티모달데이터_Gemma3.ipynb
```

> 실행 전 `codeset/.env`의 `ADAPTER_PATH` 값을 파인튜닝 결과 폴더 경로로 업데이트하세요.

---

## 4. DB 테이블 생성 (최초 1회)

```bash
mysql -u your_user -p your_db < SimulApp/db_create.sql
```

---

## 5. 디렉토리 구조

```
llm_multimodal/
├── codeset/
│   ├── .env                  ← 환경변수 (직접 생성)
│   ├── .env.example          ← 환경변수 템플릿
│   ├── requirements.txt
│   ├── ★01. 데이터 준비_멀티모달데이터_Gemma3_스포츠.ipynb
│   ├── ★01. 데이터 준비_멀티모달데이터_Gemma3_스포츠.py
│   ├── ★02. 파인튜닝_멀티모달데이터_Gemma3.ipynb
│   ★02. 파인튜닝_멀티모달데이터_Gemma3.py
│   ├── ★03_데이터병합 및 저장_멀티모달데이터_Gemma3.ipynb
│   ├── dataset/              ← 학습 데이터
│   ├── models/               ← 학습 결과 어댑터
│   └── docs/                 ← 각 파일 모델 정의서
├── SimulApp/
│   ├── server.js             ← Express 서버
│   ├── package.json
│   ├── db_create.sql         ← DB 테이블 생성 SQL
│   └── public/index.html     ← 웹 UI
└── docs/
    └── rev_log.md            ← 수정 이력
```
