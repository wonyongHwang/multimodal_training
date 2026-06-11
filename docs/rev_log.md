# 수정 로그

## 2026-06-03

### ★01. 데이터 준비_멀티모달데이터_Gemma3_OCR.ipynb

**문제:** 대용량 OCR 데이터 처리 시 PIL 이미지를 메모리에 전부 누적하다 약 5분 후 OOM으로 커널 사망

**변경 내용:**

| 셀 ID | 변경 전 | 변경 후 |
|-------|---------|---------|
| `f8a9b0c1` | 전체 루프 → `image_data` 리스트에 PIL 객체 누적 | 500개 단위 청크 처리 → JPEG bytes로 변환 후 pickle 저장, resume 지원 |
| `a9b0c1d2` | `Dataset.from_list(image_data)` | `Dataset.from_generator(ocrDataGenerator)` — 스트리밍 방식 |
| `d2e3f4a5` | `push_to_hub(repo, token)` | `push_to_hub(repo, token, max_shard_size="500MB")` — 샤드 분할 |

**효과:**
- 청크별 저장으로 중단 시 이어서 재처리 가능 (`./dataset/chunks_ocr/chunk_NNNN.pkl`)
- PIL → JPEG bytes 변환으로 처리 중 메모리 약 60~70% 절감
- HF Hub 업로드 시 대용량 파일 자동 분할 처리

---

## 2026-06-11

### ★01. 데이터 준비_멀티모달데이터_Gemma3_OCR.ipynb

**변경 내용:** 하드코딩된 설정값을 `.env` 파일에서 로드하도록 전환

| 셀 ID | 변경 전 | 변경 후 |
|-------|---------|---------|
| `b8c9d0e1` | `HF_DATASET_REPO`, `MAX_OCR_SAMPLES` 하드코딩 | `load_dotenv()` 통합, 전체 설정 env화 |
| `c9d0e1f2` | OCR/텍스트 경로 하드코딩 | `os.getenv()` 로 env에서 로드 |
| `d0e1f2a3` | `from dotenv import load_dotenv`, `load_dotenv()` 중복 호출 | `hf_token = HF_TOKEN` 표시만 (중복 제거) |
| `f8a9b0c1` | `CHUNK_SIZE=500`, `instrText`/`inputText` 하드코딩 | `CHUNK_SIZE`, `OCR_INSTR_TEXT`, `OCR_INPUT_TEXT` env화 |
| `471eb9f3` | `HF_DATASET_REPO = 'hyokwan/ocr_dataset'` 중복 선언 | 확인 출력으로 변경 (중복 제거) |
| `c7d019a4` | `ARROW_DIR = "./dataset/ocr_arrow"` 하드코딩 | `os.getenv("ARROW_DIR", ...)` 로 env화 |

**신규 `.env` 항목:** `HF_DATASET_REPO`, `OCR_ROOT`, `TEXT_JSON_PATH`, `MAX_OCR_SAMPLES`, `CHUNK_SIZE`, `DATASET_DIR`, `SEED`, `OCR_INSTR_TEXT`, `OCR_INPUT_TEXT`

### ★01. 데이터 준비_멀티모달데이터_Gemma3_OCR.ipynb (저장 구조 개선)

**변경 내용:** pickle 청크(`CHUNK_DIR`) + Arrow(`ARROW_DIR`) 이중 저장을 Arrow shard 단일 저장(`DATASET_DIR`)으로 통합

| 셀 ID | 변경 전 | 변경 후 |
|-------|---------|---------|
| `f8a9b0c1` | JPEG bytes → pickle 저장 | Arrow shard 직접 저장 (`Dataset.save_to_disk`) |
| `402fc61b` | pickle 로드 → PIL 변환 → concatenate | Arrow shard 로드 → concatenate |
| `3dad128c`, `a9b0c1d2`, `f4ce9aec`, `c7d019a4` | 주석 처리된 대안 코드·구 Arrow 로드 셀 | 삭제 |

**`.env` 변경:** `CHUNK_DIR` + `ARROW_DIR` → `DATASET_DIR` 단일화

---

## 2026-06-11 (추가 2)

### ★01. 데이터 준비_멀티모달데이터_Gemma3_스포츠.ipynb

**변경 내용:** `.env` 전체 통합, 하드코딩 제거, 코드 정리

| 셀 ID | 변경 내용 |
|-------|-----------|
| `460134ee` | 미사용 `import json`, `from pathlib import Path` 제거, 섹션 주석 정리 |
| `ba6a9f1b` | `load_dotenv()` + 전체 설정값 통합 (`IMAGE_ROOT`, `IMAGE_META_ROOT`, `TEXT_JSON_PATH`, `IMAGE_INSTRUCTION`, `IMAGE_INPUT`, `MAX_IMAGE_SAMPLES_PER_CLASS`, `SEED`) |
| `eb199edf` | 삭제 (경로 하드코딩 셀 → `ba6a9f1b`에 통합) |
| `d8200afb` | 삭제 (하드코딩 토큰 셀 → `ba6a9f1b`에 통합) |
| `8c245bf9` | `text_json_path` → `TEXT_JSON_PATH` 변수 사용 |
| `60d477a4` | `text_dataset` → `textDataset` (camelCase) |
| `35332348` | `image_meta_root` → `IMAGE_META_ROOT`, 로드 결과 출력 추가 |
| `3f14f6bb` | `image_data` → `imageData`, `for i in range(0, len(df))` 통일, `IMAGE_INSTRUCTION`/`IMAGE_INPUT` env 변수 사용, `MAX_IMAGE_SAMPLES_PER_CLASS` 실제 동작 구현 |
| `867ef52b` | `image_dataset` → `imageDataset` (camelCase) |
| `afa9b05d` | `text_dataset`/`image_dataset`/`all_dataset` → camelCase 변수명 통일 |
| `703f56f6` | `all_dataset` → `allDataset`, 하드코딩 `42` → `SEED` 변수 사용 |
| `1bfb3eee` | `all_dataset` → `allDataset` |

**신규 `.env` 항목:** `IMAGE_ROOT`, `IMAGE_META_ROOT`, `MAX_IMAGE_SAMPLES_PER_CLASS`, `IMAGE_INSTRUCTION`, `IMAGE_INPUT`

### ★01. 데이터 준비_멀티모달데이터_Gemma3_스포츠.py (신규 생성)

**변경 내용:** 노트북을 독립 실행 가능한 `.py` 스크립트로 변환

| 함수 | 내용 |
|------|------|
| `loadConfig()` | `.env` 전체 파싱 → 딕셔너리 단일 진입점 |
| `loadTextData(cfg)` | 텍스트 JSON 로드 → Dataset 반환, try-except 에러 처리 |
| `loadImageData(cfg)` | 이미지 CSV+PIL 로드, `MAX_IMAGE_SAMPLES_PER_CLASS` 제한, try-except |
| `mergeAndUpload(textDataset, imageDataset, cfg)` | 캐스팅·병합·셔플·Hub 업로드, try-except |
| `main()` | 파이프라인 조립 (각 단계 None 체크 포함) |

**실행:** `python "★01. 데이터 준비_멀티모달데이터_Gemma3_스포츠.py"`

---

## 2026-06-11 (추가)

### 02. 파인튜닝_멀티모달데이터_Gemma3.ipynb

**변경 내용:** 구조 전면 개선 — 라이브러리 통합, 한글 폰트, .env 하이퍼파라미터 시뮬레이션, TensorBoard, DB 저장

| 셀 ID | 변경 내용 |
|-------|-----------|
| `f1fa00ac` | 임포트 최상단 통합 (matplotlib, PIL, mysql.connector 추가) |
| `8ca492e5` (신규) | `setKoreanFont()` — matplotlib 한글 깨짐 방지 (맑은 고딕 우선) |
| `34708266` | 하드코딩 → 전체 `.env` 로드로 전환 (`TENSORBOARD_DIR` 추가) |
| `3fcbdab0` | `build_messages` → `buildMessages` (camelCase), 독스트링 추가 |
| `35954667` | `collate_fn` — `buildMessages` 호출로 갱신, for 루프 정비 |
| `8cfdace8` | `r/lora_alpha/dropout` → 변수 참조, `report_to='tensorboard'`, `logging_dir=TENSORBOARD_DIR` |
| `88c1d0dd` | `train_result = trainer.train()` (결과 변수 저장) |
| `90bee0ad` (신규) | `plotTrainingLoss()` — 학습 손실 그래프 (한글 폰트 적용) |
| `292bff0d` (신규) | `saveTrainingResultToDB()` — MySQL `training_runs` 테이블 자동 생성 + 저장 |
| `84b4c0d9` | `infer_unified` 독스트링 추가 |
| `c188f5c0` | PIL 중복 임포트 제거 |
| 삭제 | 중복 `load_dataset` 셀, 인라인 테스트 셀 3개, 주석처리 셀 2개 |

**신규 `.env` 항목:** `DATASET_REPO`, `BASE_MODEL`, `OUTPUT_BASE_DIR`, `MAX_SEQ_LENGTH`, `PER_DEVICE_TRAIN_BATCH_SIZE`, `PER_DEVICE_EVAL_BATCH_SIZE`, `GRAD_ACCUM`, `NUM_EPOCHS`, `LEARNING_RATE`, `LOGGING_STEPS`, `SAVE_STEPS`, `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`

### ★02. 파인튜닝_멀티모달데이터_Gemma3.py (신규 생성)

**변경 내용:** 노트북을 독립 실행 가능한 `.py` 스크립트로 변환

| 함수 | 내용 |
|------|------|
| `loadConfig()` (신규) | `.env` 전체 파싱 → 딕셔너리 반환 (전역 cfg 단일 진입점) |
| `getCollateFn(processor, model)` (신규) | 전역 변수 의존 제거 — 클로저로 processor/model 캡처 |
| `inferUnified(...)` | 파라미터로 model/processor 전달 (전역 의존 제거) |
| `main()` | 전체 학습 파이프라인 (데이터 로드→학습→시각화→DB저장→모델저장) |

**실행:** `python "★02. 파인튜닝_멀티모달데이터_Gemma3.py"`

---

## 2026-06-11 (추가 4)

### ★03_데이터병합 및 저장_멀티모달데이터_Gemma3.ipynb

**변경 내용:** `.env` 전체 통합, 하드코딩 토큰 제거

| 셀 ID | 변경 내용 |
|-------|-----------|
| `51267a36` | 하드코딩 토큰 제거 → `load_dotenv()` + `os.getenv("HF_TOKEN", "YOUR_HF_TOKEN")` + `login(hf_token)` |
| `d71acf41` | 경로 5개 전부 `.env` 전환 (`BASE_MODEL`, `ADAPTER_PATH`, `MERGED_LOCAL_DIR`, `MERGED_MODEL_REPO`, `TEST_IMAGE_PATH`) |
| `c433b74f` | 베이스 모델·프로세서 로드에 `token=hf_token` 추가 |

**신규 `.env` 항목:** `ADAPTER_PATH`, `MERGED_LOCAL_DIR`, `MERGED_MODEL_REPO`, `TEST_IMAGE_PATH`

### codeset/docs/★03_데이터병합 및 저장_멀티모달데이터_Gemma3.md (신규 생성)

**변경 내용:** 노트북 모델 정의서 작성 (셀 구성, .env 설정, 처리 흐름, 알려진 경고)

---

## 2026-06-11 (추가 3 — SimulApp)

### ★02. 파인튜닝_멀티모달데이터_Gemma3.ipynb (DB 연동 적용)

**변경 내용:** .py와 동일한 DB 연동 구조를 노트북에도 반영

| 셀 ID | 변경 내용 |
|-------|-----------|
| `f1fa00ac` | `mysql.connector`, `TrainerCallback`, `matplotlib.font_manager`, `gridspec` 임포트 추가 |
| `ea23b15e` (신규) | `setKoreanFont()` — matplotlib 한글 폰트 자동 설정 (맑은 고딕 우선) |
| `053d9fd6` | 하드코딩 토큰 제거 → `load_dotenv()` + `os.getenv("HF_TOKEN")` 만 유지 |
| `34708266` | 전체 `.env` 설정 전환 (DB_CFG, TENSORBOARD_DIR, 모든 LoRA·학습 하이퍼파라미터) |
| `eea23c1a` (신규) | `getDbConn`, `getNextSimSeq`, `insertRunStart`, `updateRunEnd`, `DbLogCallback` + `SIM_SEQ` 발급 |
| `3fcbdab0` | `build_messages` → `buildMessages` (camelCase), 독스트링 추가 |
| `35954667` | `collate_fn` — `buildMessages` 호출, `for i in range(0, len())` 루프 통일 |
| `8cfdace8` | LoRA r/α/dropout → 변수 참조, `report_to='tensorboard'`, `logging_dir=TENSORBOARD_DIR`, `DbLogCallback` 추가 |
| `88c1d0dd` | `train_result = trainer.train()` + `updateRunEnd(SIM_SEQ, ...)` |
| `b7a785d3` | 그래프 한글 레이블 적용, 변수명 camelCase 통일, LoRA r/α 동적 표시 |
| `20a38b0d` | CSV 저장 → DB 조회 확인으로 교체 |
| 삭제 | `5981f985`, `53b3eb10`, `1913f7f7`, `7ba914c8`, `52acc4cb`, `95fa7bb1`, `04353c5c` (중복·테스트·주석 셀) |

---

### ★02. 파인튜닝_멀티모달데이터_Gemma3.py (DB 연동 재구축)

**변경 내용:** sim_seq 기반 DB 저장, DbLogCallback, SimulApp 연동 지원으로 전면 재작성

| 추가/변경 내용 | 설명 |
|----------------|------|
| `getNextSimSeq(dbCfg)` | `MAX(sim_seq)+1` 조회로 시뮬레이션 번호 자동 발급 |
| `insertRunStart(simSeq, cfg)` | 학습 시작 시 `simulation_runs` INSERT (status=running) |
| `updateRunEnd(simSeq, ...)` | 완료/실패 시 결과 UPDATE (train_loss, runtime_sec 등) |
| `DbLogCallback` | `TrainerCallback` 서브클래스 — 스텝마다 `simulation_logs` INSERT |
| `loadConfig()` | `os.path.dirname(os.path.abspath(__file__))` 기준 `.env` 로드 (SimulApp 다른 cwd에서 spawn 시에도 정상 동작) |
| `try-except` | `trainer.train()` 실패 시 status=failed로 DB 기록 후 예외 재발생 |

### SimulApp/ (신규 생성)

**변경 내용:** Express 기반 파인튜닝 시뮬레이션 웹 앱 신규 구성

| 파일 | 내용 |
|------|------|
| `SimulApp/db_create.sql` | `simulation_runs`, `simulation_logs` 테이블 생성 SQL |
| `SimulApp/package.json` | express, mysql2, dotenv 의존성 정의 |
| `SimulApp/server.js` | Express API 서버 — `.env` 읽기/쓰기, Python 스크립트 spawn, DB 조회, 로그 폴링 |
| `SimulApp/public/index.html` | HTML/JS UI — 설정 탭(파라미터 저장·학습 실행·로그 실시간 표시) + 결과 탭(이력 테이블·Chart.js 손실 그래프) |

**SimulApp/server.js 주요 API:**

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/config` | GET | `.env` 하이퍼파라미터 조회 |
| `/api/config` | POST | `.env` 하이퍼파라미터 저장 |
| `/api/run/start` | POST | Python 파인튜닝 스크립트 spawn |
| `/api/run/status` | GET | 실행 상태 + 로그 폴링 (offset 파라미터) |
| `/api/simulations` | GET | `simulation_runs` 이력 조회 (최근 50건) |
| `/api/simulations/:seq` | GET | 특정 시뮬레이션 상세 |
| `/api/simulations/:seq/logs` | GET | 스텝별 손실 로그 조회 |

**실행:**
```bash
cd SimulApp
npm install
node server.js
# 브라우저: http://localhost:3000
```

### codeset/docs/★02. 파인튜닝_멀티모달데이터_Gemma3.md (신규 생성)

**변경 내용:** 파인튜닝 스크립트 모델 정의서 작성 (함수 구조, .env 설정, DB 스키마, 출력 경로)

---

## 2026-06-07

### ★01. 데이터 준비_멀티모달데이터_Gemma3.ipynb

**문제:** `concatenate_datasets([image_dataset, text_dataset])` 실행 시 `ValueError: The features can't be aligned` 발생

**원인:**
- `Dataset.from_list()` (image_dataset): 문자열 컬럼 → `Value('string')`
- `Dataset.from_pandas()` (text_dataset): 문자열 컬럼 → `Value('large_string')` (최신 datasets 라이브러리 변경 사항)
- 두 타입이 불일치하여 concatenate 불가

**변경 내용:**

| 셀 ID | 변경 전 | 변경 후 |
|-------|---------|---------|
| `afa9b05d` | `concatenate_datasets([image_dataset, text_dataset])` 직접 호출 | `text_dataset.cast(image_dataset.features)` 로 타입 통일 후 concatenate |

**효과:**
- `large_string` → `string` 타입 캐스팅으로 피처 정렬 오류 해결
- Python 버전 무관, datasets 라이브러리 버전 업그레이드 시에도 안정적으로 동작
