# 모델 정의서: ★03_데이터병합 및 저장_멀티모달데이터_Gemma3.ipynb

## 개요

| 항목 | 내용 |
|------|------|
| 파일명 | `codeset/★03_데이터병합 및 저장_멀티모달데이터_Gemma3.ipynb` |
| 목적 | LoRA 어댑터를 베이스 모델에 병합 → 로컬 저장 → HuggingFace Hub 업로드 → 추론 검증 |
| 입력 | 파인튜닝된 LoRA 어댑터 (`ADAPTER_PATH`) |
| 출력 | 병합 모델 로컬 저장 (`MERGED_LOCAL_DIR`) + HF Hub 업로드 (`MERGED_MODEL_REPO`) |

---

## 노트북 구성 (셀 순서)

| # | 셀 ID | 내용 |
|---|-------|------|
| 1 | `96bc83a1` | Markdown: 노트북 소개 |
| 2 | `7cd7d504` | pip install (주석 처리) |
| 3 | `7b03fc04` | 라이브러리 임포트 (os, torch, PIL, peft, transformers, dotenv, huggingface_hub) |
| 4 | `51267a36` | `.env` 로드 + `hf_token` + `login(hf_token)` |
| 5 | `d71acf41` | **.env 전체 설정값** (BASE_MODEL, ADAPTER_PATH, MERGED_LOCAL_DIR, MERGED_MODEL_REPO, TEST_IMAGE_PATH) |
| 6 | `00c54cb0` | GPU 아키텍처 감지 + BitsAndBytesConfig (4-bit NF4 양자화) |
| 7 | `c433b74f` | 베이스 모델 + 프로세서 로드 (`token=hf_token` 포함) |
| 8 | `6b70c0da` | LoRA 어댑터 로드 → `merge_and_unload()` → 로컬 저장 |
| 9 | `c83c9c93` | HuggingFace Hub 업로드 (`push_to_hub`) |
| 10 | `84d9e724` | Hub에서 병합 모델 재로드 (`infer_model`) |
| 11 | `2b4e3e4c` | Hub에서 프로세서 재로드 (`infer_processor`) |
| 12 | `9d4590fd` | `infer_from_hub()` 함수 정의 (텍스트/이미지+텍스트 추론) |
| 13 | `91aa251a` | 이미지+텍스트 추론 테스트 |
| 14 | `533a84de` | 텍스트-only 추론 테스트 |

---

## .env 설정값

| `.env` 키 | 기본값 | 설명 |
|-----------|--------|------|
| `HF_TOKEN` | `YOUR_HF_TOKEN` | HuggingFace 액세스 토큰 |
| `BASE_MODEL` | `google/gemma-3-4b-it` | 병합 대상 베이스 모델 ID |
| `ADAPTER_PATH` | `./models/gemma3_multimodal_lora_output` | 파인튜닝된 LoRA 어댑터 경로 |
| `MERGED_LOCAL_DIR` | `./models/gemma3_multimodal_merged` | 병합 모델 로컬 저장 경로 |
| `MERGED_MODEL_REPO` | `YOUR_HF_ID/multi_modal_model_g3` | HF Hub 업로드 레포 |
| `TEST_IMAGE_PATH` | `./dataset/pilates_samples/bridge/bridge_001.png` | 추론 테스트 이미지 경로 |

> `ADAPTER_PATH`는 `★02` 파인튜닝 실행 후 생성된 실제 폴더명을 입력해야 합니다.

---

## 처리 흐름

```
BASE_MODEL (HF Hub)
    └─ AutoModelForImageTextToText.from_pretrained
        └─ PeftModel.from_pretrained(ADAPTER_PATH)
            └─ merge_and_unload()
                ├─ save_pretrained(MERGED_LOCAL_DIR)
                └─ push_to_hub(MERGED_MODEL_REPO)

MERGED_MODEL_REPO (HF Hub)
    └─ AutoModelForImageTextToText.from_pretrained  → infer_model
    └─ AutoProcessor.from_pretrained               → infer_processor
        └─ infer_from_hub(question, image_path)
```

---

## 알려진 경고

| 경고 | 내용 | 대응 |
|------|------|------|
| LoRA 4-bit merge 경고 | `Merge lora module to 4-bit linear may get different generations due to rounding errors` | 정밀도 차이 허용 범위, 무시 가능 |

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-06-11 | 초기 작성 (모델 병합·업로드·추론 기본 구조) |
| 2026-06-11 | `.env` 전체 설정 통합, 하드코딩 토큰 제거, `token=hf_token` 모델 로드에 적용 |
