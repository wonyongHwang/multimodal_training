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
