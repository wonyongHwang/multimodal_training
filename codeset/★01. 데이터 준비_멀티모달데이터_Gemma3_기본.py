"""
★01. 데이터 준비_멀티모달데이터_Gemma3_스포츠.py
필라테스 이미지 + 텍스트 데이터를 HuggingFace Dataset으로 통합·업로드하는 스크립트

실행 방법:
    python "★01. 데이터 준비_멀티모달데이터_Gemma3_스포츠.py"

설정 변경:
    .env 파일의 경로·프롬프트·샘플 수 제한을 변경해 코드 수정 없이 조정 가능
"""

# ============================================================
# 라이브러리 임포트
# ============================================================

# === 표준 라이브러리 ===
import os
import random

# === 데이터 처리 ===
import pandas as pd
from PIL import Image

# === HuggingFace ===
from datasets import Dataset, concatenate_datasets

# === 환경변수 ===
from dotenv import load_dotenv


# ============================================================
# 설정 로드
# ============================================================

def loadConfig():
    """
    .env 파일을 로드하고 전체 설정값을 딕셔너리로 반환
    .env 값이 없으면 기본값을 사용하므로 코드 수정 없이 경로·프롬프트·샘플 수 조정 가능
    """
    load_dotenv()

    _maxSamples = os.getenv("MAX_IMAGE_SAMPLES_PER_CLASS", "")

    cfg = {
        # HuggingFace
        'hf_token':        os.getenv("HF_TOKEN", ""),
        'hf_dataset_repo': os.getenv("HF_DATASET_REPO", "hyokwan/multi_modal_sample"),

        # 데이터 경로
        'image_root':      os.getenv("IMAGE_ROOT",      "./dataset/04. 멀티모달 샘플/image_samples"),
        'image_meta_root': os.getenv("IMAGE_META_ROOT", "./dataset/04. 멀티모달 샘플/image_samples/labels.csv"),
        'text_json_path':  os.getenv("TEXT_JSON_PATH",  "./dataset/04. 멀티모달 샘플/text_samples/sample_data.json"),

        # 이미지 수집 설정
        'max_image_samples_per_class': int(_maxSamples) if _maxSamples.strip() else None,
        'image_exts': {'.jpg', '.jpeg', '.png', '.webp'},

        # 이미지 프롬프트
        'image_instruction': os.getenv("IMAGE_INSTRUCTION", "이미지를 보고 필라테스 동작명을 분류하고 한 줄 설명을 제공하세요."),
        'image_input':       os.getenv("IMAGE_INPUT",       "동작명과 간단한 설명을 한국어로 답하세요."),

        # 시드
        'seed': int(os.getenv("SEED", 42)),
    }
    return cfg


# ============================================================
# 데이터 수집 함수
# ============================================================

def loadTextData(cfg):
    """
    텍스트 JSON 파일을 로드하고 Alpaca 포맷의 HuggingFace Dataset을 반환
    """
    try:
        inDf = pd.read_json(cfg['text_json_path'], lines=True)

        inDf["modality"] = "text_only"
        inDf["image"]    = None
        inDf["source"]   = "generated"
        inDf["label"]    = ""

        alpacaDf    = inDf[["modality", "image", "instruction", "input", "output", "source", "label"]]
        textDataset = Dataset.from_pandas(alpacaDf)

        print(f"텍스트 데이터 로드 완료: {len(textDataset)}건")
        return textDataset, alpacaDf

    except Exception as e:
        print({"success": False, "message": str(e)})
        return None, None


def loadImageData(cfg):
    """
    이미지 메타 CSV를 읽어 PIL 이미지와 레이블을 담은 HuggingFace Dataset을 반환
    MAX_IMAGE_SAMPLES_PER_CLASS 설정 시 클래스별 수집 수를 제한
    """
    try:
        df          = pd.read_csv(cfg['image_meta_root'])
        imageData   = []
        classCounts = {}

        print(f"이미지 메타 로드 완료: {len(df)}건  |  클래스: {df['pose_label'].nunique()}종")

        for i in range(0, len(df)):
            label     = df.loc[i, "pose_label"]
            imagePath = os.path.join(cfg['image_root'], df.loc[i, "image_path"])

            # 클래스당 최대 샘플 수 제한
            if cfg['max_image_samples_per_class'] is not None:
                currentCount = classCounts.get(label, 0)
                if currentCount >= cfg['max_image_samples_per_class']:
                    continue
                classCounts[label] = currentCount + 1

            image      = Image.open(imagePath).convert("RGB")
            outputText = f"동작명: {label}\n설명: 이 이미지는 {label} 동작 예시입니다."

            imageData.append({
                "modality":    "image_text",
                "image":       image,
                "instruction": cfg['image_instruction'],
                "input":       cfg['image_input'],
                "output":      outputText,
                "source":      "pilates_dataset",
                "label":       label
            })

        imageDataset = Dataset.from_list(imageData)
        print(f"이미지 샘플 수: {len(imageDataset)}건")
        return imageDataset

    except Exception as e:
        print({"success": False, "message": str(e)})
        return None


def mergeAndUpload(textDataset, imageDataset, cfg):
    """
    텍스트·이미지 데이터셋을 병합·셔플하고 HuggingFace Hub에 업로드
    """
    try:
        # datasets 버전 업그레이드 후 large_string/string 타입 불일치 해결
        textDataset = textDataset.cast(imageDataset.features)

        allDataset = concatenate_datasets([imageDataset, textDataset])
        print(f"전체 데이터셋: {len(allDataset)}건  (이미지: {len(imageDataset)}건 + 텍스트: {len(textDataset)}건)")

        allDataset = allDataset.shuffle(seed=cfg['seed'])
        print("셔플 완료")

        allDataset.push_to_hub(cfg['hf_dataset_repo'], token=cfg['hf_token'])
        print(f"업로드 완료 → {cfg['hf_dataset_repo']}")

    except Exception as e:
        print({"success": False, "message": str(e)})


# ============================================================
# 메인 함수
# ============================================================

def main():
    """필라테스 멀티모달 데이터 수집 및 HuggingFace Hub 업로드 메인 실행 함수"""

    # ===== 설정 로드 =====
    cfg = loadConfig()
    random.seed(cfg['seed'])

    print(f"HF 레포         : {cfg['hf_dataset_repo']}")
    print(f"이미지 경로     : {cfg['image_root']}")
    print(f"메타 CSV        : {cfg['image_meta_root']}")
    print(f"텍스트 JSON     : {cfg['text_json_path']}")
    print(f"클래스당 최대   : {cfg['max_image_samples_per_class'] if cfg['max_image_samples_per_class'] else '제한 없음'}")
    print()

    # ===== 텍스트 데이터 수집 =====
    textDataset, alpacaDf = loadTextData(cfg)
    if textDataset is None:
        return

    # ===== 이미지 데이터 수집 =====
    imageDataset = loadImageData(cfg)
    if imageDataset is None:
        return

    # ===== 병합 및 업로드 =====
    mergeAndUpload(textDataset, imageDataset, cfg)


if __name__ == '__main__':
    main()
