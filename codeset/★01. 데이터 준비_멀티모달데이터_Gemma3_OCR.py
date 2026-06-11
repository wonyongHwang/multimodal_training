"""
01_data_prep_ocr.py
OCR 손글씨 이미지 + Alpaca 텍스트 데이터를 통합하여 Hugging Face Hub에 업로드합니다.
실행: python 01_data_prep_ocr.py
"""

import os
import json
import random
import zipfile
from io import BytesIO

import pandas as pd
from PIL import Image, ImageFile
from datasets import Dataset, Image as HFImage, Features, Value, concatenate_datasets, load_from_disk
from dotenv import load_dotenv

ImageFile.LOAD_TRUNCATED_IMAGES = True


def loadSettings():
    """환경 변수에서 전체 설정을 로드하여 딕셔너리로 반환합니다."""
    load_dotenv(override=True)

    _maxOcr = os.getenv("MAX_OCR_SAMPLES", "")
    maxOcrSamples = int(_maxOcr) if _maxOcr.strip() else None

    ocrRoot = os.getenv("OCR_ROOT", "./dataset/05. 멀티모달 OCR/OCR손글씨")

    return {
        "hfToken":       os.getenv("HF_TOKEN", ""),
        "hfDatasetRepo": os.getenv("HF_DATASET_REPO", "hyokwan/ocr_dataset"),
        "imageExts":     {'.jpg', '.jpeg', '.png', '.webp'},
        "maxOcrSamples": maxOcrSamples,
        "seed":          int(os.getenv("SEED", "42")),
        "ocrZipRoot":    ocrRoot,
        "ocrLabelRoot":  os.path.join(ocrRoot, "1.Training", "라벨링데이터", "TL", "라벨"),
        "textJsonPath":  os.getenv("TEXT_JSON_PATH", "./dataset/04. 멀티모달 샘플/text_samples/sample_data.json"),
        "chunkSize":     int(os.getenv("CHUNK_SIZE", "500")),
        "datasetDir":    os.getenv("DATASET_DIR", "./dataset/ocr_dataset"),
        "instrText":     os.getenv("OCR_INSTR_TEXT", "이미지에서 손글씨로 작성된 텍스트를 모두 인식하여 전사하세요."),
        "inputText":     os.getenv("OCR_INPUT_TEXT",  "이미지에 포함된 모든 텍스트를 공백으로 구분하여 순서대로 나열해주세요."),
    }


def loadTextDataset(textJsonPath):
    """Alpaca 형식 텍스트 JSON 파일을 로드하여 HF Dataset으로 반환합니다."""
    inDf = pd.read_json(textJsonPath, lines=True).copy()
    inDf["modality"] = "text_only"
    inDf["image"]    = None
    inDf["source"]   = "generated"
    inDf["label"]    = ""
    alpacaDf = inDf[["modality", "image", "instruction", "input", "output", "source", "label"]]
    return Dataset.from_pandas(alpacaDf)


def buildZipImageMap(zipRootDir):
    """주어진 루트 폴더 아래의 모든 ZIP 파일을 재귀 탐색하여 이미지ID → (zip경로, 내부경로) 매핑을 생성합니다."""
    imageExts = {'.jpg', '.jpeg', '.png', '.webp'}
    imageMap  = {}

    for dirPath, dirNames, fileNames in os.walk(zipRootDir):
        for i in range(0, len(fileNames)):
            if not fileNames[i].endswith('.zip'):
                continue
            zipPath = os.path.join(dirPath, fileNames[i])
            try:
                with zipfile.ZipFile(zipPath, 'r') as zf:
                    namelist = zf.namelist()
                    for j in range(0, len(namelist)):
                        entry = namelist[j]
                        ext   = os.path.splitext(entry)[-1].lower()
                        if ext in imageExts:
                            imgId = os.path.splitext(os.path.basename(entry))[0]
                            imageMap[imgId] = (zipPath, entry)
            except Exception as e:
                print(f"ZIP 스캔 오류: {zipPath} - {e}")

    return imageMap


def collectJsonFiles(labelRoot):
    """라벨 루트 폴더에서 모든 JSON 파일 경로를 재귀적으로 수집합니다."""
    jsonPaths = []
    for root, dirs, files in os.walk(labelRoot):
        for f in files:
            if f.endswith('.json'):
                jsonPaths.append(os.path.join(root, f))
    return jsonPaths


def loadOcrImage(imgId, zipMap):
    """ZIP 매핑을 사용하여 이미지 ID에 해당하는 PIL 이미지를 로드합니다."""
    if imgId not in zipMap:
        return None
    zipPath, entryPath = zipMap[imgId]
    with zipfile.ZipFile(zipPath, 'r') as zf:
        imgData = zf.read(entryPath)
    img = Image.open(BytesIO(imgData))
    img.load()
    return img.convert("RGB")


def processOcrShards(jsonFiles, zipImageMap, cfg):
    """OCR JSON 라벨과 ZIP 이미지를 청크 단위로 처리하여 Arrow shard로 저장합니다."""
    datasetDir = cfg["datasetDir"]
    chunkSize  = cfg["chunkSize"]
    instrText  = cfg["instrText"]
    inputText  = cfg["inputText"]

    os.makedirs(datasetDir, exist_ok=True)

    ocrFeatures = Features({
        "modality":    Value("string"),
        "image":       HFImage(),
        "instruction": Value("string"),
        "input":       Value("string"),
        "output":      Value("string"),
        "source":      Value("string"),
        "label":       Value("string"),
    })

    totalFiles = len(jsonFiles)
    numChunks  = (totalFiles + chunkSize - 1) // chunkSize

    for chunkIdx in range(0, numChunks):
        shardPath = os.path.join(datasetDir, f"shard_{chunkIdx:04d}")

        if os.path.exists(shardPath):
            print(f"[청크 {chunkIdx+1}/{numChunks}] 이미 존재 - 건너뜀")
            continue

        startIdx = chunkIdx * chunkSize
        endIdx   = min(startIdx + chunkSize, totalFiles)

        rows = []
        for i in range(startIdx, endIdx):
            try:
                with open(jsonFiles[i], "r", encoding="utf-8") as fp:
                    labelData = json.load(fp)

                imgInfo   = labelData.get("Images", {})
                bboxList  = labelData.get("bbox", [])
                imgId     = imgInfo.get("identifier", "")
                mediaType = "P.Paper" if imgInfo.get("media_type", 0) == 0 else "T.Tablet"

                ocrTokens = []
                for j in range(0, len(bboxList)):
                    ocrTokens.append(bboxList[j].get("data", ""))
                outputText = " ".join(ocrTokens)

                pilImage = loadOcrImage(imgId, zipImageMap)
                if pilImage is None:
                    continue

                imgBuffer = BytesIO()
                pilImage.save(imgBuffer, format="JPEG", quality=85)
                pilImage.close()

                rows.append({
                    "modality":    "image_text",
                    "image":       Image.open(BytesIO(imgBuffer.getvalue())),
                    "instruction": instrText,
                    "input":       inputText,
                    "output":      outputText,
                    "source":      "OCR_handwriting",
                    "label":       mediaType
                })
            except Exception as e:
                print(f"처리 오류: {jsonFiles[i]} - {e}")

        Dataset.from_list(rows, features=ocrFeatures).save_to_disk(shardPath)
        print(f"[청크 {chunkIdx+1}/{numChunks}] {startIdx}~{endIdx-1} 범위 저장 완료 ({len(rows)}개)")

    print(f"전체 처리 완료: {numChunks}개 청크, 저장 경로 → {datasetDir}")


def loadOcrDataset(datasetDir):
    """저장된 Arrow shard를 모두 로드하여 하나의 Dataset으로 반환합니다."""
    allEntries = os.listdir(datasetDir)
    shardNames = []
    for i in range(0, len(allEntries)):
        if allEntries[i].startswith("shard_"):
            shardNames.append(allEntries[i])
    shardNames = sorted(shardNames)

    chunkDatasets = []
    for i in range(0, len(shardNames)):
        shardPath = os.path.join(datasetDir, shardNames[i])
        chunkDatasets.append(load_from_disk(shardPath))
        print(f"{shardNames[i]}: {len(chunkDatasets[-1])}개 완료")

    imageDataset = concatenate_datasets(chunkDatasets)
    print(f"OCR 이미지 샘플 수: {len(imageDataset)}")
    return imageDataset


def main():
    """전체 파이프라인 실행: 텍스트/OCR 데이터 준비 → 통합 → Hub 업로드."""
    try:
        cfg = loadSettings()
        random.seed(cfg["seed"])

        print(f"MAX_OCR_SAMPLES = {cfg['maxOcrSamples']}")
        print(f"HF_DATASET_REPO = {cfg['hfDatasetRepo']}")

        print("\n=== 텍스트 데이터 로드 ===")
        textDataset = loadTextDataset(cfg["textJsonPath"])
        print(f"텍스트 샘플 수: {len(textDataset)}")

        print("\n=== ZIP 이미지 인덱싱 ===")
        zipImageMap = buildZipImageMap(cfg["ocrZipRoot"])
        print(f"ZIP 인덱싱 완료: 총 {len(zipImageMap)}개 이미지")

        print("\n=== OCR JSON 라벨 수집 ===")
        jsonFiles = collectJsonFiles(cfg["ocrLabelRoot"])
        if cfg["maxOcrSamples"]:
            jsonFiles = jsonFiles[:cfg["maxOcrSamples"]]
        print(f"JSON 라벨 파일 수: {len(jsonFiles)}")

        print("\n=== OCR Arrow shard 처리 ===")
        processOcrShards(jsonFiles, zipImageMap, cfg)

        print("\n=== Arrow shard 로드 ===")
        imageDataset = loadOcrDataset(cfg["datasetDir"])

        print("\n=== 데이터셋 통합 및 셔플 ===")
        allDataset = concatenate_datasets([imageDataset, textDataset.cast(imageDataset.features)])
        allDataset = allDataset.shuffle(seed=cfg["seed"])
        print(f"전체 샘플 수: {len(allDataset)}")

        print(f"\n=== Hub 업로드 → {cfg['hfDatasetRepo']} ===")
        allDataset.push_to_hub(
            cfg["hfDatasetRepo"],
            token=cfg["hfToken"],
            max_shard_size="500MB"
        )
        print("업로드 완료 ->", cfg["hfDatasetRepo"])

    except Exception as e:
        print({"success": False, "message": str(e)})
        raise


if __name__ == "__main__":
    main()
