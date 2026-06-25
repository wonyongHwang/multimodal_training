"""
★03. 데이터병합 및 저장_멀티모달데이터_Gemma3.py
LoRA 어댑터를 베이스 모델에 병합하고 HuggingFace Hub에 업로드합니다.

실행 방법:
    python "★03. 데이터병합 및 저장_멀티모달데이터_Gemma3.py"

설정 변경:
    .env 파일의 ADAPTER_PATH, MERGED_LOCAL_DIR, MERGED_MODEL_REPO 값을 수정하거나
    SimulApp 웹 화면 '모델 병합' 탭에서 설정 후 실행
"""

# ============================================================
# 라이브러리 임포트
# ============================================================

import os
import shutil
from datetime import datetime

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText
from dotenv import load_dotenv
from huggingface_hub import login, HfApi


# ============================================================
# 설정 로드
# ============================================================

def loadConfig():
    """
    스크립트와 같은 디렉토리의 .env를 로드하고 전체 설정값 딕셔너리 반환
    """
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(scriptDir, '.env'), override=True)

    cfg = {
        'hf_token':          os.getenv("HF_TOKEN",          "YOUR_HF_TOKEN"),
        'base_model':        os.getenv("BASE_MODEL",         "google/gemma-3-4b-it"),
        'adapter_path':      os.getenv("ADAPTER_PATH",       "./models/gemma3_multimodal_lora_output"),
        'merged_local_dir':  os.getenv("MERGED_LOCAL_DIR",   "./models/gemma3_multimodal_merged"),
        'merged_model_repo': os.getenv("MERGED_MODEL_REPO",  "YOUR_HF_ID/multi_modal_model_g3"),
        'test_image_path':   os.getenv("TEST_IMAGE_PATH",    ""),
    }
    return cfg


# ============================================================
# HuggingFace 로그인
# ============================================================

def hfLogin(hfToken):
    """HuggingFace 토큰으로 로그인 (실패 시 경고만 출력)"""
    if hfToken and hfToken not in ('YOUR_HF_TOKEN', ''):
        try:
            login(hfToken)
            print("HuggingFace 로그인 완료")
        except Exception as e:
            print(f"[경고] HuggingFace 로그인 실패: {e}")
            print("[경고] .env의 HF_TOKEN을 확인하세요. 공개 모델만 접근 가능합니다.")
    else:
        print("[경고] HF_TOKEN이 설정되지 않았습니다. 공개 모델만 접근 가능합니다.")


# ============================================================
# 모델 로드
# ============================================================

def loadBaseModel(cfg):
    """
    베이스 모델과 프로세서를 양자화 없이 fp16/bf16으로 로드하여 반환.
    quantization_config 사용 금지 → 4-bit 상태로 push하면 HF 재로드 시 shape mismatch 발생
    """
    if torch.cuda.get_device_capability()[0] >= 8:
        torchDtype = torch.bfloat16
    else:
        torchDtype = torch.float16

    # ★ 이전 병합 결과물 삭제 (재실행 시 양자화 잔재 방지)
    mergedLocalDir = cfg['merged_local_dir']
    if os.path.exists(mergedLocalDir):
        shutil.rmtree(mergedLocalDir)
        print(f"기존 병합 폴더 삭제 완료: {mergedLocalDir}")

    print(f"베이스 모델 로드 중: {cfg['base_model']}")
    baseModel = AutoModelForImageTextToText.from_pretrained(
        cfg['base_model'],
        device_map='auto',
        torch_dtype=torchDtype,
        attn_implementation='eager',
        trust_remote_code=True,
        token=cfg['hf_token'] if cfg['hf_token'] not in ('YOUR_HF_TOKEN', '') else None,
    )
    processor = AutoProcessor.from_pretrained(
        cfg['base_model'],
        trust_remote_code=True,
        token=cfg['hf_token'] if cfg['hf_token'] not in ('YOUR_HF_TOKEN', '') else None,
    )
    print("베이스 모델 로드 완료")
    return baseModel, processor


# ============================================================
# 어댑터 병합
# ============================================================

def mergeAdapter(baseModel, adapterPath):
    """LoRA 어댑터를 베이스 모델에 병합하고 병합된 모델 반환"""
    print(f"어댑터 병합 중: {adapterPath}")
    peftModel   = PeftModel.from_pretrained(baseModel, adapterPath)
    mergedModel = peftModel.merge_and_unload()
    print("어댑터 병합 완료")
    return mergedModel


# ============================================================
# 로컬 저장 및 Hub 업로드
# ============================================================

def saveMerged(mergedModel, processor, mergedLocalDir):
    """병합된 모델과 프로세서를 로컬에 저장 (양자화 잔재 검증 포함)"""
    try:
        for name, param in mergedModel.named_parameters():
            if param.dtype in (torch.int8, torch.uint8) or (len(param.shape) == 1 and param.shape[0] > 1_000_000):
                raise RuntimeError(
                    f"양자화 잔재 감지: {name} | dtype={param.dtype} shape={param.shape}\n"
                    "→ loadBaseModel()부터 다시 실행하세요. quantization_config 없이 로드해야 합니다."
                )
        print("검증 통과: 양자화 잔재 없음, 저장 진행")
        os.makedirs(mergedLocalDir, exist_ok=True)
        # safetensors 0.8.0은 shared tensor 허용 안 함 → tied weight 키를 state_dict에서 제거 후 저장
        # config.json의 tie_word_embeddings=True 덕분에 from_pretrained 시 자동 복원됨
        stateDict = mergedModel.state_dict()
        tiedKey = "lm_head.weight"
        if tiedKey in stateDict:
            del stateDict[tiedKey]
            print(f"Tied weight 제거: {tiedKey} (로드 시 tie_word_embeddings=True로 자동 복원)")
        mergedModel.save_pretrained(mergedLocalDir, state_dict=stateDict)
        processor.save_pretrained(mergedLocalDir)
        print(f"병합 모델 로컬 저장 완료: {mergedLocalDir}")
        return True
    except Exception as e:
        print({"success": False, "message": f"로컬 저장 실패: {e}"})
        return False


def uploadToHub(mergedLocalDir, mergedModelRepo, hfToken):
    """로컬 저장된 병합 모델을 HuggingFace Hub에 업로드 후 로컬 폴더 삭제"""
    try:
        tokenVal = hfToken if hfToken not in ('YOUR_HF_TOKEN', '') else None
        print(f"Hub 업로드 중: {mergedModelRepo}")
        api = HfApi()
        api.upload_folder(
            folder_path=mergedLocalDir,
            repo_id=mergedModelRepo,
            repo_type='model',
            token=tokenVal,
        )
        print(f"Hub 업로드 완료: {mergedModelRepo}")
        shutil.rmtree(mergedLocalDir)
        print(f"로컬 병합 폴더 삭제 완료: {mergedLocalDir}")
    except Exception as e:
        print({"success": False, "message": f"Hub 업로드 실패: {e}"})


# ============================================================
# 추론 검증
# ============================================================

def loadMergedFromHub(mergedModelRepo, hfToken):
    """Hub에서 병합 모델과 프로세서를 로드하여 반환"""
    try:
        tokenVal = hfToken if hfToken not in ('YOUR_HF_TOKEN', '') else None
        print(f"Hub에서 모델 로드 중: {mergedModelRepo}")
        inferModel = AutoModelForImageTextToText.from_pretrained(
            mergedModelRepo,
            device_map='auto',
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation='eager',
            trust_remote_code=True,
            token=tokenVal,
        )
        inferProcessor = AutoProcessor.from_pretrained(
            mergedModelRepo, trust_remote_code=True, token=tokenVal
        )
        print("Hub 모델 로드 완료")
        return inferModel, inferProcessor
    except Exception as e:
        print({"success": False, "message": f"Hub 모델 로드 실패: {e}"})
        return None, None


def inferFromHub(question, inferModel, inferProcessor, imagePath=None,
                 system='당신은 멀티모달 도우미입니다.', maxNewTokens=128, temperature=0.2):
    """
    병합된 모델로 텍스트 또는 이미지+텍스트 추론 수행 후 결과 문자열 반환
    """
    messages = [
        {'role': 'system', 'content': [{'type': 'text', 'text': system}]},
        {'role': 'user', 'content': []},
    ]

    pilImage = None
    if imagePath and os.path.exists(imagePath):
        pilImage = Image.open(imagePath).convert('RGB')
        messages[1]['content'].append({'type': 'image', 'image': pilImage})

    messages[1]['content'].append({'type': 'text', 'text': question})

    prompt = inferProcessor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if pilImage is not None:
        inputs = inferProcessor(
            text=[prompt], images=[[pilImage]], return_tensors='pt', padding=True
        )
    else:
        inputs = inferProcessor(text=[prompt], return_tensors='pt', padding=True)

    inputs = {
        k: v.to(inferModel.device) if hasattr(v, 'to') else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = inferModel.generate(
            **inputs,
            max_new_tokens=maxNewTokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
        )

    answerIds = outputs[:, inputs['input_ids'].shape[1]:]
    return inferProcessor.batch_decode(answerIds, skip_special_tokens=True)[0].strip()


# ============================================================
# 메인
# ============================================================

def main():
    """Gemma3 LoRA 어댑터 병합 → 로컬 저장 → Hub 업로드 → 추론 검증"""

    cfg = loadConfig()

    print("\n====== 모델 병합 시작 ======")
    print(f"베이스 모델    : {cfg['base_model']}")
    print(f"어댑터 경로    : {cfg['adapter_path']}")
    print(f"병합 저장 경로 : {cfg['merged_local_dir']}")
    print(f"Hub 레포       : {cfg['merged_model_repo']}\n")

    if torch.cuda.is_available():
        print('GPU =', torch.cuda.get_device_name(0))

    hfLogin(cfg['hf_token'])

    # ===== 베이스 모델 로드 =====
    try:
        baseModel, processor = loadBaseModel(cfg)
    except Exception as e:
        print({"success": False, "message": f"베이스 모델 로드 실패: {e}"})
        return

    # ===== 어댑터 병합 =====
    try:
        mergedModel = mergeAdapter(baseModel, cfg['adapter_path'])
    except Exception as e:
        print({"success": False, "message": f"어댑터 병합 실패: {e}"})
        return

    # ===== 로컬 저장 =====
    if not saveMerged(mergedModel, processor, cfg['merged_local_dir']):
        print({"success": False, "message": "로컬 저장 실패 — 업로드 중단"})
        return

    # ===== Hub 업로드 (로컬 → Hub → 로컬 삭제) =====
    uploadToHub(cfg['merged_local_dir'], cfg['merged_model_repo'], cfg['hf_token'])

    # ===== Hub에서 재로드 후 추론 검증 =====
    inferModel, inferProcessor = loadMergedFromHub(cfg['merged_model_repo'], cfg['hf_token'])

    if inferModel is not None and inferProcessor is not None:
        print("\n--- 텍스트 추론 테스트 ---")
        try:
            textAnswer = inferFromHub(
                question='금융 용어를 쉽게 설명하시오.\n\n[입력]\n기준금리',
                inferModel=inferModel,
                inferProcessor=inferProcessor,
                system='당신은 금융 개념을 쉽게 설명하는 도우미입니다.',
                maxNewTokens=96,
                temperature=0.0,
            )
            print(textAnswer)
        except Exception as e:
            print({"success": False, "message": f"추론 실패: {e}"})

        if cfg['test_image_path'] and os.path.exists(cfg['test_image_path']):
            print("\n--- 이미지+텍스트 추론 테스트 ---")
            try:
                imageAnswer = inferFromHub(
                    question='이 이미지의 동작명을 말하고 한 줄 설명을 덧붙이세요.',
                    inferModel=inferModel,
                    inferProcessor=inferProcessor,
                    imagePath=cfg['test_image_path'],
                    system='당신은 필라테스 자세 분류 도우미입니다.',
                    temperature=0.0,
                )
                print(imageAnswer)
            except Exception as e:
                print({"success": False, "message": f"이미지 추론 실패: {e}"})

    print("\n====== 모델 병합 완료 ======")
    print(f"로컬 저장 경로 : {cfg['merged_local_dir']}")
    print(f"Hub 레포       : https://huggingface.co/{cfg['merged_model_repo']}")


if __name__ == '__main__':
    main()
