"""
inference_service.py
파인튜닝된 멀티모달 Gemma3 모델 추론 FastAPI 서비스 (포트 9999)
실행: .venvg3 환경에서 자동 실행 (server.js 에서 spawn)
"""

import os
import io
import torch
from PIL import Image
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForImageTextToText
from dotenv import load_dotenv

# ============================================================
# 환경 설정
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH   = os.path.join(SCRIPT_DIR, '..', 'codeset', '.env')
load_dotenv(ENV_PATH, override=True)

MODEL_REPO   = os.getenv("MERGED_MODEL_REPO", "")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
SERVICE_PORT = 9999

# ============================================================
# FastAPI 앱
# ============================================================

app = FastAPI(title="멀티모달 추론 서비스")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

inferModel     = None
inferProcessor = None
modelStatus    = "loading"

# ============================================================
# 요청 스키마
# ============================================================

class StatusResponse(BaseModel):
    success:  bool
    status:   str
    ready:    bool
    model:    str

# ============================================================
# 모델 로드 (시작 시 1회)
# ============================================================

@app.on_event("startup")
async def loadModel():
    """ 서비스 시작 시 Hub에서 병합 모델 로드 """
    global inferModel, inferProcessor, modelStatus
    try:
        if not MODEL_REPO or MODEL_REPO in ("YOUR_HF_ID/multi_modal_model_g3", ""):
            modelStatus = "no_model"
            print("[추론서비스] MERGED_MODEL_REPO 미설정 — .env 확인 필요")
            return

        tokenVal  = HF_TOKEN if HF_TOKEN not in ("YOUR_HF_TOKEN", "") else None
        torchType = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        print(f"[추론서비스] 모델 로드 중: {MODEL_REPO}")
        inferModel = AutoModelForImageTextToText.from_pretrained(
            MODEL_REPO,
            device_map="auto",
            torch_dtype=torchType,
            attn_implementation="eager",
            trust_remote_code=True,
            token=tokenVal,
        )
        inferProcessor = AutoProcessor.from_pretrained(
            MODEL_REPO, trust_remote_code=True, token=tokenVal
        )
        modelStatus = "ready"
        print(f"[추론서비스] 모델 로드 완료: {MODEL_REPO}")
    except Exception as e:
        modelStatus = "error"
        print(f"[추론서비스] 모델 로드 실패: {e}")


# ============================================================
# 추론 함수
# ============================================================

def inferGemma3(messages, maxNewTokens, temperature):
    """ 전처리된 messages 리스트를 받아 모델 추론 후 응답 텍스트 반환 """
    pilImages = []
    for i in range(0, len(messages)):
        role    = messages[i].get("role", "")
        content = messages[i].get("content", [])
        for j in range(0, len(content)):
            item = content[j]
            if isinstance(item, dict) and item.get("type") == "image":
                pilImages.append(item["image"])

    prompt = inferProcessor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if len(pilImages) > 0:
        inputs = inferProcessor(
            text=[prompt], images=[pilImages], return_tensors="pt", padding=True
        )
    else:
        inputs = inferProcessor(text=[prompt], return_tensors="pt", padding=True)

    processedInputs = {}
    inputKeys = list(inputs.keys())
    for i in range(0, len(inputKeys)):
        key = inputKeys[i]
        val = inputs[key]
        if hasattr(val, "to"):
            processedInputs[key] = val.to(inferModel.device)
        else:
            processedInputs[key] = val

    doSample = temperature > 0
    with torch.no_grad():
        outputs = inferModel.generate(
            **processedInputs,
            max_new_tokens=maxNewTokens,
            do_sample=doSample,
            temperature=max(temperature, 1e-5),
        )

    answerIds = outputs[:, processedInputs["input_ids"].shape[1]:]
    answer    = inferProcessor.batch_decode(answerIds, skip_special_tokens=True)[0].strip()
    return answer


# ============================================================
# 엔드포인트
# ============================================================

@app.get("/status")
def getStatus():
    """ 서비스 상태 및 모델 로드 여부 반환 """
    return {
        "success": True,
        "status":  modelStatus,
        "ready":   modelStatus == "ready",
        "model":   MODEL_REPO,
    }


@app.post("/chat")
async def chat(
    message:      str        = Form(...),
    systemPrompt: str        = Form("당신은 멀티모달 AI 도우미입니다."),
    image:        UploadFile = File(None),
    maxNewTokens: int        = Form(256),
    temperature:  float      = Form(0.2),
):
    """ 텍스트 또는 이미지+텍스트 추론 수행 """
    try:
        if modelStatus == "loading":
            return {"success": False, "message": "모델 로딩 중입니다. 잠시 후 다시 시도하세요."}
        if modelStatus == "no_model":
            return {"success": False, "message": ".env의 MERGED_MODEL_REPO를 설정하고 서버를 재시작하세요."}
        if modelStatus == "error":
            return {"success": False, "message": "모델 로드에 실패했습니다. 서버 로그를 확인하세요."}

        # 이미지 처리
        pilImage = None
        if image and image.filename:
            imgBytes = await image.read()
            pilImage = Image.open(io.BytesIO(imgBytes)).convert("RGB")
            if max(pilImage.width, pilImage.height) > 1024:
                pilImage.thumbnail((1024, 1024), Image.LANCZOS)

        # 메시지 구성
        userContent = []
        if pilImage:
            userContent.append({"type": "image", "image": pilImage})
        userContent.append({"type": "text", "text": message})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": systemPrompt}]},
            {"role": "user",   "content": userContent},
        ]

        # 추론 수행
        answer = inferGemma3(messages, maxNewTokens, temperature)

        return {"success": True, "response": answer}
    except Exception as e:
        return {"success": False, "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="warning")
