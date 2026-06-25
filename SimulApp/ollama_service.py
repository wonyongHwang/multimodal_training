"""
ollama_service.py
Ollama 모델 등록·관리·채팅 테스트 FastAPI 서비스 (포트 8001)
실행: .venvgguf 환경에서 python ollama_service.py
"""

import os
import base64
import subprocess
import io
from pathlib import Path

import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

try:
    from gguf import GGUFReader
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

app = FastAPI(title="Ollama 테스트 서비스")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR      = os.path.dirname(SCRIPT_DIR)
MODELFILE_DIR = os.path.join(SCRIPT_DIR, 'modelfiles')
OLLAMA_BASE   = "http://localhost:11434"

# 스캔 제외 경로 (llama.cpp 내부 vocab 파일 등)
EXCLUDE_DIRS = ['llama.cpp', '.venvg3', '.venvgguf', '__pycache__']

# ============================================================
# 아키텍처 → Modelfile 템플릿 매핑
# ============================================================

ARCH_TEMPLATES = {
    "gemma3": {
        "label": "Gemma 3",
        "stop":  "<end_of_turn>",
        "body":  (
            'TEMPLATE """{{ if .System }}<start_of_turn>user\n'
            '{{ .System }}<end_of_turn>\n'
            '{{ end }}<start_of_turn>user\n'
            '{{ .Prompt }}<end_of_turn>\n'
            '<start_of_turn>model\n'
            '{{ .Response }}<end_of_turn>\n"""\n'
            'PARAMETER stop "<end_of_turn>"\n'
            'PARAMETER num_ctx 4096'
        ),
    },
    "gemma": {
        "label": "Gemma 2",
        "stop":  "<end_of_turn>",
        "body":  (
            'TEMPLATE """{{ if .System }}<start_of_turn>user\n'
            '{{ .System }}<end_of_turn>\n'
            '{{ end }}<start_of_turn>user\n'
            '{{ .Prompt }}<end_of_turn>\n'
            '<start_of_turn>model\n'
            '{{ .Response }}<end_of_turn>\n"""\n'
            'PARAMETER stop "<end_of_turn>"\n'
            'PARAMETER num_ctx 4096'
        ),
    },
    "gemma4": {
        "label": "Gemma 4",
        "stop":  "<end_of_turn>",
        "body":  (
            'TEMPLATE """{{ if .System }}<start_of_turn>user\n'
            '{{ .System }}<end_of_turn>\n'
            '{{ end }}<start_of_turn>user\n'
            '{{ .Prompt }}<end_of_turn>\n'
            '<start_of_turn>model\n'
            '{{ .Response }}<end_of_turn>\n"""\n'
            'PARAMETER stop "<end_of_turn>"\n'
            'PARAMETER num_ctx 4096'
        ),
    },
    "llama": {
        "label": "Llama 3.x",
        "stop":  "<|eot_id|>",
        "body":  (
            'TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>\n'
            '{{ .System }}<|eot_id|>{{ end }}'
            '<|start_header_id|>user<|end_header_id|>\n'
            '{{ .Prompt }}<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>\n'
            '{{ .Response }}<|eot_id|>\n"""\n'
            'PARAMETER stop "<|eot_id|>"\n'
            'PARAMETER num_ctx 4096'
        ),
    },
    "qwen2": {
        "label": "Qwen 2 / 2.5 / 3",
        "stop":  "<|im_end|>",
        "body":  (
            'TEMPLATE """{{ if .System }}<|im_start|>system\n'
            '{{ .System }}<|im_end|>\n'
            '{{ end }}<|im_start|>user\n'
            '{{ .Prompt }}<|im_end|>\n'
            '<|im_start|>assistant\n'
            '{{ .Response }}<|im_end|>\n"""\n'
            'PARAMETER stop "<|im_end|>"\n'
            'PARAMETER num_ctx 4096'
        ),
    },
    "phi3": {
        "label": "Phi 3 / 4",
        "stop":  "<|end|>",
        "body":  (
            'TEMPLATE """{{ if .System }}<|system|>\n'
            '{{ .System }}<|end|>\n'
            '{{ end }}<|user|>\n'
            '{{ .Prompt }}<|end|>\n'
            '<|assistant|>\n'
            '{{ .Response }}<|end|>\n"""\n'
            'PARAMETER stop "<|end|>"\n'
            'PARAMETER num_ctx 4096'
        ),
    },
    "mistral": {
        "label": "Mistral / Mixtral",
        "stop":  "[/INST]",
        "body":  (
            'TEMPLATE """[INST] {{ if .System }}{{ .System }}\n\n'
            '{{ end }}{{ .Prompt }} [/INST]{{ .Response }}"""\n'
            'PARAMETER stop "[/INST]"\n'
            'PARAMETER num_ctx 4096'
        ),
    },
    "command-r": {
        "label": "Cohere Command R",
        "stop":  "<|END_OF_TURN_TOKEN|>",
        "body":  (
            'TEMPLATE """{{ if .System }}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'
            '{{ .System }}<|END_OF_TURN_TOKEN|>{{ end }}'
            '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ .Prompt }}<|END_OF_TURN_TOKEN|>'
            '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{{ .Response }}<|END_OF_TURN_TOKEN|>"""\n'
            'PARAMETER stop "<|END_OF_TURN_TOKEN|>"\n'
            'PARAMETER num_ctx 4096'
        ),
    },
}


def buildModelfile(ggufPath, arch):
    """GGUF 절대경로와 아키텍처로 Modelfile 내용 생성"""
    tmpl = ARCH_TEMPLATES.get(arch, ARCH_TEMPLATES.get("llama"))
    return f"FROM {ggufPath}\n\n{tmpl['body']}\n"


def readGgufMeta(filePath):
    """GGUF 헤더에서 general.architecture 와 general.name 추출"""
    if not GGUF_AVAILABLE:
        return None, None, "gguf 패키지 미설치"
    try:
        reader = GGUFReader(filePath, 'r')
        arch = None
        name = None
        targetKeys = ['general.architecture', 'general.name']
        for key, field in reader.fields.items():
            if key not in targetKeys:
                continue
            try:
                val = bytes(field.parts[-1]).decode('utf-8', errors='replace').strip('\x00').strip()
                if key == 'general.architecture':
                    arch = val
                elif key == 'general.name':
                    name = val
            except Exception:
                pass
        return arch, name, None
    except Exception as e:
        return None, None, str(e)


def isExcluded(filePath):
    """llama.cpp 내부 vocab 파일 등 제외 대상 여부 확인"""
    parts = Path(filePath).parts
    for i in range(0, len(EXCLUDE_DIRS)):
        if EXCLUDE_DIRS[i] in parts:
            return True
    return False


# ============================================================
# GGUF 파일 목록 / 메타데이터
# ============================================================

@app.get("/gguf/list")
def listGgufFiles():
    """프로젝트 디렉터리 하위 .gguf 파일 목록 반환 (vocab 파일 제외)"""
    try:
        result   = []
        allGgufs = sorted(Path(BASE_DIR).rglob('*.gguf'))
        for i in range(0, len(allGgufs)):
            ggufFile = allGgufs[i]
            if isExcluded(str(ggufFile)):
                continue
            sizeMb = ggufFile.stat().st_size / 1024 / 1024
            if sizeMb < 10:
                continue
            result.append({
                "name":   ggufFile.name,
                "path":   str(ggufFile.resolve()),
                "sizeMb": round(sizeMb, 1),
            })
        return {"success": True, "files": result}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/gguf/metadata")
def getGgufMetadata(filePath: str = Form(...)):
    """GGUF 파일 메타데이터 읽기 — 아키텍처 자동 감지 후 Modelfile 초안 반환"""
    try:
        if not os.path.exists(filePath):
            return {"success": False, "message": f"파일 없음: {filePath}"}
        arch, name, err = readGgufMeta(filePath)
        if err:
            return {"success": False, "message": err}
        archLabel = ARCH_TEMPLATES.get(arch, {}).get("label", arch or "알 수 없음")
        if arch:
            modelfile = buildModelfile(filePath, arch)
        else:
            modelfile = f"FROM {filePath}\n\n# 아키텍처 감지 실패 — 템플릿을 직접 입력하세요\n"
        return {
            "success":   True,
            "arch":      arch,
            "archLabel": archLabel,
            "modelName": name,
            "modelfile": modelfile,
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


# ============================================================
# Ollama 모델 관리
# ============================================================

@app.get("/ollama/models")
def listOllamaModels():
    """Ollama에 등록된 모델 목록 반환"""
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        data = resp.json()
        return {"success": True, "models": data.get("models", [])}
    except Exception as e:
        return {"success": False, "message": f"Ollama 연결 실패: {e}"}


class RegisterRequest(BaseModel):
    modelName:        str
    modelfileContent: str


@app.post("/ollama/register")
def registerModel(req: RegisterRequest):
    """Modelfile 저장 후 ollama create 실행"""
    try:
        os.makedirs(MODELFILE_DIR, exist_ok=True)
        mfPath = os.path.join(MODELFILE_DIR, f"{req.modelName}.Modelfile")
        with open(mfPath, 'w', encoding='utf-8') as f:
            f.write(req.modelfileContent)
        result = subprocess.run(
            ['ollama', 'create', req.modelName, '-f', mfPath],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            return {"success": False, "message": result.stderr or result.stdout}
        return {"success": True, "message": result.stdout or f"'{req.modelName}' 등록 완료"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.delete("/ollama/model/{modelName}")
def deleteModel(modelName: str):
    """Ollama 모델 삭제"""
    try:
        result = subprocess.run(
            ['ollama', 'rm', modelName],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return {"success": False, "message": result.stderr}
        return {"success": True, "message": f"'{modelName}' 삭제 완료"}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ============================================================
# 채팅 테스트 (멀티모달 이미지 포함)
# ============================================================

@app.post("/ollama/chat")
async def chat(
    modelName:    str        = Form(...),
    message:      str        = Form(...),
    systemPrompt: str        = Form(""),
    image:        UploadFile = File(None),
):
    """Ollama 채팅 — 이미지 업로드 포함 멀티모달 지원"""
    try:
        imageBase64 = ""
        if image and image.filename:
            imgBytes = await image.read()
            pil      = Image.open(io.BytesIO(imgBytes)).convert("RGB")
            if max(pil.width, pil.height) > 1024:
                pil.thumbnail((1024, 1024), Image.LANCZOS)
            buf = io.BytesIO()
            pil.save(buf, format='PNG')
            imageBase64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        userMsg = {"role": "user", "content": message}
        if imageBase64:
            userMsg["images"] = [imageBase64]

        messages = []
        if systemPrompt.strip():
            messages.append({"role": "system", "content": systemPrompt.strip()})
        messages.append(userMsg)

        payload = {"model": modelName, "messages": messages, "stream": False}

        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
            data = resp.json()

        content    = data.get("message", {}).get("content", "")
        evalCount  = data.get("eval_count", None)
        return {"success": True, "response": content, "tokens": evalCount}
    except Exception as e:
        return {"success": False, "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
