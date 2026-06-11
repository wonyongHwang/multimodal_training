"""
★02. 파인튜닝_멀티모달데이터_Gemma3.py
Hugging Face 통합 데이터셋을 사용한 Gemma 3 QLoRA 파인튜닝 스크립트
DB에 sim_seq 기반으로 설정값·스텝 로그·학습 결과를 저장합니다.

실행 방법:
    python "★02. 파인튜닝_멀티모달데이터_Gemma3.py"

설정 변경:
    .env 파일 값을 변경하거나 SimulApp 웹 화면에서 설정 후 학습 시작
"""

# ============================================================
# 라이브러리 임포트
# ============================================================

import os
import random
from datetime import datetime

import pandas as pd
from datasets import load_dataset

import torch
from peft import LoraConfig
from transformers import (
    AutoProcessor, AutoModelForImageTextToText,
    BitsAndBytesConfig, TrainerCallback
)
from trl import SFTTrainer, SFTConfig

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from PIL import Image
from dotenv import load_dotenv
import huggingface_hub
import mysql.connector


# ============================================================
# matplotlib 한글 폰트
# ============================================================

def setKoreanFont():
    """matplotlib 한글 폰트 설정 (Windows: 맑은 고딕 우선 탐색)"""
    fontCandidates = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'AppleGothic', 'DejaVu Sans']
    allFonts       = fm.fontManager.ttflist
    availableFonts = []
    for i in range(0, len(allFonts)):
        availableFonts.append(allFonts[i].name)

    selectedFont = None
    for i in range(0, len(fontCandidates)):
        if fontCandidates[i] in availableFonts:
            selectedFont = fontCandidates[i]
            break

    if selectedFont is None:
        selectedFont = 'DejaVu Sans'
        print(f'[경고] 한글 폰트 없음, 대체 사용: {selectedFont}')
    else:
        print(f'한글 폰트 설정: {selectedFont}')

    matplotlib.rc('font', family=selectedFont)
    matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# 설정 로드
# ============================================================

def loadConfig():
    """
    스크립트와 같은 디렉토리의 .env를 로드하고 전체 설정값 딕셔너리 반환
    .env 값만 변경해 코드 수정 없이 하이퍼파라미터 시뮬레이션 가능
    """
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(scriptDir, '.env'))

    outputBaseDir = os.getenv("OUTPUT_BASE_DIR", "./models")
    outputDir     = os.path.join(
        outputBaseDir,
        f"gemma3_multimodal_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    _trainSamples = os.getenv("MAX_TRAIN_SAMPLES", "")
    _evalSamples  = os.getenv("MAX_EVAL_SAMPLES",  "")

    cfg = {
        # HuggingFace
        'hf_token':     os.getenv("HF_TOKEN", ""),
        'dataset_repo': os.getenv("DATASET_REPO", "hyokwan/multi_modal_sample"),
        'base_model':   os.getenv("BASE_MODEL",   "google/gemma-3-4b-it"),

        # 경로
        'output_dir':      outputDir,
        'tensorboard_dir': os.path.join(outputDir, "tensorboard"),

        # 데이터 제한
        'max_train_samples': int(_trainSamples) if _trainSamples.strip() else None,
        'max_eval_samples':  int(_evalSamples)  if _evalSamples.strip()  else None,

        # 시드
        'seed': int(os.getenv("SEED", 42)),

        # 학습 하이퍼파라미터
        'max_seq_length':               int(os.getenv("MAX_SEQ_LENGTH",              768)),
        'per_device_train_batch_size':  int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", 1)),
        'per_device_eval_batch_size':   int(os.getenv("PER_DEVICE_EVAL_BATCH_SIZE",  1)),
        'grad_accum':                   int(os.getenv("GRAD_ACCUM",                  4)),
        'num_epochs':                   int(os.getenv("NUM_EPOCHS",                  5)),
        'learning_rate':                float(os.getenv("LEARNING_RATE",             2e-4)),
        'logging_steps':                int(os.getenv("LOGGING_STEPS",               10)),
        'save_steps':                   int(os.getenv("SAVE_STEPS",                  100)),

        # LoRA 하이퍼파라미터
        'lora_r':       int(os.getenv("LORA_R",       16)),
        'lora_alpha':   int(os.getenv("LORA_ALPHA",   32)),
        'lora_dropout': float(os.getenv("LORA_DROPOUT", 0.05)),

        # DB 연결 정보
        'db': {
            'host':     os.getenv("DB_HOST",     "localhost"),
            'port':     int(os.getenv("DB_PORT", 3306)),
            'database': os.getenv("DB_NAME",     ""),
            'user':     os.getenv("DB_USER",     ""),
            'password': os.getenv("DB_PASSWORD", ""),
        },
    }
    return cfg


# ============================================================
# DB 유틸리티
# ============================================================

def getDbConn(dbCfg):
    """MySQL 연결 객체 반환"""
    return mysql.connector.connect(**dbCfg)


def getNextSimSeq(dbCfg):
    """DB에서 다음 시뮬레이션 시퀀스 번호 조회 (MAX + 1)"""
    try:
        conn   = getDbConn(dbCfg)
        cursor = conn.cursor()
        cursor.execute("SELECT COALESCE(MAX(sim_seq), 0) + 1 FROM simulation_runs")
        simSeq = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return int(simSeq)
    except Exception as e:
        print(f"[DB] sim_seq 조회 실패: {e}")
        return 1


def insertRunStart(simSeq, cfg):
    """학습 시작 시 simulation_runs에 레코드 삽입 (status=running)"""
    try:
        conn   = getDbConn(cfg['db'])
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO simulation_runs (
                sim_seq, status, run_timestamp,
                model_name, dataset_repo, output_dir, tensorboard_dir,
                num_epochs, learning_rate, lora_r, lora_alpha, lora_dropout,
                max_seq_length, grad_accum, batch_size, logging_steps, save_steps
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                simSeq, 'running', datetime.now(),
                cfg['base_model'], cfg['dataset_repo'],
                cfg['output_dir'], cfg['tensorboard_dir'],
                cfg['num_epochs'],       cfg['learning_rate'],
                cfg['lora_r'],           cfg['lora_alpha'],    cfg['lora_dropout'],
                cfg['max_seq_length'],   cfg['grad_accum'],
                cfg['per_device_train_batch_size'],
                cfg['logging_steps'],    cfg['save_steps'],
            )
        )
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[DB] 시뮬레이션 #{simSeq} 시작 기록 완료")
    except Exception as e:
        print({"success": False, "message": f"insertRunStart 실패: {e}"})


def updateRunEnd(simSeq, trainResult, dbCfg, status='completed', errorMsg=None):
    """학습 완료/실패 시 simulation_runs 레코드 업데이트"""
    try:
        conn   = getDbConn(dbCfg)
        cursor = conn.cursor()

        if trainResult is not None:
            metrics = trainResult.metrics
            cursor.execute(
                """
                UPDATE simulation_runs SET
                    status=%s, end_timestamp=%s,
                    train_loss=%s, runtime_sec=%s, samples_per_sec=%s, global_step=%s
                WHERE sim_seq=%s
                """,
                (
                    status, datetime.now(),
                    metrics.get('train_loss'),
                    metrics.get('train_runtime'),
                    metrics.get('train_samples_per_second'),
                    trainResult.global_step,
                    simSeq,
                )
            )
        else:
            cursor.execute(
                "UPDATE simulation_runs SET status=%s, end_timestamp=%s, error_message=%s WHERE sim_seq=%s",
                (status, datetime.now(), errorMsg, simSeq)
            )

        conn.commit()
        cursor.close()
        conn.close()
        print(f"[DB] 시뮬레이션 #{simSeq} 종료 기록 완료 (status={status})")
    except Exception as e:
        print({"success": False, "message": f"updateRunEnd 실패: {e}"})


# ============================================================
# DB 스텝 로그 콜백
# ============================================================

class DbLogCallback(TrainerCallback):
    """학습 스텝별 손실을 simulation_logs 테이블에 실시간 저장"""

    def __init__(self, simSeq, dbCfg):
        """콜백 초기화"""
        self.simSeq = simSeq
        self.dbCfg  = dbCfg

    def on_log(self, args, state, control, logs=None, **kwargs):
        """로깅 이벤트 발생 시 DB에 스텝 로그 저장"""
        if logs is None or 'loss' not in logs:
            return
        try:
            conn   = getDbConn(self.dbCfg)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO simulation_logs (sim_seq, step, loss, learning_rate, epoch, log_timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    self.simSeq,
                    state.global_step,
                    logs.get('loss'),
                    logs.get('learning_rate'),
                    logs.get('epoch'),
                    datetime.now(),
                )
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"[DB] 스텝 로그 저장 실패 (step={state.global_step}): {e}")


# ============================================================
# 모델 유틸리티
# ============================================================

SYSTEM_MESSAGE = '당신은 멀티모달 도우미입니다. 텍스트 질문에는 정확히 답하고, 이미지가 주어지면 이미지를 보고 답하세요.'


def buildMessages(example):
    """샘플 데이터를 모달리티(텍스트/이미지+텍스트)에 따라 모델 메시지 구조로 변환"""
    instruction = str(example.get('instruction', '')).strip()
    userInput   = str(example.get('input', '')).strip()
    output      = str(example.get('output', '')).strip()
    modality    = str(example.get('modality', 'text_only')).strip()

    if instruction and userInput:
        userText = f"{instruction}\n\n[입력]\n{userInput}"
    else:
        userText = instruction or userInput

    if not userText:
        userText = '주어진 정보를 바탕으로 답하세요.'

    userContent = []
    if modality == 'image_text' and example.get('image') is not None:
        userContent.append({'type': 'image', 'image': example['image']})
    userContent.append({'type': 'text', 'text': userText})

    messages = [
        {'role': 'system',    'content': [{'type': 'text', 'text': SYSTEM_MESSAGE}]},
        {'role': 'user',      'content': userContent},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': output}]},
    ]
    return messages


def getCollateFn(processor, model):
    """processor와 model을 캡처한 collate_fn 클로저 반환"""
    def collateFn(examples):
        """텍스트 + 이미지 데이터를 모델 학습용 배치로 변환 및 불필요 토큰 loss 제외"""
        texts       = []
        images      = []
        hasAnyImage = False

        for i in range(0, len(examples)):
            example  = examples[i]
            messages = buildMessages(example)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

            exampleImages = []
            userContent   = messages[1]["content"]
            for j in range(0, len(userContent)):
                if userContent[j]["type"] == "image":
                    exampleImages.append(userContent[j]["image"])

            if len(exampleImages) > 0:
                hasAnyImage = True
                images.append(exampleImages)
            else:
                images.append([])

        if hasAnyImage:
            batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        else:
            batch = processor(text=texts, return_tensors="pt", padding=True)

        labels     = batch["input_ids"].clone()
        padTokenId = processor.tokenizer.pad_token_id
        if padTokenId is not None:
            labels[labels == padTokenId] = -100

        imageTokenId = getattr(model.config, "image_token_id", None)
        if imageTokenId is not None:
            labels[labels == imageTokenId] = -100

        labels[labels == 262144] = -100
        batch["labels"] = labels
        return batch

    return collateFn


def plotTrainingLoss(trainer, outputDir):
    """학습 손실 추이를 시각화하고 PNG 이미지로 저장"""
    rawHistory = trainer.state.log_history
    logHistory = []
    for i in range(0, len(rawHistory)):
        if 'loss' in rawHistory[i]:
            logHistory.append(rawHistory[i])

    if len(logHistory) == 0:
        print("학습 로그가 없어 그래프를 그릴 수 없습니다.")
        return

    steps  = []
    losses = []
    for i in range(0, len(logHistory)):
        steps.append(logHistory[i]['step'])
        losses.append(logHistory[i]['loss'])

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, marker='o', linewidth=2, markersize=4, color='steelblue')
    plt.title('학습 손실 추이 (Training Loss)', fontsize=14)
    plt.xlabel('스텝 (Step)', fontsize=12)
    plt.ylabel('손실 (Loss)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(outputDir, exist_ok=True)
    savePath = os.path.join(outputDir, 'training_loss.png')
    plt.savefig(savePath, dpi=150)
    plt.close()
    print(f"그래프 저장 완료: {savePath}")


# ============================================================
# 메인
# ============================================================

def main():
    """Gemma3 멀티모달 QLoRA 파인튜닝 메인 실행 함수"""

    setKoreanFont()
    cfg = loadConfig()

    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
        print('GPU =', torch.cuda.get_device_name(0))
    print('torch =', torch.__version__)

    # ===== sim_seq 발급 및 DB 시작 기록 =====
    simSeq = getNextSimSeq(cfg['db'])
    print(f"\n====== 시뮬레이션 #{simSeq} 시작 ======")
    print(f"데이터셋     : {cfg['dataset_repo']}")
    print(f"베이스 모델  : {cfg['base_model']}")
    print(f"출력 경로    : {cfg['output_dir']}")
    print(f"TensorBoard  : {cfg['tensorboard_dir']}")
    print(f"에폭         : {cfg['num_epochs']}  |  LR: {cfg['learning_rate']}  |  LoRA r/α: {cfg['lora_r']}/{cfg['lora_alpha']}\n")

    insertRunStart(simSeq, cfg)

    # ===== HuggingFace 로그인 =====
    huggingface_hub.login(cfg['hf_token'])

    # ===== 데이터셋 로드 =====
    ds       = load_dataset(cfg['dataset_repo'])
    train_ds = ds['train']
    try:
        eval_ds = ds['test']
    except KeyError:
        eval_ds = ds['validation']

    if cfg['max_train_samples']:
        train_ds = train_ds.select(range(min(cfg['max_train_samples'], len(train_ds))))
    if cfg['max_eval_samples']:
        eval_ds = eval_ds.select(range(min(cfg['max_eval_samples'], len(eval_ds))))

    print(f"학습 데이터: {len(train_ds)}건  |  평가 데이터: {len(eval_ds)}건")

    # ===== 양자화 설정 =====
    if torch.cuda.get_device_capability()[0] >= 8:
        torchDtype = torch.bfloat16
    else:
        torchDtype = torch.float16

    quantConfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torchDtype,
        bnb_4bit_use_double_quant=False
    )

    # ===== 모델 / 프로세서 로드 =====
    model = AutoModelForImageTextToText.from_pretrained(
        cfg['base_model'],
        quantization_config=quantConfig,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',
        trust_remote_code=True,
        token=cfg['hf_token']
    )
    processor = AutoProcessor.from_pretrained(
        cfg['base_model'], trust_remote_code=True, token=cfg['hf_token']
    )
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    print('pad_token_id =', tokenizer.pad_token_id)

    # ===== LoRA 설정 =====
    peftParams = LoraConfig(
        r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'],
        bias='none',
        task_type='CAUSAL_LM',
        target_modules='all-linear',
    )

    # ===== SFT 설정 =====
    sftArgs = SFTConfig(
        output_dir=cfg['output_dir'],
        logging_dir=cfg['tensorboard_dir'],
        num_train_epochs=cfg['num_epochs'],
        per_device_train_batch_size=cfg['per_device_train_batch_size'],
        per_device_eval_batch_size=cfg['per_device_eval_batch_size'],
        gradient_accumulation_steps=cfg['grad_accum'],
        gradient_checkpointing=True,
        max_seq_length=cfg['max_seq_length'],
        packing=False,
        learning_rate=cfg['learning_rate'],
        weight_decay=0.001,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        logging_steps=cfg['logging_steps'],
        report_to='tensorboard',
        eval_strategy='no',
        save_strategy='steps',
        save_steps=cfg['save_steps'],
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # ===== Trainer (DB 콜백 포함) =====
    collateFn  = getCollateFn(processor, model)
    dbCallback = DbLogCallback(simSeq, cfg['db'])

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=None,
        peft_config=peftParams,
        processing_class=processor,
        data_collator=collateFn,
        args=sftArgs,
        callbacks=[dbCallback],
    )

    print(f"TensorBoard: tensorboard --logdir {cfg['tensorboard_dir']}\n")

    # ===== 학습 실행 =====
    startTime   = datetime.now()
    trainResult = None
    try:
        trainResult = trainer.train()
        endTime = datetime.now()
        elapsed = (endTime - startTime).seconds
        print(f"\n학습 완료 — {elapsed // 60}분 {elapsed % 60}초")

        updateRunEnd(simSeq, trainResult, cfg['db'], status='completed')

    except Exception as e:
        updateRunEnd(simSeq, None, cfg['db'], status='failed', errorMsg=str(e))
        raise

    # ===== 손실 시각화 =====
    plotTrainingLoss(trainer, cfg['output_dir'])

    # ===== 모델 저장 =====
    trainer.model.save_pretrained(cfg['output_dir'])
    processor.save_pretrained(cfg['output_dir'])
    print(f"\n어댑터 저장 완료 : {cfg['output_dir']}")
    print(f"TensorBoard     : tensorboard --logdir {cfg['tensorboard_dir']}")
    print(f"시뮬레이션 #{simSeq} 완료")


if __name__ == '__main__':
    main()
