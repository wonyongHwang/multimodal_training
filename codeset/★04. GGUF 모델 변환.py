"""
★04. GGUF 모델 변환.py
HuggingFace Hub에서 병합 모델을 다운로드하고 GGUF 포맷으로 변환한 뒤 다시 Hub에 업로드합니다.

실행 방법:
    python "★04. GGUF 모델 변환.py"

설정 변경:
    .env 파일의 GGUF_* 값을 변경하거나 SimulApp 웹 화면 'GGUF 변환' 탭에서 설정 후 실행
"""

# ============================================================
# 라이브러리 임포트
# ============================================================

import os
import subprocess
import sys
from datetime import datetime

from dotenv import load_dotenv
from huggingface_hub import snapshot_download, HfApi, login


# ============================================================
# 설정 로드
# ============================================================

def loadConfig():
    """
    스크립트와 같은 디렉토리의 .env를 로드하고 GGUF 관련 설정값 딕셔너리 반환
    """
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(scriptDir, '.env'), override=True)

    ggufLocalDir = os.getenv('GGUF_LOCAL_DIR', '').strip()

    cfg = {
        'hf_token':        os.getenv('HF_TOKEN',          ''),
        'source_repo':     os.getenv('GGUF_SOURCE_REPO',  'hyokwan/modal_merge_test'),
        'local_dir':       ggufLocalDir,
        'outfile':         os.getenv('GGUF_OUTFILE',      'model.gguf'),
        'outtype':         os.getenv('GGUF_OUTTYPE',      'f16'),
        'hf_repo':         os.getenv('GGUF_HF_REPO',      ''),
        'llamacpp_dir':    os.getenv('GGUF_LLAMACPP_DIR', './llama.cpp'),
    }

    # 로컬 다운로드 경로 자동 생성 (비워두면 gguf_{모델명}_{날짜})
    if not cfg['local_dir']:
        today     = datetime.now().strftime('%Y%m%d')
        modelName = cfg['source_repo'].split('/')[-1]
        cfg['local_dir'] = f"gguf_{modelName}_{today}"

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
    else:
        print("[경고] HF_TOKEN이 설정되지 않았습니다. 비공개 모델 접근 불가.")


# ============================================================
# 모델 다운로드
# ============================================================

def downloadModel(cfg):
    """HuggingFace Hub에서 병합 모델을 로컬에 다운로드"""
    try:
        tokenVal = cfg['hf_token'] if cfg['hf_token'] not in ('YOUR_HF_TOKEN', '') else None
        print(f"모델 다운로드 중: {cfg['source_repo']} → {cfg['local_dir']}")
        snapshot_download(
            repo_id=cfg['source_repo'],
            local_dir=cfg['local_dir'],
            revision='main',
            max_workers=4,
            token=tokenVal,
        )
        print(f"다운로드 완료: {cfg['local_dir']}")
    except Exception as e:
        print({"success": False, "message": f"모델 다운로드 실패: {e}"})
        raise


# ============================================================
# GGUF 변환
# ============================================================

def convertToGguf(cfg):
    """llama.cpp convert_hf_to_gguf.py를 사용해 HF 모델을 GGUF로 변환"""
    scriptDir     = os.path.dirname(os.path.abspath(__file__))
    llamacppDir   = os.path.join(scriptDir, cfg['llamacpp_dir'].lstrip('./'))
    convertScript = os.path.join(llamacppDir, 'convert_hf_to_gguf.py')
    localDirAbs   = os.path.join(scriptDir, cfg['local_dir'])

    if not os.path.exists(convertScript):
        raise FileNotFoundError(f"convert_hf_to_gguf.py 없음: {convertScript}")
    if not os.path.exists(localDirAbs):
        raise FileNotFoundError(f"다운로드 폴더 없음: {localDirAbs}")

    outfileAbs = os.path.join(scriptDir, cfg['outfile'])

    cmd = [sys.executable, convertScript, localDirAbs, '--outfile', outfileAbs, '--outtype', cfg['outtype']]
    print(f"GGUF 변환 실행: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, cwd=scriptDir)
        print(f"GGUF 변환 완료: {cfg['outfile']}")
    except subprocess.CalledProcessError as e:
        print({"success": False, "message": f"GGUF 변환 실패 (code={e.returncode}): {e}"})
        raise


# ============================================================
# HuggingFace 업로드
# ============================================================

def uploadGguf(cfg):
    """변환된 GGUF 파일을 HuggingFace Hub에 업로드"""
    if not cfg['hf_repo']:
        raise ValueError("GGUF_HF_REPO가 .env에 설정되지 않았습니다.")

    scriptDir  = os.path.dirname(os.path.abspath(__file__))
    outfilePath = os.path.join(scriptDir, cfg['outfile'])

    if not os.path.exists(outfilePath):
        raise FileNotFoundError(f"GGUF 파일 없음: {outfilePath}")

    tokenVal = cfg['hf_token'] if cfg['hf_token'] not in ('YOUR_HF_TOKEN', '') else None

    try:
        api = HfApi()

        # 레포가 없으면 자동 생성
        try:
            api.repo_info(repo_id=cfg['hf_repo'], repo_type='model', token=tokenVal)
        except Exception:
            api.create_repo(repo_id=cfg['hf_repo'], repo_type='model', token=tokenVal)
            print(f"레포 생성 완료: {cfg['hf_repo']}")

        print(f"HF 업로드 중: {cfg['hf_repo']}")
        api.upload_file(
            path_or_fileobj=outfilePath,
            path_in_repo=cfg['outfile'],
            repo_id=cfg['hf_repo'],
            repo_type='model',
            token=tokenVal,
        )
        print(f"업로드 완료: https://huggingface.co/{cfg['hf_repo']}")
    except Exception as e:
        print({"success": False, "message": f"HF 업로드 실패: {e}"})
        raise


# ============================================================
# 메인
# ============================================================

def main():
    """HF 모델 다운로드 → GGUF 변환 → HF 업로드 전체 파이프라인 실행"""

    cfg = loadConfig()

    print("\n====== GGUF 변환 시작 ======")
    print(f"소스 HF 레포    : {cfg['source_repo']}")
    print(f"로컬 다운 경로  : {cfg['local_dir']}")
    print(f"GGUF 파일명     : {cfg['outfile']}")
    print(f"출력 타입       : {cfg['outtype']}")
    print(f"업로드 HF 레포  : {cfg['hf_repo']}")
    print(f"llama.cpp 경로  : {cfg['llamacpp_dir']}\n")

    hfLogin(cfg['hf_token'])

    # ===== 모델 다운로드 =====
    try:
        downloadModel(cfg)
    except Exception as e:
        print({"success": False, "message": f"다운로드 단계 실패: {e}"})
        return

    # ===== GGUF 변환 =====
    try:
        convertToGguf(cfg)
    except Exception as e:
        print({"success": False, "message": f"변환 단계 실패: {e}"})
        return

    # ===== HF 업로드 =====
    try:
        uploadGguf(cfg)
    except Exception as e:
        print({"success": False, "message": f"업로드 단계 실패: {e}"})
        return

    print("\n====== GGUF 변환 완료 ======")
    print(f"GGUF 파일     : {cfg['outfile']}")
    print(f"HF 레포       : https://huggingface.co/{cfg['hf_repo']}")


if __name__ == '__main__':
    main()
