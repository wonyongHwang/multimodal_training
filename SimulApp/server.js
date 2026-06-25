/**
 * server.js — Gemma3 파인튜닝 시뮬레이션 관리 Express 서버
 *
 * 실행: node server.js
 *   - codeset/.env 읽기/쓰기 (하이퍼파라미터 관리)
 *   - Python 파인튜닝 스크립트 spawn (detached — Node 재시작 후에도 학습 유지)
 *   - MySQL DB 조회 (시뮬레이션 이력·스텝 로그)
 *   - codeset/logs/ 에 로그 파일 저장 (서버 재시작 후 로그 복구)
 *
 * 의존성 설치: npm install
 */

'use strict';

const express = require('express');
const path    = require('path');
const fs      = require('fs');
const { spawn } = require('child_process');
const mysql2  = require('mysql2/promise');
require('dotenv').config({ path: path.join(__dirname, '..', 'codeset', '.env') });

// ============================================================
// 설정
// ============================================================

const PORT          = 3000;
const CODESET_DIR   = path.join(__dirname, '..', 'codeset');
const ENV_PATH      = path.join(CODESET_DIR, '.env');
const PY_SCRIPT     = path.join(CODESET_DIR, '★02. 파인튜닝_멀티모달데이터_Gemma3.py');
const PY_MERGE      = path.join(CODESET_DIR, '★03. 데이터병합 및 저장_멀티모달데이터_Gemma3.py');
const PY_GGUF       = path.join(CODESET_DIR, '★04. GGUF 모델 변환.py');
const MODELS_DIR    = path.join(CODESET_DIR, 'models');

// 로그 / PID 파일 저장 경로
const LOGS_DIR       = path.join(CODESET_DIR, 'logs');
const TRAIN_LOG_FILE = path.join(LOGS_DIR, 'train.log');
const MERGE_LOG_FILE = path.join(LOGS_DIR, 'merge.log');
const GGUF_LOG_FILE  = path.join(LOGS_DIR, 'gguf.log');
const TRAIN_PID_FILE = path.join(LOGS_DIR, 'train.pid');
const MERGE_PID_FILE = path.join(LOGS_DIR, 'merge.pid');
const GGUF_PID_FILE  = path.join(LOGS_DIR, 'gguf.pid');

// logs/ 디렉터리 생성
if (!fs.existsSync(LOGS_DIR)) {
  fs.mkdirSync(LOGS_DIR, { recursive: true });
}

// 파인튜닝·병합 전용 venv (.venvg3) 탐색
const VENVG3_PY_WIN = path.join(__dirname, '..', '.venvg3', 'Scripts', 'python.exe');
const VENVG3_PY_NIX = path.join(__dirname, '..', '.venvg3', 'bin', 'python');
let PYTHON_CMD = 'python';
if (fs.existsSync(VENVG3_PY_WIN))      { PYTHON_CMD = VENVG3_PY_WIN; }
else if (fs.existsSync(VENVG3_PY_NIX)) { PYTHON_CMD = VENVG3_PY_NIX; }
console.log(`파인튜닝/병합 Python 경로: ${PYTHON_CMD}`);

// GGUF 전용 venv (.venvgguf) 탐색
const VENVGGUF_PY_WIN = path.join(__dirname, '..', '.venvgguf', 'Scripts', 'python.exe');
const VENVGGUF_PY_NIX = path.join(__dirname, '..', '.venvgguf', 'bin', 'python');
let PYTHON_GGUF_CMD = 'python';
if (fs.existsSync(VENVGGUF_PY_WIN))      { PYTHON_GGUF_CMD = VENVGGUF_PY_WIN; }
else if (fs.existsSync(VENVGGUF_PY_NIX)) { PYTHON_GGUF_CMD = VENVGGUF_PY_NIX; }
console.log(`GGUF Python 경로: ${PYTHON_GGUF_CMD}`);

// Ollama 테스트 서비스 (ollama_service.py, 포트 8001)
const OLLAMA_SERVICE_SCRIPT = path.join(__dirname, 'ollama_service.py');
const OLLAMA_SERVICE_PORT   = 8001;
let   ollamaServiceProc     = null;

// ============================================================
// DB 풀
// ============================================================

const dbPool = mysql2.createPool({
  host:               process.env.DB_HOST     || 'localhost',
  port:               parseInt(process.env.DB_PORT || '3306'),
  database:           process.env.DB_NAME     || '',
  user:               process.env.DB_USER     || '',
  password:           process.env.DB_PASSWORD || '',
  waitForConnections: true,
  connectionLimit:    5,
  connectTimeout:     5000,
  acquireTimeout:     5000,
});

let dbAvailable = false;
(async () => {
  try {
    const conn = await dbPool.getConnection();
    await conn.ping();
    conn.release();
    dbAvailable = true;
    console.log(`[DB] 연결 성공: ${process.env.DB_HOST}/${process.env.DB_NAME}`);
  } catch (e) {
    dbAvailable = false;
    console.warn(`[DB] 연결 실패 — 시뮬레이션 이력 탭 비활성화: ${e.message}`);
  }
})();

// ============================================================
// .env 파싱 / 업데이트 유틸
// ============================================================

/**
 * .env 파일을 읽어 { key: value } 객체로 반환
 */
function parseEnv(filePath) {
  const result = {};
  if (!fs.existsSync(filePath)) return result;

  const lines = fs.readFileSync(filePath, 'utf8').split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line || line.startsWith('#')) continue;

    const eqIdx = line.indexOf('=');
    if (eqIdx < 0) continue;

    const key = line.slice(0, eqIdx).trim();
    const val = line.slice(eqIdx + 1).trim();
    result[key] = val;
  }
  return result;
}

/**
 * 기존 .env 파일에서 updates 객체의 키만 덮어씀
 */
function updateEnv(filePath, updates) {
  let lines = fs.existsSync(filePath)
    ? fs.readFileSync(filePath, 'utf8').split('\n')
    : [];

  const handled = {};

  for (let i = 0; i < lines.length; i++) {
    const line  = lines[i];
    const eqIdx = line.indexOf('=');
    if (eqIdx < 0) continue;

    const key = line.slice(0, eqIdx).trim();
    if (key in updates) {
      lines[i]    = `${key}=${updates[key]}`;
      handled[key] = true;
    }
  }

  const updateKeys = Object.keys(updates);
  for (let i = 0; i < updateKeys.length; i++) {
    const k = updateKeys[i];
    if (!handled[k]) {
      lines.push(`${k}=${updates[k]}`);
    }
  }

  fs.writeFileSync(filePath, lines.join('\n'), 'utf8');
}

// ============================================================
// 프로세스 유틸리티
// ============================================================

/**
 * 주어진 PID 가 현재 실행 중인지 확인 (signal 0 전송)
 */
function isPidRunning(pid) {
  if (!pid || isNaN(pid)) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return false;
  }
}

/**
 * PID 파일에서 PID 읽기. 파일 없으면 null 반환.
 */
function readPidFile(pidFile) {
  if (!fs.existsSync(pidFile)) return null;
  const raw = fs.readFileSync(pidFile, 'utf8').trim();
  const pid = parseInt(raw);
  return isNaN(pid) ? null : pid;
}

/**
 * 로그 파일 전체를 읽어 logsArray 에 추가
 */
function loadLogFile(logFile, logsArray) {
  if (!fs.existsSync(logFile)) return;
  try {
    const content = fs.readFileSync(logFile, 'utf8');
    const lines   = content.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      logsArray.push({ ts: new Date().toISOString(), msg: line });
    }
    console.log(`[복구] 로그 ${logsArray.length}줄 로드: ${path.basename(logFile)}`);
  } catch (e) {
    console.error('[복구] 로그 파일 읽기 실패:', e.message);
  }
}

/**
 * 로그 파일을 1초마다 감시하며 새 줄을 logsArray 에 추가
 * 반환값: clearInterval 에 전달할 watcher ID
 */
function startLogWatch(logFile, logsArray) {
  let readPos = fs.existsSync(logFile) ? fs.statSync(logFile).size : 0;
  let partial = '';

  const watcher = setInterval(() => {
    if (!fs.existsSync(logFile)) return;
    try {
      const stat = fs.statSync(logFile);
      if (stat.size <= readPos) return;

      const fd       = fs.openSync(logFile, 'r');
      const buf      = Buffer.alloc(stat.size - readPos);
      const bytesRead = fs.readSync(fd, buf, 0, buf.length, readPos);
      fs.closeSync(fd);

      readPos += bytesRead;
      const text  = partial + buf.slice(0, bytesRead).toString('utf8');
      const parts = text.split('\n');
      partial     = parts.pop();

      for (let i = 0; i < parts.length; i++) {
        // \r 포함 시(tqdm 진행바) 마지막 값만 사용
        const crParts = parts[i].split('\r');
        const line    = crParts[crParts.length - 1].trim();
        if (!line) continue;
        const entry = { ts: new Date().toISOString(), msg: line };
        logsArray.push(entry);
        console.log('[LOG]', line);
      }
    } catch (e) {
      console.error('[LOG WATCH]', e.message);
    }
  }, 1000);

  return watcher;
}

/**
 * Python 스크립트를 Node.js 와 분리된 독립 프로세스로 실행
 * 출력은 logFile 로 리디렉션, PID 를 pidFile 에 저장
 * 반환: Promise<pid|null>
 */
function spawnDetached(pythonCmd, scriptPath, logFile, pidFile) {
  // shell 안에서 경로 안전하게 인용
  const qPy     = `'${pythonCmd.replace(/'/g, "'\\''")}'`;
  const qScript = `'${scriptPath.replace(/'/g, "'\\''")}'`;
  const qLog    = `'${logFile.replace(/'/g, "'\\''")}'`;

  // -u : Python 출력 버퍼링 비활성화 → 파일에 즉시 한 줄씩 기록
  const shellCmd = `${qPy} -u ${qScript} >> ${qLog} 2>&1 & echo $!`;

  const proc = spawn('bash', ['-c', shellCmd], {
    cwd:      CODESET_DIR,
    detached: true,
    stdio:    ['ignore', 'pipe', 'ignore'],
  });
  proc.unref();

  let pidStr = '';
  proc.stdout.on('data', (d) => { pidStr += d.toString(); });

  return new Promise((resolve) => {
    proc.stdout.on('close', () => {
      const pid = parseInt(pidStr.trim());
      if (!isNaN(pid)) {
        fs.writeFileSync(pidFile, String(pid), 'utf8');
        resolve(pid);
      } else {
        resolve(null);
      }
    });
    setTimeout(() => {
      const pid = parseInt(pidStr.trim());
      if (!isNaN(pid)) {
        fs.writeFileSync(pidFile, String(pid), 'utf8');
        resolve(pid);
      } else {
        resolve(null);
      }
    }, 5000);
  });
}

// ============================================================
// 프로세스 상태 추적 (파인튜닝 / 병합 / GGUF 독립)
// ============================================================

let currentProcPid  = null;
let currentSimSeq   = null;
let trainLogs       = [];
let trainLogWatcher = null;

let mergeProcPid    = null;
let mergeLogs       = [];
let mergeLogWatcher = null;

let ggufProcPid     = null;
let ggufLogs        = [];
let ggufLogWatcher  = null;

// ============================================================
// 서버 시작 시 프로세스 복구
// ============================================================

/**
 * ollama_service.py 를 .venvgguf Python 으로 실행 (포트 8001)
 * fastapi/uvicorn 미설치 시 경고만 출력하고 계속 진행
 */
function startOllamaService() {
  if (!fs.existsSync(OLLAMA_SERVICE_SCRIPT)) {
    console.warn('[Ollama서비스] ollama_service.py 없음 — 스킵');
    return;
  }
  const pyCmd = fs.existsSync(VENVGGUF_PY_NIX) ? VENVGGUF_PY_NIX : 'python';
  try {
    ollamaServiceProc = spawn(pyCmd, ['-u', OLLAMA_SERVICE_SCRIPT], {
      detached: false,
      stdio:    ['ignore', 'ignore', 'pipe'],
    });
    ollamaServiceProc.stderr.on('data', (d) => {
      const msg = d.toString().trim();
      if (msg && !msg.includes('INFO') && !msg.includes('WARNING')) {
        console.warn('[Ollama서비스]', msg);
      }
    });
    ollamaServiceProc.on('error', (err) => {
      console.warn(`[Ollama서비스] 시작 실패: ${err.message}`);
      console.warn('[Ollama서비스] pip install fastapi uvicorn python-multipart 설치 필요');
      ollamaServiceProc = null;
    });
    ollamaServiceProc.on('exit', (code) => {
      if (code !== null && code !== 0) {
        console.warn(`[Ollama서비스] 종료 (code=${code}) — Ollama 테스트 탭 사용 불가`);
      }
      ollamaServiceProc = null;
    });
    console.log(`[Ollama서비스] 시작됨 (PID=${ollamaServiceProc.pid}, port=${OLLAMA_SERVICE_PORT})`);
  } catch (e) {
    console.warn(`[Ollama서비스] spawn 실패: ${e.message}`);
  }
}

function recoverProcesses() {
  // ── 파인튜닝 복구 ──
  const trainPid = readPidFile(TRAIN_PID_FILE);
  if (trainPid !== null) {
    if (isPidRunning(trainPid)) {
      currentProcPid = trainPid;
      loadLogFile(TRAIN_LOG_FILE, trainLogs);
      trainLogWatcher = startLogWatch(TRAIN_LOG_FILE, trainLogs);
      console.log(`[복구] ▶ 파인튜닝 재연결 (PID=${trainPid}), 로그 ${trainLogs.length}줄`);
    } else {
      fs.unlinkSync(TRAIN_PID_FILE);
      loadLogFile(TRAIN_LOG_FILE, trainLogs);
      console.log(`[복구] 파인튜닝 완료됨 (PID=${trainPid}), 로그 ${trainLogs.length}줄 복원`);
    }
  }

  // ── 병합 복구 ──
  const mergePid = readPidFile(MERGE_PID_FILE);
  if (mergePid !== null) {
    if (isPidRunning(mergePid)) {
      mergeProcPid = mergePid;
      loadLogFile(MERGE_LOG_FILE, mergeLogs);
      mergeLogWatcher = startLogWatch(MERGE_LOG_FILE, mergeLogs);
      console.log(`[복구] ▶ 병합 재연결 (PID=${mergePid})`);
    } else {
      fs.unlinkSync(MERGE_PID_FILE);
      loadLogFile(MERGE_LOG_FILE, mergeLogs);
      console.log(`[복구] 병합 완료됨 (PID=${mergePid}), 로그 복원`);
    }
  }

  // ── GGUF 복구 ──
  const ggufPid = readPidFile(GGUF_PID_FILE);
  if (ggufPid !== null) {
    if (isPidRunning(ggufPid)) {
      ggufProcPid = ggufPid;
      loadLogFile(GGUF_LOG_FILE, ggufLogs);
      ggufLogWatcher = startLogWatch(GGUF_LOG_FILE, ggufLogs);
      console.log(`[복구] ▶ GGUF 재연결 (PID=${ggufPid})`);
    } else {
      fs.unlinkSync(GGUF_PID_FILE);
      loadLogFile(GGUF_LOG_FILE, ggufLogs);
      console.log(`[복구] GGUF 완료됨 (PID=${ggufPid}), 로그 복원`);
    }
  }
}

recoverProcesses();

// ============================================================
// Express 앱
// ============================================================

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ────────────────────────────────────────────────
// GET /api/ollama/status
// ────────────────────────────────────────────────

app.get('/api/ollama/status', (req, res) => {
  const running = ollamaServiceProc !== null && ollamaServiceProc.exitCode === null;
  res.json({ success: true, running });
});

// ────────────────────────────────────────────────
// GET /api/db/status
// ────────────────────────────────────────────────

app.get('/api/db/status', async (req, res) => {
  try {
    const conn = await dbPool.getConnection();
    await conn.ping();
    conn.release();
    dbAvailable = true;
    res.json({ success: true, connected: true, message: 'DB 연결 정상' });
  } catch (e) {
    dbAvailable = false;
    res.json({
      success:   true,
      connected: false,
      message:   `DB 연결 불가: ${e.message}`,
      hint:      `DB_HOST=${process.env.DB_HOST}, DB_NAME=${process.env.DB_NAME}`,
    });
  }
});

// ────────────────────────────────────────────────
// GET /api/config
// ────────────────────────────────────────────────

app.get('/api/config', (req, res) => {
  try {
    const envVals = parseEnv(ENV_PATH);
    const cfg = {
      dataset_repo:                 envVals['DATASET_REPO']                || '',
      base_model:                   envVals['BASE_MODEL']                  || '',
      output_base_dir:              envVals['OUTPUT_BASE_DIR']             || '',
      max_train_samples:            envVals['MAX_TRAIN_SAMPLES']           || '',
      max_eval_samples:             envVals['MAX_EVAL_SAMPLES']            || '',
      max_seq_length:               envVals['MAX_SEQ_LENGTH']             || '768',
      per_device_train_batch_size:  envVals['PER_DEVICE_TRAIN_BATCH_SIZE'] || '1',
      per_device_eval_batch_size:   envVals['PER_DEVICE_EVAL_BATCH_SIZE']  || '1',
      grad_accum:                   envVals['GRAD_ACCUM']                  || '4',
      num_epochs:                   envVals['NUM_EPOCHS']                  || '5',
      learning_rate:                envVals['LEARNING_RATE']               || '0.0002',
      logging_steps:                envVals['LOGGING_STEPS']               || '10',
      save_steps:                   envVals['SAVE_STEPS']                  || '100',
      lora_r:                       envVals['LORA_R']                      || '16',
      lora_alpha:                   envVals['LORA_ALPHA']                  || '32',
      lora_dropout:                 envVals['LORA_DROPOUT']                || '0.05',
    };
    res.json({ success: true, data: cfg });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// POST /api/config
// ────────────────────────────────────────────────

app.post('/api/config', (req, res) => {
  try {
    const body   = req.body || {};
    const keyMap = {
      dataset_repo:                'DATASET_REPO',
      base_model:                  'BASE_MODEL',
      output_base_dir:             'OUTPUT_BASE_DIR',
      max_train_samples:           'MAX_TRAIN_SAMPLES',
      max_eval_samples:            'MAX_EVAL_SAMPLES',
      max_seq_length:              'MAX_SEQ_LENGTH',
      per_device_train_batch_size: 'PER_DEVICE_TRAIN_BATCH_SIZE',
      per_device_eval_batch_size:  'PER_DEVICE_EVAL_BATCH_SIZE',
      grad_accum:                  'GRAD_ACCUM',
      num_epochs:                  'NUM_EPOCHS',
      learning_rate:               'LEARNING_RATE',
      logging_steps:               'LOGGING_STEPS',
      save_steps:                  'SAVE_STEPS',
      lora_r:                      'LORA_R',
      lora_alpha:                  'LORA_ALPHA',
      lora_dropout:                'LORA_DROPOUT',
    };

    const updates  = {};
    const bodyKeys = Object.keys(body);
    for (let i = 0; i < bodyKeys.length; i++) {
      const k = bodyKeys[i];
      if (keyMap[k] !== undefined) {
        updates[keyMap[k]] = body[k];
      }
    }

    updateEnv(ENV_PATH, updates);
    res.json({ success: true, message: '.env 저장 완료' });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// POST /api/run/start
// ────────────────────────────────────────────────

app.post('/api/run/start', async (req, res) => {
  if (currentProcPid !== null && isPidRunning(currentProcPid)) {
    return res.status(409).json({ success: false, message: '이미 학습이 실행 중입니다.' });
  }

  // 이전 상태 초기화
  currentProcPid = null;
  currentSimSeq  = null;
  trainLogs      = [];
  if (trainLogWatcher) { clearInterval(trainLogWatcher); trainLogWatcher = null; }
  fs.writeFileSync(TRAIN_LOG_FILE, '', 'utf8');

  try {
    const pid = await spawnDetached(PYTHON_CMD, PY_SCRIPT, TRAIN_LOG_FILE, TRAIN_PID_FILE);
    if (!pid) {
      return res.status(500).json({ success: false, message: '프로세스 시작 실패: PID를 가져올 수 없습니다.' });
    }
    currentProcPid  = pid;
    trainLogWatcher = startLogWatch(TRAIN_LOG_FILE, trainLogs);
    console.log(`[TRAIN] 학습 시작 (PID=${pid})`);
    res.json({ success: true, message: '학습 시작됨' });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// POST /api/run/stop
// ────────────────────────────────────────────────

app.post('/api/run/stop', async (req, res) => {
  if (currentProcPid === null || !isPidRunning(currentProcPid)) {
    return res.status(409).json({ success: false, message: '실행 중인 학습이 없습니다.' });
  }
  try {
    process.kill(currentProcPid, 'SIGTERM');
    const stoppedPid = currentProcPid;
    currentProcPid   = null;
    if (trainLogWatcher) { clearInterval(trainLogWatcher); trainLogWatcher = null; }
    if (fs.existsSync(TRAIN_PID_FILE)) fs.unlinkSync(TRAIN_PID_FILE);

    if (dbAvailable) {
      try {
        await dbPool.query(
          "UPDATE simulation_runs SET status='stopped', end_timestamp=NOW() WHERE status='running'"
        );
      } catch (dbErr) {
        console.warn('[STOP] DB 상태 업데이트 실패:', dbErr.message);
      }
    }

    console.log(`[TRAIN] 학습 중지 (PID=${stoppedPid})`);
    res.json({ success: true, message: '학습이 중지되었습니다.' });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// GET /api/run/status
// ────────────────────────────────────────────────

app.get('/api/run/status', (req, res) => {
  const offset  = parseInt(req.query.offset || '0');
  const running = currentProcPid !== null && isPidRunning(currentProcPid);

  // 프로세스가 끝났으면 정리
  if (!running && currentProcPid !== null) {
    console.log(`[TRAIN] 프로세스 종료 감지 (PID=${currentProcPid})`);
    currentProcPid = null;
    if (trainLogWatcher) { clearInterval(trainLogWatcher); trainLogWatcher = null; }
    if (fs.existsSync(TRAIN_PID_FILE)) fs.unlinkSync(TRAIN_PID_FILE);
  }

  res.json({
    success:  true,
    running,
    simSeq:   currentSimSeq,
    logCount: trainLogs.length,
    logs:     trainLogs.slice(offset),
  });
});

// ────────────────────────────────────────────────
// GET /api/simulations
// ────────────────────────────────────────────────

app.get('/api/simulations', async (req, res) => {
  if (!dbAvailable) {
    return res.status(503).json({
      success: false,
      dbError: true,
      message: `DB 연결 불가 (${process.env.DB_HOST}). VPN/네트워크 확인 후 서버를 재시작하거나 [DB 재연결] 버튼을 클릭하세요.`,
    });
  }
  try {
    const [rows] = await dbPool.query(
      `SELECT sim_seq, status, run_timestamp, end_timestamp,
              model_name, dataset_repo, num_epochs, learning_rate,
              lora_r, lora_alpha, lora_dropout, max_seq_length,
              grad_accum, batch_size, train_loss, runtime_sec,
              global_step, error_message
       FROM simulation_runs
       ORDER BY sim_seq DESC
       LIMIT 50`
    );
    res.json({ success: true, data: rows });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// GET /api/simulations/:seq
// ────────────────────────────────────────────────

app.get('/api/simulations/:seq', async (req, res) => {
  try {
    const [rows] = await dbPool.query(
      'SELECT * FROM simulation_runs WHERE sim_seq = ?',
      [req.params.seq]
    );
    if (rows.length === 0) {
      return res.status(404).json({ success: false, message: '시뮬레이션 없음' });
    }
    res.json({ success: true, data: rows[0] });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// GET /api/simulations/:seq/logs
// ────────────────────────────────────────────────

app.get('/api/simulations/:seq/logs', async (req, res) => {
  try {
    const [rows] = await dbPool.query(
      `SELECT step, loss, learning_rate, epoch, log_timestamp
       FROM simulation_logs
       WHERE sim_seq = ?
       ORDER BY step ASC`,
      [req.params.seq]
    );
    res.json({ success: true, data: rows });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// GET /api/models
// ────────────────────────────────────────────────

app.get('/api/models', (req, res) => {
  try {
    if (!fs.existsSync(MODELS_DIR)) {
      return res.json({ success: true, data: [] });
    }
    const entries = fs.readdirSync(MODELS_DIR, { withFileTypes: true });
    const dirs    = [];
    for (let i = 0; i < entries.length; i++) {
      if (entries[i].isDirectory()) {
        dirs.push(entries[i].name);
      }
    }
    res.json({ success: true, data: dirs });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// GET /api/merge/config
// ────────────────────────────────────────────────

app.get('/api/merge/config', (req, res) => {
  try {
    const envVals = parseEnv(ENV_PATH);
    const cfg = {
      base_model:        envVals['BASE_MODEL']        || 'google/gemma-3-4b-it',
      adapter_path:      envVals['ADAPTER_PATH']      || '',
      merged_local_dir:  envVals['MERGED_LOCAL_DIR']  || './models/gemma3_multimodal_merged',
      merged_model_repo: envVals['MERGED_MODEL_REPO'] || '',
      test_image_path:   envVals['TEST_IMAGE_PATH']   || '',
    };
    res.json({ success: true, data: cfg });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// POST /api/merge/config
// ────────────────────────────────────────────────

app.post('/api/merge/config', (req, res) => {
  try {
    const body   = req.body || {};
    const keyMap = {
      base_model:        'BASE_MODEL',
      adapter_path:      'ADAPTER_PATH',
      merged_local_dir:  'MERGED_LOCAL_DIR',
      merged_model_repo: 'MERGED_MODEL_REPO',
      test_image_path:   'TEST_IMAGE_PATH',
    };
    const updates  = {};
    const bodyKeys = Object.keys(body);
    for (let i = 0; i < bodyKeys.length; i++) {
      const k = bodyKeys[i];
      if (keyMap[k] !== undefined) {
        updates[keyMap[k]] = body[k];
      }
    }
    updateEnv(ENV_PATH, updates);
    res.json({ success: true, message: '.env 병합 설정 저장 완료' });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// POST /api/merge/start
// ────────────────────────────────────────────────

app.post('/api/merge/start', async (req, res) => {
  if (mergeProcPid !== null && isPidRunning(mergeProcPid)) {
    return res.status(409).json({ success: false, message: '이미 병합이 실행 중입니다.' });
  }
  if (currentProcPid !== null && isPidRunning(currentProcPid)) {
    return res.status(409).json({ success: false, message: '파인튜닝이 실행 중입니다. 완료 후 병합하세요.' });
  }

  mergeProcPid = null;
  mergeLogs    = [];
  if (mergeLogWatcher) { clearInterval(mergeLogWatcher); mergeLogWatcher = null; }
  fs.writeFileSync(MERGE_LOG_FILE, '', 'utf8');

  try {
    const pid = await spawnDetached(PYTHON_CMD, PY_MERGE, MERGE_LOG_FILE, MERGE_PID_FILE);
    if (!pid) {
      return res.status(500).json({ success: false, message: '병합 프로세스 시작 실패' });
    }
    mergeProcPid    = pid;
    mergeLogWatcher = startLogWatch(MERGE_LOG_FILE, mergeLogs);
    console.log(`[MERGE] 병합 시작 (PID=${pid})`);
    res.json({ success: true, message: '병합 시작됨' });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// GET /api/merge/status
// ────────────────────────────────────────────────

app.get('/api/merge/status', (req, res) => {
  const offset  = parseInt(req.query.offset || '0');
  const running = mergeProcPid !== null && isPidRunning(mergeProcPid);

  if (!running && mergeProcPid !== null) {
    console.log(`[MERGE] 프로세스 종료 감지 (PID=${mergeProcPid})`);
    mergeProcPid = null;
    if (mergeLogWatcher) { clearInterval(mergeLogWatcher); mergeLogWatcher = null; }
    if (fs.existsSync(MERGE_PID_FILE)) fs.unlinkSync(MERGE_PID_FILE);
  }

  res.json({
    success:  true,
    running,
    logCount: mergeLogs.length,
    logs:     mergeLogs.slice(offset),
  });
});

// ────────────────────────────────────────────────
// GET /api/gguf/config
// ────────────────────────────────────────────────

app.get('/api/gguf/config', (req, res) => {
  try {
    const envVals = parseEnv(ENV_PATH);
    const cfg = {
      source_repo:  envVals['GGUF_SOURCE_REPO'] || '',
      local_dir:    envVals['GGUF_LOCAL_DIR']   || '',
      outfile:      envVals['GGUF_OUTFILE']      || 'model.gguf',
      outtype:      envVals['GGUF_OUTTYPE']      || 'f16',
      hf_repo:      envVals['GGUF_HF_REPO']      || '',
      llamacpp_dir: envVals['GGUF_LLAMACPP_DIR'] || './llama.cpp',
    };
    res.json({ success: true, data: cfg });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// POST /api/gguf/config
// ────────────────────────────────────────────────

app.post('/api/gguf/config', (req, res) => {
  try {
    const body   = req.body || {};
    const keyMap = {
      source_repo:  'GGUF_SOURCE_REPO',
      local_dir:    'GGUF_LOCAL_DIR',
      outfile:      'GGUF_OUTFILE',
      outtype:      'GGUF_OUTTYPE',
      hf_repo:      'GGUF_HF_REPO',
      llamacpp_dir: 'GGUF_LLAMACPP_DIR',
    };
    const updates  = {};
    const bodyKeys = Object.keys(body);
    for (let i = 0; i < bodyKeys.length; i++) {
      const k = bodyKeys[i];
      if (keyMap[k] !== undefined) {
        updates[keyMap[k]] = body[k];
      }
    }
    updateEnv(ENV_PATH, updates);
    res.json({ success: true, message: '.env GGUF 설정 저장 완료' });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// POST /api/gguf/start
// ────────────────────────────────────────────────

app.post('/api/gguf/start', async (req, res) => {
  if (ggufProcPid !== null && isPidRunning(ggufProcPid)) {
    return res.status(409).json({ success: false, message: '이미 GGUF 변환이 실행 중입니다.' });
  }
  if (currentProcPid !== null && isPidRunning(currentProcPid)) {
    return res.status(409).json({ success: false, message: '파인튜닝이 실행 중입니다. 완료 후 진행하세요.' });
  }
  if (mergeProcPid !== null && isPidRunning(mergeProcPid)) {
    return res.status(409).json({ success: false, message: '모델 병합이 실행 중입니다. 완료 후 진행하세요.' });
  }

  ggufProcPid = null;
  ggufLogs    = [];
  if (ggufLogWatcher) { clearInterval(ggufLogWatcher); ggufLogWatcher = null; }
  fs.writeFileSync(GGUF_LOG_FILE, '', 'utf8');

  try {
    const pid = await spawnDetached(PYTHON_GGUF_CMD, PY_GGUF, GGUF_LOG_FILE, GGUF_PID_FILE);
    if (!pid) {
      return res.status(500).json({ success: false, message: 'GGUF 프로세스 시작 실패' });
    }
    ggufProcPid    = pid;
    ggufLogWatcher = startLogWatch(GGUF_LOG_FILE, ggufLogs);
    console.log(`[GGUF] 변환 시작 (PID=${pid})`);
    res.json({ success: true, message: 'GGUF 변환 시작됨' });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// GET /api/gguf/status
// ────────────────────────────────────────────────

app.get('/api/gguf/status', (req, res) => {
  const offset  = parseInt(req.query.offset || '0');
  const running = ggufProcPid !== null && isPidRunning(ggufProcPid);

  if (!running && ggufProcPid !== null) {
    console.log(`[GGUF] 프로세스 종료 감지 (PID=${ggufProcPid})`);
    ggufProcPid = null;
    if (ggufLogWatcher) { clearInterval(ggufLogWatcher); ggufLogWatcher = null; }
    if (fs.existsSync(GGUF_PID_FILE)) fs.unlinkSync(GGUF_PID_FILE);
  }

  res.json({
    success:  true,
    running,
    logCount: ggufLogs.length,
    logs:     ggufLogs.slice(offset),
  });
});

// ============================================================
// 서버 시작
// ============================================================

app.listen(PORT, () => {
  console.log(`SimulApp 서버 실행 중: http://localhost:${PORT}`);
  console.log(`codeset 경로: ${CODESET_DIR}`);
  console.log(`로그 저장 경로: ${LOGS_DIR}`);
  console.log(`DB: ${process.env.DB_HOST}/${process.env.DB_NAME}`);
  startOllamaService();
});

// 서버 종료 시 Ollama 서비스도 함께 종료
process.on('exit',   () => { if (ollamaServiceProc) ollamaServiceProc.kill(); });
process.on('SIGINT',  () => { if (ollamaServiceProc) ollamaServiceProc.kill(); process.exit(0); });
process.on('SIGTERM', () => { if (ollamaServiceProc) ollamaServiceProc.kill(); process.exit(0); });
