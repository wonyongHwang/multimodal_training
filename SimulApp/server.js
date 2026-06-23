/**
 * server.js — Gemma3 파인튜닝 시뮬레이션 관리 Express 서버
 *
 * 실행: node server.js
 *   - codeset/.env 읽기/쓰기 (하이퍼파라미터 관리)
 *   - Python 파인튜닝 스크립트 spawn
 *   - MySQL DB 조회 (시뮬레이션 이력·스텝 로그)
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
  connectTimeout:     5000,   // 5초 내 연결 안 되면 즉시 에러
  acquireTimeout:     5000,
});

// 서버 시작 시 DB 연결 상태 확인
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
    console.warn(`     DB_HOST=${process.env.DB_HOST}, DB_NAME=${process.env.DB_NAME}`);
  }
})();

// ============================================================
// .env 파싱 / 업데이트 유틸
// ============================================================

/**
 * .env 파일을 읽어 { key: value } 객체로 반환
 * 빈 줄·주석(#) 보존, 부작용 없음
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
 * 없는 키는 파일 끝에 추가, 기존 주석·빈 줄 보존
 */
function updateEnv(filePath, updates) {
  let lines = fs.existsSync(filePath)
    ? fs.readFileSync(filePath, 'utf8').split('\n')
    : [];

  const handled = {};

  for (let i = 0; i < lines.length; i++) {
    const line   = lines[i];
    const eqIdx  = line.indexOf('=');
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
// 프로세스 상태 추적 (파인튜닝 / 병합 각각 독립)
// ============================================================

let currentProc   = null;
let currentSimSeq = null;
let trainLogs     = [];

let mergeProc     = null;
let mergeLogs     = [];

let ggufProc      = null;
let ggufLogs      = [];

// ============================================================
// Express 앱
// ============================================================

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ────────────────────────────────────────────────
// GET /api/db/status — DB 연결 상태 확인
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
      success: true,
      connected: false,
      message: `DB 연결 불가: ${e.message}`,
      hint: `DB_HOST=${process.env.DB_HOST}, DB_NAME=${process.env.DB_NAME}`,
    });
  }
});

// ────────────────────────────────────────────────
// GET /api/config — .env 하이퍼파라미터 조회
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
// POST /api/config — .env 하이퍼파라미터 저장
// ────────────────────────────────────────────────

app.post('/api/config', (req, res) => {
  try {
    const body    = req.body || {};
    const keyMap  = {
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

    const updates = {};
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
// POST /api/run/start — 파인튜닝 스크립트 실행
// ────────────────────────────────────────────────

app.post('/api/run/start', (req, res) => {
  if (currentProc !== null) {
    return res.status(409).json({ success: false, message: '이미 학습이 실행 중입니다.' });
  }

  trainLogs     = [];
  currentSimSeq = null;

  try {
    currentProc = spawn(PYTHON_CMD, [PY_SCRIPT], {
      cwd:   CODESET_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    currentProc.stdout.on('data', (data) => {
      const lines = data.toString().split('\n');
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const logEntry = { ts: new Date().toISOString(), msg: line };
        trainLogs.push(logEntry);
        console.log('[PY]', line);
      }
    });

    currentProc.stderr.on('data', (data) => {
      const lines = data.toString().split('\n');
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const logEntry = { ts: new Date().toISOString(), msg: '[ERR] ' + line };
        trainLogs.push(logEntry);
      }
    });

    currentProc.on('close', (code) => {
      console.log(`[PY] 프로세스 종료 (code=${code})`);
      currentProc = null;
    });

    res.json({ success: true, message: '학습 시작됨' });
  } catch (e) {
    currentProc = null;
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// POST /api/run/stop — 학습 강제 중지
// ────────────────────────────────────────────────

app.post('/api/run/stop', async (req, res) => {
  if (currentProc === null) {
    return res.status(409).json({ success: false, message: '실행 중인 학습이 없습니다.' });
  }
  try {
    currentProc.kill('SIGTERM');
    currentProc = null;

    if (dbAvailable) {
      try {
        await dbPool.query(
          "UPDATE simulation_runs SET status='stopped', end_timestamp=NOW() WHERE status='running'"
        );
      } catch (dbErr) {
        console.warn('[STOP] DB 상태 업데이트 실패:', dbErr.message);
      }
    }

    res.json({ success: true, message: '학습이 중지되었습니다.' });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// GET /api/run/status — 현재 실행 상태·로그
// ────────────────────────────────────────────────

app.get('/api/run/status', (req, res) => {
  const offset  = parseInt(req.query.offset || '0');
  const sliced  = trainLogs.slice(offset);

  res.json({
    success:  true,
    running:  currentProc !== null,
    simSeq:   currentSimSeq,
    logCount: trainLogs.length,
    logs:     sliced,
  });
});

// ────────────────────────────────────────────────
// GET /api/simulations — 전체 시뮬레이션 이력
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
// GET /api/simulations/:seq — 특정 시뮬레이션 상세
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
// GET /api/simulations/:seq/logs — 스텝 로그
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
// GET /api/models — models/ 폴더 내 어댑터 목록
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
// GET /api/merge/config — 병합 설정 조회
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
// POST /api/merge/config — 병합 설정 저장
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
// POST /api/merge/start — 병합 스크립트 실행
// ────────────────────────────────────────────────

app.post('/api/merge/start', (req, res) => {
  if (mergeProc !== null) {
    return res.status(409).json({ success: false, message: '이미 병합이 실행 중입니다.' });
  }
  if (currentProc !== null) {
    return res.status(409).json({ success: false, message: '파인튜닝이 실행 중입니다. 완료 후 병합하세요.' });
  }

  mergeLogs = [];

  try {
    mergeProc = spawn(PYTHON_CMD, [PY_MERGE], {
      cwd:   CODESET_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    mergeProc.stdout.on('data', (data) => {
      const lines = data.toString().split('\n');
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        mergeLogs.push({ ts: new Date().toISOString(), msg: line });
        console.log('[MERGE]', line);
      }
    });

    mergeProc.stderr.on('data', (data) => {
      const lines = data.toString().split('\n');
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        mergeLogs.push({ ts: new Date().toISOString(), msg: '[ERR] ' + line });
      }
    });

    mergeProc.on('close', (code) => {
      console.log(`[MERGE] 프로세스 종료 (code=${code})`);
      mergeProc = null;
    });

    res.json({ success: true, message: '병합 시작됨' });
  } catch (e) {
    mergeProc = null;
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// GET /api/merge/status — 병합 실행 상태·로그
// ────────────────────────────────────────────────

app.get('/api/merge/status', (req, res) => {
  const offset = parseInt(req.query.offset || '0');
  res.json({
    success:  true,
    running:  mergeProc !== null,
    logCount: mergeLogs.length,
    logs:     mergeLogs.slice(offset),
  });
});

// ────────────────────────────────────────────────
// GET /api/gguf/config — GGUF 설정 조회
// ────────────────────────────────────────────────

app.get('/api/gguf/config', (req, res) => {
  try {
    const envVals = parseEnv(ENV_PATH);
    const cfg = {
      source_repo:     envVals['GGUF_SOURCE_REPO'] || '',
      local_dir:       envVals['GGUF_LOCAL_DIR']   || '',
      outfile:         envVals['GGUF_OUTFILE']      || 'model.gguf',
      outtype:         envVals['GGUF_OUTTYPE']      || 'f16',
      hf_repo:         envVals['GGUF_HF_REPO']      || '',
      llamacpp_dir:    envVals['GGUF_LLAMACPP_DIR'] || './llama.cpp',
    };
    res.json({ success: true, data: cfg });
  } catch (e) {
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// POST /api/gguf/config — GGUF 설정 저장
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
// POST /api/gguf/start — GGUF 변환 스크립트 실행
// ────────────────────────────────────────────────

app.post('/api/gguf/start', (req, res) => {
  if (ggufProc !== null) {
    return res.status(409).json({ success: false, message: '이미 GGUF 변환이 실행 중입니다.' });
  }
  if (currentProc !== null) {
    return res.status(409).json({ success: false, message: '파인튜닝이 실행 중입니다. 완료 후 진행하세요.' });
  }
  if (mergeProc !== null) {
    return res.status(409).json({ success: false, message: '모델 병합이 실행 중입니다. 완료 후 진행하세요.' });
  }

  ggufLogs = [];

  try {
    ggufProc = spawn(PYTHON_GGUF_CMD, [PY_GGUF], {
      cwd:   CODESET_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    ggufProc.stdout.on('data', (data) => {
      const lines = data.toString().split('\n');
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        ggufLogs.push({ ts: new Date().toISOString(), msg: line });
        console.log('[GGUF]', line);
      }
    });

    ggufProc.stderr.on('data', (data) => {
      const lines = data.toString().split('\n');
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        ggufLogs.push({ ts: new Date().toISOString(), msg: '[ERR] ' + line });
      }
    });

    ggufProc.on('close', (code) => {
      console.log(`[GGUF] 프로세스 종료 (code=${code})`);
      ggufProc = null;
    });

    res.json({ success: true, message: 'GGUF 변환 시작됨' });
  } catch (e) {
    ggufProc = null;
    res.status(500).json({ success: false, message: e.message });
  }
});

// ────────────────────────────────────────────────
// GET /api/gguf/status — GGUF 변환 상태·로그
// ────────────────────────────────────────────────

app.get('/api/gguf/status', (req, res) => {
  const offset = parseInt(req.query.offset || '0');
  res.json({
    success:  true,
    running:  ggufProc !== null,
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
  console.log(`DB: ${process.env.DB_HOST}/${process.env.DB_NAME}`);
});
