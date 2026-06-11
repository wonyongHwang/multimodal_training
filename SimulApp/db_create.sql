-- ============================================================
-- Gemma3 파인튜닝 시뮬레이션 테이블 생성 스크립트
-- 실행: mysql -u hkcode -p hkcodedb < db_create.sql
-- ============================================================

-- 시뮬레이션 실행 이력 (설정값 스냅샷 + 학습 결과)
CREATE TABLE IF NOT EXISTS simulation_runs (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    sim_seq          INT NOT NULL UNIQUE     COMMENT '순차 시뮬레이션 번호',
    status           VARCHAR(20) NOT NULL DEFAULT 'running'  COMMENT 'running|completed|failed',
    run_timestamp    DATETIME    NOT NULL    COMMENT '학습 시작 시각',
    end_timestamp    DATETIME               COMMENT '학습 종료 시각',
    model_name       VARCHAR(255),
    dataset_repo     VARCHAR(255),
    output_dir       VARCHAR(512),
    tensorboard_dir  VARCHAR(512),
    num_epochs       INT,
    learning_rate    FLOAT,
    lora_r           INT,
    lora_alpha       INT,
    lora_dropout     FLOAT,
    max_seq_length   INT,
    grad_accum       INT,
    batch_size       INT,
    logging_steps    INT,
    save_steps       INT,
    train_loss       FLOAT       COMMENT '최종 학습 손실',
    runtime_sec      FLOAT       COMMENT '소요 시간(초)',
    samples_per_sec  FLOAT       COMMENT '초당 샘플 수',
    global_step      INT         COMMENT '총 학습 스텝',
    error_message    TEXT,
    INDEX idx_sim_seq (sim_seq),
    INDEX idx_status  (status)
) COMMENT='파인튜닝 시뮬레이션 실행 이력';


-- 스텝별 학습 손실 로그
CREATE TABLE IF NOT EXISTS simulation_logs (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    sim_seq       INT      NOT NULL  COMMENT '시뮬레이션 번호',
    step          INT      NOT NULL  COMMENT '학습 스텝',
    loss          FLOAT              COMMENT '스텝 손실',
    learning_rate FLOAT              COMMENT '스텝 학습률',
    epoch         FLOAT              COMMENT '에폭',
    log_timestamp DATETIME NOT NULL  COMMENT '로그 기록 시각',
    INDEX idx_sim_seq (sim_seq),
    INDEX idx_step    (sim_seq, step)
) COMMENT='파인튜닝 스텝별 손실 로그';
