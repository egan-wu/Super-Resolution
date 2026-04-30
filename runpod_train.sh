#!/bin/bash
# runpod_train.sh — Phase 3 完整訓練（在 RunPod pod 內執行）
# Usage: bash runpod_train.sh
#
# 此腳本可在 SSH 進去 pod 後直接執行，也可被 runpod_run.sh 呼叫。
# 所有資料 / checkpoint 都放在 /workspace（volume，persistent）。

set -e

# ── 路徑（全部在 volume 上） ─────────────────────────
WORKSPACE="/workspace/Super-Resolution"
cd "$WORKSPACE"

echo "================================================"
echo " Super-Resolution Phase 3 完整訓練"
echo " Workspace: $WORKSPACE"
echo " Date: $(date)"
echo "================================================"

# ── 訓練設定（最完整方案） ──────────────────────────
# 設計理念：
#   - DIV2K + Flickr2K (~2750 張圖) 解決資料不足
#   - 啟用所有 Phase 3 增強：sched-sampling / curriculum / temp loss
#   - batch_size 16，配合 seq_len 4→6→8 curriculum
#   - 5000 epoch（A100/H100 上約數小時）
#   - grad-clip 1.0 防止長序列爆梯度

PHASE=3
EPOCHS=5000
BATCH_SIZE=16          # A100 40GB / H100 適用；如果 OOM 改成 8
SEQ_LEN=4              # curriculum 起始長度
LR=1e-4
VAL_EVERY=200          # 比預設 500 更頻繁，方便監看
TEMP_LOSS=0.1
SCHED_RAMP=2000
GRAD_CLIP=1.0

LOG_DIR="$WORKSPACE/assets"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_p${PHASE}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=== Training config ==="
echo "  phase            : $PHASE"
echo "  epochs           : $EPOCHS"
echo "  batch_size       : $BATCH_SIZE"
echo "  seq_len (init)   : $SEQ_LEN  (curriculum 4→6→8)"
echo "  lr               : $LR"
echo "  val_every        : $VAL_EVERY"
echo "  temp_loss_weight : $TEMP_LOSS"
echo "  sched_sample_ramp: $SCHED_RAMP"
echo "  grad_clip        : $GRAD_CLIP"
echo "  datasets         : DIV2K + Flickr2K (~2750 imgs)"
echo "  log              : $LOG_FILE"

# ── GPU 檢查 ────────────────────────────────────────
echo ""
echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# ── 開始訓練 ────────────────────────────────────────
echo ""
echo "=== Start training ==="
# 用 tee 同時輸出到 stdout 和 log file
python -u src/train.py \
    --phase $PHASE \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --lr $LR \
    --val-every $VAL_EVERY \
    --div2k --flickr2k \
    --sched-sampling \
    --sched-sample-ramp $SCHED_RAMP \
    --curriculum \
    --temp-loss-weight $TEMP_LOSS \
    --grad-clip $GRAD_CLIP \
    --save-dir "$WORKSPACE/checkpoints" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "================================================"
echo " 訓練完成！"
echo " Best checkpoint: $WORKSPACE/checkpoints/p${PHASE}_best.pth"
echo " Log: $LOG_FILE"
echo "================================================"
