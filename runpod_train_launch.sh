#!/bin/bash
# runpod_train_launch.sh — 在 pod 內以 nohup 啟動訓練（背景執行，SSH 斷線不會中斷）
# 此腳本由 ssh 從本地呼叫，會把訓練放到背景並回傳 PID

set -e
cd /workspace/Super-Resolution

LOG_FILE=/workspace/Super-Resolution/assets/train.log
PID_FILE=/workspace/Super-Resolution/assets/train.pid

# 如果已有訓練在跑，先檢查
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "ERROR: 已有訓練在跑 PID=$OLD_PID. 如要重啟請先 kill $OLD_PID"
        exit 1
    fi
fi

# Phase 3 完整訓練（給 RTX 4090 24GB 的調整版）
nohup python -u src/train.py \
    --phase 3 \
    --epochs 2000 \
    --batch-size 8 \
    --seq-len 4 \
    --lr 1e-4 \
    --val-every 100 \
    --div2k --flickr2k \
    --hidden-channels 128 \
    --pad-mode reflection \
    --augment \
    --cosine-anneal \
    --sched-sampling \
    --sched-sample-ramp 800 \
    --curriculum \
    --temp-loss-weight 0.1 \
    --tv-loss-weight 1e-5 \
    --grad-clip 1.0 \
    --save-dir /workspace/Super-Resolution/checkpoints \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "================================================"
echo " 訓練已在背景啟動"
echo " PID:      $PID"
echo " Log:      $LOG_FILE"
echo " PID file: $PID_FILE"
echo ""
echo " 監看進度:  tail -f $LOG_FILE"
echo " 檢查狀態:  ps -p $PID"
echo " 停止訓練:  kill $PID"
echo "================================================"
