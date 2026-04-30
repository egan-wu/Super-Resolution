#!/bin/bash
# runpod_run.sh — 全自動上傳 → 安裝 → 訓練 → 下載 checkpoint
#
# Usage:
#   bash runpod_run.sh <HOST> <PORT> [KEY]
#
# Example:
#   bash runpod_run.sh 157.157.221.29 33962
#   bash runpod_run.sh 157.157.221.29 33962 ~/.ssh/id_ed25519
#
# 重要：所有資料和 checkpoint 都會放在 /workspace (50GB volume)
#       Pod 重啟 / 重建時資料會保留，重新訓練不需要再下載 dataset

set -e

# ── 必要參數 ─────────────────────────────────────────
HOST="${1:?請提供 HOST，例如: bash runpod_run.sh 157.157.221.29 33962}"
PORT="${2:?請提供 PORT}"
KEY="${3:-$HOME/.ssh/id_ed25519}"

# ── 路徑設定 ─────────────────────────────────────────
LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
# 重要：REMOTE_DIR 在 /workspace（volume），不是 ~/（container disk）
REMOTE_DIR="/workspace/Super-Resolution"
SSH_OPTS="-i $KEY -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10"
SCP_OPTS="-i $KEY -P $PORT -o StrictHostKeyChecking=no"
TARGET="root@$HOST"

PHASE=3
CHECKPOINT_FILE="checkpoints/p${PHASE}_best.pth"

echo "================================================"
echo " RunPod 自動訓練腳本（Super-Resolution）"
echo " Host  : $HOST:$PORT"
echo " Key   : $KEY"
echo " Local : $LOCAL_ROOT"
echo " Remote: $REMOTE_DIR (volume!)"
echo "================================================"
echo ""

# ── 1. 確認 /workspace 是 volume ──────────────────────
echo "[1/6] 確認 /workspace volume..."
ssh $SSH_OPTS $TARGET "
    if ! mountpoint -q /workspace 2>/dev/null && [ ! -d /workspace ]; then
        echo 'ERROR: /workspace 不存在，請確認 pod 有掛載 volume'
        exit 1
    fi
    df -h /workspace || df -h /
    echo 'OK: /workspace 可用'
"

# ── 2. 建立遠端目錄結構（在 volume 上）─────────────────
echo ""
echo "[2/6] 建立遠端目錄（/workspace）..."
ssh $SSH_OPTS $TARGET "mkdir -p \
    $REMOTE_DIR/src \
    $REMOTE_DIR/data \
    $REMOTE_DIR/checkpoints \
    $REMOTE_DIR/assets"

# ── 3. 上傳程式碼（不傳 dataset / checkpoint，那些在 volume 上自動產生）─
echo ""
echo "[3/6] 上傳程式碼..."
scp $SCP_OPTS \
    "$LOCAL_ROOT/src/dataset.py" \
    "$LOCAL_ROOT/src/model.py" \
    "$LOCAL_ROOT/src/train.py" \
    "$LOCAL_ROOT/src/inference.py" \
    "$LOCAL_ROOT/src/video_inference.py" \
    "$LOCAL_ROOT/src/utils.py" \
    "$LOCAL_ROOT/src/warp.py" \
    "$TARGET:$REMOTE_DIR/src/"

scp $SCP_OPTS \
    "$LOCAL_ROOT/setup_runpod.sh" \
    "$LOCAL_ROOT/runpod_train.sh" \
    "$TARGET:$REMOTE_DIR/"

# 也傳 README + revision_history 方便在 pod 內查閱
scp $SCP_OPTS \
    "$LOCAL_ROOT/README.md" \
    "$LOCAL_ROOT/revision_history.md" \
    "$TARGET:$REMOTE_DIR/" 2>/dev/null || true

echo "   上傳完成"

# ── 4. 安裝環境 ──────────────────────────────────────
echo ""
echo "[4/6] 安裝環境（首次約 2-3 分鐘，第二次幾乎瞬間）..."
ssh $SSH_OPTS $TARGET "cd $REMOTE_DIR && bash setup_runpod.sh"

# ── 5. 訓練（最耗時，輸出即時顯示）──────────────────
echo ""
echo "[5/6] 開始訓練（A100 約 4-8 小時，輸出即時顯示）..."
echo "      首次執行會自動下載 DIV2K (~770MB) + Flickr2K (~3.2GB) 到 /workspace/data"
echo "      之後重跑就會直接使用快取的 dataset"
echo ""
# -t 強制 pseudo-tty 以便看到 tqdm 進度條
ssh -t $SSH_OPTS $TARGET "cd $REMOTE_DIR && bash runpod_train.sh"

# ── 6. 下載結果（best checkpoint + log）───────────────
echo ""
echo "[6/6] 下載結果..."
mkdir -p "$LOCAL_ROOT/checkpoints"
mkdir -p "$LOCAL_ROOT/assets"

# 下載 best checkpoint
scp $SCP_OPTS \
    "$TARGET:$REMOTE_DIR/$CHECKPOINT_FILE" \
    "$LOCAL_ROOT/$CHECKPOINT_FILE"

# 下載最新的 log（用 ls -t 抓最新）
LATEST_LOG=$(ssh $SSH_OPTS $TARGET "ls -t $REMOTE_DIR/assets/train_p${PHASE}_*.log 2>/dev/null | head -1" || echo "")
if [ -n "$LATEST_LOG" ]; then
    scp $SCP_OPTS "$TARGET:$LATEST_LOG" "$LOCAL_ROOT/assets/" || true
fi

echo ""
echo "================================================"
echo " 完成！"
echo " Checkpoint: $LOCAL_ROOT/$CHECKPOINT_FILE"
echo " Log:        $LOCAL_ROOT/assets/"
echo ""
echo " 下一步："
echo "   1. 本地推論測試:"
echo "      python src/inference.py --phase 3 \\"
echo "          --checkpoint $CHECKPOINT_FILE \\"
echo "          --image data/samples/sample_00.jpg"
echo ""
echo "   2. ★ 記得到 RunPod 網頁 STOP 或 TERMINATE pod 省錢 ★"
echo "      （Volume 不會被刪除，下次重新建 pod 掛同一顆 volume 就能繼續）"
echo "================================================"
