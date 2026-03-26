#!/bin/bash
# =============================================================
# full_training_v13 - 仅 Phase B 重启脚本
#
# 用途：Phase A 已完成时，清空 Phase B 旧数据后重新启动 Phase B。
# 区分：与 train_full_v12 / train_phase_b_v3 等脚本区分，v13 对应 full_training_v13。
# =============================================================

set -e

# 项目根目录（脚本在 scripts/ 下）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- 配置 ----------
GPU_ID=0
EXPERIMENT_NAME="full_training_v13"
RUN_BASE="${PROJECT_ROOT}/results/MyEnv/MyEnv/${EXPERIMENT_NAME}"
PHASE_B_DIR="${RUN_BASE}/run1/phase_b_ppo"
PHASE_A_BEST_MODEL="${RUN_BASE}/run1/phase_a_astar/best_model"
LOG_FILE="${PROJECT_ROOT}/full_training_v13_phaseb.log"

# ---------- 停止旧进程（避免多个训练堆叠占用 GPU/写乱日志） ----------
# 训练进程会通过 setproctitle 改名为 mappo-MyEnv-<experiment_name>
echo "[停止] 尝试停止旧训练进程: mappo-MyEnv-${EXPERIMENT_NAME}"
pkill -f "mappo-MyEnv-${EXPERIMENT_NAME}" 2>/dev/null || true
pkill -f "train/train.py" 2>/dev/null || true
sleep 2

# ---------- 清理旧数据，防止堆积 ----------
# 1) 删除所有 run*/phase_b_ppo
# 2) 删除 run2、run3 等多余 run 目录，使下次训练使用 run1
if [ -d "$RUN_BASE" ]; then
    echo "[清理] 实验目录: $RUN_BASE"
    for run_dir in "$RUN_BASE"/run*; do
        [ -d "$run_dir" ] || continue
        if [ -d "$run_dir/phase_b_ppo" ]; then
            echo "[清理] 删除 Phase B: $run_dir/phase_b_ppo"
            rm -rf "$run_dir/phase_b_ppo"
        fi
        # 只保留 run1（Phase A 结果），删除 run2、run3 等
        run_name="$(basename "$run_dir")"
        if [ "$run_name" != "run1" ]; then
            echo "[清理] 删除多余 run 目录: $run_dir"
            rm -rf "$run_dir"
        fi
    done
    echo "[清理] 完成."
else
    echo "[清理] 实验目录不存在，跳过: $RUN_BASE"
fi
# 3) 清空旧日志，避免堆积
if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
    echo "[清理] 清空旧日志: $LOG_FILE"
    : > "$LOG_FILE"
fi
echo ""

# ---------- 检查 Phase A 最佳模型 ----------
if [ ! -f "$PHASE_A_BEST_MODEL/actor.pt" ]; then
    echo "错误: Phase A 最佳模型不存在: $PHASE_A_BEST_MODEL/actor.pt"
    exit 1
fi

echo "══════════════════════════════════════════════════════════════════════"
echo "              full_training_v13 - Phase B 重启"
echo "══════════════════════════════════════════════════════════════════════"
echo "  实验名称: $EXPERIMENT_NAME"
echo "  预训练: $PHASE_A_BEST_MODEL"
echo "  日志: $LOG_FILE"
echo "  GPU: $GPU_ID"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=$GPU_ID

nohup python -u train/train.py \
    --env_name "MyEnv" \
    --experiment_name "$EXPERIMENT_NAME" \
    --training_mode "phase_b_only" \
    --model_dir "$PHASE_A_BEST_MODEL" \
    --use_random_map \
    --map_size_min 12 \
    --map_size_max 20 \
    --obstacle_density_min 0.15 \
    --obstacle_density_max 0.25 \
    --phase_b_episodes 800 \
    --phase_b_eval_interval 50 \
    --phase_b_eval_scenarios 20 \
    --n_rollout_threads 4 \
    --episode_length 100 \
    --num_env_steps 1000000 \
    --hidden_size 256 \
    --layer_N 4 \
    --lr 3e-4 \
    --critic_lr 3e-4 \
    --cuda \
    > "$LOG_FILE" 2>&1 &

echo "Phase B 已启动，PID: $!"
echo "查看日志: tail -f $LOG_FILE"
