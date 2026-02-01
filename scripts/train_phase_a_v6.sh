#!/bin/bash
# Phase A v6 训练脚本 - 每回合保存+验证，历史遗忘检测
# 
# 使用方法:
#   chmod +x scripts/train_phase_a_v6.sh
#   ./scripts/train_phase_a_v6.sh
#
# 或指定 GPU:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/train_phase_a_v6.sh

# 默认参数
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"astar_bc_phase_a_v6"}
PHASE_A_EPISODES=${PHASE_A_EPISODES:-100}
EPISODE_LENGTH=${EPISODE_LENGTH:-1000}
BC_LR=${BC_LR:-0.0003}
BC_UPDATES=${BC_UPDATES:-10}
COMPARISON_SEGMENTS_PER_EP=${COMPARISON_SEGMENTS_PER_EP:-10}
MAX_HISTORY_EPISODES=${MAX_HISTORY_EPISODES:-20}
ASTAR_STEP_MULTIPLIER=${ASTAR_STEP_MULTIPLIER:-3.0}

# 切换到项目目录
cd "$(dirname "$0")/.."

echo "=================================================="
echo "  Phase A v6 训练 - 每回合保存+验证"
echo "=================================================="
echo "  实验名称: $EXPERIMENT_NAME"
echo "  Episodes: $PHASE_A_EPISODES"
echo "  Episode 长度: $EPISODE_LENGTH"
echo "  BC 学习率: $BC_LR"
echo "  BC 更新次数/episode: $BC_UPDATES"
echo "  每回合对比碎片数: $COMPARISON_SEGMENTS_PER_EP"
echo "  最多回顾历史回合: $MAX_HISTORY_EPISODES"
echo "  判定倍数: ≤ ${ASTAR_STEP_MULTIPLIER}×A*步数"
echo "=================================================="

python train/train.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --training_mode "phase_a_only" \
    --phase_a_episodes $PHASE_A_EPISODES \
    --episode_length $EPISODE_LENGTH \
    --phase_a_bc_lr $BC_LR \
    --bc_updates_per_episode $BC_UPDATES \
    --comparison_segments_per_ep $COMPARISON_SEGMENTS_PER_EP \
    --max_history_episodes $MAX_HISTORY_EPISODES \
    --astar_step_multiplier $ASTAR_STEP_MULTIPLIER \
    --n_rollout_threads 1 \
    --gif_fps 10 \
    "$@"
