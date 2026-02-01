#!/bin/bash
# Phase A 训练脚本 - A* 在线学习
# 
# 使用方法:
#   chmod +x scripts/train_phase_a.sh
#   ./scripts/train_phase_a.sh
#
# 或指定 GPU:
#   CUDA_VISIBLE_DEVICES=2 ./scripts/train_phase_a.sh

# 默认参数
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"phase_a_test"}
PHASE_A_EPISODES=${PHASE_A_EPISODES:-100}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-10}
EPISODE_LENGTH=${EPISODE_LENGTH:-1000}
BC_LR=${BC_LR:-0.001}
BC_UPDATES=${BC_UPDATES:-10}

# 切换到项目目录
cd "$(dirname "$0")/.."

echo "=================================================="
echo "  Phase A 训练 - A* 在线学习"
echo "=================================================="
echo "  实验名称: $EXPERIMENT_NAME"
echo "  Episodes: $PHASE_A_EPISODES"
echo "  Checkpoint 间隔: $CHECKPOINT_INTERVAL"
echo "  Episode 长度: $EPISODE_LENGTH"
echo "  BC 学习率: $BC_LR"
echo "  BC 更新次数/episode: $BC_UPDATES"
echo "=================================================="

python train/train.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --training_mode "phase_a_only" \
    --phase_a_episodes $PHASE_A_EPISODES \
    --phase_a_checkpoint_interval $CHECKPOINT_INTERVAL \
    --episode_length $EPISODE_LENGTH \
    --phase_a_bc_lr $BC_LR \
    --bc_updates_per_episode $BC_UPDATES \
    --n_rollout_threads 1 \
    --comparison_max_steps 100 \
    --gif_fps 10 \
    "$@"
