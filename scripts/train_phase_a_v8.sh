#!/bin/bash
# Phase A v8 训练脚本 - 观测增强 + 经验回放
# 
# 使用方法:
#   chmod +x scripts/train_phase_a_v8.sh
#   ./scripts/train_phase_a_v8.sh
#
# 或指定 GPU:
#   CUDA_VISIBLE_DEVICES=2 ./scripts/train_phase_a_v8.sh

# 默认参数
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"astar_bc_phase_a_v8"}
PHASE_A_EPISODES=${PHASE_A_EPISODES:-50}
EPISODE_LENGTH=${EPISODE_LENGTH:-3000}
BC_LR=${BC_LR:-0.001}
BC_UPDATES=${BC_UPDATES:-50}
REPLAY_BUFFER_SIZE=${REPLAY_BUFFER_SIZE:-100000}

# 切换到项目目录
cd "$(dirname "$0")/.."

echo "=================================================="
echo "  Phase A v8 训练 - 观测增强 + 经验回放"
echo "=================================================="
echo "  实验名称: $EXPERIMENT_NAME"
echo "  Episodes: $PHASE_A_EPISODES"
echo "  Episode 长度: $EPISODE_LENGTH"
echo "  BC 学习率: $BC_LR"
echo "  BC 更新次数/episode: $BC_UPDATES"
echo "  经验池大小: $REPLAY_BUFFER_SIZE"
echo "  观测增强: 目标方向 + 位置"
echo "=================================================="

python train/train.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --training_mode "phase_a_only" \
    --phase_a_episodes $PHASE_A_EPISODES \
    --episode_length $EPISODE_LENGTH \
    --phase_a_bc_lr $BC_LR \
    --bc_updates_per_episode $BC_UPDATES \
    --n_rollout_threads 1 \
    --obs_include_goal_direction \
    --obs_include_position \
    --use_replay_buffer \
    --replay_buffer_size $REPLAY_BUFFER_SIZE \
    --comparison_segments_per_ep 10 \
    --max_history_episodes 20 \
    --astar_step_multiplier 3.0 \
    --gif_fps 10 \
    "$@"
