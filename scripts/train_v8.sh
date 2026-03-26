#!/bin/bash
# =============================================================
# V8 训练脚本 - Phase A (BC) + Phase B (PPO)
#
# V8 改进 (相比 V7):
#   - obs_radius=3 (7x7视野, 更大感知范围)
#   - 均衡课程采样 (分层桶采样, 消除分布偏移)
#   - 奖励优化 (归一化距离惩罚, 增强抗循环, 步数惩罚)
#   - 40场景评估 + 按尺寸分组统计
#   - 全局最佳模型自动追踪保存
#   - 256并行环境 + ppo_epoch=5 (加速训练)
# =============================================================

GPU_ID=0
EXPERIMENT_NAME="full_v8"

echo "══════════════════════════════════════════════════════════════════════"
echo "      V8: Phase A (BC 200ep) + Phase B (PPO 2000ep)"
echo "══════════════════════════════════════════════════════════════════════"
echo "  GPU: $GPU_ID | 并行: 256 | obs_radius: 3 | ppo_epoch: 5"
echo "══════════════════════════════════════════════════════════════════════"

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$GPU_ID

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/full_v8_${TIMESTAMP}.log"
mkdir -p logs

echo "训练日志: $LOG_FILE"
echo "开始训练..."

stdbuf -oL python -u train/train.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --training_mode "two_phase" \
    \
    --phase_a_episodes 200 \
    --phase_a_checkpoint_interval 10 \
    \
    --phase_b_episodes 2000 \
    --phase_b_eval_interval 50 \
    --phase_b_eval_scenarios 40 \
    \
    --episode_length 256 \
    --n_rollout_threads 256 \
    --obs_radius 3 \
    \
    --map_size_min 8 \
    --map_size_max 20 \
    --map_change_interval 5 \
    \
    --hidden_size 256 \
    --layer_N 4 \
    \
    --lr 5e-5 \
    --critic_lr 5e-5 \
    --ppo_epoch 5 \
    --num_mini_batch 4 \
    --clip_param 0.2 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --entropy_coef 0.05 \
    --max_grad_norm 10.0 \
    \
    --log_interval 10 \
    \
    --obs_include_goal_direction \
    --obs_include_position \
    \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "训练完成! 日志: $LOG_FILE"
