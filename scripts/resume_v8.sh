#!/bin/bash
# =============================================================
# V8 断点续训脚本 - 从已有 checkpoint 恢复 Phase B 训练
# =============================================================

GPU_ID=0
EXPERIMENT_NAME="full_v8"

echo "══════════════════════════════════════════════════════════════════════"
echo "      V8 断点续训: 复用 full_v8/run1 目录"
echo "══════════════════════════════════════════════════════════════════════"
echo "  GPU: $GPU_ID | 并行: 256 | obs_radius: 3 | ppo_epoch: 5"
echo "══════════════════════════════════════════════════════════════════════"

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$GPU_ID

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/full_v8_resume_${TIMESTAMP}.log"
mkdir -p logs

echo "训练日志: $LOG_FILE"
echo "开始续训..."

stdbuf -oL python -u train/train.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --training_mode "two_phase" \
    --resume_run "run1" \
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
