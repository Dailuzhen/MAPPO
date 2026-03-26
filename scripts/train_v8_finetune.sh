#!/bin/bash
# =============================================================
# V8 微调脚本 - 基于 ep1350 模型，专注大地图训练
#
# 策略:
#   - 加载 V8 ep1350 checkpoint (独立验证最佳 73.8%)
#   - curriculum_stages=1: 无课程分阶段，14-22 均匀采样
#   - 学习率 1e-5 (防止遗忘小地图知识)
#   - 早停: 每 50ep 全尺寸评估，连续 3 次无提升则停止
#   - 目标: 22x22 从 50% 提升到 70%+，综合 ≥80%
# =============================================================

GPU_ID=0
EXPERIMENT_NAME="v8_finetune"

V8_CHECKPOINT="results/MyEnv/MyEnv/full_v8/run1/phase_b_ppo/checkpoint_ep1350/models"

echo "══════════════════════════════════════════════════════════════════════"
echo "      V8 微调: Phase B Only (基于 ep1350, 大地图专项)"
echo "══════════════════════════════════════════════════════════════════════"
echo "  GPU: $GPU_ID | 基础模型: ep1350 | 地图: 14-22 | lr: 1e-5"
echo "  早停: 每50ep验证, 耐心=3 | 预计 6-10 小时"
echo "══════════════════════════════════════════════════════════════════════"

if [ ! -f "$V8_CHECKPOINT/actor.pt" ]; then
    echo "[ERROR] V8 ep1350 模型不存在: $V8_CHECKPOINT/actor.pt"
    exit 1
fi

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$GPU_ID

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/v8_finetune_${TIMESTAMP}.log"
mkdir -p logs

echo "训练日志: $LOG_FILE"
echo "开始微调..."

stdbuf -oL python -u train/train.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --training_mode "phase_b_only" \
    --model_dir "$V8_CHECKPOINT" \
    \
    --phase_b_episodes 800 \
    --phase_b_eval_interval 50 \
    --phase_b_eval_scenarios 40 \
    \
    --episode_length 256 \
    --n_rollout_threads 256 \
    --obs_radius 3 \
    \
    --map_size_min 14 \
    --map_size_max 22 \
    --map_change_interval 5 \
    --curriculum_stages 1 \
    \
    --hidden_size 256 \
    --layer_N 4 \
    \
    --lr 1e-5 \
    --critic_lr 1e-5 \
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
    --val_interval 50 \
    --val_scenarios 160 \
    --val_patience 3 \
    \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "微调完成! 日志: $LOG_FILE"
echo ""
echo "验证命令:"
echo "  python scripts/validate_models.py \\"
echo "    --phase_b_dir results/MyEnv/MyEnv/$EXPERIMENT_NAME/run1/phase_b_ppo \\"
echo "    --checkpoints 50 100 150 200 250 300 \\"
echo "    --obs_radius 3"
echo "══════════════════════════════════════════════════════════════════════"
