#!/bin/bash
# =============================================================
# V7 训练脚本 - Phase A (多地图BC) + Phase B (课程学习PPO)
#
# 相比 V6 改进:
#   - Phase A: episode_length 内部自动提升到 1000
#   - 评估: 纯数据对比（无GIF），修复卡住重置bug
#   - 每10回合评估一次，输出简洁数据表格
# =============================================================

GPU_ID=2

EXPERIMENT_NAME="full_v7"

echo "══════════════════════════════════════════════════════════════════════"
echo "      V7: Phase A (多地图BC 200ep) + Phase B (课程学习PPO 2000ep)"
echo "══════════════════════════════════════════════════════════════════════"
echo "  实验名称: $EXPERIMENT_NAME"
echo "  使用 GPU: $GPU_ID"
echo ""
echo "  Phase A (行为克隆):"
echo "    - 回合数: 200, 每回合 1000 步"
echo "    - 地图尺寸: 8-20 随机"
echo "    - 每10回合评估（纯数据，无GIF）"
echo ""
echo "  Phase B (PPO):"
echo "    - 回合数: 2000"
echo "    - 课程学习 4 阶段"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "使用GPU进行训练..."
nvidia-smi -i $GPU_ID --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "GPU 信息获取失败"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/full_v7_${TIMESTAMP}.log"
mkdir -p logs

stdbuf -oL python -u train/train.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --training_mode "two_phase" \
    \
    --phase_a_episodes 200 \
    --phase_a_checkpoint_interval 10 \
    \
    --phase_b_episodes 2000 \
    --phase_b_eval_interval 50 \
    --phase_b_eval_scenarios 20 \
    \
    --episode_length 256 \
    --n_rollout_threads 128 \
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
    --ppo_epoch 10 \
    --num_mini_batch 2 \
    --clip_param 0.2 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --entropy_coef 0.03 \
    --max_grad_norm 10.0 \
    \
    --log_interval 10 \
    \
    --obs_include_goal_direction \
    --obs_include_position \
    \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "训练日志: $LOG_FILE"
