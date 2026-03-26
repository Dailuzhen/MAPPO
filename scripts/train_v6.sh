#!/bin/bash
# =============================================================
# V6 训练脚本 - Phase A (多地图BC) + Phase B (课程学习PPO)
#
# 改进点:
#   1. Phase A: 在 8-20 多种地图尺寸上做 A* 行为克隆
#   2. Phase B: 4阶段课程学习（地图尺寸+距离同时递进）
#   3. 抗卡死惩罚（重复位置 -0.1）
#   4. A* 奖励塑形（偏离路径重新计算）
#   5. 每个环境独立随机地图（去掉 shared_map）
# =============================================================

GPU_ID=2

EXPERIMENT_NAME="full_v6"

echo "══════════════════════════════════════════════════════════════════════"
echo "      V6: Phase A (多地图BC 200ep) + Phase B (课程学习PPO 2000ep)"
echo "══════════════════════════════════════════════════════════════════════"
echo "  实验名称: $EXPERIMENT_NAME"
echo "  使用 GPU: $GPU_ID"
echo ""
echo "  Phase A (行为克隆):"
echo "    - 回合数: 200"
echo "    - 地图尺寸: 8-20 随机"
echo "    - 每回合切换地图尺寸+障碍布局"
echo ""
echo "  Phase B (PPO):"
echo "    - 回合数: 2000"
echo "    - 课程学习 4 阶段:"
echo "      ep1-500:    8-12  dist<=10"
echo "      ep500-1000: 8-16  dist<=18"
echo "      ep1000-1500:8-20  dist<=30"
echo "      ep1500-2000:8-22  无限制"
echo "    - A* 奖励塑形 +0.1/-0.02"
echo "    - 抗卡死惩罚 -0.1"
echo ""
echo "  训练配置:"
echo "    - 并行环境: 128"
echo "    - 回合步数: 256"
echo "    - 网络结构: 256 x 4 层"
echo "    - 学习率: 5e-5"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "使用GPU进行训练..."
nvidia-smi -i $GPU_ID --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "GPU 信息获取失败"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/full_v6_${TIMESTAMP}.log"
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
