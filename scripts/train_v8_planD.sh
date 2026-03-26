#!/bin/bash
# =============================================================
# 方案 D: 奖励修复 + Phase B 重训练
#
# 基于 V8 Phase A ep190 (BC 预训练模型)，重新跑 Phase B
# 与 V8 原始 Phase B 的区别:
#   - stay 动作惩罚 (-0.05)
#   - anti_stuck 系数增强 (0.1→0.2, 窗口 20→30 步)
#   - 卡住检测漏洞已修复 (所有 agent 不动即触发)
#   - 地图范围扩展到 8-22 (V8 原始是 8-20)
#   - 独立验证 + 早停 (每 100ep, 160 场景, 耐心=5)
#   - 新增训练诊断日志 (地图分布 + reward 分量 + 评估诊断)
# =============================================================

GPU_ID=0
EXPERIMENT_NAME="v8_planD"

PHASE_A_MODEL="results/MyEnv/MyEnv/full_v8/run1/phase_a_astar/checkpoint_ep190/models"

echo "══════════════════════════════════════════════════════════════════════"
echo "      方案 D: Phase B 重训练 (基于 Phase A ep190)"
echo "══════════════════════════════════════════════════════════════════════"
echo "  GPU: $GPU_ID | 基础模型: Phase A ep190 | 地图: 8-22"
echo "  修改: stay惩罚(-0.05) + anti_stuck增强(0.2) + 卡住检测修复"
echo "  验证: 每100ep独立验证(160场景), 早停耐心=5"
echo "══════════════════════════════════════════════════════════════════════"

if [ ! -f "$PHASE_A_MODEL/actor.pt" ]; then
    echo "[ERROR] Phase A 模型不存在: $PHASE_A_MODEL/actor.pt"
    exit 1
fi

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$GPU_ID

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/v8_planD_${TIMESTAMP}.log"
mkdir -p logs

echo "训练日志: $LOG_FILE"
echo "开始训练..."

stdbuf -oL python -u train/train.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --training_mode "phase_b_only" \
    --model_dir "$PHASE_A_MODEL" \
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
    --map_size_max 22 \
    --map_change_interval 5 \
    --curriculum_stages 4 \
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
    --val_interval 100 \
    --val_scenarios 160 \
    --val_patience 5 \
    \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "训练完成! 日志: $LOG_FILE"
echo ""
echo "验证命令:"
echo "  python scripts/validate_models.py \\"
echo "    --phase_b_dir results/MyEnv/MyEnv/$EXPERIMENT_NAME/run1/phase_b_ppo \\"
echo "    --checkpoints 500 1000 1500 2000 \\"
echo "    --scenarios_per_size 50 \\"
echo "    --obs_radius 3"
echo "══════════════════════════════════════════════════════════════════════"
