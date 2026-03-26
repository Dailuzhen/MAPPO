#!/bin/bash
# =============================================================
# 方案 C: 课程加速 (GPU 3)
#
# 不改观测空间，但用双倍 episode_length + 2 阶段课程
# 核心假设: 大地图需要更多训练时间，快速跳到大地图训练
# episode_length 512 + n_rollout_threads 128 = 保持总步数 65536/更新
# curriculum_stages 2: 8-15 -> 8-22，后半段全在大地图训练
# =============================================================

GPU_ID=3
EXPERIMENT_NAME="v8_planE_C"
PHASE_A_MODEL="results/MyEnv/MyEnv/full_v8/run1/phase_a_astar/checkpoint_ep190/models"

echo "══════════════════════════════════════════════════════════════════════"
echo "  方案 C: 课程加速 (GPU $GPU_ID)"
echo "══════════════════════════════════════════════════════════════════════"
echo "  核心: 方案A全部改进 + episode_length=512 + 2阶段课程"
echo "  观测: obs_dim=55 (无 visit_history)"
echo "  训练: 2000 回合, 无早停, 每 100 回合评估"
echo "══════════════════════════════════════════════════════════════════════"

if [ ! -f "$PHASE_A_MODEL/actor.pt" ]; then
    echo "[ERROR] Phase A 模型不存在: $PHASE_A_MODEL/actor.pt"
    exit 1
fi

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$GPU_ID

LOG_FILE="logs/v8_planE_C_20260311_193925.log"
mkdir -p logs

echo "训练日志: $LOG_FILE (追加)"
echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  [续训] $(date '+%Y-%m-%d %H:%M:%S') - 继续训练至 2000 ep"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

stdbuf -oL python -u train/train.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --training_mode "phase_b_only" \
    --model_dir "$PHASE_A_MODEL" \
    --resume_run "run1" \
    \
    --phase_b_episodes 2000 \
    --phase_b_eval_interval 100 \
    --phase_b_eval_scenarios 40 \
    \
    --episode_length 512 \
    --n_rollout_threads 128 \
    --obs_radius 3 \
    \
    --map_size_min 8 \
    --map_size_max 22 \
    --map_change_interval 5 \
    --curriculum_stages 2 \
    \
    --hidden_size 256 \
    --layer_N 4 \
    \
    --lr 3e-5 \
    --critic_lr 3e-5 \
    --ppo_epoch 5 \
    --num_mini_batch 4 \
    --clip_param 0.2 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --entropy_coef 0.02 \
    --max_grad_norm 10.0 \
    \
    --log_interval 10 \
    \
    --obs_include_goal_direction \
    --obs_include_position \
    \
    --use_first_reach_reward \
    --use_wall_penalty \
    --use_pbrs \
    \
    --val_interval 0 \
    \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "方案 C 训练完成! 日志: $LOG_FILE"
echo "══════════════════════════════════════════════════════════════════════"
