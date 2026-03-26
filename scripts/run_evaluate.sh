#!/bin/bash
# Phase B 模型评估启动脚本

# 配置参数
MODEL_PATH="/workspace/zj/light_mappo-main/results/MyEnv/MyEnv/mappo_phase_b_v3/run1/phase_b_ppo/checkpoint_ep400/models"
MAP_FILE="/workspace/zj/light_mappo-main/results/MyEnv/MyEnv/astar_bc_phase_a_v8/run1/phase_a_map.txt"
OUTPUT_DIR="/workspace/zj/light_mappo-main/results/evaluation_phase_b_v3"

# 评估参数
NUM_EPISODES=50       # 评估回合数
NUM_AGENTS=2          # 智能体数量
MAX_STEPS=100         # 每回合最大步数
SEED=42               # 随机种子

# GIF 参数
GIF_FPS=1.0           # GIF 帧率 (越小越慢，方便观看，1.0 = 每帧1秒)

# 模型参数 (需与训练一致)
HIDDEN_SIZE=128
LAYER_N=4

echo "=========================================="
echo "Phase B 模型评估"
echo "=========================================="
echo "模型路径: ${MODEL_PATH}"
echo "地图文件: ${MAP_FILE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "回合数: ${NUM_EPISODES}"
echo "智能体数: ${NUM_AGENTS}"
echo "最大步数: ${MAX_STEPS}"
echo "GIF 帧率: ${GIF_FPS} fps"
echo "=========================================="

# 切换到项目根目录
cd /workspace/zj/light_mappo-main

# 运行评估
python scripts/evaluate.py \
    --model_path "${MODEL_PATH}" \
    --map_file "${MAP_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_episodes ${NUM_EPISODES} \
    --num_agents ${NUM_AGENTS} \
    --max_steps ${MAX_STEPS} \
    --seed ${SEED} \
    --gif_fps ${GIF_FPS} \
    --hidden_size ${HIDDEN_SIZE} \
    --layer_N ${LAYER_N} \
    --cuda

echo "=========================================="
echo "评估完成!"
echo "结果保存在: ${OUTPUT_DIR}"
echo "=========================================="
