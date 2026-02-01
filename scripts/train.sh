#!/bin/bash
set -e

# 训练脚本（适配当前 MyEnv + train/train.py）
ENV_NAME="MyEnv"
SCENARIO_NAME="MyEnv"
EXP_NAME="check"
NUM_AGENTS=2
SEED=1

# 训练参数
N_ROLLOUT_THREADS=1
EPISODE_LENGTH=1000
NUM_ENV_STEPS=1000000
SAVE_INTERVAL=100
LOG_INTERVAL=10
EVAL_INTERVAL=100
EVAL_EPISODES=32
PPO_EPOCH=20
NUM_MINI_BATCH=128
HIDDEN_SIZE=128
N_TRAINING_THREADS=8

echo "Starting training: env=${ENV_NAME}, scenario=${SCENARIO_NAME}, exp=${EXP_NAME}, agents=${NUM_AGENTS}"

CUDA_VISIBLE_DEVICES=3 python /workspace/lxx4/light_mappo-main/train/train.py \
  --env_name ${ENV_NAME} \
  --scenario_name ${SCENARIO_NAME} \
  --experiment_name ${EXP_NAME} \
  --num_agents ${NUM_AGENTS} \
  --seed ${SEED} \
  --n_training_threads ${N_TRAINING_THREADS} \
  --n_rollout_threads ${N_ROLLOUT_THREADS} \
  --episode_length ${EPISODE_LENGTH} \
  --num_env_steps ${NUM_ENV_STEPS} \
  --ppo_epoch ${PPO_EPOCH} \
  --num_mini_batch ${NUM_MINI_BATCH} \
  --hidden_size ${HIDDEN_SIZE} \
  --save_interval ${SAVE_INTERVAL} \
  --log_interval ${LOG_INTERVAL} \
  --eval_interval ${EVAL_INTERVAL} \
  --eval_episodes ${EVAL_EPISODES}
