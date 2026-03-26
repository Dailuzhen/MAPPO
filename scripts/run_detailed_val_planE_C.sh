#!/bin/bash
# =============================================================
# 方案 C 最佳模型详细验证
#
# 从 v8_planE_C 训练结果中取成功率最高的模型（best_model），
# 运行 detailed_validate 格式的验证，输出到便于识别的独立目录。
# =============================================================

cd "$(dirname "$0")/.."

MODEL_DIR="results/MyEnv/MyEnv/v8_planE_C/run1/phase_b_ppo/best_model"
BASE_DIR="logs/detailed_val_v8_planE_C_best"
OUT_DIR="${BASE_DIR}/对比_修改后"

if [ ! -f "$MODEL_DIR/actor.pt" ]; then
    echo "[ERROR] 模型不存在: $MODEL_DIR/actor.pt"
    exit 1
fi

if [ -f "$MODEL_DIR/info.json" ]; then
    EP=$(python3 -c "import json; print(json.load(open('$MODEL_DIR/info.json')).get('episode','?'))" 2>/dev/null || echo "?")
    SR=$(python3 -c "import json; d=json.load(open('$MODEL_DIR/info.json')); print(f\"{d.get('success_rate',0)*100:.1f}%\")" 2>/dev/null || echo "?")
    echo "══════════════════════════════════════════════════════════════════════"
    echo "  方案 C 最佳模型: ep${EP} 训练评估成功率 ${SR}"
    echo "══════════════════════════════════════════════════════════════════════"
fi

echo ""
echo "  模型: $MODEL_DIR"
echo "  输出: $OUT_DIR (8x8, 10x10 等子目录)"
echo "  GIF 速度: 1.5s/帧 (较慢便于观看)"
echo ""

python -u scripts/detailed_validate.py \
    --model_dir "$MODEL_DIR" \
    --out_dir "$OUT_DIR" \
    --scenarios 20 \
    --map_sizes 8 10 12 14 16 18 20 22 \
    --gif_duration 1.5

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  验证完成! 结果目录: $OUT_DIR"
echo "  子目录: 8x8/, 10x10/, 12x12/, ... (日志+GIF)"
echo "══════════════════════════════════════════════════════════════════════"
