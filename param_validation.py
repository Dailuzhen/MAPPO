#!/usr/bin/env python
import os
import sys
import time
import multiprocessing
import subprocess
import json
import argparse
from pathlib import Path

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "train", "train.py")

# 定义要验证的参数组合
PARAM_COMBINATIONS = [
    # num_mini_batch参数组合
    {
        "name": "num_mini_batch_32",
        "params": ["--num_mini_batch", "32"]
    },
    {
        "name": "num_mini_batch_64",
        "params": ["--num_mini_batch", "64"]
    },
    {
        "name": "num_mini_batch_128",
        "params": ["--num_mini_batch", "128"]
    },
    {
        "name": "num_mini_batch_256",
        "params": ["--num_mini_batch", "256"]
    },
    {
        "name": "num_mini_batch_512",
        "params": ["--num_mini_batch", "512"]
    },
    # 隐藏层大小参数组合
    {
        "name": "hidden_64",
        "params": ["--hidden_size", "64"]
    },
    {
        "name": "hidden_128",
        "params": ["--hidden_size", "128"]
    },
    {
        "name": "hidden_256",
        "params": ["--hidden_size", "256"]
    },
    {
        "name": "hidden_512",
        "params": ["--hidden_size", "512"]
    },
    {
        "name": "hidden_1024",
        "params": ["--hidden_size", "1024"]
    },
    # PPO参数组合
    {
        "name": "ppo_epoch_10",
        "params": ["--ppo_epoch", "10"]
    },
    {
        "name": "ppo_epoch_15",
        "params": ["--ppo_epoch", "15"]
    },
    {
        "name": "ppo_epoch_20",
        "params": ["--ppo_epoch", "20"]
    },
    {
        "name": "ppo_epoch_25",
        "params": ["--ppo_epoch", "25"]
    },
    {
        "name": "ppo_epoch_30",
        "params": ["--ppo_epoch", "30"]
    },
    # Phase A参数组合
    {
        "name": "phase_a_bc_lr_1e-4",
        "params": ["--phase_a_bc_lr", "1e-4"]
    },
    {
        "name": "phase_a_bc_lr_3e-4",
        "params": ["--phase_a_bc_lr", "3e-4"]
    },
    {
        "name": "phase_a_bc_lr_5e-4",
        "params": ["--phase_a_bc_lr", "5e-4"]
    },
    {
        "name": "phase_a_bc_lr_1e-3",
        "params": ["--phase_a_bc_lr", "1e-3"]
    },
    {
        "name": "phase_a_bc_lr_3e-3",
        "params": ["--phase_a_bc_lr", "3e-3"]
    },
]

# 基础参数
BASE_PARAMS = [
    "--num_env_steps", "500000",  # 减少训练步数，加快验证
    "--eval_interval", "25",
    "--log_interval", "10",
    "--save_interval", "20",
    "--cuda",  # 确保使用GPU训练
    "--n_training_threads", "4",
    "--n_rollout_threads", "1",
]

def run_training(param_combination):
    """运行单个参数组合的训练
    
    Args:
        param_combination: 参数组合字典
    
    Returns:
        训练结果字典
    """
    name = param_combination["name"]
    params = param_combination["params"]
    
    print(f"开始运行参数组合: {name}")
    print(f"参数: {params}")
    
    # 构建命令
    cmd = [
        sys.executable,
        TRAIN_SCRIPT,
        "--experiment_name", f"param_validation_{name}",
        *BASE_PARAMS,
        *params
    ]
    
    # 运行训练
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=7200  # 设置2小时超时
        )
        runtime = time.time() - start_time
        
        # 分析结果
        success = result.returncode == 0
        
        # 尝试提取训练结果
        metrics = extract_metrics(name)
        
        return {
            "name": name,
            "params": params,
            "success": success,
            "runtime": runtime,
            "metrics": metrics,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "params": params,
            "success": False,
            "runtime": 7200,
            "metrics": {},
            "stdout": "",
            "stderr": "训练超时"
        }
    except Exception as e:
        return {
            "name": name,
            "params": params,
            "success": False,
            "runtime": time.time() - start_time,
            "metrics": {},
            "stdout": "",
            "stderr": str(e)
        }

def extract_metrics(param_name):
    """从训练结果中提取指标
    
    Args:
        param_name: 参数组合名称
    
    Returns:
        指标字典
    """
    metrics = {}
    
    # 查找结果目录
    results_dir = os.path.join(PROJECT_ROOT, "results", "MyEnv", "MyEnv", f"param_validation_{param_name}")
    if not os.path.exists(results_dir):
        return metrics
    
    # 查找最新的run目录
    run_dirs = [d for d in os.listdir(results_dir) if d.startswith("run")]
    if not run_dirs:
        return metrics
    
    latest_run = sorted(run_dirs, key=lambda x: int(x[3:]))[-1]
    run_path = os.path.join(results_dir, latest_run)
    
    # 尝试读取summary.json
    summary_file = os.path.join(run_path, "summary.json")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 提取关键指标
                if "episode_reward" in data:
                    metrics["average_reward"] = data["episode_reward"][-1] if data["episode_reward"] else 0
                if "value_loss" in data:
                    metrics["final_value_loss"] = data["value_loss"][-1] if data["value_loss"] else 0
                if "policy_loss" in data:
                    metrics["final_policy_loss"] = data["policy_loss"][-1] if data["policy_loss"] else 0
        except Exception:
            pass
    
    return metrics

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多进程并行验证参数设置合理性")
    parser.add_argument("--num_processes", type=int, default=20, help="并行进程数")
    args = parser.parse_args()
    
    print("=" * 80)
    print("多进程并行参数验证脚本")
    print("=" * 80)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"训练脚本: {TRAIN_SCRIPT}")
    print(f"并行进程数: {args.num_processes}")
    print(f"参数组合数: {len(PARAM_COMBINATIONS)}")
    print("=" * 80)
    
    # 使用多进程运行参数验证
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        results = pool.map(run_training, PARAM_COMBINATIONS)
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("参数验证结果汇总")
    print("=" * 80)
    
    # 打印结果表格
    print(f"{'参数组合':<30} {'状态':<10} {'运行时间(s)':<15} {'平均奖励':<15} {'价值损失':<15} {'策略损失':<15}")
    print("-" * 100)
    
    for result in results:
        status = "成功" if result["success"] else "失败"
        runtime = f"{result['runtime']:.2f}"
        avg_reward = f"{result['metrics'].get('average_reward', 0):.2f}"
        value_loss = f"{result['metrics'].get('final_value_loss', 0):.4f}"
        policy_loss = f"{result['metrics'].get('final_policy_loss', 0):.4f}"
        print(f"{result['name']:<30} {status:<10} {runtime:<15} {avg_reward:<15} {value_loss:<15} {policy_loss:<15}")
    
    # 保存详细结果
    output_file = os.path.join(PROJECT_ROOT, "param_validation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()