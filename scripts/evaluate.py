#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase B 模型评估脚本

功能:
1. 加载训练好的 PPO 模型
2. 在固定地图上运行 50 回合评估
3. 每回合生成 GIF 动图（与 Phase A 格式一致）
4. 输出汇总统计

使用方法:
    python scripts/evaluate.py \
        --model_path /path/to/checkpoint/models \
        --map_file /path/to/map.txt \
        --num_episodes 50 \
        --num_agents 2 \
        --max_steps 100 \
        --output_dir results/evaluation
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')  # 非交互式后端

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

from PIL import Image, ImageDraw, ImageFont

from envs.env import Env
from algorithms.algorithm.actor_critic import Actor


def parse_args():
    parser = argparse.ArgumentParser(description="Phase B 模型评估")
    
    # 必需参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径 (包含 actor.pt 的目录)")
    parser.add_argument("--map_file", type=str, required=True,
                        help="固定地图文件路径 (.txt 或 .npy)")
    
    # 评估参数
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="评估回合数 (默认: 50)")
    parser.add_argument("--num_agents", type=int, default=2,
                        help="智能体数量 (默认: 2)")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="每回合最大步数 (默认: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="results/evaluation",
                        help="输出目录 (默认: results/evaluation)")
    parser.add_argument("--gif_fps", type=float, default=0.1,
                        help="GIF 帧率，越小越慢 (默认: 0.1 fps，即每帧约10秒)")
    
    # 模型参数 (需要与训练时一致)
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="网络隐藏层大小 (默认: 128)")
    parser.add_argument("--layer_N", type=int, default=4,
                        help="网络层数 (默认: 4)")
    
    # 设备
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="使用 GPU")
    
    return parser.parse_args()


class ModelArgs:
    """模型参数包装类"""
    def __init__(self, hidden_size=128, layer_N=4):
        self.hidden_size = hidden_size
        self.layer_N = layer_N
        self.use_orthogonal = True
        self.use_ReLU = True
        self.use_feature_normalization = True
        self.use_policy_active_masks = True
        self.gain = 0.01
        self.stacked_frames = 1
        self.use_stacked_frames = False


def get_font(size):
    """获取字体"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    
    return ImageFont.load_default()


class Evaluator:
    """评估器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 设置随机种子
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.gifs_dir = self.output_dir / "gifs"
        self.gifs_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建环境
        self.env = self._create_env()
        
        # 加载模型
        self.actor = self._load_model()
        
        # 结果存储
        self.episode_results = []
        
    def _create_env(self):
        """创建评估环境"""
        env = Env(
            agent_num=self.args.num_agents,
            map_file=self.args.map_file,
            max_episode_steps=self.args.max_steps * 10,  # 设大一点，我们自己控制
            use_astar_shaping=False,
            use_astar_first_episode=False,
            obs_include_goal_direction=True,
            obs_include_position=True
        )
        
        print(f"地图尺寸: {env.grid_size}x{env.grid_size}")
        print(f"障碍物数量: {np.sum(env.map)}")
        print(f"智能体数量: {env.agent_num}")
        print(f"观测维度: {env.obs_dim}")
        
        return env
    
    def _load_model(self):
        """加载 Actor 模型"""
        model_args = ModelArgs(
            hidden_size=self.args.hidden_size,
            layer_N=self.args.layer_N
        )
        
        # 创建 Actor
        actor = Actor(
            model_args,
            self.env.observation_space[0],
            self.env.action_space[0],
            device=self.device
        )
        
        # 加载权重
        actor_path = Path(self.args.model_path) / "actor.pt"
        if not actor_path.exists():
            raise FileNotFoundError(f"Actor 模型不存在: {actor_path}")
        
        state_dict = torch.load(str(actor_path), map_location=self.device)
        actor.load_state_dict(state_dict)
        actor.eval()
        
        print(f"模型已加载: {actor_path}")
        return actor
    
    def _get_astar_steps(self):
        """
        计算 A* 最优步数
        
        因为所有智能体同时移动，所以取各智能体 A* 步数的最大值
        （而非总和，否则会导致 ratio < 1）
        """
        agent_steps = []
        for agent_idx in range(self.env.agent_num):
            start = tuple(self.env.agents_pos[agent_idx])
            goal = tuple(self.env.goals_pos[agent_idx])
            
            if start == goal:
                agent_steps.append(0)
                continue
            
            path = self.env._astar_path(start, goal)
            if path:
                agent_steps.append(len(path) - 1)
            else:
                # 无路径，用曼哈顿距离估计
                agent_steps.append(abs(start[0] - goal[0]) + abs(start[1] - goal[1]))
        
        # 取最大值（因为智能体同时移动，完成时间取决于最慢的智能体）
        return max(1, max(agent_steps) if agent_steps else 1)
    
    def _check_all_reached(self):
        """检查是否所有智能体都到达终点"""
        for i in range(self.env.agent_num):
            if self.env.agents_pos[i] != self.env.goals_pos[i]:
                return False
        return True
    
    def _count_reached(self):
        """统计到达终点的智能体数量"""
        count = 0
        for i in range(self.env.agent_num):
            if self.env.agents_pos[i] == self.env.goals_pos[i]:
                count += 1
        return count
    
    @torch.no_grad()
    def _run_episode(self, episode_idx):
        """运行单个回合"""
        # 强制清除环境缓存，确保每次 reset 都随机生成新的起点终点
        self.env.start_pos = None
        self.env.fixed_goals = None
        self.env.agents_pos = None
        self.env.goals_pos = None
        
        # 禁用环境的阻塞检测（防止提前重置）
        self.env._stuck_counter = 0
        
        # 重置环境
        obs, _ = self.env.reset()
        
        # 记录初始位置
        start_positions = [tuple(pos) for pos in self.env.agents_pos]
        goal_positions = [tuple(pos) for pos in self.env.goals_pos]
        
        # 保存起点信息到环境（用于渲染）
        self.env.start_pos = list(self.env.agents_pos)
        
        astar_steps = self._get_astar_steps()
        
        # 帧收集（使用环境的 render 方法）
        frames = []
        
        # 渲染初始帧
        initial_frame = self.env.render(mode="rgb_array")
        frames.append(initial_frame)
        
        # 运行回合
        total_steps = 0
        collision_count = 0
        all_reached = False
        inference_times = []
        
        for step in range(self.args.max_steps):
            # 禁用阻塞检测（每步重置计数器）
            self.env._stuck_counter = 0
            
            # 策略推理
            obs_tensor = torch.tensor(
                obs.reshape(-1, obs.shape[-1]),
                dtype=torch.float32,
                device=self.device
            )
            
            start_time = time.perf_counter()
            actions, _ = self.actor(obs_tensor, deterministic=True)
            inference_time = (time.perf_counter() - start_time) * 1000  # ms
            inference_times.append(inference_time)
            
            actions = actions.cpu().numpy().flatten().tolist()
            
            # 在执行动作前记录当前位置（用于检测到达终点）
            positions_before_step = list(self.env.agents_pos)
            goals_before_step = list(self.env.goals_pos)
            
            # 执行动作
            obs, _, rewards, dones, info = self.env.step(actions)
            total_steps += 1
            
            # 统计碰撞
            collision_list = info.get("collision", [])
            if isinstance(collision_list, list):
                collision_count += sum(collision_list)
            
            # 检查是否全部到达（looped_to_start=True 表示所有智能体都到达了终点）
            if info.get("looped_to_start", False):
                all_reached = True
                # 环境已重置，我们需要渲染"到达终点"的状态
                # 临时恢复到达终点时的位置来渲染最终帧
                saved_agents_pos = list(self.env.agents_pos)
                saved_goals_pos = list(self.env.goals_pos)
                saved_start_pos = list(self.env.start_pos) if self.env.start_pos else None
                
                # 恢复到达终点时的状态
                self.env.agents_pos = goals_before_step  # 到达终点时，位置就是目标位置
                self.env.goals_pos = goals_before_step
                self.env.start_pos = start_positions  # 恢复原始起点用于正确渲染
                
                # 渲染到达终点的帧
                final_frame = self.env.render(mode="rgb_array")
                frames.append(final_frame)
                
                # 恢复环境状态（虽然后面不用了，但保持一致性）
                self.env.agents_pos = saved_agents_pos
                self.env.goals_pos = saved_goals_pos
                if saved_start_pos:
                    self.env.start_pos = saved_start_pos
                break
            
            # 渲染当前帧
            frame = self.env.render(mode="rgb_array")
            frames.append(frame)
        
        # 计算指标
        # 如果 all_reached，说明所有智能体都到达了
        agents_reached = self.env.agent_num if all_reached else self._count_reached()
        success = all_reached
        step_ratio = total_steps / astar_steps if success else None
        total_inference_time = sum(inference_times)
        avg_inference_time = total_inference_time / total_steps if total_steps > 0 else 0
        
        result = {
            "episode": episode_idx + 1,
            "all_reached": all_reached,
            "agents_reached": agents_reached,
            "total_agents": self.env.agent_num,
            "success": success,
            "total_steps": total_steps,
            "max_steps": self.args.max_steps,
            "astar_steps": astar_steps,
            "step_ratio": step_ratio,
            "inference_time_ms": round(total_inference_time, 3),
            "avg_inference_per_step_ms": round(avg_inference_time, 3),
            "collision_count": collision_count,
            "start_positions": start_positions,
            "goal_positions": goal_positions,
            "frames": frames  # 临时保存帧用于生成 GIF
        }
        
        return result
    
    def _create_gif(self, result, episode_idx):
        """为回合创建 GIF 动图（带指标信息）"""
        frames = result["frames"]
        
        # 首先统一所有原始帧的尺寸
        target_size = None
        for frame in frames:
            if target_size is None:
                target_size = (frame.shape[1], frame.shape[0])  # (width, height)
            else:
                # 找最大尺寸
                target_size = (max(target_size[0], frame.shape[1]), 
                              max(target_size[1], frame.shape[0]))
        
        # 创建带指标的帧
        annotated_frames = []
        final_size = None
        
        for i, frame in enumerate(frames):
            # 调整帧尺寸
            img = Image.fromarray(frame)
            if img.size != target_size:
                img = img.resize(target_size, Image.LANCZOS)
            frame = np.array(img)
            
            annotated_frame = self._add_metrics_to_frame(
                frame, 
                result, 
                current_step=i,
                is_final=(i == len(frames) - 1)
            )
            
            # 记录最终帧尺寸
            if final_size is None:
                final_size = annotated_frame.shape[:2]
            
            # 确保所有帧尺寸一致
            if annotated_frame.shape[:2] != final_size:
                img = Image.fromarray(annotated_frame)
                img = img.resize((final_size[1], final_size[0]), Image.LANCZOS)
                annotated_frame = np.array(img)
            
            annotated_frames.append(annotated_frame)
        
        # 不重复最后一帧，直接停留在到达终点的画面
        
        # 保存 GIF
        gif_path = self.gifs_dir / f"episode_{episode_idx+1:03d}.gif"
        duration = 1.0 / self.args.gif_fps
        imageio.mimsave(str(gif_path), annotated_frames, duration=duration)
        
        # 清理帧数据（节省内存）
        del result["frames"]
        
        return str(gif_path)
    
    def _add_metrics_to_frame(self, frame, result, current_step, is_final):
        """在帧上添加指标信息"""
        # 转换为 PIL Image
        img = Image.fromarray(frame)
        
        # 创建更大的画布以添加信息
        orig_w, orig_h = img.size
        header_h = 50
        footer_h = 100  # 统一 footer 高度，确保所有帧尺寸一致
        
        canvas = Image.new('RGB', (orig_w, orig_h + header_h + footer_h), color='white')
        canvas.paste(img, (0, header_h))
        
        draw = ImageDraw.Draw(canvas)
        font = get_font(14)
        font_small = get_font(11)
        font_large = get_font(16)
        
        # 头部信息 (使用英文避免乱码)
        status_text = "SUCCESS" if (is_final and result["success"]) else \
                      ("FAILED (Timeout)" if (is_final and not result["success"]) else "Running...")
        
        header_text = f"Episode {result['episode']}/{self.args.num_episodes}   Step: {current_step}/{result['total_steps']}   {status_text}"
        draw.text((10, 15), header_text, fill=(0, 0, 0), font=font)
        
        # 底部信息
        y_offset = orig_h + header_h + 5
        
        # 起点终点信息
        start_str = "Start: " + ", ".join([f"A{i+1}:{s}" for i, s in enumerate(result["start_positions"])])
        goal_str = "Goal: " + ", ".join([f"A{i+1}:{g}" for i, g in enumerate(result["goal_positions"])])
        draw.text((10, y_offset), start_str, fill=(0, 100, 0), font=font_small)
        draw.text((10, y_offset + 15), goal_str, fill=(200, 0, 0), font=font_small)
        
        # 最终帧添加详细指标
        if is_final:
            y_offset += 35
            
            # 是否全部到达
            if result['all_reached']:
                reached_text = f"All Reached: YES ({result['agents_reached']}/{result['total_agents']})"
                reached_color = (0, 128, 0)
            else:
                reached_text = f"All Reached: NO ({result['agents_reached']}/{result['total_agents']})"
                reached_color = (200, 0, 0)
            draw.text((10, y_offset), reached_text, fill=reached_color, font=font)
            
            # 步数信息
            y_offset += 18
            steps_text = f"Steps: {result['total_steps']} (A* optimal: {result['astar_steps']})"
            draw.text((10, y_offset), steps_text, fill=(0, 0, 0), font=font_small)
            
            # 步数比
            if result['step_ratio']:
                ratio_text = f"Ratio: {result['step_ratio']:.2f}x"
                ratio_color = (0, 128, 0) if result['step_ratio'] <= 1.5 else (200, 128, 0)
            else:
                ratio_text = "Ratio: - (incomplete)"
                ratio_color = (200, 0, 0)
            draw.text((280, y_offset), ratio_text, fill=ratio_color, font=font_small)
            
            # 推理时间
            y_offset += 18
            time_text = f"Inference: {result['inference_time_ms']:.2f}ms (avg {result['avg_inference_per_step_ms']:.3f}ms/step)"
            draw.text((10, y_offset), time_text, fill=(0, 0, 128), font=font_small)
            
            # 碰撞次数
            collision_text = f"Collisions: {result['collision_count']}"
            collision_color = (0, 0, 0) if result['collision_count'] == 0 else (200, 0, 0)
            draw.text((350, y_offset), collision_text, fill=collision_color, font=font_small)
        
        return np.array(canvas)
    
    def run(self):
        """运行完整评估"""
        print(f"\n{'='*60}")
        print(f"开始评估: {self.args.num_episodes} 回合")
        print(f"{'='*60}\n")
        
        for episode_idx in range(self.args.num_episodes):
            print(f"回合 {episode_idx + 1}/{self.args.num_episodes}...", end=" ")
            
            # 运行回合
            result = self._run_episode(episode_idx)
            
            # 创建 GIF
            gif_path = self._create_gif(result, episode_idx)
            result["gif_path"] = gif_path
            
            self.episode_results.append(result)
            
            # 打印结果
            status = "✓" if result["success"] else "✗"
            ratio_str = f"{result['step_ratio']:.2f}x" if result['step_ratio'] else "-"
            print(f"{status} | 步数: {result['total_steps']}/{result['max_steps']} | "
                  f"A*: {result['astar_steps']} | 比率: {ratio_str} | "
                  f"碰撞: {result['collision_count']} | 推理: {result['avg_inference_per_step_ms']:.3f}ms/步")
        
        # 保存结果
        self._save_results()
        
        # 打印汇总
        self._print_summary()
    
    def _save_results(self):
        """保存评估结果"""
        # 保存详细结果
        details_path = self.output_dir / "episode_details.json"
        
        # 转换为可序列化格式
        serializable_results = []
        for r in self.episode_results:
            sr = {k: v for k, v in r.items() if k != "frames"}
            sr["start_positions"] = [list(p) for p in r["start_positions"]]
            sr["goal_positions"] = [list(p) for p in r["goal_positions"]]
            serializable_results.append(sr)
        
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 计算汇总统计
        successes = sum(1 for r in self.episode_results if r["success"])
        success_results = [r for r in self.episode_results if r["success"]]
        
        summary = {
            "config": {
                "map_file": str(self.args.map_file),
                "map_size": f"{self.env.grid_size}x{self.env.grid_size}",
                "obstacles": int(np.sum(self.env.map)),
                "num_agents": self.args.num_agents,
                "model_path": str(self.args.model_path),
                "total_episodes": self.args.num_episodes,
                "max_steps_per_episode": self.args.max_steps,
                "seed": self.args.seed,
                "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": {
                "success_count": successes,
                "fail_count": self.args.num_episodes - successes,
                "success_rate": successes / self.args.num_episodes,
                "avg_steps": np.mean([r["total_steps"] for r in success_results]) if success_results else 0,
                "avg_astar_steps": np.mean([r["astar_steps"] for r in success_results]) if success_results else 0,
                "avg_step_ratio": np.mean([r["step_ratio"] for r in success_results]) if success_results else 0,
                "avg_inference_time_ms": np.mean([r["avg_inference_per_step_ms"] for r in self.episode_results]),
                "total_collisions": sum(r["collision_count"] for r in self.episode_results),
                "failed_episodes": [r["episode"] for r in self.episode_results if not r["success"]]
            }
        }
        
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 保存 CSV
        csv_path = self.output_dir / "summary.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Episode,All_Reached,Reached,Steps,A*_Steps,Ratio,Time_ms,Collisions\n")
            for r in self.episode_results:
                ratio_str = f"{r['step_ratio']:.2f}" if r['step_ratio'] else "-"
                f.write(f"{r['episode']},{r['all_reached']},{r['agents_reached']}/{r['total_agents']},"
                       f"{r['total_steps']},{r['astar_steps']},{ratio_str},"
                       f"{r['avg_inference_per_step_ms']:.3f},{r['collision_count']}\n")
        
        print(f"\n结果已保存:")
        print(f"  - 详细数据: {details_path}")
        print(f"  - 汇总统计: {summary_path}")
        print(f"  - CSV 表格: {csv_path}")
        print(f"  - GIF 动图: {self.gifs_dir}/")
    
    def _print_summary(self):
        """打印汇总统计"""
        successes = sum(1 for r in self.episode_results if r["success"])
        success_results = [r for r in self.episode_results if r["success"]]
        
        print(f"\n{'='*60}")
        print("评估汇总")
        print(f"{'='*60}")
        print(f"总回合数: {self.args.num_episodes}")
        print(f"成功: {successes} ({successes/self.args.num_episodes*100:.1f}%)")
        print(f"失败: {self.args.num_episodes - successes}")
        
        if success_results:
            avg_steps = np.mean([r["total_steps"] for r in success_results])
            avg_astar = np.mean([r["astar_steps"] for r in success_results])
            avg_ratio = np.mean([r["step_ratio"] for r in success_results])
            print(f"平均步数: {avg_steps:.1f} (A*: {avg_astar:.1f})")
            print(f"平均步数比: {avg_ratio:.2f}x")
        
        avg_inference = np.mean([r["avg_inference_per_step_ms"] for r in self.episode_results])
        total_collisions = sum(r["collision_count"] for r in self.episode_results)
        print(f"平均推理时间: {avg_inference:.3f} ms/步")
        print(f"总碰撞次数: {total_collisions}")
        
        failed = [r["episode"] for r in self.episode_results if not r["success"]]
        if failed:
            print(f"失败回合: {failed}")
        
        print(f"{'='*60}\n")


def main():
    args = parse_args()
    evaluator = Evaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
