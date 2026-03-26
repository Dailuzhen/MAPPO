"""
Phase A Runner 模块 (v2.0)

优化内容：
1. 分层经验池：按地图大小分类，均衡采样
2. 智能评估：快速评估(每5ep) + 完整评估(每20ep)
3. 早停机制：BC 准确率达标时提前停止
4. 检查点滚动：只保留最近 N 个检查点
5. 简化日志：去除 Segment 详细输出
"""

import os
import time
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
import random
import shutil

from utils.phase_a_utils import (
    save_json, load_json,
    make_gif_from_frames, save_frames_to_dir,
    generate_comparison_gif,
    generate_checkpoint_summary, create_scene_config,
    setup_phase_a_directories, get_checkpoint_dir, get_episode_dir,
    print_phase_a_header, print_episode_header, print_checkpoint_summary,
    calculate_bc_accuracy
)


class StratifiedReplayBuffer:
    """
    分层经验回放缓冲区 (v2.0)
    
    按地图大小分类存储，确保各尺寸地图均衡采样
    """
    
    def __init__(self, max_size_per_tier=20000):
        self.max_size_per_tier = max_size_per_tier
        
        # 三个分层缓冲区
        self.tiers = {
            'small': {'obs': [], 'acts': [], 'episodes': []},   # 10-14
            'medium': {'obs': [], 'acts': [], 'episodes': []},  # 15-17
            'large': {'obs': [], 'acts': [], 'episodes': []}    # 18-25
        }
        
        # 地图大小到分层的映射边界
        self.small_max = 14
        self.medium_max = 17
    
    def _get_tier(self, map_size):
        """根据地图大小确定分层"""
        if map_size <= self.small_max:
            return 'small'
        elif map_size <= self.medium_max:
            return 'medium'
        else:
            return 'large'
    
    def add_episode(self, episode_data, map_size):
        """添加一个回合的数据到对应分层"""
        tier_name = self._get_tier(map_size)
        tier = self.tiers[tier_name]
        
        episode_num = episode_data.get("episode", 0)
        
        for seg in episode_data["segments"]:
            for obs, acts in zip(seg["obs_list"], seg["act_list"]):
                for agent_id in range(len(acts)):
                    tier['obs'].append(obs[agent_id])
                    tier['acts'].append(acts[agent_id])
        
        tier['episodes'].append(episode_num)
        
        # FIFO 限制大小
        if len(tier['obs']) > self.max_size_per_tier:
            remove_count = len(tier['obs']) - self.max_size_per_tier
            tier['obs'] = tier['obs'][remove_count:]
            tier['acts'] = tier['acts'][remove_count:]
    
    def sample(self, batch_size):
        """从三个分层均衡采样"""
        samples_per_tier = batch_size // 3
        
        all_obs = []
        all_acts = []
        
        for tier_name, tier in self.tiers.items():
            if len(tier['obs']) > 0:
                n_samples = min(samples_per_tier, len(tier['obs']))
                indices = np.random.choice(len(tier['obs']), n_samples, replace=False)
                for i in indices:
                    all_obs.append(tier['obs'][i])
                    all_acts.append(tier['acts'][i])
        
        if len(all_obs) == 0:
            return np.array([]), np.array([])
        
        return np.array(all_obs), np.array(all_acts)
    
    def __len__(self):
        return sum(len(tier['obs']) for tier in self.tiers.values())
    
    def get_stats(self):
        """获取各分层统计信息"""
        return {
            'small': len(self.tiers['small']['obs']),
            'medium': len(self.tiers['medium']['obs']),
            'large': len(self.tiers['large']['obs']),
            'total': len(self)
        }


class PhaseARunnerV2:
    """
    Phase A 运行器 v2.0
    
    主要优化：
    1. 分层经验池
    2. 智能评估策略
    3. 早停机制
    4. 检查点滚动
    5. 多环境并行数据收集
    """
    
    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.policy = config['policy']
        self.trainer = config['trainer']
        self.device = config['device']
        self.run_dir = Path(config['run_dir'])
        self.num_agents = config['num_agents']
        
        # 多环境参数
        self.n_envs = len(self.envs.envs)  # 环境数量
        
        # 基础参数
        self.phase_a_episodes = getattr(self.all_args, 'phase_a_episodes', 200)
        self.bc_lr = getattr(self.all_args, 'phase_a_bc_lr', 3e-4)
        self.bc_updates_per_episode = getattr(self.all_args, 'bc_updates_per_episode', 80)
        self.bc_batch_size = getattr(self.all_args, 'bc_batch_size', 2048)
        self.episode_length = self.all_args.episode_length
        self.gif_fps = getattr(self.all_args, 'gif_fps', 2.0)
        
        # 评估参数 (v2.0)
        self.quick_eval_interval = 5      # 快速评估间隔
        self.full_eval_interval = 20      # 完整评估间隔
        self.segments_per_ep = 5          # 每回合评估的 segment 数
        self.quick_eval_history = 3       # 快速评估的历史回合数
        self.full_eval_history = 15       # 完整评估的历史回合数
        self.astar_step_multiplier = 3.0
        
        # 早停参数
        self.early_stop_accuracy = 0.995  # 早停阈值
        self.early_stop_check_interval = 20  # 检查间隔
        
        # 检查点参数
        self.max_checkpoints = 10         # 最大保留检查点数
        
        # 随机地图参数
        self.use_random_map = getattr(self.all_args, 'use_random_map', False)
        self.map_size_min = getattr(self.all_args, 'map_size_min', 12)
        self.map_size_max = getattr(self.all_args, 'map_size_max', 20)
        self.obstacle_density_min = getattr(self.all_args, 'obstacle_density_min', 0.15)
        self.obstacle_density_max = getattr(self.all_args, 'obstacle_density_max', 0.25)
        
        # 创建目录结构
        self.dirs = setup_phase_a_directories(self.run_dir)
        self.dirs["evaluations"] = self.dirs["phase_a_root"] / "evaluations"
        self.dirs["evaluations"].mkdir(parents=True, exist_ok=True)
        self.dirs["best_model"] = self.dirs["phase_a_root"] / "best_model"
        self.dirs["best_model"].mkdir(parents=True, exist_ok=True)
        self.dirs["checkpoints"] = self.dirs["phase_a_root"] / "checkpoints"
        self.dirs["checkpoints"].mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(str(self.dirs["logs"]))
        
        # BC 优化器
        self.bc_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(),
            lr=self.bc_lr
        )
        
        # 分层经验池
        self.replay_buffer = StratifiedReplayBuffer(max_size_per_tier=20000)
        
        # 统计变量
        self.all_bc_losses = []
        self.all_bc_accuracies = []
        self.episode_map_sizes = {}  # 记录每个 episode 的地图大小
        self.best_history_rate = 0.0
        self.checkpoint_list = []    # 检查点列表（用于滚动删除）
    
    def run(self):
        """运行 Phase A 主循环（支持多环境并行）"""
        self._print_header()
        start_time = time.time()
        
        # 计算实际需要的迭代次数（多环境时每次迭代处理多个 episode）
        episodes_per_iter = self.n_envs
        total_iters = (self.phase_a_episodes + episodes_per_iter - 1) // episodes_per_iter
        
        current_ep = 0
        
        for iter_idx in range(total_iters):
            iter_start = time.time()
            
            # 计算本次迭代处理的 episode 范围
            start_ep = current_ep + 1
            end_ep = min(current_ep + episodes_per_iter, self.phase_a_episodes)
            n_active_envs = end_ep - start_ep + 1
            
            # 为每个环境重建随机地图
            map_sizes = []
            if self.use_random_map:
                for env_idx in range(n_active_envs):
                    map_info = self._rebuild_random_map_for_env(env_idx)
                    map_sizes.append(map_info['grid_size'])
            else:
                map_sizes = [self.envs.envs[i].grid_size for i in range(n_active_envs)]
            
            # 并行收集数据
            all_episode_data = self._run_astar_episode_parallel(start_ep, n_active_envs)
            
            # 保存数据并添加到经验池
            for i, episode_data in enumerate(all_episode_data):
                ep = start_ep + i
                map_size = map_sizes[i]
                self.episode_map_sizes[ep] = map_size
                
                # 只保存第一个环境的数据用于评估
                if i == 0:
                    self._save_episode_data(ep, episode_data)
                
                # 添加到经验池
                self.replay_buffer.add_episode(episode_data, map_size)
            
            # BC 更新（使用所有环境的数据）
            bc_info = self._bc_update_from_buffer()
            self.all_bc_losses.append(bc_info['avg_loss'])
            self.all_bc_accuracies.append(bc_info['accuracy'])
            
            # 日志
            for i, episode_data in enumerate(all_episode_data):
                ep = start_ep + i
                self._log_episode_metrics(ep, episode_data, bc_info)
            
            # 简化输出
            buffer_stats = self.replay_buffer.get_stats()
            iter_time = time.time() - iter_start
            
            # 显示本轮所有 episode
            map_str = "/".join([f"{s}" for s in map_sizes])
            total_segs = sum(d['total_segments'] for d in all_episode_data)
            
            print(f"[Ep {start_ep:3d}-{end_ep:3d}/{self.phase_a_episodes}] "
                  f"地图: {map_str} | "
                  f"Seg: {total_segs:3d} | "
                  f"BC: {bc_info['avg_loss']:.4f}, {bc_info['accuracy']:.1%} | "
                  f"池: S={buffer_stats['small']//1000}k/M={buffer_stats['medium']//1000}k/L={buffer_stats['large']//1000}k | "
                  f"{iter_time:.1f}s"
                  + (f" (早停)" if bc_info.get('early_stopped') else ""))
            
            # 保存检查点（以最后一个 episode 为准）
            self._save_checkpoint(end_ep)
            
            # 评估（以最后一个 episode 为准）
            if end_ep % self.quick_eval_interval == 0 or any((start_ep + i) % self.quick_eval_interval == 0 for i in range(n_active_envs)):
                eval_ep = end_ep
                if eval_ep % self.full_eval_interval == 0:
                    self._run_full_evaluation(eval_ep)
                elif eval_ep % self.quick_eval_interval == 0:
                    self._run_quick_evaluation(eval_ep)
            
            current_ep = end_ep
        
        elapsed = time.time() - start_time
        print(f"\n[Phase A] 完成! 总耗时: {elapsed/60:.1f} 分钟")
        print(f"[Phase A] 最佳历史成功率: {self.best_history_rate:.1%}")
        self.writer.close()
        return True
    
    def _print_header(self):
        """打印训练头部信息"""
        print("\n" + "═" * 60)
        print("              Phase A 随机地图训练 v2.0")
        print("═" * 60)
        print(f"  地图: {self.map_size_min}×{self.map_size_min} ~ {self.map_size_max}×{self.map_size_max}")
        print(f"  Episode: {self.phase_a_episodes}")
        print(f"  并行环境数: {self.n_envs}")
        print(f"  评估: 快速(每{self.quick_eval_interval}ep) + 完整(每{self.full_eval_interval}ep)")
        print(f"  早停阈值: {self.early_stop_accuracy:.1%}")
        print("═" * 60 + "\n")
    
    def _rebuild_random_map_for_env(self, env_idx):
        """为指定环境重建随机地图"""
        size_range = (self.map_size_min, self.map_size_max)
        density_range = (self.obstacle_density_min, self.obstacle_density_max)
        
        env = self.envs.envs[env_idx]
        map_info = env.rebuild_map_random(
            size_range=size_range,
            density_range=density_range
        )
        return map_info
    
    def _run_astar_episode_parallel(self, start_ep, n_envs):
        """并行执行多个环境的 A* Episode（优化：只收集数据，不渲染帧）"""
        all_episode_data = []
        
        for env_idx in range(n_envs):
            env = self.envs.envs[env_idx]
            ep_num = start_ep + env_idx
            
            # 只有第一个环境需要保存 GIF（用于评估）
            save_frames = (env_idx == 0)
            
            original_astar_flag = env.use_astar_first_episode
            env.use_astar_first_episode = True
            obs, share_obs = env.reset()
            
            episode_data = {
                "episode": ep_num,
                "map_size": env.grid_size,
                "segments": []
            }
            
            total_steps = 0
            segment_id = 1
            
            while total_steps < self.episode_length:
                segment_data = self._collect_segment_fast(env, obs, segment_id, total_steps, save_frames)
                if segment_data is None:
                    break
                episode_data["segments"].append(segment_data)
                total_steps = segment_data["end_step"]
                segment_id += 1
                obs = np.stack([env._get_local_obs(i) for i in range(env.agent_num)])
            
            episode_data["total_steps"] = total_steps
            episode_data["total_segments"] = len(episode_data["segments"])
            env.use_astar_first_episode = original_astar_flag
            
            all_episode_data.append(episode_data)
        
        return all_episode_data
    
    def _collect_segment_fast(self, env, init_obs, segment_id, start_step, save_frames=False):
        """快速收集单个 Segment（可选渲染帧）"""
        segment_data = {
            "segment_id": segment_id,
            "agents": [],
            "start_step": start_step,
            "obs_list": [],
            "act_list": [],
            "frames": []
        }
        
        for i in range(env.agent_num):
            segment_data["agents"].append({
                "agent_id": i,
                "start": list(env.agents_pos[i]),
                "goal": list(env.goals_pos[i])
            })
        
        obs = init_obs
        current_step = start_step
        segment_done = False
        
        while not segment_done and current_step < self.episode_length:
            # 只在需要时渲染帧
            if save_frames:
                frame = env.render(mode="rgb_array")
                segment_data["frames"].append(frame)
            
            actions = [env._action_from_path(i) for i in range(env.agent_num)]
            segment_data["obs_list"].append(obs.copy())
            segment_data["act_list"].append(actions.copy())
            
            obs, share_obs, rewards, dones, info = env.step(actions)
            current_step += 1
            
            if info.get("looped_to_start", False):
                segment_done = True
                # 修复：渲染到达终点的帧
                if save_frames:
                    saved_agents = list(env.agents_pos)
                    saved_goals = list(env.goals_pos)
                    target_goals = [tuple(a["goal"]) for a in segment_data["agents"]]
                    env.agents_pos = target_goals
                    env.goals_pos = target_goals
                    final_frame = env.render(mode="rgb_array")
                    segment_data["frames"].append(final_frame)
                    env.agents_pos = saved_agents
                    env.goals_pos = saved_goals
        
        segment_data["end_step"] = current_step
        segment_data["astar_steps"] = current_step - start_step
        return segment_data
    
    def _bc_update_from_buffer(self):
        """从经验池采样进行 BC 更新"""
        if len(self.replay_buffer) == 0:
            return {"avg_loss": 0.0, "accuracy": 0.0, "early_stopped": False}
        
        self.trainer.prep_training()
        losses = []
        early_stopped = False
        
        for update_idx in range(self.bc_updates_per_episode):
            obs_batch, act_batch = self.replay_buffer.sample(self.bc_batch_size)
            
            if len(obs_batch) == 0:
                continue
            
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
            act_tensor = torch.tensor(act_batch.reshape(-1, 1), dtype=torch.int64, device=self.device)
            
            self.bc_optimizer.zero_grad()
            action_log_probs, _ = self.policy.actor.evaluate_actions(obs_tensor, act_tensor)
            bc_loss = -action_log_probs.mean()
            bc_loss.backward()
            
            if self.all_args.use_max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.all_args.max_grad_norm)
            self.bc_optimizer.step()
            losses.append(bc_loss.item())
            
            # 早停检查
            if (update_idx + 1) % self.early_stop_check_interval == 0:
                with torch.no_grad():
                    self.trainer.prep_rollout()
                    eval_obs, eval_acts = self.replay_buffer.sample(min(2000, len(self.replay_buffer)))
                    if len(eval_obs) > 0:
                        eval_obs_tensor = torch.tensor(eval_obs, dtype=torch.float32, device=self.device)
                        pred_actions = self.policy.act(eval_obs_tensor, deterministic=True)
                        pred_actions = pred_actions.cpu().numpy().flatten()
                        accuracy = (pred_actions == eval_acts).mean()
                        if accuracy >= self.early_stop_accuracy:
                            early_stopped = True
                            break
                self.trainer.prep_training()
        
        # 最终准确率评估
        with torch.no_grad():
            self.trainer.prep_rollout()
            eval_obs, eval_acts = self.replay_buffer.sample(min(2000, len(self.replay_buffer)))
            if len(eval_obs) > 0:
                eval_obs_tensor = torch.tensor(eval_obs, dtype=torch.float32, device=self.device)
                pred_actions = self.policy.act(eval_obs_tensor, deterministic=True)
                pred_actions = pred_actions.cpu().numpy().flatten()
                accuracy = (pred_actions == eval_acts).mean()
            else:
                accuracy = 0.0
        
        return {
            "avg_loss": np.mean(losses) if losses else 0.0,
            "accuracy": accuracy,
            "early_stopped": early_stopped
        }
    
    def _rebuild_random_map(self):
        """重建随机地图"""
        size_range = (self.map_size_min, self.map_size_max)
        density_range = (self.obstacle_density_min, self.obstacle_density_max)
        
        map_info = None
        for env in self.envs.envs:
            map_info = env.rebuild_map_random(
                size_range=size_range,
                density_range=density_range
            )
        return map_info
    
    def _run_astar_episode(self, episode_num):
        """执行 A* Episode"""
        env = self.envs.envs[0]
        original_astar_flag = env.use_astar_first_episode
        env.use_astar_first_episode = True
        obs, share_obs = env.reset()
        
        episode_data = {
            "episode": episode_num,
            "map_size": env.grid_size,
            "segments": []
        }
        
        total_steps = 0
        segment_id = 1
        
        while total_steps < self.episode_length:
            segment_data = self._collect_segment(env, obs, segment_id, total_steps)
            if segment_data is None:
                break
            episode_data["segments"].append(segment_data)
            total_steps = segment_data["end_step"]
            segment_id += 1
            obs = np.stack([env._get_local_obs(i) for i in range(env.agent_num)])
        
        episode_data["total_steps"] = total_steps
        episode_data["total_segments"] = len(episode_data["segments"])
        env.use_astar_first_episode = original_astar_flag
        return episode_data
    
    def _collect_segment(self, env, init_obs, segment_id, start_step):
        """收集单个 Segment"""
        segment_data = {
            "segment_id": segment_id,
            "agents": [],
            "start_step": start_step,
            "obs_list": [],
            "act_list": [],
            "frames": []
        }
        
        for i in range(env.agent_num):
            segment_data["agents"].append({
                "agent_id": i,
                "start": list(env.agents_pos[i]),
                "goal": list(env.goals_pos[i])
            })
        
        obs = init_obs
        current_step = start_step
        segment_done = False
        
        while not segment_done and current_step < self.episode_length:
            frame = env.render(mode="rgb_array")
            segment_data["frames"].append(frame)
            
            actions = [env._action_from_path(i) for i in range(env.agent_num)]
            segment_data["obs_list"].append(obs.copy())
            segment_data["act_list"].append(actions.copy())
            
            obs, share_obs, rewards, dones, info = env.step(actions)
            current_step += 1
            
            if info.get("looped_to_start", False):
                segment_done = True
                # 修复：渲染到达终点的帧
                saved_agents = list(env.agents_pos)
                saved_goals = list(env.goals_pos)
                target_goals = [tuple(a["goal"]) for a in segment_data["agents"]]
                env.agents_pos = target_goals
                env.goals_pos = target_goals
                final_frame = env.render(mode="rgb_array")
                segment_data["frames"].append(final_frame)
                env.agents_pos = saved_agents
                env.goals_pos = saved_goals
        
        segment_data["end_step"] = current_step
        segment_data["astar_steps"] = current_step - start_step
        return segment_data
    
    def _bc_update(self, episode_data, map_size):
        """BC 更新（分层采样 + 早停）"""
        # 添加到分层缓冲区
        self.replay_buffer.add_episode(episode_data, map_size)
        
        if len(self.replay_buffer) == 0:
            return {"avg_loss": 0.0, "accuracy": 0.0, "early_stopped": False}
        
        self.trainer.prep_training()
        losses = []
        early_stopped = False
        
        for update_idx in range(self.bc_updates_per_episode):
            obs_batch, act_batch = self.replay_buffer.sample(self.bc_batch_size)
            
            if len(obs_batch) == 0:
                continue
            
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
            act_tensor = torch.tensor(act_batch.reshape(-1, 1), dtype=torch.int64, device=self.device)
            
            self.bc_optimizer.zero_grad()
            action_log_probs, _ = self.policy.actor.evaluate_actions(obs_tensor, act_tensor)
            bc_loss = -action_log_probs.mean()
            bc_loss.backward()
            
            if self.all_args.use_max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.all_args.max_grad_norm)
            self.bc_optimizer.step()
            losses.append(bc_loss.item())
            
            # 早停检查
            if (update_idx + 1) % self.early_stop_check_interval == 0:
                with torch.no_grad():
                    self.trainer.prep_rollout()
                    eval_obs, eval_acts = self.replay_buffer.sample(min(2000, len(self.replay_buffer)))
                    if len(eval_obs) > 0:
                        eval_obs_tensor = torch.tensor(eval_obs, dtype=torch.float32, device=self.device)
                        pred_actions = self.policy.act(eval_obs_tensor, deterministic=True)
                        pred_actions = pred_actions.cpu().numpy().flatten()
                        accuracy = (pred_actions == eval_acts).mean()
                        if accuracy >= self.early_stop_accuracy:
                            early_stopped = True
                            break
                self.trainer.prep_training()
        
        # 最终准确率评估
        with torch.no_grad():
            self.trainer.prep_rollout()
            eval_obs, eval_acts = self.replay_buffer.sample(min(2000, len(self.replay_buffer)))
            if len(eval_obs) > 0:
                eval_obs_tensor = torch.tensor(eval_obs, dtype=torch.float32, device=self.device)
                pred_actions = self.policy.act(eval_obs_tensor, deterministic=True)
                pred_actions = pred_actions.cpu().numpy().flatten()
                accuracy = (pred_actions == eval_acts).mean()
            else:
                accuracy = 0.0
        
        return {
            "avg_loss": np.mean(losses) if losses else 0.0,
            "accuracy": accuracy,
            "early_stopped": early_stopped
        }
    
    def _save_episode_data(self, episode_num, episode_data):
        """保存 Episode 数据（只保留随机 N 个 GIF）"""
        ep_dir = get_episode_dir(self.dirs["episodes"], episode_num)
        render_dir = ep_dir / "astar_render"
        
        # 保存地图
        if self.use_random_map:
            env = self.envs.envs[0]
            map_info = env.save_map(ep_dir, prefix="map")
            episode_data["map_info"] = map_info
        
        # 保存 scene_config
        scene_config = create_scene_config(
            episode_data["episode"],
            episode_data["map_size"],
            episode_data["total_steps"],
            episode_data["segments"]
        )
        
        if self.use_random_map and "map_info" in episode_data:
            scene_config["map_info"] = episode_data["map_info"]
        
        # 只随机保留 N 个 GIF
        all_segments = episode_data["segments"]
        n_save = min(self.segments_per_ep, len(all_segments))
        selected_segments = random.sample(all_segments, n_save) if len(all_segments) > n_save else all_segments
        saved_seg_ids = [seg['segment_id'] for seg in selected_segments]
        
        for seg in selected_segments:
            seg_dir = render_dir / f"segment_{seg['segment_id']:03d}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            gif_path = seg_dir / f"seg_{seg['segment_id']:03d}_astar.gif"
            make_gif_from_frames(seg["frames"], gif_path, fps=self.gif_fps)
        
        scene_config["saved_gif_segments"] = saved_seg_ids
        save_json(scene_config, ep_dir / "scene_config.json")
    
    def _save_checkpoint(self, episode_num):
        """保存检查点（滚动保留）"""
        checkpoint_dir = self.dirs["checkpoints"] / f"checkpoint_ep{episode_num:03d}"
        models_dir = checkpoint_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.policy.actor.state_dict(), str(models_dir / "actor.pt"))
        torch.save(self.policy.critic.state_dict(), str(models_dir / "critic.pt"))
        
        self.checkpoint_list.append(checkpoint_dir)
        
        # 滚动删除旧检查点
        while len(self.checkpoint_list) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_list.pop(0)
            if old_checkpoint.exists():
                shutil.rmtree(old_checkpoint)
    
    def _save_best_model(self, history_rate, episode_num):
        """保存最佳模型"""
        if history_rate > self.best_history_rate:
            self.best_history_rate = history_rate
            best_dir = self.dirs["best_model"]
            
            torch.save(self.policy.actor.state_dict(), str(best_dir / "actor.pt"))
            torch.save(self.policy.critic.state_dict(), str(best_dir / "critic.pt"))
            
            info = {
                "episode": episode_num,
                "history_rate": history_rate,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_json(info, best_dir / "info.json")
            return True
        return False
    
    def _run_quick_evaluation(self, current_ep):
        """快速评估（当前 + 最近 N 个历史）"""
        print(f"\n  ┌─ 快速评估 (ep {current_ep}) ─────────────────────┐")
        
        env = self.envs.envs[0]
        results = {}
        
        # 评估回合列表
        start_ep = max(1, current_ep - self.quick_eval_history)
        eval_episodes = list(range(start_ep, current_ep + 1))
        
        for ep in eval_episodes:
            result = self._evaluate_episode(ep, env)
            if result:
                results[ep] = result
                marker = "★ 当前" if ep == current_ep else ""
                print(f"  │ ep{ep:3d}: {result['reached']}/{result['total']} ({result['rate']:.0%}) | {result['avg_ratio']:.2f}x {marker}")
        
        # 计算历史平均
        history_results = [r for ep, r in results.items() if ep < current_ep]
        if history_results:
            history_rate = np.mean([r['rate'] for r in history_results])
            current_rate = results.get(current_ep, {}).get('rate', 0)
            print(f"  │ 📊 历史平均: {history_rate:.0%} | 遗忘差: {current_rate - history_rate:+.0%}")
            
            # 保存最佳模型
            if self._save_best_model(history_rate, current_ep):
                print(f"  │ 🏆 新最佳模型已保存 (历史率: {history_rate:.0%})")
        
        print(f"  └────────────────────────────────────────────┘\n")
        
        # 保存评估结果
        eval_dir = self.dirs["evaluations"] / f"quick_eval_ep{current_ep:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        save_json({"episode": current_ep, "results": results}, eval_dir / "summary.json")
    
    def _run_full_evaluation(self, current_ep):
        """完整评估（分层采样历史）"""
        print(f"\n  ┌─ 完整评估 (ep {current_ep}) ─────────────────────┐")
        
        env = self.envs.envs[0]
        
        # 按地图大小分层采样历史回合
        small_eps = [ep for ep, size in self.episode_map_sizes.items() if ep < current_ep and size <= 14]
        medium_eps = [ep for ep, size in self.episode_map_sizes.items() if ep < current_ep and 15 <= size <= 17]
        large_eps = [ep for ep, size in self.episode_map_sizes.items() if ep < current_ep and size >= 18]
        
        samples_per_tier = self.full_eval_history // 3
        
        sampled_small = random.sample(small_eps, min(samples_per_tier, len(small_eps))) if small_eps else []
        sampled_medium = random.sample(medium_eps, min(samples_per_tier, len(medium_eps))) if medium_eps else []
        sampled_large = random.sample(large_eps, min(samples_per_tier, len(large_eps))) if large_eps else []
        
        tier_results = {'small': [], 'medium': [], 'large': []}
        
        # 评估小地图
        if sampled_small:
            print(f"  │ 小地图 (10-14): ", end="")
            for ep in sampled_small:
                result = self._evaluate_episode(ep, env)
                if result:
                    tier_results['small'].append(result)
                    print(f"{result['reached']}/{result['total']} ", end="")
            if tier_results['small']:
                avg = np.mean([r['rate'] for r in tier_results['small']])
                print(f"| 平均: {avg:.0%}")
            else:
                print()
        
        # 评估中地图
        if sampled_medium:
            print(f"  │ 中地图 (15-17): ", end="")
            for ep in sampled_medium:
                result = self._evaluate_episode(ep, env)
                if result:
                    tier_results['medium'].append(result)
                    print(f"{result['reached']}/{result['total']} ", end="")
            if tier_results['medium']:
                avg = np.mean([r['rate'] for r in tier_results['medium']])
                print(f"| 平均: {avg:.0%}")
            else:
                print()
        
        # 评估大地图
        if sampled_large:
            print(f"  │ 大地图 (18+):   ", end="")
            for ep in sampled_large:
                result = self._evaluate_episode(ep, env)
                if result:
                    tier_results['large'].append(result)
                    print(f"{result['reached']}/{result['total']} ", end="")
            if tier_results['large']:
                avg = np.mean([r['rate'] for r in tier_results['large']])
                print(f"| 平均: {avg:.0%}")
            else:
                print()
        
        # 评估当前回合
        current_result = self._evaluate_episode(current_ep, env)
        if current_result:
            print(f"  │ 当前 ep{current_ep}: {current_result['reached']}/{current_result['total']} ({current_result['rate']:.0%}) ★")
        
        # 计算总体历史平均
        all_history = tier_results['small'] + tier_results['medium'] + tier_results['large']
        if all_history:
            history_rate = np.mean([r['rate'] for r in all_history])
            current_rate = current_result['rate'] if current_result else 0
            print(f"  │")
            print(f"  │ 📊 总体历史平均: {history_rate:.0%} | 遗忘差: {current_rate - history_rate:+.0%}")
            
            # 保存最佳模型
            if self._save_best_model(history_rate, current_ep):
                print(f"  │ 🏆 新最佳模型已保存 (历史率: {history_rate:.0%})")
        
        print(f"  └────────────────────────────────────────────┘\n")
        
        # 保存评估结果
        eval_dir = self.dirs["evaluations"] / f"full_eval_ep{current_ep:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        save_json({
            "episode": current_ep,
            "tier_results": {k: [r for r in v] for k, v in tier_results.items()},
            "current_result": current_result
        }, eval_dir / "summary.json")
    
    def _evaluate_episode(self, episode_num, env):
        """评估单个历史回合"""
        ep_dir = get_episode_dir(self.dirs["episodes"], episode_num)
        scene_config_path = ep_dir / "scene_config.json"
        
        if not scene_config_path.exists():
            return None
        
        scene_config = load_json(scene_config_path)
        
        # 加载地图
        if self.use_random_map:
            map_path = ep_dir / "map.npy"
            if map_path.exists():
                env.map = np.load(str(map_path))
                env.grid_size = env.map.shape[0]
                env.map_locked = True
        
        all_segments = scene_config["segments"]
        n_eval = min(self.segments_per_ep, len(all_segments))
        selected_segments = random.sample(all_segments, n_eval) if len(all_segments) > n_eval else all_segments
        
        reached_count = 0
        step_ratios = []
        
        for seg_config in selected_segments:
            astar_steps = seg_config["astar_steps"]
            max_allowed_steps = int(astar_steps * self.astar_step_multiplier)
            
            result = self._infer_segment(env, seg_config, max_allowed_steps)
            
            if result["reached"]:
                reached_count += 1
                step_ratios.append(result["steps"] / astar_steps)
            else:
                step_ratios.append(self.astar_step_multiplier)
        
        return {
            "episode": episode_num,
            "total": n_eval,
            "reached": reached_count,
            "rate": reached_count / n_eval if n_eval > 0 else 0,
            "avg_ratio": np.mean(step_ratios) if step_ratios else 0
        }
    
    def _infer_segment(self, env, seg_config, max_steps):
        """使用策略推理 Segment"""
        starts = [tuple(a["start"]) for a in seg_config["agents"]]
        goals = [tuple(a["goal"]) for a in seg_config["agents"]]
        
        env.set_start_goal(starts, goals)
        obs = env.reset_to_segment(seg_config)
        
        target_goals = list(goals)
        steps = 0
        reached = False
        
        for step in range(max_steps):
            all_reached = all(
                env.agents_pos[i] == target_goals[i]
                for i in range(env.agent_num)
            )
            if all_reached:
                reached = True
                break
            
            self.trainer.prep_rollout()
            obs_tensor = torch.tensor(
                obs.reshape(-1, obs.shape[-1]),
                dtype=torch.float32,
                device=self.device
            )
            action = self.policy.act(obs_tensor, deterministic=True)
            actions = action.cpu().numpy().flatten().tolist()
            
            obs, share_obs, rewards, dones, info = env.step(actions)
            steps += 1
            
            if info.get("looped_to_start", False):
                reached = True
                break
        
        return {"steps": steps, "reached": reached}
    
    def _log_episode_metrics(self, ep, episode_data, bc_info):
        """记录 TensorBoard 指标"""
        self.writer.add_scalar("train/bc_loss", bc_info["avg_loss"], ep)
        self.writer.add_scalar("train/bc_accuracy", bc_info["accuracy"], ep)
        self.writer.add_scalar("data/episode_segments", episode_data["total_segments"], ep)
        self.writer.add_scalar("data/episode_total_steps", episode_data["total_steps"], ep)
        
        buffer_stats = self.replay_buffer.get_stats()
        self.writer.add_scalar("buffer/small", buffer_stats["small"], ep)
        self.writer.add_scalar("buffer/medium", buffer_stats["medium"], ep)
        self.writer.add_scalar("buffer/large", buffer_stats["large"], ep)
