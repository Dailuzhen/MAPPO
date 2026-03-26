"""
Phase A Runner 模块 (v8)

实现 Phase A（A* 在线学习）的核心训练逻辑：
- A* Episode 执行和数据收集
- 行为克隆（BC）更新 + 经验回放
- 策略推理对比
- Checkpoint 保存和统计

v8 更新：
- 增加经验回放缓冲区，避免灾难性遗忘
- 配合观测增强（目标方向+位置），提升 BC 准确率

该模块被 Runner 类调用，作为两阶段训练的第一阶段。
"""

import os
import time
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter

from utils.phase_a_utils import (
    save_json, load_json,
    make_gif_from_frames, save_frames_to_dir,
    generate_comparison_gif,
    generate_checkpoint_summary, create_scene_config,
    setup_phase_a_directories, get_checkpoint_dir, get_episode_dir,
    print_phase_a_header, print_episode_header, print_checkpoint_summary,
    calculate_bc_accuracy
)


class ExperienceReplayBuffer:
    """
    经验回放缓冲区（v8）
    
    存储历史回合的 (obs, action) 数据，支持随机采样训练
    """
    
    def __init__(self, max_size=100000):
        self.obs = []
        self.acts = []
        self.max_size = max_size
        self.episode_boundaries = []  # 记录每个回合的数据边界
    
    def add_episode(self, episode_data):
        """
        添加一个回合的数据
        
        Args:
            episode_data: 回合数据字典，包含 segments
        """
        start_idx = len(self.obs)
        
        for seg in episode_data["segments"]:
            for obs, acts in zip(seg["obs_list"], seg["act_list"]):
                for agent_id in range(len(acts)):
                    self.obs.append(obs[agent_id])
                    self.acts.append(acts[agent_id])
        
        end_idx = len(self.obs)
        self.episode_boundaries.append((start_idx, end_idx))
        
        # 限制大小（FIFO）
        if len(self.obs) > self.max_size:
            remove_count = len(self.obs) - self.max_size
            self.obs = self.obs[remove_count:]
            self.acts = self.acts[remove_count:]
            
            # 更新边界索引
            new_boundaries = []
            for start, end in self.episode_boundaries:
                new_start = max(0, start - remove_count)
                new_end = max(0, end - remove_count)
                if new_end > new_start:
                    new_boundaries.append((new_start, new_end))
            self.episode_boundaries = new_boundaries
    
    def sample(self, batch_size):
        """
        随机采样
        
        Returns:
            obs_array, act_array: numpy 数组
        """
        n_samples = min(batch_size, len(self.obs))
        indices = np.random.choice(len(self.obs), n_samples, replace=False)
        obs_batch = np.array([self.obs[i] for i in indices])
        act_batch = np.array([self.acts[i] for i in indices])
        return obs_batch, act_batch
    
    def __len__(self):
        return len(self.obs)
    
    def get_stats(self):
        """获取缓冲区统计信息"""
        return {
            "total_samples": len(self.obs),
            "num_episodes": len(self.episode_boundaries),
            "max_size": self.max_size
        }


class PhaseARunner:
    """
    Phase A 运行器：A* 在线学习
    
    负责执行 Phase A 的完整训练流程：
    1. 使用 A* 算法引导智能体，收集 (obs, action) 数据
    2. 每个 episode 后进行 BC 更新
    3. 每 N 个 episode 保存 checkpoint 并进行策略对比评估
    
    Args:
        config: 配置字典，包含:
            - all_args: 命令行参数
            - envs: 向量化环境
            - policy: MAPPO 策略
            - trainer: MAPPO 训练器
            - device: 计算设备
            - run_dir: 运行目录
    """
    
    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.policy = config['policy']
        self.trainer = config['trainer']
        self.device = config['device']
        self.run_dir = Path(config['run_dir'])
        self.num_agents = config['num_agents']
        
        # Phase A 参数（v8：经验回放 + 观测增强）
        self.phase_a_episodes = getattr(self.all_args, 'phase_a_episodes', 100)
        self.bc_lr = getattr(self.all_args, 'phase_a_bc_lr', 1e-3)
        self.bc_updates_per_episode = getattr(self.all_args, 'bc_updates_per_episode', 50)  # v8: 增加更新次数
        self.bc_batch_size = getattr(self.all_args, 'bc_batch_size', 256)
        self.comparison_segments_per_ep = getattr(self.all_args, 'comparison_segments_per_ep', 10)
        self.max_history_episodes = getattr(self.all_args, 'max_history_episodes', 20)
        self.astar_step_multiplier = getattr(self.all_args, 'astar_step_multiplier', 5.0)
        self.episode_length = max(self.all_args.episode_length, 1000)
        self.gif_fps = getattr(self.all_args, 'gif_fps', 10.0)
        self.save_frames = getattr(self.all_args, 'save_frames', False)
        
        # v8: 经验回放参数
        self.use_replay_buffer = getattr(self.all_args, 'use_replay_buffer', True)
        self.replay_buffer_size = getattr(self.all_args, 'replay_buffer_size', 100000)
        
        # 创建目录结构
        self.dirs = setup_phase_a_directories(self.run_dir)
        
        # 创建 TensorBoard writer
        self.writer = SummaryWriter(str(self.dirs["logs"]))
        
        # BC 优化器（独立学习率）
        self.bc_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(),
            lr=self.bc_lr
        )
        
        # 统计变量
        self.all_bc_losses = []
        self.all_bc_accuracies = []
        
        # v8: 初始化经验回放缓冲区
        if self.use_replay_buffer:
            self.replay_buffer = ExperienceReplayBuffer(max_size=self.replay_buffer_size)
            print(f"[Phase A v8] 经验回放缓冲区已初始化 (最大容量: {self.replay_buffer_size})")
        else:
            self.replay_buffer = None
    
    def _try_resume(self):
        """
        尝试从已有 checkpoint 恢复训练。
        返回起始 episode 编号（从 1 开始），如无可恢复的则返回 1。
        """
        episodes_dir = self.dirs["episodes"]
        if not episodes_dir.exists():
            return 1
        
        existing_eps = sorted([
            int(d.name.replace("ep_", ""))
            for d in episodes_dir.iterdir()
            if d.is_dir() and d.name.startswith("ep_")
        ])
        if not existing_eps:
            return 1
        
        last_ep = existing_eps[-1]
        
        # 找到最近的 checkpoint（模型文件）
        checkpoint_ep = (last_ep // 10) * 10
        if checkpoint_ep == 0:
            checkpoint_ep = 10 if last_ep >= 10 else 0
        
        if checkpoint_ep > 0:
            ckpt_dir = get_checkpoint_dir(self.dirs["phase_a_root"], checkpoint_ep)
            model_path = ckpt_dir / "models" / "actor.pt"
            if model_path.exists():
                self.policy.actor.load_state_dict(
                    torch.load(str(model_path), map_location=self.device)
                )
                critic_path = ckpt_dir / "models" / "critic.pt"
                if critic_path.exists():
                    self.policy.critic.load_state_dict(
                        torch.load(str(critic_path), map_location=self.device)
                    )
                print(f"[Phase A] 从 checkpoint ep{checkpoint_ep} 恢复模型")
            
            # 尝试恢复优化器状态
            optim_path = ckpt_dir / "models" / "bc_optimizer.pt"
            if optim_path.exists():
                self.bc_optimizer.load_state_dict(
                    torch.load(str(optim_path), map_location=self.device)
                )
                print(f"[Phase A] 恢复 BC 优化器状态")
        
        resume_ep = last_ep + 1
        print(f"[Phase A] 跳过已有 {last_ep} 个 episodes, 从 Episode {resume_ep} 继续训练")
        return resume_ep
    
    def run(self):
        """
        运行 Phase A 主循环（支持断点续训）
        """
        print_phase_a_header()
        start_time = time.time()
        
        start_ep = self._try_resume()
        
        for ep in range(start_ep, self.phase_a_episodes + 1):
            print_episode_header(ep, self.phase_a_episodes)
            episode_data = self._run_astar_episode(ep)
            self._save_episode_data(ep, episode_data)
            bc_info = self._bc_update(episode_data)
            self.all_bc_losses.append(bc_info['avg_loss'])
            self.all_bc_accuracies.append(bc_info['accuracy'])
            self._log_episode_metrics(ep, episode_data, bc_info)
            buffer_info = ""
            if self.use_replay_buffer and self.replay_buffer:
                buffer_stats = self.replay_buffer.get_stats()
                buffer_info = f", 经验池: {buffer_stats['total_samples']}"
            
            print(f"  Segments: {episode_data['total_segments']}, "
                  f"Steps: {episode_data['total_steps']}, "
                  f"BC Loss: {bc_info['avg_loss']:.4f}, "
                  f"Accuracy: {bc_info['accuracy']:.2%}{buffer_info}")
            if ep % 10 == 0 or ep == self.phase_a_episodes:
                self._save_checkpoint_and_compare(ep)
        
        elapsed = time.time() - start_time
        print(f"\n[Phase A] 完成! 总耗时: {elapsed/60:.1f} 分钟")
        self.writer.close()
        return True
    
    def _run_astar_episode(self, episode_num):
        """执行单个 A* Episode（单环境 envs.envs[0]，每回合切换地图）"""
        import random as _rnd
        env = self.envs.envs[0]

        map_size_min = getattr(self.all_args, 'map_size_min', 8)
        map_size_max = getattr(self.all_args, 'map_size_max', 20)
        new_size = _rnd.randint(map_size_min, map_size_max)
        env.rebuild_map_new_size(new_size)

        original_astar_flag = env.use_astar_first_episode
        original_max_steps = env.max_episode_steps
        env.use_astar_first_episode = True
        env.max_episode_steps = self.episode_length
        obs, share_obs = env.reset()
        episode_data = {"episode": episode_num, "map_size": env.grid_size, "segments": []}
        total_steps = 0
        segment_id = 1
        skip_count = 0
        max_consecutive_skips = 10
        while total_steps < self.episode_length:
            segment_data = self._collect_segment(env, obs, segment_id, total_steps)
            if segment_data is None:
                skip_count += 1
                if skip_count >= max_consecutive_skips:
                    print(f"    [WARN] 连续 {skip_count} 个 segment 失败，结束本 episode")
                    break
                obs = np.stack([env._get_local_obs(i) for i in range(env.agent_num)])
                continue
            skip_count = 0
            episode_data["segments"].append(segment_data)
            total_steps = segment_data["end_step"]
            print(f"    Segment {segment_id}: {segment_data['astar_steps']} steps "
                  f"(total: {total_steps}/{self.episode_length})")
            segment_id += 1
            obs = np.stack([env._get_local_obs(i) for i in range(env.agent_num)])
        episode_data["total_steps"] = total_steps
        episode_data["total_segments"] = len(episode_data["segments"])
        env.use_astar_first_episode = original_astar_flag
        env.max_episode_steps = original_max_steps
        return episode_data
    
    def _bc_update(self, episode_data):
        """
        使用经验回放进行 BC 更新（v8）
        
        1. 将当前回合数据加入经验池
        2. 从经验池采样训练（均匀混合历史和当前数据）
        """
        # v8: 将当前回合数据加入经验池
        if self.use_replay_buffer and self.replay_buffer:
            self.replay_buffer.add_episode(episode_data)
            
            # 如果经验池为空，返回
            if len(self.replay_buffer) == 0:
                return {"avg_loss": 0.0, "accuracy": 0.0, "grad_norm": 0.0}
            
            # 从经验池采样训练
            self.trainer.prep_training()
            losses = []
            
            for _ in range(self.bc_updates_per_episode):
                obs_batch, act_batch = self.replay_buffer.sample(self.bc_batch_size)
                
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
            
            # 计算整体准确率（采样一批数据评估）
            with torch.no_grad():
                self.trainer.prep_rollout()
                eval_obs, eval_acts = self.replay_buffer.sample(min(2000, len(self.replay_buffer)))
                eval_obs_tensor = torch.tensor(eval_obs, dtype=torch.float32, device=self.device)
                pred_actions = self.policy.act(eval_obs_tensor, deterministic=True)
                pred_actions = pred_actions.cpu().numpy().flatten()
                accuracy = (pred_actions == eval_acts).mean()
            
            return {"avg_loss": np.mean(losses), "accuracy": accuracy, "grad_norm": 0.0}
        
        else:
            # 旧版：仅使用当前回合数据
            all_obs, all_acts = [], []
            for seg in episode_data["segments"]:
                for obs, acts in zip(seg["obs_list"], seg["act_list"]):
                    for agent_id in range(len(acts)):
                        all_obs.append(obs[agent_id])
                        all_acts.append(acts[agent_id])
            if len(all_obs) == 0:
                return {"avg_loss": 0.0, "accuracy": 0.0, "grad_norm": 0.0}
            obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32, device=self.device)
            act_tensor = torch.tensor(np.array(all_acts).reshape(-1, 1), dtype=torch.int64, device=self.device)
            self.trainer.prep_training()
            losses = []
            batch_size = min(self.bc_batch_size, len(obs_tensor))
            for _ in range(self.bc_updates_per_episode):
                indices = np.random.choice(len(obs_tensor), batch_size, replace=False)
                self.bc_optimizer.zero_grad()
                action_log_probs, _ = self.policy.actor.evaluate_actions(obs_tensor[indices], act_tensor[indices])
                bc_loss = -action_log_probs.mean()
                bc_loss.backward()
                if self.all_args.use_max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.all_args.max_grad_norm)
                self.bc_optimizer.step()
                losses.append(bc_loss.item())
            with torch.no_grad():
                self.trainer.prep_rollout()
                pred_actions = self.policy.act(obs_tensor, deterministic=True)
                pred_actions = pred_actions.cpu().numpy().flatten()
                expert_actions = act_tensor.cpu().numpy().flatten()
                accuracy = (pred_actions == expert_actions).mean()
            return {"avg_loss": np.mean(losses), "accuracy": accuracy, "grad_norm": 0.0}
    
    def _log_episode_metrics(self, ep, episode_data, bc_info):
        """记录 Episode 指标到 TensorBoard"""
        self.writer.add_scalar("train/bc_loss", bc_info["avg_loss"], ep)
        self.writer.add_scalar("train/bc_accuracy", bc_info["accuracy"], ep)
        self.writer.add_scalar("data/episode_segments", episode_data["total_segments"], ep)
        self.writer.add_scalar("data/episode_total_steps", episode_data["total_steps"], ep)
    
    def _collect_segment(self, env, init_obs, segment_id, start_step):
        """
        收集单个 Segment 的数据
        
        Args:
            env: 环境实例
            init_obs: 初始观测
            segment_id: Segment 编号
            start_step: 在 episode 中的起始步
            
        Returns:
            segment_data: Segment 数据字典，或 None（如果无法继续）
        """
        segment_data = {
            "segment_id": segment_id,
            "agents": [],
            "start_step": start_step,
            "obs_list": [],
            "act_list": [],
            "frames": []
        }
        
        # 记录起点终点
        for i in range(env.agent_num):
            segment_data["agents"].append({
                "agent_id": i,
                "start": list(env.agents_pos[i]),
                "goal": list(env.goals_pos[i])
            })
        
        # 预估A*最优步数作为上限参考（防止卡死）
        max_segment_steps = min(150, self.episode_length - start_step)
        
        obs = init_obs
        current_step = start_step
        segment_done = False
        segment_steps = 0
        prev_positions = None
        stuck_count = 0
        
        while not segment_done and current_step < self.episode_length and segment_steps < max_segment_steps:
            # 渲染当前帧
            frame = env.render(mode="rgb_array")
            segment_data["frames"].append(frame)
            
            # 获取 A* 动作
            actions = [env._action_from_path(i) for i in range(env.agent_num)]
            
            # 检测是否所有智能体都选择 stay(0) — A* 无路可走
            if all(a == 0 for a in actions) and not all(
                env.agents_pos[i] == env.goals_pos[i] for i in range(env.agent_num)
            ):
                # A* 无法找到路径，跳过此 segment 并触发重新生成
                env._regenerate_start_goal()
                return None
            
            # 记录训练数据
            segment_data["obs_list"].append(obs.copy())
            segment_data["act_list"].append(actions.copy())
            
            # 检测位置卡住（连续不动）
            curr_positions = list(env.agents_pos)
            if curr_positions == prev_positions:
                stuck_count += 1
                if stuck_count >= 10:
                    env._regenerate_start_goal()
                    return None
            else:
                stuck_count = 0
            prev_positions = curr_positions
            
            # 执行动作
            obs, share_obs, rewards, dones, info = env.step(actions)
            current_step += 1
            segment_steps += 1
            
            # 检查是否完成 segment
            if info.get("all_goals_reached", False) or info.get("looped_to_start", False):
                segment_done = True
                final_frame = env.render(mode="rgb_array")
                segment_data["frames"].append(final_frame)
        
        if not segment_done:
            # segment 超时，强制重新生成起终点
            env._regenerate_start_goal()
            return None
        
        segment_data["end_step"] = current_step
        segment_data["astar_steps"] = segment_steps
        
        return segment_data
    
    def _save_episode_data(self, episode_num, episode_data):
        """
        保存 Episode 数据（scene_config 和 A* 渲染）
        
        Args:
            episode_num: Episode 编号
            episode_data: Episode 数据
        """
        ep_dir = get_episode_dir(self.dirs["episodes"], episode_num)
        render_dir = ep_dir / "astar_render"
        
        # 保存 scene_config.json
        scene_config = create_scene_config(
            episode_data["episode"],
            episode_data["map_size"],
            episode_data["total_steps"],
            episode_data["segments"]
        )
        save_json(scene_config, ep_dir / "scene_config.json")

        # 保存地图（Phase B 评估需要）
        env = self.envs.envs[0]
        np.save(str(ep_dir / "map.npy"), env.map)
        
        # 保存 A* 渲染 GIF
        for seg in episode_data["segments"]:
            seg_dir = render_dir / f"segment_{seg['segment_id']:03d}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存 GIF
            gif_path = seg_dir / f"seg_{seg['segment_id']:03d}_astar.gif"
            make_gif_from_frames(seg["frames"], gif_path, fps=self.gif_fps)
            
            # 可选：保存逐帧图像
            if self.save_frames:
                frames_dir = seg_dir / "frames"
                save_frames_to_dir(seg["frames"], frames_dir)
    
    def _save_checkpoint_and_compare(self, current_ep):
        """保存模型并进行策略对比评估（纯数据版）"""
        checkpoint_dir = get_checkpoint_dir(self.dirs["phase_a_root"], current_ep)
        models_dir = checkpoint_dir / "models"
        
        models_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(
            self.policy.actor.state_dict(),
            str(models_dir / "actor.pt")
        )
        torch.save(
            self.policy.critic.state_dict(),
            str(models_dir / "critic.pt")
        )
        torch.save(
            self.bc_optimizer.state_dict(),
            str(models_dir / "bc_optimizer.pt")
        )
        print(f"  模型已保存: {models_dir}")
        
        comparison_results = self._run_policy_comparison_all_episodes(current_ep, checkpoint_dir)
        
        # 保存评估结果 JSON
        save_json(comparison_results, checkpoint_dir / "eval_results.json")
        
        # 记录到 TensorBoard
        self._log_comparison_metrics(current_ep, comparison_results)
        
        # 打印汇总表格
        self._print_comparison_results(current_ep, comparison_results)
    
    def _run_policy_comparison_all_episodes(self, current_ep, comparison_dir):
        """
        对所有历史回合进行策略对比（纯数据版，无 GIF）
        每个 episode 取 10 个 segment，使用原始地图和场景进行对比。
        """
        import random
        
        env = self.envs.envs[0]
        
        start_ep = max(1, current_ep - self.max_history_episodes + 1)
        episodes_to_compare = list(range(start_ep, current_ep + 1))
        total_segments = len(episodes_to_compare) * self.comparison_segments_per_ep
        
        print(f"\n  策略对比评估（限制: A*×{self.astar_step_multiplier}, "
              f"共 {len(episodes_to_compare)} 回合 × {self.comparison_segments_per_ep} segments）")
        
        comparison_results = {}
        grand_reached = 0
        grand_total = 0
        
        for ep in episodes_to_compare:
            ep_dir = get_episode_dir(self.dirs["episodes"], ep)
            scene_config_path = ep_dir / "scene_config.json"
            
            if not scene_config_path.exists():
                continue
            
            scene_config = load_json(scene_config_path)
            all_segments = scene_config["segments"]
            
            # 加载该回合对应的地图
            map_path = ep_dir / "map.npy"
            if map_path.exists():
                saved_map = np.load(str(map_path))
                env.grid_size = saved_map.shape[0]
                env.map = saved_map
            
            n_compare = min(self.comparison_segments_per_ep, len(all_segments))
            selected_segments = random.sample(all_segments, n_compare)
            
            reached_count = 0
            timeout_count = 0
            collision_count = 0
            step_ratios = []
            
            for seg_config in selected_segments:
                astar_steps = seg_config["astar_steps"]
                max_allowed_steps = int(astar_steps * self.astar_step_multiplier)
                
                seg_result = self._infer_segment_with_limit(env, seg_config, max_allowed_steps)
                
                if seg_result["reached"]:
                    reached_count += 1
                    step_ratios.append(seg_result["steps"] / astar_steps)
                if seg_result["timeout"]:
                    timeout_count += 1
                if seg_result["collision"]:
                    collision_count += 1
            
            reach_rate = reached_count / n_compare if n_compare > 0 else 0
            avg_step_ratio = np.mean(step_ratios) if step_ratios else 0
            
            grand_reached += reached_count
            grand_total += n_compare
            
            print(f"    ep{ep:<3d}: {reached_count}/{n_compare} ({reach_rate:.0%})  "
                  f"avg_ratio={avg_step_ratio:.2f}x  "
                  f"(map:{env.grid_size}x{env.grid_size})")
            
            comparison_results[f"ep{ep:03d}"] = {
                "episode": ep,
                "total_compared": n_compare,
                "reached": reached_count,
                "timeout": timeout_count,
                "collision": collision_count,
                "reach_rate": reach_rate,
                "avg_step_ratio": avg_step_ratio,
            }
        
        grand_rate = grand_reached / grand_total if grand_total > 0 else 0
        print(f"  ────────────────────────────────────")
        print(f"  总计: {grand_reached}/{grand_total} ({grand_rate:.1%})")
        
        return comparison_results
    
    def _run_policy_comparison(self, start_ep, end_ep, comparison_dir):
        """
        从 Episode 1~end_ep 随机选择 N 个 segment 进行策略推理对比
        
        Args:
            start_ep: 起始 episode
            end_ep: 结束 episode
            comparison_dir: 对比结果保存目录
            
        Returns:
            comparison_stats: 统计数据
        """
        import random
        
        env = self.envs.envs[0]
        
        stats = {
            "total_segments": 0,
            "reached": 0,
            "timeout": 0,
            "collision": 0,
            "total_astar_steps": 0,
            "total_policy_steps": 0,
            "selected_from_episodes": []
        }
        
        # 收集所有可用的 (episode, segment) 对
        all_candidates = []
        for ep in range(1, end_ep + 1):
            ep_dir = get_episode_dir(self.dirs["episodes"], ep)
            scene_config_path = ep_dir / "scene_config.json"
            if scene_config_path.exists():
                scene_config = load_json(scene_config_path)
                for seg in scene_config["segments"]:
                    all_candidates.append({
                        "episode": ep,
                        "segment": seg,
                        "ep_dir": ep_dir
                    })
        
        total_candidates = len(all_candidates)
        
        # 随机选择 N 个 segment
        n_compare = min(self.comparison_segments, total_candidates)
        selected = random.sample(all_candidates, n_compare)
        
        # 按 episode 和 segment_id 排序
        selected.sort(key=lambda x: (x["episode"], x["segment"]["segment_id"]))
        
        print(f"\n  开始策略对比评估 (从 ep1-{end_ep} 随机选择 {n_compare} 个 segment)...")
        
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        all_details = []
        
        for item in selected:
            ep = item["episode"]
            seg_config = item["segment"]
            ep_dir = item["ep_dir"]
            
            seg_result = self._infer_segment(env, seg_config)
            
            seg_id = seg_config["segment_id"]
            seg_comparison_dir = comparison_dir / f"ep{ep:03d}_seg{seg_id:03d}"
            seg_comparison_dir.mkdir(parents=True, exist_ok=True)
            
            # 提取起始点和终点
            starts = [tuple(a["start"]) for a in seg_config["agents"]]
            goals = [tuple(a["goal"]) for a in seg_config["agents"]]
            
            # 保存策略 GIF
            policy_gif_path = seg_comparison_dir / f"ep{ep:03d}_seg{seg_id:03d}_policy.gif"
            make_gif_from_frames(seg_result["frames"], policy_gif_path, fps=self.gif_fps)
            
            # 加载 A* 帧并生成对比 GIF
            astar_render_dir = ep_dir / "astar_render"
            astar_seg_dir = astar_render_dir / f"segment_{seg_id:03d}"
            astar_gif_path = astar_seg_dir / f"seg_{seg_id:03d}_astar.gif"
            
            if astar_gif_path.exists():
                try:
                    import imageio.v2 as imageio
                except ImportError:
                    import imageio
                astar_frames = list(imageio.mimread(str(astar_gif_path)))
                
                compare_gif_path = seg_comparison_dir / f"ep{ep:03d}_seg{seg_id:03d}_compare.gif"
                generate_comparison_gif(
                    astar_frames, seg_result["frames"],
                    compare_gif_path,
                    ep, seg_id,
                    seg_config["astar_steps"], seg_result["steps"],
                    seg_result["reached"],
                    fps=self.gif_fps,
                    starts=starts,
                    goals=goals
                )
            
            # 更新统计
            seg_detail = {
                "episode": ep,
                "segment_id": seg_id,
                "starts": starts,
                "goals": goals,
                "astar_steps": seg_config["astar_steps"],
                "policy_steps": seg_result["steps"],
                "reached": seg_result["reached"],
                "timeout": seg_result["timeout"],
                "collision": seg_result["collision"]
            }
            all_details.append(seg_detail)
            
            stats["total_segments"] += 1
            stats["total_astar_steps"] += seg_config["astar_steps"]
            
            status = "✓" if seg_result["reached"] else "✗"
            print(f"    ep{ep}-seg{seg_id}: {status} (A*:{seg_config['astar_steps']} Policy:{seg_result['steps']})")
            
            if seg_result["reached"]:
                stats["reached"] += 1
                stats["total_policy_steps"] += seg_result["steps"]
            if seg_result["timeout"]:
                stats["timeout"] += 1
            if seg_result["collision"]:
                stats["collision"] += 1
        
        # 计算成功率
        reach_rate = stats["reached"] / max(1, n_compare)
        
        # 保存汇总
        summary = {
            "total_compared": n_compare,
            "reached": stats["reached"],
            "timeout": stats["timeout"],
            "collision": stats["collision"],
            "reach_rate": round(reach_rate, 4),
            "details": all_details
        }
        save_json(summary, comparison_dir / "comparison_summary.json")
        
        print(f"  对比结果: {stats['reached']}/{n_compare} reached ({reach_rate*100:.1f}%)")
        
        return stats
    
    def _infer_segment_with_limit(self, env, seg_config, max_steps):
        """
        使用策略推理单个 Segment（纯数据版，无渲染）
        """
        saved_astar_flag = env.use_astar_first_episode
        env.use_astar_first_episode = False
        env._stuck_counter = 0
        
        starts = [tuple(a["start"]) for a in seg_config["agents"]]
        goals = [tuple(a["goal"]) for a in seg_config["agents"]]
        
        env.set_start_goal(starts, goals)
        obs = env.reset_to_segment(seg_config)
        
        target_goals = list(goals)
        steps = 0
        reached = False
        timeout = False
        collision = False
        
        for step in range(max_steps):
            all_at_goal = all(
                env.agents_pos[i] == target_goals[i]
                for i in range(env.agent_num)
            )
            if all_at_goal:
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
            
            if any(info.get("collision", [])):
                collision = True
            
            if info.get("all_goals_reached", False):
                reached = True
                break
            
            # 卡住重置但非真正到达 → 目标已被篡改，视为失败
            if info.get("looped_to_start", False) and not info.get("all_goals_reached", False):
                break
        
        if not reached:
            timeout = True
        
        env.use_astar_first_episode = saved_astar_flag
        
        return {
            "steps": steps,
            "reached": reached,
            "timeout": timeout,
            "collision": collision
        }
    
    def _infer_segment(self, env, seg_config):
        """
        使用策略推理单个 Segment（旧版，保留兼容）
        
        策略从指定起点出发，尝试到达终点。
        到达终点即停止，不重置环境，以便与 A* 路径进行公平对比。
        
        Args:
            env: 环境实例
            seg_config: Segment 配置
            
        Returns:
            result: 包含 frames, steps, reached, timeout, collision 的字典
        """
        # 设置环境到指定场景
        starts = [tuple(a["start"]) for a in seg_config["agents"]]
        goals = [tuple(a["goal"]) for a in seg_config["agents"]]
        
        env.set_start_goal(starts, goals)
        obs = env.reset_to_segment(seg_config)
        
        # 保存目标位置（因为 env.step 可能会在到达后重置）
        target_goals = list(goals)
        
        frames = []
        steps = 0
        reached = False
        timeout = False
        collision = False
        
        # 渲染初始帧
        frame = env.render(mode="rgb_array")
        frames.append(frame)
        
        max_steps = getattr(self, 'comparison_max_steps', 100)
        for step in range(max_steps):
            # 检查是否全部到达目标（使用保存的目标位置）
            all_reached = all(
                env.agents_pos[i] == target_goals[i]
                for i in range(env.agent_num)
            )
            if all_reached:
                reached = True
                break
            
            # 策略推理
            self.trainer.prep_rollout()
            obs_tensor = torch.tensor(
                obs.reshape(-1, obs.shape[-1]),
                dtype=torch.float32,
                device=self.device
            )
            action = self.policy.act(obs_tensor, deterministic=True)
            actions = action.cpu().numpy().flatten().tolist()
            
            # 执行动作
            obs, share_obs, rewards, dones, info = env.step(actions)
            steps += 1
            
            # 检查碰撞
            if any(info.get("collision", [])):
                collision = True
            
            # 检查是否刚刚到达终点（env.step 内部会重置，但我们用保存的目标判断）
            # 如果 looped_to_start 为 True，说明所有智能体刚刚到达终点
            if info.get("looped_to_start", False):
                reached = True
                # 渲染到达终点时的帧（在重置之前的状态已经丢失，用当前帧近似）
                # 实际上这个帧是重置后的，但我们标记已到达
                break
            
            # 渲染当前帧
            frame = env.render(mode="rgb_array")
            frames.append(frame)
        
        # 添加最终帧
        if reached:
            # 到达终点，添加一帧表示成功
            final_frame = env.render(mode="rgb_array")
            frames.append(final_frame)
        else:
            # 超时，添加最终帧
            final_frame = env.render(mode="rgb_array")
            frames.append(final_frame)
        
        if not reached:
            timeout = True
        
        return {
            "frames": frames,
            "steps": steps,
            "reached": reached,
            "timeout": timeout,
            "collision": collision
        }
    
    def _generate_segment_comparison_gif(self, episode, ep_dir, seg_config, seg_result, output_path):
        """生成单个 segment 的对比 GIF"""
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio
        
        seg_id = seg_config["segment_id"]
        astar_render_dir = ep_dir / "astar_render"
        astar_seg_dir = astar_render_dir / f"segment_{seg_id:03d}"
        astar_gif_path = astar_seg_dir / f"seg_{seg_id:03d}_astar.gif"
        
        if not astar_gif_path.exists():
            return
        
        # 加载 A* 帧
        astar_frames = list(imageio.mimread(str(astar_gif_path)))
        
        # 生成对比 GIF
        starts = [tuple(a["start"]) for a in seg_config["agents"]]
        goals = [tuple(a["goal"]) for a in seg_config["agents"]]
        
        generate_comparison_gif(
            astar_frames, seg_result["frames"],
            output_path,
            episode, seg_id,
            seg_config["astar_steps"], seg_result["steps"],
            seg_result["reached"],
            fps=self.gif_fps,
            starts=starts,
            goals=goals
        )
    
    def _generate_comparison_summary(self, current_ep, comparison_results):
        """生成对比总结"""
        summary = {
            "current_episode": current_ep,
            "comparison_results": comparison_results,
            "statistics": {}
        }
        
        # 计算当前和历史的统计
        if comparison_results:
            current_key = f"ep{current_ep:03d}"
            if current_key in comparison_results:
                summary["statistics"]["current_reach_rate"] = comparison_results[current_key]["reach_rate"]
            
            # 历史平均（不包括当前）
            history_rates = [
                res["reach_rate"] for key, res in comparison_results.items()
                if res["episode"] < current_ep
            ]
            if history_rates:
                summary["statistics"]["history_avg_reach_rate"] = np.mean(history_rates)
                summary["statistics"]["oldest_reach_rate"] = history_rates[0] if history_rates else 0
                summary["statistics"]["forgetting_gap"] = (
                    summary["statistics"]["current_reach_rate"] - summary["statistics"]["history_avg_reach_rate"]
                )
        
        return summary
    
    def _log_comparison_metrics(self, current_ep, comparison_results):
        """记录对比指标到 TensorBoard"""
        for key, result in comparison_results.items():
            ep = result["episode"]
            self.writer.add_scalar(f"eval/ep{ep:03d}_reach_rate", result["reach_rate"], current_ep)
            self.writer.add_scalar(f"eval/ep{ep:03d}_avg_step_ratio", result["avg_step_ratio"], current_ep)
        
        # 记录汇总指标
        current_key = f"ep{current_ep:03d}"
        if current_key in comparison_results:
            self.writer.add_scalar("eval/current_reach_rate", 
                                 comparison_results[current_key]["reach_rate"], current_ep)
        
        # 历史平均
        history_rates = [
            res["reach_rate"] for key, res in comparison_results.items()
            if res["episode"] < current_ep
        ]
        if history_rates:
            history_avg = np.mean(history_rates)
            self.writer.add_scalar("eval/history_avg_reach_rate", history_avg, current_ep)
            self.writer.add_scalar("eval/oldest_reach_rate", history_rates[0], current_ep)
            
            current_rate = comparison_results[current_key]["reach_rate"]
            forgetting_gap = current_rate - history_avg
            self.writer.add_scalar("eval/forgetting_gap", forgetting_gap, current_ep)
    
    def _print_comparison_results(self, current_ep, comparison_results):
        """打印对比结果"""
        print(f"\n  ┌─────────┬──────────┬──────────┐")
        print(f"  │  回合   │  到达率   │ 平均倍率  │")
        print(f"  ├─────────┼──────────┼──────────┤")
        
        # 按回合倒序排列（最新的在上面）
        sorted_results = sorted(comparison_results.items(), 
                              key=lambda x: x[1]["episode"], 
                              reverse=True)
        
        for key, result in sorted_results:
            ep = result["episode"]
            reach_rate = result["reach_rate"]
            avg_ratio = result["avg_step_ratio"]
            reached = result["reached"]
            total = result["total_compared"]
            
            marker = "  ★ 当前" if ep == current_ep else ""
            if ep == min(r["episode"] for r in comparison_results.values()):
                marker = "  ← 最旧"
            
            print(f"  │   ep{ep:<3d} │  {reached}/{total:<2d}   │  {avg_ratio:.2f}x   │{marker}")
        
        print(f"  └─────────┴──────────┴──────────┘")
        
        # 打印汇总
        current_key = f"ep{current_ep:03d}"
        if current_key in comparison_results:
            current_rate = comparison_results[current_key]["reach_rate"]
            
            history_rates = [
                res["reach_rate"] for key, res in comparison_results.items()
                if res["episode"] < current_ep
            ]
            
            if history_rates:
                history_avg = np.mean(history_rates)
                forgetting_gap = current_rate - history_avg
                print(f"  📊 当前: {current_rate:.0%} | 历史平均: {history_avg:.0%} | 遗忘差: {forgetting_gap:+.0%}")
            else:
                print(f"  📊 当前: {current_rate:.0%}")
    
    def _log_checkpoint_metrics(self, checkpoint_ep, summary):
        """记录 Checkpoint 指标到 TensorBoard（旧版，保留兼容）"""
        stats = summary["policy_statistics"]
        
        self.writer.add_scalar("eval/reach_rate", stats["reach_rate"], checkpoint_ep)
        self.writer.add_scalar("eval/timeout_rate", stats["timeout_rate"], checkpoint_ep)
        self.writer.add_scalar("eval/avg_extra_steps", stats["avg_extra_steps"], checkpoint_ep)
        self.writer.add_scalar("eval/avg_extra_steps_ratio", stats["avg_extra_steps_ratio"], checkpoint_ep)
        
        if stats["avg_policy_steps"] > 0:
            self.writer.add_scalar("eval/avg_policy_steps", stats["avg_policy_steps"], checkpoint_ep)
