"""
Phase B Runner: PPO 强化学习训练（v9 改进版）

基于 Phase A 预训练模型进行 PPO 训练，带有：
- A* 奖励塑形（不覆盖动作）
- 5 阶段课程学习（v9：地图 8-24，距离递进）
- 独立验证 + 早停（v9）
- 势能差分奖励塑形 PBRS（v9）
"""
import os, json, time, numpy as np, torch
from pathlib import Path
from tensorboardX import SummaryWriter
try:
    import imageio
except ImportError:
    imageio = None

def _t2n(x):
    """将 PyTorch 张量转换为 NumPy 数组"""
    return x.detach().cpu().numpy()


class PhaseBRunner:
    """Phase B Runner: PPO 强化学习训练（v9 改进版）"""

    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.policy = config["policy"]
        self.trainer = config["trainer"]
        self.buffer = config["buffer"]
        self.device = config["device"]
        self.run_dir = Path(config["run_dir"])
        self.num_agents = config["num_agents"]
        self.episodes = getattr(self.all_args, "phase_b_episodes", 500)
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.eval_interval = getattr(self.all_args, "phase_b_eval_interval", 50)
        self.eval_scenarios = getattr(self.all_args, "phase_b_eval_scenarios", 20)
        self.save_interval = self.eval_interval
        self.log_interval = getattr(self.all_args, "log_interval", 10)
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.use_centralized_V = self.all_args.use_centralized_V
        self.phase_b_dir = self.run_dir / "phase_b_ppo"
        self.phase_b_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.phase_b_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.eval_history = []
        self.best_success_rate = -1.0
        self.best_episode = -1
        
        # v9: 独立验证 + 早停
        self.val_interval = getattr(self.all_args, "val_interval", 0)
        self.val_scenarios = getattr(self.all_args, "val_scenarios", 100)
        self.val_patience = getattr(self.all_args, "val_patience", 3)
        self.curriculum_stages = getattr(self.all_args, "curriculum_stages", 4)
        self.map_size_max_train = getattr(self.all_args, "map_size_max", 20)
        self.val_history = []
        self.best_val_success_rate = -1.0
        self.best_val_episode = -1
        self._val_no_improve_count = 0

        # 训练诊断：地图尺寸分布 + reward 分量追踪
        self._map_size_counter = {}
        self._reward_component_sums = {"goal": 0.0, "collision": 0.0, "distance": 0.0,
                                        "astar": 0.0, "anti_stuck": 0.0, "pbrs": 0.0,
                                        "stay": 0.0, "wall": 0.0}
        self._reward_component_count = 0

    def _try_resume_phase_b(self):
        """
        扫描已有 Phase B checkpoint，加载最新的模型权重，返回起始 episode。
        """
        if not self.phase_b_dir.exists():
            return 1

        existing = sorted([
            int(d.name.replace("checkpoint_ep", ""))
            for d in self.phase_b_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint_ep")
        ])
        if not existing or max(existing) == 0:
            return 1

        last_ep = max(existing)
        ckpt_dir = self.phase_b_dir / f"checkpoint_ep{last_ep:03d}"
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
            print(f"[Phase B] 从 checkpoint ep{last_ep} 恢复模型")

        resume_ep = last_ep + 1
        print(f"[Phase B] 跳过已有 {last_ep} episodes, 从 Episode {resume_ep} 继续训练")
        return resume_ep

    def run(self):
        """运行 Phase B 训练（支持断点续训 + v9 独立验证 + 早停）"""
        print("\n══════════════════════════════════════════════════════════════════════")
        print("                    Phase B: PPO 训练")
        print("══════════════════════════════════════════════════════════════════════")
        print(f"  预训练模型: {self.all_args.model_dir}")
        print(f"  训练回合数: {self.episodes}")
        print(f"  评估间隔: 每 {self.eval_interval} 回合 (自动保存模型)")
        print(f"  并行环境: {self.n_rollout_threads}")
        print(f"  网络结构: {self.all_args.hidden_size} × {self.all_args.layer_N} 层")
        if self.val_interval > 0:
            print(f"  独立验证: 每 {self.val_interval} 回合 ({self.val_scenarios} 场景, 早停耐心={self.val_patience})")
            map_min = getattr(self.all_args, 'map_size_min', 8)
            print(f"  课程阶段: {self.curriculum_stages} 阶段, 地图 {map_min}-{self.map_size_max_train}")
        print("══════════════════════════════════════════════════════════════════════\n")
        self._disable_astar_guidance()

        start_ep = self._try_resume_phase_b()

        if start_ep <= 1:
            print("──────────────────────────────────────────────────────────────────────")
            print("[Phase B] Episode 0/{} - 基线评估".format(self.episodes))
            print("──────────────────────────────────────────────────────────────────────")
            baseline_result = self._evaluate(episode=0, is_baseline=True)
            self._save_checkpoint(0, baseline_result, is_baseline=True)
            self.eval_history.append((0, baseline_result))

        self._warmup()
        start_time = time.time()
        early_stopped = False
        for episode in range(start_ep, self.episodes + 1):
            self._update_curriculum(episode)
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, self.episodes)
            episode_reward = self._collect_rollout()
            self.buffer.compute_returns(self._get_next_values(), self.trainer.value_normalizer)
            train_info = self.trainer.train(self.buffer)
            train_info["explained_variance"] = self._compute_explained_variance()
            self.buffer.after_update()
            total_steps = episode * self.episode_length * self.n_rollout_threads
            if episode % self.log_interval == 0:
                self._log_training(episode, train_info, episode_reward, total_steps, start_time)
            if episode % self.eval_interval == 0 or episode == self.episodes:
                print("\n──────────────────────────────────────────────────────────────────────")
                print(f"[Phase B] Episode {episode}/{self.episodes} - 训练评估 & 保存")
                print("──────────────────────────────────────────────────────────────────────")
                eval_result = self._evaluate(episode)
                self._save_checkpoint(episode, eval_result)
                self.eval_history.append((episode, eval_result))
                if self.val_interval <= 0:
                    if eval_result["success_rate"] > self.best_success_rate:
                        self.best_success_rate = eval_result["success_rate"]
                        self.best_episode = episode
                        self._save_best_model(episode, eval_result["success_rate"], source="train_eval")
                        print(f"  ★ 新最佳模型! ep{episode} ({eval_result['success_rate']*100:.1f}%)")
                self.writer.add_scalar("eval/success_rate", eval_result["success_rate"], total_steps)
                self.writer.add_scalar("eval/avg_steps", eval_result["avg_steps"], total_steps)
                self.writer.add_scalar("eval/avg_step_ratio", eval_result["avg_step_ratio"], total_steps)
                self.writer.add_scalar("eval/timeout_rate", eval_result["timeout_rate"], total_steps)
                self.writer.add_scalar("eval/collision_rate", eval_result["collision_rate"], total_steps)
                for bk, s in eval_result.get("size_stats", {}).items():
                    r = s["success"] / s["total"] if s["total"] > 0 else 0
                    self.writer.add_scalar(f"eval/{bk}", r, total_steps)

            # v9: 独立验证 + 早停
            if self.val_interval > 0 and episode % self.val_interval == 0 and episode > 0:
                print("\n──────────────────────────────────────────────────────────────────────")
                print(f"[Phase B] Episode {episode}/{self.episodes} - 独立验证 ({self.val_scenarios}场景, A*完全禁用)")
                print("──────────────────────────────────────────────────────────────────────")
                val_result = self._independent_validate(episode)
                self.val_history.append((episode, val_result))
                self.writer.add_scalar("val/success_rate", val_result["success_rate"], total_steps)
                self.writer.add_scalar("val/timeout_rate", val_result["timeout_rate"], total_steps)
                self.writer.add_scalar("val/collision_rate", val_result["collision_rate"], total_steps)
                for bk, s in val_result.get("size_stats", {}).items():
                    r = s["success"] / s["total"] if s["total"] > 0 else 0
                    self.writer.add_scalar(f"val/{bk}", r, total_steps)
                
                self._save_validation_history()
                
                if val_result["success_rate"] > self.best_val_success_rate:
                    self.best_val_success_rate = val_result["success_rate"]
                    self.best_val_episode = episode
                    self._val_no_improve_count = 0
                    self._save_best_model(episode, val_result["success_rate"], source="val")
                    print(f"  ★ 新验证最佳! ep{episode} ({val_result['success_rate']*100:.1f}%)")
                else:
                    self._val_no_improve_count += 1
                    print(f"  [验证最佳: ep{self.best_val_episode} ({self.best_val_success_rate*100:.1f}%)]")
                    if self._val_no_improve_count >= self.val_patience:
                        print(f"\n[早停] 连续{self.val_patience}次验证无提升，在 ep{episode} 停止训练")
                        early_stopped = True
                        break

        self._print_summary(start_time, early_stopped)
        return True

    def _disable_astar_guidance(self):
        """
        配置 Phase B 环境：
        - 禁用 A* 动作覆盖（模型自主决策）
        - 启用 A* 奖励塑形（提供密集引导信号）
        """
        for env in self.envs.envs:
            env.use_astar_first_episode = False
            env.use_astar_shaping = True
            env.progress_reward = 0.2
            env.no_progress_penalty = -0.05
            env.episode_count = 0
            env.map_locked = False
            env.fixed_first_episode_only = True
            env.fixed_first_episode_used = True

    def _update_curriculum(self, episode):
        """课程学习：分层均衡采样 + A* 距离递进（支持 1/4/5 阶段）"""
        import random as _rnd

        if self.curriculum_stages == 1:
            map_min = getattr(self.all_args, 'map_size_min', 8)
            sizes_all = list(range(map_min, self.map_size_max_train + 1))
            buckets = [sizes_all]
            max_dist = None
            stage = 1
        elif self.curriculum_stages == 2:
            if episode <= self.episodes * 0.40:
                buckets = [[8, 9, 10, 11, 12], [13, 14, 15]]
                max_dist = 20
                stage = 1
            else:
                sizes_all = list(range(8, self.map_size_max_train + 1))
                buckets = [sizes_all]
                max_dist = None
                stage = 2
        elif self.curriculum_stages == 5:
            if episode <= self.episodes * 0.20:
                buckets = [[8, 9, 10], [11, 12, 13]]
                max_dist = 10
                stage = 1
            elif episode <= self.episodes * 0.40:
                buckets = [[8, 9, 10], [11, 12, 13, 14], [15, 16, 17]]
                max_dist = 20
                stage = 2
            elif episode <= self.episodes * 0.60:
                buckets = [[8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21]]
                max_dist = 35
                stage = 3
            elif episode <= self.episodes * 0.80:
                buckets = [[8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22], [23, 24]]
                max_dist = 50
                stage = 4
            else:
                sizes_all = list(range(8, self.map_size_max_train + 1))
                buckets = [sizes_all]
                max_dist = None
                stage = 5
        else:
            if episode <= self.episodes * 0.25:
                buckets = [[8, 9, 10], [11, 12]]
                max_dist = 10
                stage = 1
            elif episode <= self.episodes * 0.50:
                buckets = [[8, 9, 10], [11, 12, 13], [14, 15, 16]]
                max_dist = 18
                stage = 2
            elif episode <= self.episodes * 0.75:
                buckets = [[8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18, 19, 20]]
                max_dist = 30
                stage = 3
            else:
                buckets = [[8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]]
                max_dist = None
                stage = 4
        
        if not hasattr(self, '_last_stage') or self._last_stage != stage:
            max_size = max(max(b) for b in buckets)
            dist_str = str(max_dist) if max_dist else "无限制"
            min_size = min(min(b) for b in buckets)
            print(f"\n[课程阶段 {stage}/{self.curriculum_stages}] 地图: {min_size}-{max_size}, max_dist: {dist_str}")
            self._last_stage = stage

        for env in self.envs.envs:
            env.max_astar_distance = max_dist
            bucket = _rnd.choice(buckets)
            new_size = _rnd.choice(bucket)
            env.rebuild_map_new_size(new_size)
            env.map_locked = False
            self._map_size_counter[new_size] = self._map_size_counter.get(new_size, 0) + 1

        obs = self.envs.reset()
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    def _warmup(self):
        """环境热身 - 确保所有环境正确初始化"""
        for env in self.envs.envs:
            env.fixed_first_episode_only = True
            env.fixed_first_episode_used = True
            env.map_locked = False
            env.start_pos = None
            env.fixed_goals = None

        obs = self.envs.reset()
        for i, env in enumerate(self.envs.envs):
            if env.agents_pos is None:
                raise RuntimeError(f"[Phase B] 环境 {i} 初始化失败: agents_pos is None")

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat((self.num_agents), axis=1)
        else:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    def _collect_rollout(self):
        """收集一个回合的数据"""
        episode_rewards = []
        for step in range(self.episode_length):
            values, actions, action_log_probs, actions_env = self._collect_step(step)
            obs, rewards, dones, infos = self.envs.step(actions_env)
            episode_rewards.append(rewards)
            self._insert_buffer(obs, rewards, dones, values, actions, action_log_probs)

            for info in infos:
                if isinstance(info, dict):
                    rc_list = info.get("reward_components", [])
                    for rc in rc_list:
                        if isinstance(rc, dict):
                            for k in self._reward_component_sums:
                                self._reward_component_sums[k] += rc.get(k, 0.0)
                            self._reward_component_count += 1

        return np.mean(episode_rewards) * self.episode_length

    @torch.no_grad()
    def _collect_step(self, step):
        """采样单步动作"""
        self.trainer.prep_rollout()
        value, action, action_log_prob = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]), np.concatenate(self.buffer.obs[step]))
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        return (
         values, actions, action_log_probs, actions_env)

    def _insert_buffer(self, obs, rewards, dones, values, actions, action_log_probs):
        """插入数据到缓冲区"""
        if rewards.ndim == 2:
            rewards = rewards[..., np.newaxis]

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def _get_next_values(self):
        """获取下一状态价值"""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]))
        return np.array(np.split(_t2n(next_values), self.n_rollout_threads))

    def _evaluate(self, episode, is_baseline=False):
        """训练评估：使用课程匹配的场景"""
        import random as _rnd
        env = self.envs.envs[0]
        self.trainer.prep_rollout()

        saved_astar_flag = env.use_astar_first_episode
        saved_shaping_flag = env.use_astar_shaping
        env.use_astar_first_episode = False
        env.use_astar_shaping = False

        if self.curriculum_stages == 1:
            size_min = getattr(self.all_args, 'map_size_min', 8)
            size_max = self.map_size_max_train
        elif self.curriculum_stages == 2:
            if episode <= self.episodes * 0.40:
                size_min, size_max = 8, 15
            else:
                size_min, size_max = 8, self.map_size_max_train
        elif self.curriculum_stages == 5:
            if episode <= self.episodes * 0.20:
                size_min, size_max = 8, 13
            elif episode <= self.episodes * 0.40:
                size_min, size_max = 8, 17
            elif episode <= self.episodes * 0.60:
                size_min, size_max = 8, 21
            else:
                size_min, size_max = 8, self.map_size_max_train
        else:
            if episode <= self.episodes * 0.25:
                size_min, size_max = 8, 12
            elif episode <= self.episodes * 0.50:
                size_min, size_max = 8, 16
            elif episode <= self.episodes * 0.75:
                size_min, size_max = 8, 20
            else:
                size_min, size_max = 8, 22

        large_label = f"large(18-{size_max})" if size_max > 17 else "large(17+)"
        bucket_names = ["small(8-12)", "medium(13-17)", large_label]

        result = self._run_eval_scenarios(
            env, self.eval_scenarios, size_min, size_max, bucket_names
        )
        result["episode"] = episode
        result["type"] = "baseline" if is_baseline else "training"

        print(f"  训练评估: {result['successes']}/{result['total_scenarios']} "
              f"({result['success_rate'] * 100:.1f}%)  "
              f"ratio={result['avg_step_ratio']:.2f}x  "
              f"timeout={result['timeouts']}  collision={result['collisions']}")
        diag = result.get("diag", {})
        for bk in bucket_names:
            if bk in result.get("size_stats", {}):
                s = result["size_stats"][bk]
                r = s["success"] / s["total"] * 100 if s["total"] > 0 else 0
                d = diag.get(bk, {})
                extra = ""
                if s["timeout"] > 0:
                    extra += f"  timeout_dist={d.get('avg_timeout_dist', 0):.1f}"
                extra += f"  stay={d.get('avg_stay_ratio', 0)*100:.0f}%"
                extra += f"  loop={d.get('avg_loop_rate', 0)*100:.0f}%"
                print(f"    {bk}: {s['success']}/{s['total']} ({r:.0f}%){extra}")

        env.use_astar_first_episode = saved_astar_flag
        env.use_astar_shaping = saved_shaping_flag
        env.fixed_first_episode_only = True
        env.fixed_first_episode_used = True
        return result

    def _independent_validate(self, episode):
        """v9: 独立验证 —— 固定种子、全尺寸均匀采样、A* 完全禁用、GIF 录制"""
        import random as _rnd
        import matplotlib
        matplotlib.use('Agg')
        
        env = self.envs.envs[0]
        self.trainer.prep_rollout()

        saved_astar_flag = env.use_astar_first_episode
        saved_shaping_flag = env.use_astar_shaping
        saved_pbrs = getattr(env, 'use_pbrs', False)
        saved_max_astar_dist = env.max_astar_distance
        env.use_astar_first_episode = False
        env.use_astar_shaping = False
        env.use_pbrs = False
        env.max_astar_distance = None

        old_np_state = np.random.get_state()
        old_py_state = _rnd.getstate()
        np.random.seed(42)
        _rnd.seed(42)

        val_sizes = list(range(8, self.map_size_max_train + 1, 2))
        if val_sizes[-1] != self.map_size_max_train:
            val_sizes.append(self.map_size_max_train)
        per_size = max(1, self.val_scenarios // len(val_sizes))

        gif_dir = self.phase_b_dir / "val_gifs" / f"ep{episode:04d}"
        record_gif = (imageio is not None)
        if record_gif:
            gif_dir.mkdir(parents=True, exist_ok=True)
        gif_sizes_to_record = set(val_sizes)

        size_stats = {}
        successes = 0
        timeouts = 0
        collisions = 0
        total_policy_steps = 0
        total_astar_steps = 0
        total_scenarios = 0

        for map_size in val_sizes:
            bucket = f"{map_size}x{map_size}"
            size_stats[bucket] = {"success": 0, "total": 0, "timeout": 0, "collision": 0,
                                  "stay_ratios": [], "loop_rates": [], "timeout_dists": []}
            for sc_idx in range(per_size):
                env.rebuild_map_new_size(map_size)
                env.map_locked = False
                env.start_pos = None
                env.fixed_goals = None
                obs, _ = env.reset()
                target_goals = list(env.goals_pos)
                astar_steps = self._get_astar_optimal_steps(env)
                max_steps = max(200, astar_steps * 5)
                policy_steps = 0
                reached = False
                collision_flag = False
                env._stuck_counter = 0

                stay_count = 0
                position_history = []

                should_record = (record_gif and sc_idx == 0
                                 and map_size in gif_sizes_to_record)
                frames = []
                if should_record:
                    try:
                        frames.append(env.render(mode="rgb_array"))
                    except Exception:
                        should_record = False

                for step in range(max_steps):
                    obs_tensor = torch.tensor(
                        obs.reshape(-1, obs.shape[-1]),
                        dtype=torch.float32, device=self.device
                    )
                    action = self.policy.act(obs_tensor, deterministic=True)
                    actions = action.cpu().numpy().flatten().tolist()

                    for a in actions:
                        if int(a) == 0:
                            stay_count += 1

                    obs, _, rewards, dones, info = env.step(actions)
                    policy_steps += 1
                    position_history.append(tuple(tuple(p) for p in env.agents_pos))

                    if should_record and step % max(1, max_steps // 80) == 0:
                        try:
                            frames.append(env.render(mode="rgb_array"))
                        except Exception:
                            should_record = False

                    all_at_original_goal = all(
                        env.agents_pos[i] == target_goals[i]
                        for i in range(env.agent_num)
                    )
                    if info.get("all_goals_reached", False) or all_at_original_goal:
                        reached = True
                        if should_record:
                            try:
                                frames.append(env.render(mode="rgb_array"))
                            except Exception:
                                pass
                        break
                    if info.get("looped_to_start", False) and not info.get("all_goals_reached", False):
                        break
                    collision_list = info.get("collision", [])
                    if isinstance(collision_list, list) and any(collision_list):
                        collision_flag = True

                if should_record and len(frames) >= 2:
                    tag = "ok" if reached else ("col" if collision_flag else "timeout")
                    gif_path = gif_dir / f"{map_size}x{map_size}_{tag}.gif"
                    try:
                        imageio.mimsave(str(gif_path), frames, duration=0.15, loop=0)
                    except Exception as e:
                        print(f"    [GIF] 保存失败 {gif_path}: {e}")

                total_actions = policy_steps * env.agent_num
                stay_ratio = stay_count / total_actions if total_actions > 0 else 0

                last_n = position_history[-20:] if len(position_history) >= 20 else position_history
                if len(last_n) > 1:
                    unique_pos = len(set(last_n))
                    loop_rate = 1.0 - unique_pos / len(last_n)
                else:
                    loop_rate = 0.0

                size_stats[bucket]["total"] += 1
                size_stats[bucket]["stay_ratios"].append(stay_ratio)
                size_stats[bucket]["loop_rates"].append(loop_rate)
                total_scenarios += 1
                if reached:
                    successes += 1
                    total_policy_steps += policy_steps
                    total_astar_steps += astar_steps
                    size_stats[bucket]["success"] += 1
                elif collision_flag:
                    collisions += 1
                    size_stats[bucket]["collision"] += 1
                else:
                    timeouts += 1
                    size_stats[bucket]["timeout"] += 1
                    dist_to_goal = sum(
                        abs(env.agents_pos[i][0] - target_goals[i][0]) + abs(env.agents_pos[i][1] - target_goals[i][1])
                        for i in range(env.agent_num)
                    ) / env.agent_num
                    size_stats[bucket]["timeout_dists"].append(dist_to_goal)

        np.random.set_state(old_np_state)
        _rnd.setstate(old_py_state)

        env.use_astar_first_episode = saved_astar_flag
        env.use_astar_shaping = saved_shaping_flag
        env.use_pbrs = saved_pbrs
        env.max_astar_distance = saved_max_astar_dist
        env.fixed_first_episode_only = True
        env.fixed_first_episode_used = True

        success_rate = successes / total_scenarios if total_scenarios > 0 else 0
        timeout_rate = timeouts / total_scenarios if total_scenarios > 0 else 0
        collision_rate = collisions / total_scenarios if total_scenarios > 0 else 0
        avg_steps = total_policy_steps / successes if successes > 0 else 0
        avg_step_ratio = total_policy_steps / total_astar_steps if total_astar_steps > 0 else 0

        # 聚合诊断数据
        diag_summary = {}
        for bk in size_stats:
            s = size_stats[bk]
            diag_summary[bk] = {
                "avg_stay_ratio": float(np.mean(s["stay_ratios"])) if s["stay_ratios"] else 0,
                "avg_loop_rate": float(np.mean(s["loop_rates"])) if s["loop_rates"] else 0,
                "avg_timeout_dist": float(np.mean(s["timeout_dists"])) if s["timeout_dists"] else 0,
            }
            del s["stay_ratios"], s["loop_rates"], s["timeout_dists"]

        result = {
            'episode': episode,
            'type': 'validation',
            'success_rate': success_rate,
            'successes': successes,
            'timeouts': timeouts,
            'collisions': collisions,
            'total_scenarios': total_scenarios,
            'avg_steps': avg_steps,
            'avg_step_ratio': avg_step_ratio,
            'timeout_rate': timeout_rate,
            'collision_rate': collision_rate,
            'size_stats': size_stats,
            'diag': diag_summary,
        }

        print(f"  验证: {successes}/{total_scenarios} ({success_rate * 100:.1f}%)")
        for sz in val_sizes:
            bk = f"{sz}x{sz}"
            if bk in size_stats:
                s = size_stats[bk]
                r = s["success"] / s["total"] * 100 if s["total"] > 0 else 0
                d = diag_summary.get(bk, {})
                extra = ""
                if s["timeout"] > 0:
                    extra += f" tdist={d.get('avg_timeout_dist', 0):.1f}"
                extra += f" stay={d.get('avg_stay_ratio', 0)*100:.0f}%"
                extra += f" loop={d.get('avg_loop_rate', 0)*100:.0f}%"
                print(f"    {bk}: {s['success']}/{s['total']} ({r:.0f}%){extra}")
        print()
        if record_gif:
            gif_count = len(list(gif_dir.glob("*.gif"))) if gif_dir.exists() else 0
            if gif_count > 0:
                print(f"  GIF 已保存: {gif_dir} ({gif_count} 个)")

        return result

    def _run_eval_scenarios(self, env, total_scenarios, size_min, size_max, bucket_names):
        """通用场景评估（训练评估和独立验证共享逻辑）"""
        import random as _rnd
        successes = 0
        timeouts = 0
        collisions_count = 0
        total_policy_steps = 0
        total_astar_steps = 0
        size_stats = {}

        # 诊断统计
        timeout_distances = []
        all_stay_ratios = []
        all_loop_rates = []

        for scenario_idx in range(total_scenarios):
            map_size = _rnd.randint(size_min, size_max)
            env.rebuild_map_new_size(map_size)
            env.map_locked = False
            env.start_pos = None
            env.fixed_goals = None
            obs, _ = env.reset()
            target_goals = list(env.goals_pos)
            astar_steps = self._get_astar_optimal_steps(env)
            max_steps = max(200, astar_steps * 5)
            policy_steps = 0
            reached = False
            collision = False
            env._stuck_counter = 0

            stay_count = 0
            position_history = []

            for step in range(max_steps):
                obs_tensor = torch.tensor(
                    obs.reshape(-1, obs.shape[-1]),
                    dtype=torch.float32, device=self.device
                )
                action = self.policy.act(obs_tensor, deterministic=True)
                actions = action.cpu().numpy().flatten().tolist()

                for a in actions:
                    if int(a) == 0:
                        stay_count += 1

                obs, _, rewards, dones, info = env.step(actions)
                policy_steps += 1
                position_history.append(tuple(tuple(p) for p in env.agents_pos))

                all_at_original_goal = all(
                    env.agents_pos[i] == target_goals[i]
                    for i in range(env.agent_num)
                )
                if info.get("all_goals_reached", False) or all_at_original_goal:
                    reached = True
                    break
                if info.get("looped_to_start", False) and not info.get("all_goals_reached", False):
                    break
                collision_list = info.get("collision", [])
                if isinstance(collision_list, list) and any(collision_list):
                    collision = True

            total_actions = policy_steps * env.agent_num
            stay_ratio = stay_count / total_actions if total_actions > 0 else 0
            all_stay_ratios.append(stay_ratio)

            last_n = position_history[-20:] if len(position_history) >= 20 else position_history
            if len(last_n) > 1:
                unique_pos = len(set(last_n))
                loop_rate = 1.0 - unique_pos / len(last_n)
            else:
                loop_rate = 0.0
            all_loop_rates.append(loop_rate)

            if map_size <= 12:
                bucket = "small(8-12)"
            elif map_size <= 17:
                bucket = "medium(13-17)"
            else:
                bucket = bucket_names[-1] if len(bucket_names) > 2 else f"large(18-{size_max})"

            if bucket not in size_stats:
                size_stats[bucket] = {"success": 0, "total": 0, "timeout": 0, "collision": 0,
                                      "stay_ratios": [], "loop_rates": [], "timeout_dists": []}
            size_stats[bucket]["total"] += 1

            if reached:
                successes += 1
                total_policy_steps += policy_steps
                total_astar_steps += astar_steps
                size_stats[bucket]["success"] += 1
            elif collision:
                collisions_count += 1
                size_stats[bucket]["collision"] += 1
            else:
                timeouts += 1
                size_stats[bucket]["timeout"] += 1
                dist_to_goal = sum(
                    abs(env.agents_pos[i][0] - target_goals[i][0]) + abs(env.agents_pos[i][1] - target_goals[i][1])
                    for i in range(env.agent_num)
                ) / env.agent_num
                timeout_distances.append(dist_to_goal)
                size_stats[bucket]["timeout_dists"].append(dist_to_goal)

            size_stats[bucket]["stay_ratios"].append(stay_ratio)
            size_stats[bucket]["loop_rates"].append(loop_rate)

        success_rate = successes / total_scenarios if total_scenarios > 0 else 0
        timeout_rate = timeouts / total_scenarios if total_scenarios > 0 else 0
        collision_rate = collisions_count / total_scenarios if total_scenarios > 0 else 0
        avg_steps = total_policy_steps / successes if successes > 0 else 0
        avg_step_ratio = total_policy_steps / total_astar_steps if total_astar_steps > 0 else 0

        # 聚合诊断：去掉列表（不影响 JSON 保存）
        diag_summary = {}
        for bk in size_stats:
            s = size_stats[bk]
            diag_summary[bk] = {
                "avg_stay_ratio": float(np.mean(s["stay_ratios"])) if s["stay_ratios"] else 0,
                "avg_loop_rate": float(np.mean(s["loop_rates"])) if s["loop_rates"] else 0,
                "avg_timeout_dist": float(np.mean(s["timeout_dists"])) if s["timeout_dists"] else 0,
            }
            del s["stay_ratios"], s["loop_rates"], s["timeout_dists"]

        return {
            'success_rate': success_rate,
            'successes': successes,
            'timeouts': timeouts,
            'collisions': collisions_count,
            'total_scenarios': total_scenarios,
            'avg_steps': avg_steps,
            'avg_step_ratio': avg_step_ratio,
            'timeout_rate': timeout_rate,
            'collision_rate': collision_rate,
            'size_stats': size_stats,
            'diag': diag_summary,
            'avg_timeout_dist': float(np.mean(timeout_distances)) if timeout_distances else 0,
            'avg_stay_ratio': float(np.mean(all_stay_ratios)) if all_stay_ratios else 0,
            'avg_loop_rate': float(np.mean(all_loop_rates)) if all_loop_rates else 0,
        }

    def _load_phase_a_segments(self):
        """
        加载 Phase A 训练过的碎片场景
        
        Returns:
            碎片列表，每个元素包含地图路径和碎片信息
        """
        import json, random
        model_dir = str(self.all_args.model_dir)
        possible_dirs = []
        if "best_model" in model_dir:
            phase_a_dir = model_dir.replace("/best_model", "")
            possible_dirs.append(os.path.join(phase_a_dir, "episodes"))
        if "checkpoint_ep" in model_dir:
            phase_a_dir = os.path.dirname(model_dir)
            possible_dirs.append(os.path.join(phase_a_dir, "episodes"))
        parts = model_dir.split("/")
        for i in range(len(parts)):
            test_path = "/".join(parts[:i + 1]) + "/phase_a_astar/episodes"
            possible_dirs.append(test_path)

        segments = []
        for episodes_dir in possible_dirs:
            if not os.path.exists(episodes_dir):
                continue
            episode_dirs = sorted([d for d in os.listdir(episodes_dir) if d.startswith("ep_")])
            for ep_dir in episode_dirs:
                scene_config_path = os.path.join(episodes_dir, ep_dir, "scene_config.json")
                map_path = os.path.join(episodes_dir, ep_dir, "map.npy")
                if os.path.exists(scene_config_path) and os.path.exists(map_path):
                    try:
                        with open(scene_config_path, "r") as f:
                            scene_config = json.load(f)
                        for seg in scene_config.get("segments", []):
                            segments.append({'map_path':map_path, 
                             'segment':seg, 
                             'episode':ep_dir})

                    except Exception as e:
                        try:
                            print(f"  [警告] 加载 {scene_config_path} 失败: {e}")
                        finally:
                            e = None
                            del e

            if segments:
                break

        if segments:
            random.shuffle(segments)
        return segments

    def _get_astar_optimal_steps(self, env):
        """获取 A* 最优步数（取所有智能体中的最大值）"""
        try:
            agent_steps = []
            for agent_idx in range(env.agent_num):
                start = tuple(env.agents_pos[agent_idx])
                goal = tuple(env.goals_pos[agent_idx])
                if start == goal:
                    agent_steps.append(0)
                    continue
                path = env._astar_path(start, goal)
                if path:
                    agent_steps.append(len(path) - 1)
                else:
                    agent_steps.append(abs(start[0] - goal[0]) + abs(start[1] - goal[1]))
            return max(1, max(agent_steps) if agent_steps else 1)
        except Exception as e:
            try:
                print(f"[警告] A* 计算失败: {e}")
                return 50
            finally:
                e = None
                del e

    def _save_checkpoint(self, episode, eval_result, is_baseline=False):
        """保存检查点"""
        checkpoint_dir = self.phase_b_dir / f"checkpoint_ep{episode:03d}"
        models_dir = checkpoint_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.actor.state_dict(), str(models_dir / "actor.pt"))
        torch.save(self.policy.critic.state_dict(), str(models_dir / "critic.pt"))
        save_result = {k: v for k, v in eval_result.items() if k != "size_stats"}
        save_result["source_model"] = str(self.all_args.model_dir) if is_baseline else None
        save_result["size_stats"] = eval_result.get("size_stats", {})
        with open(checkpoint_dir / "evaluation.json", "w") as f:
            json.dump(save_result, f, indent=2)

    def _save_best_model(self, episode, success_rate, source="train_eval"):
        """保存当前最佳模型到 best_model/ 目录"""
        best_dir = self.phase_b_dir / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.actor.state_dict(), str(best_dir / "actor.pt"))
        torch.save(self.policy.critic.state_dict(), str(best_dir / "critic.pt"))
        with open(best_dir / "info.json", "w") as f:
            json.dump({
                "episode": episode,
                "success_rate": success_rate,
                "source": source
            }, f, indent=2)

    def _save_validation_history(self):
        """保存独立验证历史到 JSON"""
        history_file = self.phase_b_dir / "validation_history.json"
        records = []
        for ep, result in self.val_history:
            rec = {k: v for k, v in result.items()}
            records.append(rec)
        with open(history_file, "w") as f:
            json.dump(records, f, indent=2)

    def _compute_explained_variance(self):
        """计算 Explained Variance: 衡量 Critic 预测质量，< 0 表示预测比均值还差"""
        values = self.buffer.value_preds[:-1].flatten()
        returns = self.buffer.returns[:-1].flatten()
        var_returns = np.var(returns)
        if var_returns < 1e-8:
            return 0.0
        return float(1.0 - np.var(returns - values) / var_returns)

    def _log_training(self, episode, train_info, episode_reward, total_steps, start_time):
        """记录训练日志"""
        elapsed = time.time() - start_time
        fps = total_steps / elapsed if elapsed > 0 else 0
        eta_min = (self.episodes - episode) / max(1, episode) * elapsed / 60
        ev = train_info.get("explained_variance", 0.0)
        print(f"  ep{episode:>4d}/{self.episodes}  reward={episode_reward:>7.1f}  "
              f"ploss={train_info['policy_loss']:.4f}  entropy={train_info['dist_entropy']:.3f}  "
              f"EV={ev:.3f}  FPS={fps:.0f}  ETA={eta_min:.0f}min")

        # reward 分量均值
        if self._reward_component_count > 0:
            rc_strs = []
            for k in ["goal", "collision", "distance", "astar", "anti_stuck", "stay", "pbrs", "wall"]:
                mean_v = self._reward_component_sums[k] / self._reward_component_count
                rc_strs.append(f"{k}={mean_v:+.3f}")
                self.writer.add_scalar(f"reward/{k}_mean", mean_v, total_steps)
            print(f"         reward分量: {' | '.join(rc_strs)}")
            self._reward_component_sums = {k: 0.0 for k in self._reward_component_sums}
            self._reward_component_count = 0

        # 地图尺寸分布
        if self._map_size_counter:
            total_maps = sum(self._map_size_counter.values())
            sorted_sizes = sorted(self._map_size_counter.keys())
            dist_parts = [f"{s}:{self._map_size_counter[s]}" for s in sorted_sizes]
            print(f"         地图分布(n={total_maps}): {' '.join(dist_parts)}")
            for s in sorted_sizes:
                self.writer.add_scalar(f"map_dist/size_{s}", self._map_size_counter[s], total_steps)
            self._map_size_counter = {}

        self.writer.add_scalar("train/value_loss", train_info["value_loss"], total_steps)
        self.writer.add_scalar("train/policy_loss", train_info["policy_loss"], total_steps)
        self.writer.add_scalar("train/dist_entropy", train_info["dist_entropy"], total_steps)
        self.writer.add_scalar("train/actor_grad_norm", train_info["actor_grad_norm"], total_steps)
        self.writer.add_scalar("train/critic_grad_norm", train_info["critic_grad_norm"], total_steps)
        self.writer.add_scalar("train/ratio", train_info["ratio"], total_steps)
        self.writer.add_scalar("train/explained_variance", ev, total_steps)
        self.writer.add_scalar("env/average_episode_rewards", episode_reward, total_steps)

    def _print_summary(self, start_time, early_stopped=False):
        """打印训练总结"""
        elapsed = time.time() - start_time
        print("\n══════════════════════════════════════════════════════════════════════")
        if early_stopped:
            print("[Phase B] 训练早停！")
        else:
            print("[Phase B] 训练完成！")
        print("══════════════════════════════════════════════════════════════════════")
        print(f"  总耗时: {elapsed / 60:.1f} 分钟")
        if self.val_interval > 0 and self.best_val_episode >= 0:
            print(f"  最佳验证模型: ep{self.best_val_episode} ({self.best_val_success_rate*100:.1f}%)")
        if self.best_episode >= 0 and self.val_interval <= 0:
            print(f"  最佳训练评估: ep{self.best_episode} ({self.best_success_rate*100:.1f}%)")
        print(f"  最佳模型位置: {self.phase_b_dir}/best_model/")
        print()
        print("  训练评估对比:")
        print("  ┌──────────┬──────────┬──────────┬─────────────┐")
        print("  │   回合   │  成功率   │ 步数比   │   模型      │")
        print("  ├──────────┼──────────┼──────────┼─────────────┤")
        for ep, result in self.eval_history:
            sr = result["success_rate"] * 100
            ratio = result["avg_step_ratio"]
            label = "baseline" if result.get("type") == "baseline" else "saved"
            if ep == self.episodes:
                label = "final"
            if ep == self.best_episode and self.val_interval <= 0:
                label = "BEST"
            print(f"  │   ep{ep:03d}  │  {sr:5.1f}%  │  {ratio:5.2f}x  │ ✓ {label:9s} │")
        print("  └──────────┴──────────┴──────────┴─────────────┘")
        
        if self.val_history:
            print()
            print("  独立验证对比:")
            print("  ┌──────────┬──────────┬──────────┬─────────────┐")
            print("  │   回合   │  成功率   │ 步数比   │   状态      │")
            print("  ├──────────┼──────────┼──────────┼─────────────┤")
            for ep, result in self.val_history:
                sr = result["success_rate"] * 100
                ratio = result["avg_step_ratio"]
                label = "★ BEST" if ep == self.best_val_episode else "val"
                print(f"  │   ep{ep:04d} │  {sr:5.1f}%  │  {ratio:5.2f}x  │ {label:11s} │")
            print("  └──────────┴──────────┴──────────┴─────────────┘")
        
        print()
        print(f"  共保存 {len(self.eval_history)} 个 checkpoint")
        print(f"  保存位置: {self.phase_b_dir}")
        print("══════════════════════════════════════════════════════════════════════\n")
        self.writer.close()
