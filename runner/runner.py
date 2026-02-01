import os
import shutil
import time
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from utils.shared_buffer import SharedReplayBuffer
try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

# Phase A Runner（延迟导入，避免循环依赖）
PhaseARunner = None


def _t2n(x):
    """
    将PyTorch张量转换为NumPy数组。
    
    参数：
        x: (torch.Tensor) 要转换的PyTorch张量
        
    返回：
        numpy.ndarray: 转换后的NumPy数组
    """
    return x.detach().cpu().numpy()


class Runner(object):
    """
    MAPPO算法的运行器类，负责管理训练流程，包括环境交互、数据收集、模型更新和结果记录。
    
    参数：
        config: (dict) 包含训练参数的配置字典
    """
    def __init__(self, config):
        # 从配置中提取参数
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.render_envs = config.get('render_envs', None)  # 渲染环境，可选
        
        # 算法和环境基本参数
        self.env_name = self.all_args.env_name
        self.algorithm_name = "mappo"  # 硬编码为MAPPO算法
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V  # 是否使用中心化价值函数
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state  # 是否使用观察代替状态
        self.num_env_steps = self.all_args.num_env_steps  # 总训练环境步数
        self.episode_length = self.all_args.episode_length  # 每个episode的最大步数
        self.n_rollout_threads = self.all_args.n_rollout_threads  # 采样线程数
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads  # 评估线程数
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads  # 渲染线程数
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay  # 是否使用线性学习率衰减
        self.hidden_size = self.all_args.hidden_size  # 网络隐藏层大小
        self.use_render = self.all_args.use_render  # 是否渲染
        self.use_astar_bc = getattr(self.all_args, "use_astar_bc", False)
        self.astar_bc_updates = getattr(self.all_args, "astar_bc_updates", 0)
        for env in self.envs.envs:
            env.fixed_first_episode_only = True

        # 间隔参数
        self.save_interval = self.all_args.save_interval  # 模型保存间隔
        self.use_eval = self.all_args.use_eval  # 是否进行评估
        self.eval_interval = self.all_args.eval_interval  # 评估间隔
        self.log_interval = self.all_args.log_interval  # 日志记录间隔

        # 目录设置
        self.model_dir = self.all_args.model_dir  # 预训练模型目录
        self.run_dir = config["run_dir"]  # 运行目录
        self.log_dir = str(self.run_dir / 'logs')  # 日志目录
        self.save_dir = str(self.run_dir / 'models')  # 模型保存目录
        self.gif_dir = None
        self.frames_dir = None
        
        # 创建目录
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.all_args.save_gifs or self.all_args.save_frames:
            gif_dir = self.all_args.gif_dir
            frames_dir = self.all_args.frames_dir
            if not os.path.isabs(gif_dir):
                gif_dir = os.path.join(str(self.run_dir), gif_dir)
            if not os.path.isabs(frames_dir):
                frames_dir = os.path.join(str(self.run_dir), frames_dir)
            self.gif_dir = gif_dir
            self.frames_dir = frames_dir
            os.makedirs(self.gif_dir, exist_ok=True)
            os.makedirs(self.frames_dir, exist_ok=True)
        
        # 初始化TensorBoard写入器
        self.writter = SummaryWriter(self.log_dir)

        # 导入算法和策略
        from algorithms.algorithm.mappo import MAPPO as TrainAlgo
        from algorithms.algorithm.MAPPOPolicy import MAPPOPolicy as Policy

        # 确定共享观察空间
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # 初始化策略网络
        self.policy = Policy(
            self.all_args,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0],
            device=self.device
        )

        # 加载预训练模型（如果有）
        if self.model_dir is not None:
            self.restore()

        # 初始化算法训练器
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
        
        # 初始化共享回放缓冲区
        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0]
        )

    def run(self):
        """
        运行训练主入口，根据 training_mode 选择训练模式。
        
        支持的模式:
        - two_phase: 完整两阶段训练（Phase A + Phase B）
        - phase_a_only: 仅执行 Phase A（A* 在线学习）
        - phase_b_only: 仅执行 Phase B（PPO 训练）
        - legacy: 旧版训练模式（兼容原有代码）
        """
        training_mode = getattr(self.all_args, 'training_mode', 'legacy')
        
        if training_mode == 'two_phase':
            self._run_two_phase_training()
        elif training_mode == 'phase_a_only':
            self._run_phase_a()
        elif training_mode == 'phase_b_only':
            self._run_phase_b()
        else:  # legacy
            self._run_legacy()
    
    def _run_two_phase_training(self):
        """运行完整的两阶段训练"""
        print("\n" + "=" * 70)
        print("              两阶段训练模式")
        print("=" * 70)
        print(f"  Phase A: {self.all_args.phase_a_episodes} episodes (A* 在线学习)")
        print(f"  Phase B: {self.all_args.phase_b_episodes} episodes (PPO 训练)")
        print("=" * 70 + "\n")
        
        # Phase A
        phase_a_success = self._run_phase_a()
        
        if not phase_a_success:
            print("[警告] Phase A 未成功完成，跳过 Phase B")
            return
        
        # Phase B
        self._run_phase_b()
    
    def _run_phase_a(self):
        """
        运行 Phase A: A* 在线学习
        
        Returns:
            bool: 是否成功完成
        """
        global PhaseARunner
        if PhaseARunner is None:
            from runner.phase_a_runner import PhaseARunner
        
        # 创建 Phase A Runner 配置
        phase_a_config = {
            'all_args': self.all_args,
            'envs': self.envs,
            'policy': self.policy,
            'trainer': self.trainer,
            'device': self.device,
            'run_dir': self.run_dir,
            'num_agents': self.num_agents
        }
        
        # 创建并运行 Phase A Runner
        phase_a_runner = PhaseARunner(phase_a_config)
        success = phase_a_runner.run()
        
        return success
    
    def _run_phase_b(self):
        """
        运行 Phase B: 纯策略 PPO 训练
        """
        print("\n" + "=" * 70)
        print("                    Phase B: PPO 训练")
        print("=" * 70)
        
        # 确保禁用 A* 引导
        for env in self.envs.envs:
            env.use_astar_first_episode = False
            env.use_astar_shaping = False
        self._reset_env_episode_count()
        
        # Phase B 目录
        phase_b_dir = Path(self.run_dir) / "phase_b_policy"
        phase_b_dir.mkdir(parents=True, exist_ok=True)
        
        # 更新保存目录
        self.save_dir = str(phase_b_dir / "models")
        self.log_dir = str(phase_b_dir / "logs")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 重新创建 TensorBoard writer
        self.writter = SummaryWriter(self.log_dir)
        
        # 执行 PPO 训练
        self._run_ppo_training(
            episodes=self.all_args.phase_b_episodes,
            render_interval=getattr(self.all_args, 'phase_b_render_interval', 100)
        )
        
        print(f"\n[Phase B] 完成! 模型保存于: {self.save_dir}")
    
    def _run_legacy(self):
        """
        运行旧版训练模式（兼容原有代码）
        """
        if self.use_astar_bc and self.astar_bc_updates > 0:
            self._behavior_clone_from_first_episode()
            for env in self.envs.envs:
                env.use_astar_first_episode = False
                env.use_astar_shaping = False
            self._reset_env_episode_count()
        
        self._run_ppo_training()
    
    def _run_ppo_training(self, episodes=None, render_interval=None):
        """
        PPO 训练核心循环
        
        Args:
            episodes: 训练的 episode 数量（None 则使用 num_env_steps 计算）
            render_interval: 渲染间隔（None 则不额外渲染）
        """
        self.warmup()
        
        start = time.time()
        
        if episodes is None:
            episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # 采样动作
                (
                    values,
                    actions,
                    action_log_probs,
                    actions_env,
                ) = self.collect(step)

                # 执行动作，获取下一个观察、奖励、完成状态和信息
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                )

                # 将数据插入到回放缓冲区
                self.insert(data)

            # 计算回报并更新网络
            self.compute()
            train_infos = self.train()

            # 计算总步数
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # 保存模型
            if episode % 20 == 0 or episode == episodes - 1:
                self.save()

            # 记录日志
            if episode % 10 == 0:
                end = time.time()
                print(
                    "\n 场景 {} 算法 {} 实验 {} 更新 {}/{} episode, 总步数 {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                # 计算平均episode奖励
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("平均episode奖励: {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)

            # 评估模型
            if episode % 200 == 0 and self.use_eval:
                self.eval(total_num_steps)
            
            # Phase B 额外渲染
            if render_interval and episode > 0 and episode % render_interval == 0:
                self._save_phase_b_render(episode)
    
    def _save_phase_b_render(self, episode):
        """保存 Phase B 渲染 GIF"""
        if imageio is None:
            return
        
        phase_b_dir = Path(self.run_dir) / "phase_b_policy" / "renders"
        ep_render_dir = phase_b_dir / f"ep_{episode:04d}"
        ep_render_dir.mkdir(parents=True, exist_ok=True)
        
        env = self.envs.envs[0]
        obs, _ = env.reset()
        frames = [env.render(mode="rgb_array")]
        
        for step in range(min(200, self.episode_length)):  # 最多渲染 200 步
            self.trainer.prep_rollout()
            obs_tensor = torch.tensor(
                obs.reshape(-1, obs.shape[-1]),
                dtype=torch.float32,
                device=self.device
            )
            action = self.policy.act(obs_tensor, deterministic=True)
            actions = action.cpu().numpy().flatten().tolist()
            
            obs, _, _, dones, info = env.step(actions)
            frames.append(env.render(mode="rgb_array"))
            
            if info.get("looped_to_start", False):
                break
        
        gif_path = ep_render_dir / f"ep_{episode:04d}.gif"
        imageio.mimsave(str(gif_path), frames, duration=1.0/self.all_args.gif_fps)
        print(f"  [Phase B] 渲染保存: {gif_path}")

    def warmup(self):
        """
        环境热身，初始化回放缓冲区。
        """
        # 重置环境
        obs = self.envs.reset()  # shape = [env_num, agent_num, obs_dim]

        # 处理共享观察空间
        if self.use_centralized_V:
            # 中心化价值函数：将所有智能体的观察拼接成一个共享观察
            share_obs = obs.reshape(self.n_rollout_threads, -1)  # shape = [env_num, agent_num * obs_dim]
            share_obs = np.expand_dims(share_obs, 1).repeat(
                self.num_agents, axis=1
            )  # shape = [env_num, agent_num， agent_num * obs_dim]
        else:
            # 非中心化价值函数：使用智能体各自的观察作为共享观察
            share_obs = obs

        # 将初始观察插入缓冲区
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        """
        收集数据，生成动作并记录相关信息。
        
        参数：
            step: (int) 当前episode的步数
            
        返回：
            values: (np.ndarray) 价值函数预测
            actions: (np.ndarray) 生成的动作
            action_log_probs: (np.ndarray) 动作的对数概率
            actions_env: (np.ndarray) 环境可执行的动作格式
        """
        # 将策略设置为评估模式
        self.trainer.prep_rollout()
        
        # 获取动作
        value, action, action_log_prob = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
        )
        
        # 转换为NumPy数组并拆分到各个线程
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))  # [env_num, agent_num, 1]
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))  # [env_num, agent_num, action_dim]
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )  # [env_num, agent_num, 1]
        
        # 处理动作格式，转换为环境可执行的格式
        if self.envs.action_space[0].__class__.__name__ == "Discrete":
            # 离散动作空间：将动作索引转换为one-hot编码
            # actions --> actions_env : shape:[10, 1] --> [5, 2, 5]
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError("不支持的动作空间: {}".format(self.envs.action_space[0].__class__.__name__))

        return values, actions, action_log_probs, actions_env

    def insert(self, data):
        """
        将收集的数据插入到回放缓冲区。
        
        参数：
            data: (tuple) 包含观察、奖励、完成状态、信息、价值预测、动作和动作对数概率的数据元组
        """
        # 解包数据
        obs, rewards, dones, infos, values, actions, action_log_probs = data

        if rewards.ndim == 2:
            rewards = rewards[..., np.newaxis]
        
        # 生成掩码，用于处理完成状态
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # 处理共享观察
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        # 插入数据到缓冲区
        self.buffer.insert(
            share_obs,
            obs,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )
    
    @torch.no_grad()
    def compute(self):
        """
        计算收集数据的回报值。
        """
        # 将策略设置为评估模式
        self.trainer.prep_rollout()
        
        # 获取下一个状态的价值预测
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        
        # 计算回报
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """
        使用缓冲区中的数据训练策略。
        
        返回：
            train_infos: (dict) 包含训练信息的字典
        """
        # 将策略设置为训练模式
        self.trainer.prep_training()
        
        # 执行训练
        train_infos = self.trainer.train(self.buffer)
        
        # 训练后处理
        self.buffer.after_update()
        
        return train_infos

    def save(self):
        """保存策略的actor和critic网络。"""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self):
        """从保存的模型中恢复策略网络。"""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
 
    def log_train(self, train_infos, total_num_steps):
        """
        记录训练信息。
        
        参数：
            train_infos: (dict) 训练更新的信息
            total_num_steps: (int) 训练环境的总步数
        """
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        记录环境信息。
        
        参数：
            env_infos: (dict) 环境状态的信息
            total_num_steps: (int) 训练环境的总步数
        """
        for k, v in env_infos.items():
            if len(v)>0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action = self.trainer.policy.act(
                np.concatenate(eval_obs),
                deterministic=False,
            )
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError("不支持的动作空间: {}".format(self.eval_envs.action_space[0].__class__.__name__))

            # 执行动作，获取下一个观察、奖励、完成状态和信息
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos["eval_average_episode_rewards"])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """可视化环境，渲染智能体的行为。"""
        envs = self.envs

        if self.use_astar_bc and self.astar_bc_updates > 0:
            with torch.enable_grad():
                self._behavior_clone_from_astar()
            self._reset_env_episode_count()

        save_frames = self.all_args.save_frames or self.all_args.save_gifs
        frame_duration = 1.0 / self.all_args.gif_fps if self.all_args.gif_fps > 0 else self.all_args.ifi
        if (self.all_args.save_gifs or save_frames) and imageio is None:
            raise ImportError("Saving GIFs/frames requires imageio. Please install it via 'pip install imageio'")
        if save_frames and self.frames_dir is not None:
            for name in os.listdir(self.frames_dir):
                path = os.path.join(self.frames_dir, name)
                if name.startswith("frame_") and name.endswith(".png") and os.path.isfile(path):
                    os.remove(path)
                elif name.startswith("episode_") and os.path.isdir(path):
                    shutil.rmtree(path)
        for episode in range(self.all_args.render_episodes):
            all_frames = []
            frame_index = 0
            episode_dir = None
            astar_actions_log = []
            policy_actions_log = []
            if save_frames:
                episode_dir = os.path.join(self.frames_dir, f"episode_{episode + 1:03d}")
                os.makedirs(episode_dir, exist_ok=True)
            obs = envs.reset()
            if self.all_args.save_gifs or save_frames:
                image = envs.render("rgb_array")[0]
                if self.all_args.save_gifs:
                    all_frames.append(image)
                if save_frames:
                    imageio.imwrite(os.path.join(episode_dir, f"frame_{frame_index:06d}.png"), image)
                    frame_index += 1
            else:
                envs.render("human")

            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                # 由于移除了RNN支持，act方法不再返回rnn_states
                action = self.trainer.policy.act(
                    np.concatenate(obs),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError("不支持的动作空间: {}".format(envs.action_space[0].__class__.__name__))

                if episode < 2:
                    env0 = envs.envs[0]
                    astar_actions = [env0._action_from_path(i) for i in range(env0.agent_num)]
                    policy_actions = np.argmax(actions_env[0], axis=-1).tolist()
                    astar_actions_log.append(astar_actions)
                    policy_actions_log.append(policy_actions)
                if episode == 1:
                    print(f"[STEP] Episode 2 step {step + 1} policy actions_env: {actions_env[0].tolist()}")

                # 执行动作，获取下一个观察、奖励、完成状态和信息
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs or save_frames:
                    image = envs.render("rgb_array")[0]
                    if self.all_args.save_gifs:
                        all_frames.append(image)
                    if save_frames:
                        imageio.imwrite(os.path.join(episode_dir, f"frame_{frame_index:06d}.png"), image)
                        frame_index += 1
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < frame_duration:
                        time.sleep(frame_duration - elapsed)
                else:
                    envs.render("human")

            print("平均episode奖励: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
            if episode < 2 and len(astar_actions_log) > 0:
                preview_len = min(20, len(astar_actions_log))
                label = "policy(pre-override)" if episode == 0 else "policy"
                print(f"[VERIFY] Episode {episode + 1} A* actions (first {preview_len}): {astar_actions_log[:preview_len]}")
                print(f"[VERIFY] Episode {episode + 1} {label} actions (first {preview_len}): {policy_actions_log[:preview_len]}")
                matches = np.zeros(len(astar_actions_log[0]), dtype=np.int64)
                for t in range(len(astar_actions_log)):
                    for i, (a, p) in enumerate(zip(astar_actions_log[t], policy_actions_log[t])):
                        if a == p:
                            matches[i] += 1
                ratios = matches / len(astar_actions_log)
                print(f"[VERIFY] Episode {episode + 1} match ratio vs A*: {ratios.tolist()}")
            if self.all_args.save_gifs:
                gif_path = os.path.join(self.gif_dir, f"render_ep{episode + 1}.gif")
                imageio.mimsave(gif_path, all_frames, duration=frame_duration)

    def _behavior_clone_from_astar(self):
        """使用A*轨迹对actor进行行为克隆初始化。"""
        env = self.envs.envs[0]
        prev_astar_flag = env.use_astar_first_episode
        env.use_astar_first_episode = True
        obs, _ = env.reset()
        obs_seq = []
        act_seq = []

        for _ in range(self.episode_length):
            actions = [env._action_from_path(i) for i in range(env.agent_num)]
            obs_seq.append(obs.copy())
            act_seq.append(np.array(actions, dtype=np.int64))
            obs, _share_obs, _rewards, dones, _info = env.step(actions)
            if np.all(dones):
                break

        if len(obs_seq) == 0:
            return

        obs_batch = np.concatenate([o.reshape(-1, o.shape[-1]) for o in obs_seq], axis=0)
        act_batch = np.concatenate([a.reshape(-1, 1) for a in act_seq], axis=0)

        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(act_batch, dtype=torch.int64, device=self.device)

        self.trainer.prep_training()
        for _ in range(self.astar_bc_updates):
            self.policy.actor_optimizer.zero_grad()
            action_log_probs, _ = self.policy.actor.evaluate_actions(obs_t, act_t)
            bc_loss = -action_log_probs.mean()
            bc_loss.backward()
            self.policy.actor_optimizer.step()

        env.use_astar_first_episode = prev_astar_flag

    def _behavior_clone_from_first_episode(self):
        """用第1回合A*轨迹进行行为克隆初始化。"""
        env = self.envs.envs[0]
        prev_astar_flag = env.use_astar_first_episode
        env.use_astar_first_episode = True

        obs, _ = env.reset()
        obs_seq = []
        act_seq = []

        for _ in range(self.episode_length):
            actions = [env._action_from_path(i) for i in range(env.agent_num)]
            obs_seq.append(obs.copy())
            act_seq.append(np.array(actions, dtype=np.int64))
            obs, _share_obs, _rewards, _dones, _info = env.step(actions)

        if len(obs_seq) == 0:
            env.use_astar_first_episode = prev_astar_flag
            return

        obs_batch = np.concatenate([o.reshape(-1, o.shape[-1]) for o in obs_seq], axis=0)
        act_batch = np.concatenate([a.reshape(-1, 1) for a in act_seq], axis=0)

        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(act_batch, dtype=torch.int64, device=self.device)

        self.trainer.prep_training()
        for _ in range(self.astar_bc_updates):
            self.policy.actor_optimizer.zero_grad()
            action_log_probs, _ = self.policy.actor.evaluate_actions(obs_t, act_t)
            bc_loss = -action_log_probs.mean()
            bc_loss.backward()
            self.policy.actor_optimizer.step()

        env.use_astar_first_episode = prev_astar_flag

    def _reset_env_episode_count(self):
        for env in self.envs.envs:
            env.episode_count = 0
