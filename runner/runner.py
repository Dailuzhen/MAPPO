import os
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter
from utils.shared_buffer import SharedReplayBuffer


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
        
        # 创建目录
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
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
        运行MAPPO算法的主训练循环。
        """
        self.warmup()

        start = time.time()
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
                    action_log_probs,
                )

                # 将数据插入到回放缓冲区
                self.insert(data)

            # 计算回报并更新网络
            self.compute()
            train_infos = self.train()

            # 计算总步数
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # 保存模型
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # 记录日志
            if episode % self.log_interval == 0:
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
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

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
                deterministic=True,
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

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render("rgb_array")[0][0]
                all_frames.append(image)
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

                # 执行动作，获取下一个观察、奖励、完成状态和信息
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render("human")

            print("平均episode奖励: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        # if self.all_args.save_gifs:
        #     imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
