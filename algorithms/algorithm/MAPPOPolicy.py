"""
MAPPO 策略类。封装了actor和critic网络，用于计算动作和价值函数预测。
"""

import torch
from algorithms.algorithm.actor_critic import Actor, Critic
from utils.util import update_linear_schedule


class MAPPOPolicy:
    """
    MAPPO 策略类。封装了actor和critic网络，用于计算动作和价值函数预测。

    参数：
        args: (argparse.Namespace) 包含相关模型和策略信息的参数。
        obs_space: (gym.Space) 观察空间。
        cent_obs_space: (gym.Space) 价值函数输入空间（MAPPO使用中心化输入，IPPO使用去中心化输入）。
        action_space: (gym.Space) 动作空间。
        device: (torch.device) 指定运行设备（cpu/gpu）。
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        # 初始化actor和critic网络
        self.actor = Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = Critic(args, self.share_obs_space, self.device)

        # 初始化优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        线性衰减actor和critic的学习率。
        
        参数：
            episode: (int) 当前训练轮次。
            episodes: (int) 总训练轮次。
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, available_actions=None, deterministic=False):
        """
        为给定输入计算动作和价值函数预测。
        
        参数：
            cent_obs (np.ndarray): 传入critic的中心化输入。
            obs (np.ndarray): 传入actor的本地智能体输入。
            available_actions: (np.ndarray) 表示智能体可用的动作（如果为None，则所有动作可用）。
            deterministic: (bool) 动作是否应该是分布的模式还是应该被采样。

        返回：
            values: (torch.Tensor) 价值函数预测。
            actions: (torch.Tensor) 要执行的动作。
            action_log_probs: (torch.Tensor) 所选动作的对数概率。
        """
        actions, action_log_probs = self.actor(obs, available_actions, deterministic)
        values = self.critic(cent_obs)
        return values, actions, action_log_probs

    def get_values(self, cent_obs):
        """
        获取价值函数预测。
        
        参数：
            cent_obs (np.ndarray): 传入critic的中心化输入。

        返回：
            values: (torch.Tensor) 价值函数预测。
        """
        values = self.critic(cent_obs)
        return values

    def evaluate_actions(self, cent_obs, obs, action, available_actions=None, active_masks=None):
        """
        为actor更新获取动作对数概率/熵和价值函数预测。
        
        参数：
            cent_obs (np.ndarray): 传入critic的中心化输入。
            obs (np.ndarray): 传入actor的本地智能体输入。
            action: (np.ndarray) 要计算对数概率和熵的动作。
            available_actions: (np.ndarray) 表示智能体可用的动作（如果为None，则所有动作可用）。
            active_masks: (torch.Tensor) 表示智能体是活跃还是死亡。

        返回：
            values: (torch.Tensor) 价值函数预测。
            action_log_probs: (torch.Tensor) 输入动作的对数概率。
            dist_entropy: (torch.Tensor) 给定输入的动作分布熵。
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, action, available_actions, active_masks)
        values = self.critic(cent_obs)
        return values, action_log_probs, dist_entropy

    def act(self, obs, available_actions=None, deterministic=False):
        """
        为给定输入计算动作。
        
        参数：
            obs (np.ndarray): 传入actor的本地智能体输入。
            available_actions: (np.ndarray) 表示智能体可用的动作（如果为None，则所有动作可用）。
            deterministic: (bool) 动作是否应该是分布的模式还是应该被采样。

        返回：
            actions: (torch.Tensor) 要执行的动作。
        """
        actions, _ = self.actor(obs, available_actions, deterministic)
        return actions
