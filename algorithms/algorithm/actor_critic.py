"""
# MAPPO算法的Actor-Critic网络实现
"""

import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.act import ACTLayer
from algorithms.utils.popart import PopArt
from utils.util import get_shape_from_obs_space


class Actor(nn.Module):
    """
    MAPPO算法的Actor网络类，根据观察输出动作。
    
    参数：
        args: (argparse.Namespace) 包含相关模型信息的参数
        obs_space: (gym.Space) 观察空间
        action_space: (gym.Space) 动作空间
        device: (torch.device) 指定运行设备（cpu/gpu）
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self.tpdv = dict(dtype=torch.float32, device=device)

        # 获取观察空间形状，选择CNN或MLP作为基础网络
        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        # 初始化动作层
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, available_actions=None, deterministic=False):
        """
        根据给定输入计算动作。
        
        参数：
            obs: (np.ndarray / torch.Tensor) 输入到网络的观察
            available_actions: (np.ndarray / torch.Tensor) 表示智能体可用的动作（如果为None，则所有动作可用）
            deterministic: (bool) 是否从动作分布中采样或返回概率最高的动作

        返回：
            actions: (torch.Tensor) 要执行的动作
            action_log_probs: (torch.Tensor) 执行动作的对数概率
        """
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs

    def evaluate_actions(self, obs, action, available_actions=None, active_masks=None):
        """
        计算给定动作的对数概率和熵。
        
        参数：
            obs: (torch.Tensor) 输入到网络的观察
            action: (torch.Tensor) 要评估熵和对数概率的动作
            available_actions: (torch.Tensor) 表示智能体可用的动作（如果为None，则所有动作可用）
            active_masks: (torch.Tensor) 表示智能体是活跃还是死亡

        返回：
            action_log_probs: (torch.Tensor) 输入动作的对数概率
            dist_entropy: (torch.Tensor) 给定输入的动作分布熵
        """
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)
        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features, action, available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None
        )

        return action_log_probs, dist_entropy


class Critic(nn.Module):
    """
    MAPPO算法的Critic网络类，根据中心化输入（MAPPO）或局部观察（IPPO）输出价值函数预测。
    
    参数：
        args: (argparse.Namespace) 包含相关模型信息的参数
        cent_obs_space: (gym.Space) 中心化观察空间
        device: (torch.device) 指定运行设备（cpu/gpu）
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        # 获取中心化观察空间形状，选择CNN或MLP作为基础网络
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        # 定义初始化函数
        #定义一个通用的层初始化函数
        #m -要初始化的网络层
        #init_method -初始化权重
        #nn.init.constant_(x, 0) -初始化偏置为0
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # 初始化价值输出层
        if self._use_popart:
            #self.hidden_size -隐藏层大小
            #1 -输出层大小（价值函数预测）
            #device -指定运行设备（cpu/gpu）
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs):
        """
        计算值函数预测。
        
        参数：
            cent_obs: (np.ndarray / torch.Tensor) 输入到网络的观察

        返回：
            values: (torch.Tensor) 价值函数预测
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        critic_features = self.base(cent_obs)
        values = self.v_out(critic_features)

        return values
