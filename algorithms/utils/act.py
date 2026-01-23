from .distributions import Categorical
import torch
import torch.nn as nn

class ACTLayer(nn.Module):
    """
    动作生成层，用于从网络输出计算动作和动作对数概率。
    
    参数：
        action_space: (gym.Space) 动作空间
        inputs_dim: (int) 网络输入维度
        use_orthogonal: (bool) 是否使用正交初始化
        gain: (float) 网络输出层的增益
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()

        # 检查动作空间类型，目前仅支持Discrete
        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        else:
            raise NotImplementedError("不支持的动作空间: {}".format(action_space.__class__.__name__))
    
    def forward(self, x, available_actions=None, deterministic=False):
        """
        从给定输入计算动作和动作对数概率。
        
        参数：
            x: (torch.Tensor) 网络输入
            available_actions: (torch.Tensor) 表示智能体可用的动作（如果为None，则所有动作可用）
            deterministic: (bool) 是否从动作分布中采样或返回模式（即贪婪策略）

        返回：
            actions: (torch.Tensor) 要执行的动作
            action_log_probs: (torch.Tensor) 执行动作的对数概率
        """
        # 生成动作分布
        action_logits = self.action_out(x, available_actions)
        # 根据deterministic选择贪婪动作或采样动作
        actions = action_logits.mode() if deterministic else action_logits.sample() 
        # 计算动作的对数概率
        action_log_probs = action_logits.log_probs(actions)
        
        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        """
        从输入计算动作概率。
        
        参数：
            x: (torch.Tensor) 网络输入
            available_actions: (torch.Tensor) 表示智能体可用的动作（如果为None，则所有动作可用）

        返回：
            action_probs: (torch.Tensor) 动作概率
        """
        action_logits = self.action_out(x, available_actions)
        action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """
        计算给定动作的对数概率和熵。
        
        参数：
            x: (torch.Tensor) 网络输入
            action: (torch.Tensor) 要评估的动作
            available_actions: (torch.Tensor) 表示智能体可用的动作（如果为None，则所有动作可用）
            active_masks: (torch.Tensor) 表示智能体是否活跃的掩码

        返回：
            action_log_probs: (torch.Tensor) 输入动作的对数概率
            dist_entropy: (torch.Tensor) 给定输入的动作分布熵
        """
        # 生成动作分布
        action_logits = self.action_out(x, available_actions)
        # 计算动作的对数概率
        action_log_probs = action_logits.log_probs(action)
        
        # 计算动作分布的熵
        if active_masks is not None:
            # 如果提供了active_masks，则仅计算活跃智能体的熵
            dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
        else:
            # 否则计算所有智能体的平均熵
            dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy
