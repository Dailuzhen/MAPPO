import torch
import torch.nn as nn
from .util import init

"""
标准化PyTorch分布接口，使其与本代码库兼容。
目前仅支持Discrete动作空间，因此只保留Categorical相关类。
"""


# Categorical分布类，用于离散动作空间
class FixedCategorical(torch.distributions.Categorical):
    """
    固定的Categorical分布类，用于处理离散动作空间。
    扩展了PyTorch的Categorical分布，添加了sample、log_probs和mode方法。
    """
    def sample(self):
        """采样动作，返回形状为[batch_size, 1]的张量。"""
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        """计算给定动作的对数概率。"""
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        """返回概率最大的动作（即贪婪策略）。"""
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    """
    Categorical策略网络的输出层，用于生成离散动作分布。
    """
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        """
        参数：
            num_inputs: (int) 输入特征维度
            num_outputs: (int) 输出动作维度
            use_orthogonal: (bool) 是否使用正交初始化
            gain: (float) 初始化增益
        """
        super(Categorical, self).__init__()
        # 选择初始化方法
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        # 创建线性层
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        """
        前向传播，生成动作分布。
        
        参数：
            x: (torch.Tensor) 输入特征
            available_actions: (torch.Tensor) 可用动作掩码（可选）
            
        返回：
            FixedCategorical: 动作分布
        """
        x = self.linear(x)
        # 如果提供了可用动作掩码，则将不可用动作的logits设为负无穷
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        # 返回固定的Categorical分布
        return FixedCategorical(logits=x)
