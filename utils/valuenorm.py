import numpy as np
import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    """价值归一化层，用于对价值函数的输出进行归一化
    
    在第一个norm_axes维度上对观察向量进行归一化
    """

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape  # 输入形状
        self.norm_axes = norm_axes  # 归一化的轴数
        self.epsilon = epsilon  # 防止除零的小值
        self.beta = beta  # 衰减系数，用于更新统计信息
        self.per_element_update = per_element_update  # 是否按元素更新
        self.tpdv = dict(dtype=torch.float32, device=device)  # 张量数据类型和设备

        # 不可学习参数：用于统计归一化的均值、方差等
        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)
        
        self.reset_parameters()  # 初始化参数

    def reset_parameters(self):
        """重置归一化参数"""
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        """计算去偏后的均值和方差"""
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector):
        """更新归一化统计信息
        
        Args:
            input_vector: 输入向量
        """
        # 处理numpy数组输入
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        # 确保输入在正确的设备上
        input_vector = input_vector.to(**self.tpdv)

        # 计算当前批次的均值和平方均值
        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        # 根据是否按元素更新选择权重
        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[:self.norm_axes])
            weight = self.beta ** batch_size
        else:
            weight = self.beta

        # 更新移动平均统计信息
        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        """对输入向量进行归一化
        
        Args:
            input_vector: 输入向量
        
        Returns:
            归一化后的向量
        """
        # 处理numpy数组输入
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        # 确保输入在正确的设备上
        input_vector = input_vector.to(**self.tpdv)

        # 获取去偏后的均值和方差
        mean, var = self.running_mean_var()
        # 归一化输入
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        
        return out

    def denormalize(self, input_vector):
        """将归一化向量反归一化回原始分布
        
        Args:
            input_vector: 归一化后的向量
        
        Returns:
            反归一化后的向量
        """
        # 处理numpy数组输入
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        # 确保输入在正确的设备上
        input_vector = input_vector.to(**self.tpdv)

        # 获取去偏后的均值和方差
        mean, var = self.running_mean_var()
        # 反归一化输入
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        
        # 转换为numpy数组返回
        out = out.cpu().numpy()
        
        return out
