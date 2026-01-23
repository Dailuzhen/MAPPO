import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PopArt(torch.nn.Module):
    """PopArt 归一化层，用于自适应调整价值函数的输出范围"""
    
    def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-5, device=torch.device("cpu")):
        
        super(PopArt, self).__init__()

        self.beta = beta  # 衰减系数，用于更新统计信息
        self.epsilon = epsilon  # 防止除零的小值
        self.norm_axes = norm_axes  # 归一化的轴数
        self.tpdv = dict(dtype=torch.float32, device=device)  # 张量数据类型和设备

        self.input_shape = input_shape  # 输入形状
        self.output_shape = output_shape  # 输出形状

        # 可学习参数：权重和偏置
        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(**self.tpdv)
        self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)
        
        # 不可学习参数：用于统计归一化的均值、方差等
        self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

        self.reset_parameters()  # 初始化参数

    def reset_parameters(self):
        """初始化网络参数"""
        # Kaiming 均匀初始化权重
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # 初始化偏置
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        # 重置统计参数
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, input_vector):
        """前向传播，执行线性变换"""
        # 处理 numpy 数组输入
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        # 确保输入在正确的设备上
        input_vector = input_vector.to(**self.tpdv)

        return F.linear(input_vector, self.weight, self.bias)
    
    @torch.no_grad()
    def update(self, input_vector):
        """更新归一化统计信息"""
        # 处理 numpy 数组输入
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        # 确保输入在正确的设备上
        input_vector = input_vector.to(**self.tpdv)
        
        # 保存旧的均值和标准差
        old_mean, old_stddev = self.mean, self.stddev

        # 计算当前批次的均值和平方均值
        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        # 更新移动平均统计信息
        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        # 计算新的标准差
        self.stddev = (self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4)

        # 调整权重和偏置以保持输出范围一致
        self.weight = self.weight * old_stddev / self.stddev
        self.bias = (old_stddev * self.bias + old_mean - self.mean) / self.stddev

    def debiased_mean_var(self):
        """计算去偏后的均值和方差"""
        # 计算去偏均值
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        # 计算去偏平方均值
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        # 计算去偏方差
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        """对输入向量进行归一化"""
        # 处理 numpy 数组输入
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        # 确保输入在正确的设备上
        input_vector = input_vector.to(**self.tpdv)

        # 获取去偏后的均值和方差
        mean, var = self.debiased_mean_var()
        # 归一化输入
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        
        return out

    def denormalize(self, input_vector):
        """对归一化向量进行反归一化"""
        # 处理 numpy 数组输入
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        # 确保输入在正确的设备上
        input_vector = input_vector.to(**self.tpdv)

        # 获取去偏后的均值和方差
        mean, var = self.debiased_mean_var()
        # 反归一化输入
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        
        # 转换为 numpy 数组返回
        out = out.cpu().numpy()

        return out
