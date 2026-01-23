import copy
import numpy as np

import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    """初始化模型参数
    
    Args:
        module: 要初始化的模型模块
        weight_init: 权重初始化函数
        bias_init: 偏置初始化函数
        gain: 初始化增益
    
    Returns:
        初始化后的模块
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    """创建模型模块的多个副本
    
    Args:
        module: 要复制的模型模块
        N: 副本数量
    
    Returns:
        包含N个模块副本的ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def check(input):
    """检查输入类型，将numpy数组转换为torch张量
    
    Args:
        input: 输入数据，可以是numpy数组或torch张量
    
    Returns:
        torch张量
    """
    return torch.from_numpy(input) if isinstance(input, np.ndarray) else input
