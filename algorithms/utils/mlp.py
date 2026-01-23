import torch.nn as nn
from .util import init, get_clones

"""MLP（多层感知机）模块。"""

class MLPLayer(nn.Module):
    """MLP隐藏层实现"""
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        # 激活函数选择：True使用ReLU，False使用Tanh
        active_func = nn.ReLU() if use_ReLU else nn.Tanh()
        # 初始化方法选择：True使用正交初始化，False使用Xavier均匀初始化
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
        # 根据激活函数计算增益
        gain = nn.init.calculate_gain('relu' if use_ReLU else 'tanh')

        # 自定义初始化函数
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        # 第一层全连接
        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        # 隐藏层
        self.fc_h = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        # 创建多个隐藏层克隆
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        """前向传播"""
        x = self.fc1(x)
        for fc_layer in self.fc2:
            x = fc_layer(x)
        return x


class MLPBase(nn.Module):
    """MLP基础网络，用于特征提取"""
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        # 获取观察空间维度
        obs_dim = obs_shape[0]

        # 特征归一化层
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        # 创建MLP网络
        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        """前向传播"""
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x