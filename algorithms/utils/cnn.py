import torch.nn as nn
from .util import init

"""CNN（卷积神经网络）模块和工具。"""

class Flatten(nn.Module):
    """将卷积层输出展平为全连接层输入"""
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNLayer(nn.Module):
    """CNN卷积层实现"""
    def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU, kernel_size=3, stride=1):
        super(CNNLayer, self).__init__()

        # 激活函数选择：True使用ReLU，False使用Tanh
        active_func = nn.ReLU() if use_ReLU else nn.Tanh()
        # 初始化方法选择：True使用正交初始化，False使用Xavier均匀初始化
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
        # 根据激活函数计算增益
        gain = nn.init.calculate_gain('relu' if use_ReLU else 'tanh')

        # 自定义初始化函数
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        # 获取输入形状
        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        # 计算卷积后特征图的尺寸
        output_size = (input_width - kernel_size + stride) * (input_height - kernel_size + stride)

        # CNN网络结构
        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=hidden_size // 2,
                            kernel_size=kernel_size,
                            stride=stride)
                  ),
            active_func,
            Flatten(),
            init_(nn.Linear(hidden_size // 2 * output_size, hidden_size)),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)),
            active_func)

    def forward(self, x):
        """前向传播"""
        # 图像归一化到[0, 1]
        x = x / 255.0
        x = self.cnn(x)
        return x


class CNNBase(nn.Module):
    """CNN基础网络，用于图像特征提取"""
    def __init__(self, args, obs_shape):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size

        # 创建CNN层
        self.cnn = CNNLayer(obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        """前向传播"""
        x = self.cnn(x)
        return x
