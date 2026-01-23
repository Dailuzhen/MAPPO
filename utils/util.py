import numpy as np
import math
import torch

def check(input):
    """检查输入类型，将numpy数组转换为torch张量
    
    Args:
        input: 输入数据，可以是numpy数组或torch张量
    
    Returns:
        torch张量
    """
    return torch.from_numpy(input) if isinstance(input, np.ndarray) else input
        
def get_gard_norm(it):
    """计算梯度范数
    
    Args:
        it: 模型参数迭代器
    
    Returns:
        梯度范数
    """
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """线性调整学习率
    
    Args:
        optimizer: 优化器
        epoch: 当前epoch
        total_num_epochs: 总epoch数
        initial_lr: 初始学习率
    """
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    """计算Huber损失
    
    Args:
        e: 误差
        d: Huber损失的阈值
    
    Returns:
        Huber损失值
    """
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    """计算均方误差损失
    
    Args:
        e: 误差
    
    Returns:
        均方误差损失值
    """
    return e**2/2

def get_shape_from_obs_space(obs_space):
    """从观察空间获取形状
    
    Args:
        obs_space: 观察空间
    
    Returns:
        观察空间的形状
    """
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError(f"不支持的观察空间类型: {obs_space.__class__.__name__}")
    return obs_shape

def get_shape_from_act_space(act_space):
    """从动作空间获取形状
    
    Args:
        act_space: 动作空间
    
    Returns:
        动作空间的形状
    """
    # 只支持Discrete动作空间
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    else:
        raise NotImplementedError(f"只支持Discrete动作空间，当前类型: {act_space.__class__.__name__}")
    return act_shape


def tile_images(img_nhwc):
    """将多个图像拼接成一个大图像
    
    Args:
        img_nhwc: 图像列表或数组，形状为[N, H, W, C]
            N = 批次索引, H = 高度, W = 宽度, C = 通道数
    
    Returns:
        拼接后的大图像，形状为[H*P, W*Q, C]，其中P和Q尽可能接近
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    # 计算拼接后的行数和列数
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    # 填充不足的图像
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    # 拼接图像
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c