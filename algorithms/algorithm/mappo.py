"""
# MAPPO算法实现，用于多智能体策略优化
"""

import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.valuenorm import ValueNorm
from algorithms.utils.util import check

class MAPPO():
    """
    MAPPO算法训练器类，用于更新策略网络。
    
    参数：
        args: (argparse.Namespace) 包含模型、策略和环境相关参数的命名空间
        policy: (MAPPOPolicy) 需要更新的策略网络
        device: (torch.device) 指定运行设备（cpu/gpu）
    """

    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)  # 张量数据类型和设备配置
        self.policy = policy  # 策略网络实例

        # PPO核心超参数
        self.clip_param = args.clip_param  # PPO裁剪参数
        self.ppo_epoch = args.ppo_epoch  # PPO训练轮数
        self.num_mini_batch = args.num_mini_batch  # 每轮训练的小批量数量
        self.value_loss_coef = args.value_loss_coef  # 价值损失系数
        self.entropy_coef = args.entropy_coef  # 熵正则化系数
        self.max_grad_norm = args.max_grad_norm  # 梯度裁剪阈值
        self.huber_delta = args.huber_delta  # Huber损失的delta参数

        # 训练配置标志
        self._use_max_grad_norm = args.use_max_grad_norm  # 是否使用梯度裁剪
        self._use_clipped_value_loss = args.use_clipped_value_loss  # 是否使用裁剪的价值损失
        self._use_huber_loss = args.use_huber_loss  # 是否使用Huber损失
        self._use_popart = args.use_popart  # 是否使用PopArt归一化
        self._use_valuenorm = args.use_valuenorm  # 是否使用价值归一化
        self._use_value_active_masks = args.use_value_active_masks  # 是否在价值损失中使用active masks
        self._use_policy_active_masks = args.use_policy_active_masks  # 是否在策略损失中使用active masks

        # 确保PopArt和ValueNorm不同时使用
        #PopArt 是在网络层内部做数学变换，ValueNorm 是在网络层外部（或者作为一层包装）做变换。 
        #如果你两个都开，数据会被“缩放两次”，导致梯度计算（模型学习）时出现混乱，模型根本不知道该听谁的
        assert not (self._use_popart and self._use_valuenorm), "PopArt和ValueNorm不能同时设置为True"

        # 初始化价值归一化器
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out  # 使用PopArt归一化
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)  # 使用ValueNorm归一化
        else:
            self.value_normalizer = None  # 不使用价值归一化

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        计算价值函数损失。
        
        参数：
            values: (torch.Tensor) 当前价值函数预测值
            value_preds_batch: (torch.Tensor) 数据批次中的"旧"价值预测值（用于价值裁剪损失）
            return_batch: (torch.Tensor) 回报目标值
            active_masks_batch: (torch.Tensor) 表示智能体在给定时间步是否活跃的掩码
            
        返回：
            value_loss: (torch.Tensor) 价值函数损失
        """
        # 计算裁剪后的价值预测
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        
        # 计算价值误差
        if self._use_popart or self._use_valuenorm:
            # 使用归一化处理
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            # 直接计算误差
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values
        
        # 计算价值损失
        if self._use_huber_loss:
            # 使用Huber损失
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            # 使用均方误差
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)
        
        # 应用价值裁剪
        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original
        
        # 应用active masks（如果启用）
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()
        
        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        执行一次PPO更新，同时更新策略网络和价值网络。
        
        参数：
            sample: (Tuple) 包含用于更新网络的数据批次
            update_actor: (bool) 是否更新策略网络
            
        返回：
            value_loss: (torch.Tensor) 价值函数损失
            critic_grad_norm: (torch.Tensor) 价值网络梯度范数
            policy_loss: (torch.Tensor) 策略损失
            dist_entropy: (torch.Tensor) 动作分布的熵
            actor_grad_norm: (torch.Tensor) 策略网络梯度范数
            imp_weights: (torch.Tensor) 重要性采样权重
        """
        # 解包数据样本
        share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch = sample

        # 将张量转移到指定设备
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # 一次前向传播计算所有步骤的价值、动作对数概率和熵
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, actions_batch, available_actions_batch, active_masks_batch
        )
        
        # 策略网络更新
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)  # 重要性采样权重
        surr1 = imp_weights * adv_targ  # 未裁剪的优势函数
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ  # 裁剪后的优势函数

        # 计算策略损失
        if self._use_policy_active_masks:
            # 使用active masks加权
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            # 直接取均值
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        # 策略网络梯度更新
        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            # 加上熵正则化项
            #公式意思：策略损失 = 优势函数 * 重要性采样权重 - 熵正则化项 * 熵系数
            (policy_loss - dist_entropy * self.entropy_coef).backward()#公式意思：

        # 计算并应用梯度裁剪
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # 价值网络更新
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()

        # 计算并应用梯度裁剪
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        使用小批量梯度下降执行PPO训练更新。
        
        参数：
            buffer: (SharedReplayBuffer) 包含训练数据的共享回放缓冲区
            update_actor: (bool) 是否更新策略网络
            
        返回：
            train_info: (dict) 包含训练更新相关信息的字典（如损失、梯度范数等）
        """
        # 计算优势函数
        if self._use_popart or self._use_valuenorm:
            # 使用归一化处理
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            # 直接计算优势函数
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        
        # 优势函数归一化
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan  # 屏蔽无效数据
        mean_advantages = np.nanmean(advantages_copy)  # 计算均值（忽略NaN）
        std_advantages = np.nanstd(advantages_copy)  # 计算标准差（忽略NaN）
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)  # 归一化优势函数

        # 初始化训练信息字典
        train_info = {
            'value_loss': 0,       # 价值损失
            'policy_loss': 0,      # 策略损失
            'dist_entropy': 0,     # 动作分布熵
            'actor_grad_norm': 0,  # 策略网络梯度范数
            'critic_grad_norm': 0, # 价值网络梯度范数
            'ratio': 0             # 重要性采样权重比
        }

        # 执行PPO训练轮数
        for _ in range(self.ppo_epoch):
            # 获取数据生成器
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)
            
            # 遍历每个小批量数据
            for sample in data_generator:
                # 执行PPO更新
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.ppo_update(sample, update_actor)
                
                # 累加训练信息
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        # 计算平均训练信息
        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """
        准备训练模式，将策略网络和价值网络设置为训练模式。
        """
        self.policy.actor.train()  # 策略网络设为训练模式
        self.policy.critic.train()  # 价值网络设为训练模式

    def prep_rollout(self):
        """
        准备采样模式，将策略网络和价值网络设置为评估模式。
        """
        self.policy.actor.eval()  # 策略网络设为评估模式
        self.policy.critic.eval()  # 价值网络设为评估模式

