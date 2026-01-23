import torch # 导入PyTorch库，用于深度学习计算
import numpy as np # 导入NumPy库，用于数值计算和数组操作
from utils.util import get_shape_from_obs_space, get_shape_from_act_space # 从工具文件中导入获取观测空间和动作空间形状的函数


class SharedReplayBuffer(object):
    """
    用于存储训练数据的缓冲区。
    
    参数：
        args: (argparse.Namespace) 包含相关模型、策略和环境信息的参数
        num_agents: (int) 环境中的智能体数量
        obs_space: (gym.Space) 智能体的观察空间
        cent_obs_space: (gym.Space) 智能体的中心化观察空间
        act_space: (gym.Space) 智能体的动作空间
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        self.episode_length = args.episode_length # 获取每个episode的时长（步数）
        self.n_rollout_threads = args.n_rollout_threads # 获取并行运行环境的线程数量
        self.gamma = args.gamma # 获取折扣因子gamma，用于计算回报
        self.gae_lambda = args.gae_lambda # 获取GAE（广义优势估计）的lambda参数
        self._use_gae = args.use_gae # 获取是否使用GAE的标志
        self._use_popart = args.use_popart # 获取是否使用PopArt归一化技术的标志
        self._use_valuenorm = args.use_valuenorm # 获取是否使用价值归一化技术的标志
        self._use_proper_time_limits = args.use_proper_time_limits # 获取是否使用正确时间限制处理的标志

        obs_shape = get_shape_from_obs_space(obs_space) # 获取观测空间的形状
        share_obs_shape = get_shape_from_obs_space(cent_obs_space) # 获取共享（中心化）观测空间的形状

        if type(obs_shape[-1]) == list: # 检查观测形状的最后一个元素是否为列表（通常用于处理混合观测空间）
            obs_shape = obs_shape[:1] # 如果是，则只取第一个元素

        if type(share_obs_shape[-1]) == list: # 检查共享观测形状的最后一个元素是否为列表
            share_obs_shape = share_obs_shape[:1] # 如果是，则只取第一个元素

        # 初始化共享观测数据的存储数组，形状为(episode长度+1, 线程数, 智能体数, *共享观测维度)
        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape),
                                  dtype=np.float32)
        # 初始化观测数据的存储数组，形状为(episode长度+1, 线程数, 智能体数, *观测维度)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        # 初始化价值预测值的存储数组，形状为(episode长度+1, 线程数, 智能体数, 1)
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        # 初始化回报值的存储数组，形状与价值预测值相同
        self.returns = np.zeros_like(self.value_preds)

        if act_space.__class__.__name__ == 'Discrete': # 检查动作空间是否为离散空间
            # 如果是离散动作空间，初始化可用动作的掩码，形状为(episode长度+1, 线程数, 智能体数, 动作维度)
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n),
                                             dtype=np.float32)
        else: # 如果不是离散动作空间
            self.available_actions = None # 设置可用动作为None

        act_shape = get_shape_from_act_space(act_space) # 获取动作空间的形状

        # 初始化动作数据的存储数组，形状为(episode长度, 线程数, 智能体数, 动作维度)
        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        # 初始化动作对数概率的存储数组，形状为(episode长度, 线程数, 智能体数, 动作维度)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        # 初始化奖励值的存储数组，形状为(episode长度, 线程数, 智能体数, 1)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        # 初始化掩码（用于标记环境是否结束），形状为(episode长度+1, 线程数, 智能体数, 1)
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        # 初始化bad_masks（用于处理非正常结束的情况），形状与masks相同
        self.bad_masks = np.ones_like(self.masks)
        # 初始化active_masks（用于标记智能体是否存活/激活），形状与masks相同
        self.active_masks = np.ones_like(self.masks)

        self.step = 0 # 初始化当前步数为0

    def insert(self, share_obs, obs, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        向缓冲区插入数据。
        
        参数：
            share_obs: (np.ndarray) 共享观察
            obs: (np.ndarray) 本地智能体观察
            actions: (np.ndarray) 智能体采取的动作
            action_log_probs: (np.ndarray) 动作的对数概率
            value_preds: (np.ndarray) 价值函数预测
            rewards: (np.ndarray) 收集的奖励
            masks: (np.ndarray) 表示环境是否终止的掩码
            bad_masks: (np.ndarray) 表示是否为真实终止状态或由于episode限制的掩码
            active_masks: (np.ndarray) 表示智能体是否活跃的掩码
            available_actions: (np.ndarray) 智能体可用的动作（如果为None，则所有动作可用）
        """
        self.share_obs[self.step + 1] = share_obs.copy() # 将下一时刻的共享观测存入缓冲区
        self.obs[self.step + 1] = obs.copy() # 将下一时刻的观测存入缓冲区
        
        # 插入其他数据
        self.actions[self.step] = actions.copy() # 将当前时刻采取的动作存入缓冲区
        self.action_log_probs[self.step] = action_log_probs.copy() # 将当前时刻动作的对数概率存入缓冲区
        self.value_preds[self.step] = value_preds.copy() # 将当前时刻的价值预测存入缓冲区
        self.rewards[self.step] = rewards.copy() # 将当前时刻获得的奖励存入缓冲区
        self.masks[self.step + 1] = masks.copy() # 将下一时刻的掩码存入缓冲区
        
        # 处理可选掩码
        if bad_masks is not None: # 如果提供了bad_masks
            self.bad_masks[self.step + 1] = bad_masks.copy() # 将下一时刻的bad_masks存入缓冲区
        if active_masks is not None: # 如果提供了active_masks
            self.active_masks[self.step + 1] = active_masks.copy() # 将下一时刻的active_masks存入缓冲区
        if available_actions is not None: # 如果提供了available_actions
            self.available_actions[self.step + 1] = available_actions.copy() # 将下一时刻的可用动作存入缓冲区

        self.step = (self.step + 1) % self.episode_length # 更新当前步数，并在达到episode长度时循环

    def chooseinsert(self, share_obs, obs, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        向缓冲区插入数据。此插入函数专门用于基于回合制的Hanabi游戏。
        
        参数：
            share_obs: (argparse.Namespace) 包含相关模型、策略和环境信息的参数。
            obs: (np.ndarray) 本地智能体观察。
            actions:(np.ndarray) 智能体采取的动作。
            action_log_probs:(np.ndarray) 智能体采取动作的对数概率。
            value_preds: (np.ndarray) 每一步的价值函数预测。
            rewards: (np.ndarray) 每一步收集的奖励。
            masks: (np.ndarray) 表示环境是否终止。
            bad_masks: (np.ndarray) 表示是真实终止状态还是由于episode限制。
            active_masks: (np.ndarray) 表示智能体在环境中是活跃还是死亡。
            available_actions: (np.ndarray) 每个智能体可用的动作。如果为None，则所有动作可用。
        """
        self.share_obs[self.step] = share_obs.copy() # 存储当前步的共享观测
        self.obs[self.step] = obs.copy() # 存储当前步的观测
        self.actions[self.step] = actions.copy() # 存储当前步的动作
        self.action_log_probs[self.step] = action_log_probs.copy() # 存储当前步的动作对数概率
        self.value_preds[self.step] = value_preds.copy() # 存储当前步的价值预测
        self.rewards[self.step] = rewards.copy() # 存储当前步的奖励
        self.masks[self.step + 1] = masks.copy() # 存储下一步的掩码
        if bad_masks is not None: # 如果提供了bad_masks
            self.bad_masks[self.step + 1] = bad_masks.copy() # 存储下一步的bad_masks
        if active_masks is not None: # 如果提供了active_masks
            self.active_masks[self.step] = active_masks.copy() # 存储当前步的active_masks
        if available_actions is not None: # 如果提供了available_actions
            self.available_actions[self.step] = available_actions.copy() # 存储当前步的可用动作

        self.step = (self.step + 1) % self.episode_length # 更新当前步数

    def after_update(self):
        """将最后一个时间步的数据复制到第一个索引。在模型更新后调用。"""
        self.share_obs[0] = self.share_obs[-1].copy() # 将最后一个时间步的共享观测复制到起始位置
        self.obs[0] = self.obs[-1].copy() # 将最后一个时间步的观测复制到起始位置
        self.masks[0] = self.masks[-1].copy() # 将最后一个时间步的掩码复制到起始位置
        self.bad_masks[0] = self.bad_masks[-1].copy() # 将最后一个时间步的bad_masks复制到起始位置
        self.active_masks[0] = self.active_masks[-1].copy() # 将最后一个时间步的active_masks复制到起始位置
        if self.available_actions is not None: # 如果存在可用动作数据
            self.available_actions[0] = self.available_actions[-1].copy() # 将最后一个时间步的可用动作复制到起始位置

    def chooseafter_update(self):
        """将最后一个时间步的数据复制到第一个索引。此方法用于Hanabi游戏。"""
        self.masks[0] = self.masks[-1].copy() # 将最后一个时间步的掩码复制到起始位置
        self.bad_masks[0] = self.bad_masks[-1].copy() # 将最后一个时间步的bad_masks复制到起始位置

    def compute_returns(self, next_value, value_normalizer=None):
        """
        计算回报，要么使用奖励的折扣总和，要么使用GAE。
        
        参数：
            next_value: (np.ndarray) 最后一个episode步骤之后的价值预测
            value_normalizer: (PopArt) 如果不是None，则为PopArt价值归一化器实例
        """
        if self._use_proper_time_limits: # 如果使用了正确的时间限制处理
            if self._use_gae: # 如果使用GAE（广义优势估计）
                self.value_preds[-1] = next_value # 设置最后一步的价值预测
                gae = 0 # 初始化GAE变量
                #self.reward -奖励数组，形状[时间步数，并行环境数，智能体数]
                for step in reversed(range(self.rewards.shape[0])): # 逆序遍历每一个时间步
                    if self._use_popart or self._use_valuenorm: # 如果使用了PopArt或ValueNorm归一化
                        # step + 1
                        # 计算TD误差 delta
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        # 计算GAE
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1] # 应用bad_masks
                        # 计算回报值（normalized后的value + GAE）
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else: # 如果没有使用归一化
                        # 计算TD误差 delta
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        # 计算GAE
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1] # 应用bad_masks
                        # 计算回报值
                        self.returns[step] = gae + self.value_preds[step]
            else: # 如果不使用GAE
                self.returns[-1] = next_value # 设置最后一步的回报为next_value
                for step in reversed(range(self.rewards.shape[0])): # 逆序遍历
                    if self._use_popart or self._use_valuenorm: # 如果使用了归一化
                        # 计算回报：R_t = r_t + gamma * R_{t+1}
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                    else: # 如果没有使用归一化
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else: # 如果不使用正确的时间限制处理（标准处理）
            if self._use_gae: # 如果使用GAE
                self.value_preds[-1] = next_value # 设置最后一步的价值
                gae = 0 # 初始化GAE
                for step in reversed(range(self.rewards.shape[0])): # 逆序遍历
                    if self._use_popart or self._use_valuenorm: # 如果使用归一化
                        # 计算TD误差 delta
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        # 计算GAE
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        # 计算回报
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else: # 如果不使用归一化
                        # 计算TD误差 delta
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        # 计算GAE
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        # 计算回报
                        self.returns[step] = gae + self.value_preds[step]
            else: # 如果不使用GAE
                self.returns[-1] = next_value # 设置最后一步的回报
                for step in reversed(range(self.rewards.shape[0])): # 逆序遍历
                    # 标准的回报计算 R_t = r_t + gamma * R_{t+1}
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        生成MLP策略的训练数据。
        
        参数：
            advantages: (np.ndarray) 优势估计值
            num_mini_batch: (int) 将批次分割成的小批次数量
            mini_batch_size: (int) 每个小批次中的样本数量
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3] # 获取episode长度，线程数，智能体数
        batch_size = n_rollout_threads * episode_length * num_agents # 计算总的批次大小

        if mini_batch_size is None: # 如果没有指定小批次大小
            assert batch_size >= num_mini_batch, ( # 确保总批次大小大于等于小批次数量
                "PPO要求进程数 ({}) "
                "* 步数 ({}) * 智能体数量 ({}) = {} "
                "大于或等于PPO小批次数量 ({})。"
                "".format(n_rollout_threads, episode_length, num_agents,
                          n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch # 计算小批次大小

        rand = torch.randperm(batch_size).numpy() # 生成随机排列的索引
        # TODO
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)] # 将索引分割成小批次

        # 将数据展平以便进行批处理
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:]) # 展平共享观测
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:]) # 展平观测
        actions = self.actions.reshape(-1, self.actions.shape[-1]) # 展平动作
        if self.available_actions is not None: # 如果有可用动作
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1]) # 展平可用动作
        value_preds = self.value_preds[:-1].reshape(-1, 1) # 展平价值预测
        returns = self.returns[:-1].reshape(-1, 1) # 展平回报
        masks = self.masks[:-1].reshape(-1, 1) # 展平掩码
        active_masks = self.active_masks[:-1].reshape(-1, 1) # 展平活跃掩码
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1]) # 展平动作对数概率
        advantages = advantages.reshape(-1, 1) # 展平优势值

        for indices in sampler: # 遍历每个小批次的索引
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices] # 获取当前批次的共享观测
            obs_batch = obs[indices] # 获取当前批次的观测
            actions_batch = actions[indices] # 获取当前批次的动作
            if self.available_actions is not None: # 如果有可用动作
                available_actions_batch = available_actions[indices] # 获取当前批次的可用动作
            else:
                available_actions_batch = None # 否则为None
            value_preds_batch = value_preds[indices] # 获取当前批次的价值预测
            return_batch = returns[indices] # 获取当前批次的回报
            masks_batch = masks[indices] # 获取当前批次的掩码
            active_masks_batch = active_masks[indices] # 获取当前批次的活跃掩码
            old_action_log_probs_batch = action_log_probs[indices] # 获取当前批次的旧动作对数概率
            if advantages is None: # 如果没有优势值
                adv_targ = None # 目标优势为None
            else:
                adv_targ = advantages[indices] # 获取当前批次的优势值

            # 使用yield生成器返回当前批次的数据
            yield (share_obs_batch, obs_batch, actions_batch,
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,
                  adv_targ, available_actions_batch)
