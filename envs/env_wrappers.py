"""
环境包装器，修改自 OpenAI Baselines 代码以支持多智能体环境
"""

import numpy as np


class DummyVecEnv():
    """简单的向量环境包装器，用于同步执行多个环境
    
    该包装器将多个独立的环境包装成一个向量环境，支持同步步进和重置。
    """
    
    def __init__(self, env_fns):
        """初始化向量环境
        
        Args:
            env_fns: 环境构造函数列表，每个函数返回一个独立的环境实例
        """
        # 创建环境实例列表
        self.envs = [fn() for fn in env_fns]
        # 获取第一个环境的信息，用于确定空间类型
        env = self.envs[0]
        self.num_envs = len(env_fns)  # 环境数量
        # 观察空间
        self.observation_space = env.observation_space
        # 共享观察空间（多智能体环境中使用）
        self.share_observation_space = env.share_observation_space
        # 动作空间
        self.action_space = env.action_space
        self.actions = None  # 存储当前动作

    def step(self, actions):
        """同步执行环境步进
        
        Args:
            actions: 动作数组，形状为 [num_envs, agent_num, action_dim]
        
        Returns:
            obs: 观察数组，形状为 [num_envs, agent_num, obs_dim]
            rews: 奖励数组，形状为 [num_envs, agent_num]
            dones: 终止标志数组，形状为 [num_envs, agent_num]
            infos: 信息字典列表
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        """异步设置动作
        
        Args:
            actions: 动作数组
        """
        self.actions = actions

    def step_wait(self):
        """等待异步步进完成并返回结果
        
        Returns:
            obs: 观察数组
            rews: 奖励数组
            dones: 终止标志数组
            infos: 信息字典列表
        """
        # 执行所有环境的步进
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        # 解包结果并转换为数组
        if len(results) > 0 and len(results[0]) == 5:
            obs, _share_obs, rews, dones, infos = map(np.array, zip(*results))
        else:
            obs, rews, dones, infos = map(np.array, zip(*results))

        # 处理环境重置
        for i, done in enumerate(dones):
            # 检查 done 类型
            if isinstance(done, bool):
                # 单智能体环境
                if done:
                    reset_out = self.envs[i].reset()
                    if isinstance(reset_out, tuple) and len(reset_out) == 2:
                        reset_out = reset_out[0]
                    obs[i] = reset_out
            else:
                # 多智能体环境，所有智能体都终止时才重置
                if np.all(done):
                    reset_out = self.envs[i].reset()
                    if isinstance(reset_out, tuple) and len(reset_out) == 2:
                        reset_out = reset_out[0]
                    obs[i] = reset_out

        self.actions = None  # 清空动作缓存
        return obs, rews, dones, infos

    def reset(self):
        """重置所有环境
        
        Returns:
            obs: 观察数组，形状为 [env_num, agent_num, obs_dim]
        """
        obs = [env.reset() for env in self.envs]
        if len(obs) > 0 and isinstance(obs[0], tuple) and len(obs[0]) == 2:
            obs = [o[0] for o in obs]
        return np.array(obs)

    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        """渲染所有环境
        
        Args:
            mode: 渲染模式，"human" 或 "rgb_array"
        
        Returns:
            如果 mode 为 "rgb_array"，返回渲染图像数组
        """
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError(f"不支持的渲染模式: {mode}")