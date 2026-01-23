# test_integration.py
from envs.env import Env
from envs.env_wrappers import DummyVecEnv
import numpy as np


def test_env_integration():
    """测试环境与项目的集成兼容性"""
    print("开始测试环境集成...")

    def make_test_env():
        env = Env(agent_num=2, size_range=(8, 8), obstacle_density=0.2)
        env.seed(1)  # 测试种子设置
        return env

    # 创建多个环境实例来满足 DummyVecEnv 的批量处理需求
    n_envs = 2  # 创建2个环境实例
    env_fns = [lambda: make_test_env() for _ in range(n_envs)]

    # 为每个环境创建函数设置不同的种子以避免完全相同的行为
    def make_test_env_with_seed(seed):
        def init_env():
            env = Env(agent_num=2, size_range=(8, 8), obstacle_density=0.2)
            env.seed(seed)
            return env

        return init_env

    env_fns = [make_test_env_with_seed(1 + i) for i in range(n_envs)]

    # 测试向量化环境
    print("创建向量化环境...")
    vec_env = DummyVecEnv(env_fns)

    # 测试重置
    print("测试环境重置...")
    obs, share_obs = vec_env.reset()
    print(f"VecEnv reset - Obs shape: {obs.shape}, Share obs shape: {share_obs.shape}")

    # 测试步进
    print("测试环境步进...")
    # 为每个环境提供动作：[env1_actions, env2_actions, ...]
    # 每个环境的动作是一个包含所有智能体动作的列表
    actions = [[np.random.choice(5) for _ in range(2)] for _ in range(n_envs)]  # 2个环境，每个环境2个智能体
    obs, share_obs, rewards, dones, infos = vec_env.step(actions)

    print(f"VecEnv step - Obs shape: {obs.shape}")
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
    print(f"All spaces defined correctly: {hasattr(vec_env, 'action_space')}")

    # 关闭环境
    vec_env.close()
    print("集成测试完成！")


if __name__ == "__main__":
    test_env_integration()
