import numpy as np
import random
import os
from gym import spaces


class Env(object):
    """
       多智能体路径规划环境（MAPF）

       支持两种地图模式：
         1. 从文件加载（map_file）
         2. 随机生成（size_range + obstacle_density）

       地图更换策略：
         - 文件模式：地图永不改变
         - 随机模式：每 map_change_interval 个 episode 更换障碍物布局（grid_size 固定为首次随机值）

       每次 reset() 都会重置智能体和目标位置。
       """

    def __init__(
            self,
            agent_num=2,  # 智能体数量
            obstacle_density=0.2,  # 障碍物密度（仅随机模式）
            size_range=(8, 12),  # 地图尺寸范围（仅随机模式）
            map_file=None,  # 地图文件路径（.npy 或 .txt 格式）
            map_change_interval=0,  # 0 表示不更换地图（即使随机模式）
            obs_radius=2,  # 局部观测半径
            max_episode_steps=100,  # 每个 episode 最大步数
            seed=None  # 随机种子
    ):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.agent_num = agent_num
        self.obstacle_density = obstacle_density
        self.size_range = size_range
        self.obs_radius = obs_radius
        self.max_episode_steps = max_episode_steps
        self.map_change_interval = map_change_interval
        self.episode_count = 0  # 记录已经运行了多少轮 episode，用于随机地图的更换判断

        # 判断模式
        self.use_random_map = (map_file is None)

        # ===== 地图初始化 =====
        if map_file is not None:
            if not os.path.exists(map_file):
                raise FileNotFoundError(f"Map file not found: {map_file}")
            self.map = self._load_map_from_file(map_file)
            self.grid_size = self.map.shape[0]
            print(f"[INFO] Loaded fixed map from {map_file}, size: {self.grid_size}x{self.grid_size}")
        else:
            min_sz, max_sz = size_range
            if min_sz > max_sz:
                raise ValueError("size_range must be (min, max) with min <= max")
            self.grid_size = random.randint(min_sz, max_sz)
            self._build_random_map()
            print(f"[INFO] Initialized random map mode, fixed size: {self.grid_size}x{self.grid_size}")

        # 验证地图
        assert self.map.shape == (self.grid_size, self.grid_size), "Map must be square!"
        assert self.map.dtype.kind in ['i', 'u'], "Map must be integer type"

        # 初始化状态变量
        self.agents_pos = None
        self.goals_pos = None
        self.steps = 0

        # 计算观测维度（固定！因为 grid_size 在 init 后不再变化）
        self.obs_dim = (2 * self.obs_radius + 1) ** 2
        self.share_obs_dim = self.grid_size * self.grid_size

        # 定义空间（Gym 要求 init 后不可变）
        self.observation_space = [
            spaces.Box(low=-2.0, high=4.0, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.agent_num)
        ]
        self.action_space = [spaces.Discrete(5) for _ in range(self.agent_num)]

        self.share_observation_space = [
            spaces.Box(low=0.0, high=4.0, shape=(self.share_obs_dim,), dtype=np.float32)
            for _ in range(self.agent_num)
        ]

    def _load_map_from_file(self, filepath):
        """加载 .npy 或 .txt 格式的地图"""
        if filepath.endswith('.npy'):
            return np.load(filepath).astype(int)
        elif filepath.endswith('.txt') or filepath.endswith('.csv'):
            with open(filepath, 'r') as f:
                lines = []
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#'):  # 忽略空行和注释
                        parts = stripped.split()
                        lines.append([int(x) for x in parts])
            return np.array(lines, dtype=int)
        else:
            raise ValueError("Unsupported map format. Use .npy or .txt/.csv")

    def _build_random_map(self):
        """在当前 self.grid_size 下生成新障碍物布局（不改变尺寸）"""
        total_cells = self.grid_size * self.grid_size
        num_obstacles = int(total_cells * self.obstacle_density)

        self.map = np.zeros((self.grid_size, self.grid_size), dtype=int)
        obstacle_positions = set()

        while len(obstacle_positions) < num_obstacles:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (x, y) not in obstacle_positions:
                obstacle_positions.add((x, y))
                self.map[x, y] = 1

    def _get_local_obs(self, agent_id):
        """获取局部观测（以智能体为中心的 (2r+1)x(2r+1) 区域）"""
        x, y = self.agents_pos[agent_id]
        r = self.obs_radius
        obs = np.full((2 * r + 1, 2 * r + 1), -2, dtype=np.float32)  # -2 表示视野外

        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                nx, ny = x + i, y + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if (nx, ny) == (x, y):
                        val = 2  # 自己
                    elif (nx, ny) in self.agents_pos:
                        val = 3  # 其他智能体
                    elif (nx, ny) in self.goals_pos:
                        val = 4  # 目标
                    elif self.map[nx, ny] == 1:
                        val = 1  # 障碍物
                    else:
                        val = 0  # 空地
                    obs[i + r, j + r] = val
        return obs.flatten()

    def _get_share_obs(self):
        """全局共享观测：整个地图 + 所有智能体和目标位置"""
        obs = self.map.astype(np.float32).copy()
        for x, y in self.agents_pos:
            obs[x, y] = 2
        for x, y in self.goals_pos:
            if obs[x, y] != 2:  # 如果目标和智能体重合，优先显示智能体
                obs[x, y] = 4
        return obs.flatten()

    def reset(self):
        """重置环境状态"""
        self.steps = 0
        self.episode_count += 1

        # ===== 决定是否更换地图（仅随机模式且 interval > 0）=====
        if self.use_random_map and self.map_change_interval > 0:
            if (self.episode_count - 1) % self.map_change_interval == 0:
                self._build_random_map()
                print(f"[MAP CHANGE] Episode {self.episode_count}: New obstacle layout generated.")

        # ===== 重置智能体和目标位置 =====
        free_positions = [
            (i, j) for i in range(self.grid_size) for j in range(self.grid_size)
            if self.map[i, j] == 0
        ]
        if len(free_positions) < 2 * self.agent_num:
            raise RuntimeError(
                f"Not enough free space! Need {2 * self.agent_num}, got {len(free_positions)}"
            )
        random.shuffle(free_positions)
        self.agents_pos = free_positions[:self.agent_num]
        self.goals_pos = free_positions[self.agent_num:2 * self.agent_num]

        # 构建观测
        obs = np.stack([self._get_local_obs(i) for i in range(self.agent_num)])
        share_obs = np.stack([self._get_share_obs()] * self.agent_num)

        return obs, share_obs

    def step(self, actions):
        """
        执行一步动作
        actions: list of int, length = agent_num
        Returns: (obs, share_obs, rewards, dones, info)
        """
        self.steps += 1
        next_agents_pos = []
        rewards = [0.0] * self.agent_num
        done_all = True

        # 临时存储新位置（先计算，再统一更新，避免顺序依赖）
        for i, action in enumerate(actions):
            x, y = self.agents_pos[i]
            dx, dy = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][action]  # 0: stay, 1-4: up/down/left/right
            nx, ny = x + dx, y + dy

            # 边界和障碍检查
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.map[nx, ny] == 0:
                next_agents_pos.append((nx, ny))
            else:
                next_agents_pos.append((x, y))  # 保持原位

        # 碰撞检测（智能体之间不能重叠）
        pos_set = set()
        collision = [False] * self.agent_num
        for i, pos in enumerate(next_agents_pos):
            if pos in pos_set:
                collision[i] = True
                next_agents_pos[i] = self.agents_pos[i]  # 回退
            else:
                pos_set.add(pos)

        self.agents_pos = next_agents_pos

        # 计算奖励和完成状态
        for i in range(self.agent_num):
            if collision[i]:
                rewards[i] = -0.5
            elif self.agents_pos[i] == self.goals_pos[i]:
                rewards[i] = 1.0
            else:
                # 可选：加入距离奖励（例如负曼哈顿距离）
                gx, gy = self.goals_pos[i]
                ax, ay = self.agents_pos[i]
                rewards[i] = -0.01 * (abs(ax - gx) + abs(ay - gy))

            if self.agents_pos[i] != self.goals_pos[i]:
                done_all = False

        truncated = (self.steps >= self.max_episode_steps)
        terminated = done_all
        dones = [terminated or truncated] * self.agent_num

        # 构建观测
        obs = np.stack([self._get_local_obs(i) for i in range(self.agent_num)])
        share_obs = np.stack([self._get_share_obs()] * self.agent_num)

        info = {
            "collision": collision,
            "episode_step": self.steps,
            "episode_count": self.episode_count
        }

        return obs, share_obs, rewards, dones, info

    def render(self, save_path=None):
        """
        渲染当前环境状态

        Args:
            save_path (str or None): 如果提供路径，则保存图像到该路径；否则显示窗口
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            raise ImportError("Rendering requires matplotlib. Please install it via 'pip install matplotlib'")

        # 构建渲染用的 grid（基于当前 map + agent/goal 位置）
        grid = self.map.copy().astype(float)
        for x, y in self.agents_pos:
            grid[x, y] = 2
        for x, y in self.goals_pos:
            if grid[x, y] != 2:  # 智能体优先级更高
                grid[x, y] = 4

        # 设置颜色映射
        cmap = plt.get_cmap('tab10')
        colors = {
            0: 'white',  # 空地
            1: 'black',  # 障碍物
            2: 'blue',  # 智能体
            4: 'green'  # 目标
        }

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # 使 (0,0) 在左上角，符合矩阵索引习惯
        ax.set_xticks([])
        ax.set_yticks([])

        # 绘制每个格子
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                val = grid[i, j]
                color = colors.get(val, 'white')
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=0.5, edgecolor='gray', facecolor=color)
                ax.add_patch(rect)

                # 可选：添加文字标签（用于调试小地图）
                if self.grid_size <= 12:
                    if val == 2:
                        ax.text(j, i, 'A', ha='center', va='center', fontsize=12, color='white', weight='bold')
                    elif val == 4:
                        ax.text(j, i, 'G', ha='center', va='center', fontsize=12, color='white', weight='bold')

        plt.title(f"Episode {self.episode_count}, Step {self.steps}")

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
        else:
            plt.show()

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        return [seed]


if __name__ == "__main__":
    # 创建环境实例
    env = Env(agent_num=2, size_range=(8, 8), obstacle_density=0.2)

    # 首先调用 reset() 初始化环境
    obs, share_obs = env.reset()
    print(f"Reset successful. Obs shape: {obs.shape}, Share obs shape: {share_obs.shape}")

    # 使用随机动作进行测试
    import numpy as np

    actions = [np.random.choice(5) for _ in range(env.agent_num)]  # 生成有效的动作列表
    print(f"Actions: {actions}")

    # 执行一步
    obs, share_obs, rewards, dones, info = env.step(actions)
    print(f"Step successful. Rewards: {rewards}, Dones: {dones}")
    print(f"Info: {info}")

