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
            size_range=(10, 10),  # 地图尺寸范围（仅随机模式）
            map_file=None,  # 地图文件路径（.npy 或 .txt 格式）
            shared_map=None,  # 共享地图数组（所有并行环境使用同一地图时传入）
            map_change_interval=0,  # 0 表示不更换地图（即使随机模式）
            obs_radius=2,  # 局部观测半径
            max_episode_steps=100,  # 每个 episode 最大步数
            seed=None,  # 随机种子
            use_astar_shaping=True,  # 是否使用A*路径引导奖励
            progress_reward=0.05,  # 沿参考路径前进的固定奖励
            no_progress_penalty=0.0,  # 未前进时的惩罚（可为0）
            use_astar_first_episode=True,  # 是否仅首个episode用A*动作
            obs_include_goal_direction=True,  # 是否在观测中包含目标方向（v8新增）
            obs_include_position=True  # 是否在观测中包含当前位置（v8新增）
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
        self.use_astar_shaping = use_astar_shaping
        self.progress_reward = progress_reward
        self.no_progress_penalty = no_progress_penalty
        self.use_astar_first_episode = use_astar_first_episode
        self.map_locked = False
        
        # v8: 观测增强参数
        self.obs_include_goal_direction = obs_include_goal_direction
        self.obs_include_position = obs_include_position

        # 判断模式：shared_map 或 map_file 时不再随机更换地图
        self.use_random_map = (map_file is None and shared_map is None)

        # ===== 地图初始化 =====
        if shared_map is not None:
            # 方案A：使用共享地图，保证所有并行环境障碍物布局一致
            self.map = np.asarray(shared_map, dtype=np.int64).copy()
            self.grid_size = self.map.shape[0]
            if self.grid_size != self.map.shape[1]:
                raise ValueError("shared_map must be square!")
        elif map_file is not None:
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
        self.ref_paths = None
        self.ref_remaining = None
        self.start_pos = None
        self.fixed_goals = None
        self.fixed_first_episode_only = False
        self.fixed_first_episode_used = False

        # 计算观测维度（固定！因为 grid_size 在 init 后不再变化）
        # 基础局部观测: (2r+1)^2 = 25 维
        self.local_obs_dim = (2 * self.obs_radius + 1) ** 2
        
        # v8: 额外观测维度
        self.extra_obs_dim = 0
        if self.obs_include_goal_direction:
            self.extra_obs_dim += 4  # dx, dy, dist, at_goal
        if self.obs_include_position:
            self.extra_obs_dim += 2  # x/size, y/size
        
        self.obs_dim = self.local_obs_dim + self.extra_obs_dim
        self.share_obs_dim = self.agent_num * self.obs_dim
        
        print(f"[INFO] 观测维度: {self.obs_dim} (局部:{self.local_obs_dim} + 额外:{self.extra_obs_dim})")

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

    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _astar_path(self, start, goal):
        """A*路径规划，返回包含start和goal的路径列表；无路返回None。"""
        if start == goal:
            return [start]

        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            open_set.remove(current)
            cx, cy = current
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue
                if self.map[nx, ny] == 1:
                    continue
                neighbor = (nx, ny)
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    open_set.add(neighbor)

        return None

    def _action_from_path(self, agent_id):
        """
        根据A*路径生成下一步离散动作。
        如果智能体偏离路径（如碰撞后），会重新计算A*路径。
        无路或已到终点则返回stay(0)。
        """
        # 检查是否已到终点
        if self.agents_pos[agent_id] == self.goals_pos[agent_id]:
            return 0
        
        path = None if self.ref_paths is None else self.ref_paths[agent_id]
        
        # 如果没有路径或者当前位置不在路径上，重新计算路径
        if path is None:
            path = self._astar_path(self.agents_pos[agent_id], self.goals_pos[agent_id])
            if path is None:
                return 0
            if self.ref_paths is None:
                self.ref_paths = [None] * self.agent_num
            self.ref_paths[agent_id] = path
        
        try:
            idx = path.index(self.agents_pos[agent_id])
        except ValueError:
            # 当前位置不在路径上（可能因碰撞偏离），重新计算路径
            path = self._astar_path(self.agents_pos[agent_id], self.goals_pos[agent_id])
            if path is None:
                return 0
            self.ref_paths[agent_id] = path
            idx = 0  # 新路径的起点
        
        if idx >= len(path) - 1:
            return 0  # 已到终点
        
        cx, cy = path[idx]
        nx, ny = path[idx + 1]
        if nx == cx - 1 and ny == cy:
            return 1  # up
        if nx == cx + 1 and ny == cy:
            return 2  # down
        if nx == cx and ny == cy - 1:
            return 3  # left
        if nx == cx and ny == cy + 1:
            return 4  # right
        return 0

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
        """
        获取增强观测（v8）
        
        包含:
        1. 局部网格 (2r+1)x(2r+1) = 25 维
        2. 目标相对方向 (dx, dy, dist, at_goal) = 4 维 (可选)
        3. 当前位置归一化 (x/size, y/size) = 2 维 (可选)
        """
        x, y = self.agents_pos[agent_id]
        gx, gy = self.goals_pos[agent_id]
        r = self.obs_radius
        
        # 1. 局部网格观测
        local_obs = np.full((2 * r + 1, 2 * r + 1), -2, dtype=np.float32)  # -2 表示视野外

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
                    local_obs[i + r, j + r] = val
        
        obs_parts = [local_obs.flatten()]
        
        # 2. 目标相对方向（v8新增）
        if self.obs_include_goal_direction:
            dx = (gx - x) / self.grid_size  # 归一化到 [-1, 1]
            dy = (gy - y) / self.grid_size
            dist = (abs(gx - x) + abs(gy - y)) / (2 * self.grid_size)  # 曼哈顿距离归一化
            at_goal = 1.0 if (x, y) == (gx, gy) else 0.0
            obs_parts.append(np.array([dx, dy, dist, at_goal], dtype=np.float32))
        
        # 3. 当前位置归一化（v8新增）
        if self.obs_include_position:
            pos_x = x / self.grid_size
            pos_y = y / self.grid_size
            obs_parts.append(np.array([pos_x, pos_y], dtype=np.float32))
        
        return np.concatenate(obs_parts)

    def _get_share_obs(self):
        """共享观测：拼接所有智能体的局部观测"""
        return np.concatenate([self._get_local_obs(i) for i in range(self.agent_num)], axis=0)

    def reset(self):
        """重置环境状态"""
        self.steps = 0
        self.episode_count += 1

        # 仅首回合固定起点/终点，其余回合重新采样
        if self.fixed_first_episode_only and self.fixed_first_episode_used:
            self.start_pos = None
            self.fixed_goals = None

        # ===== 决定是否更换地图（仅随机模式且 interval > 0）=====
        if self.use_random_map and self.map_change_interval > 0:
            if (self.episode_count - 1) % self.map_change_interval == 0:
                self._build_random_map()
                print(f"[MAP CHANGE] Episode {self.episode_count}: New obstacle layout generated.")

        # ===== 重置智能体和目标位置（确保可达） =====
        attempts = 0
        max_attempts = 200
        paths = None
        if self.start_pos is not None and self.fixed_goals is not None:
            self.agents_pos = list(self.start_pos)
            self.goals_pos = list(self.fixed_goals)
            paths = []
            for i in range(self.agent_num):
                path = self._astar_path(self.agents_pos[i], self.goals_pos[i])
                if path is None:
                    raise RuntimeError("Fixed start/goal are not reachable.")
                paths.append(path)
        else:
            while True:
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

                paths = []
                reachable = True
                for i in range(self.agent_num):
                    path = self._astar_path(self.agents_pos[i], self.goals_pos[i])
                    if path is None:
                        reachable = False
                        break
                    paths.append(path)

                if reachable:
                    break

                attempts += 1
                if attempts >= max_attempts:
                    if self.use_random_map and not self.map_locked:
                        self._build_random_map()
                        attempts = 0
                    else:
                        raise RuntimeError("Failed to sample reachable start/goal pairs on fixed map.")

            self.start_pos = list(self.agents_pos)
            self.fixed_goals = list(self.goals_pos)
            self.map_locked = True

        if self.fixed_first_episode_only and not self.fixed_first_episode_used:
            self.fixed_first_episode_used = True

        # 生成参考路径（A*）
        self.ref_paths = [None] * self.agent_num
        self.ref_remaining = [None] * self.agent_num
        if self.use_astar_shaping or self.use_astar_first_episode:
            for i in range(self.agent_num):
                self.ref_paths[i] = paths[i]
                self.ref_remaining[i] = len(paths[i]) - 1

        # 构建观测
        obs = np.stack([self._get_local_obs(i) for i in range(self.agent_num)])
        share_obs = np.stack([self._get_share_obs()] * self.agent_num)

        return obs, share_obs

    def _regenerate_start_goal(self):
        """
        重新随机生成新的起点和终点（在同一episode内，到达终点后调用）
        保持地图不变，只重新采样智能体位置和目标位置
        此方法保证总是能成功生成新的起点终点
        """
        max_attempts = 500  # 增加尝试次数
        
        for attempt in range(max_attempts):
            # 获取所有空闲位置
            free_positions = [
                (i, j) for i in range(self.grid_size) for j in range(self.grid_size)
                if self.map[i, j] == 0
            ]
            
            if len(free_positions) < 2 * self.agent_num:
                # 空间不足，这不应该发生
                print(f"[警告] 空闲位置不足: {len(free_positions)}")
                continue
            
            random.shuffle(free_positions)
            new_agents_pos = free_positions[:self.agent_num]
            new_goals_pos = free_positions[self.agent_num:2 * self.agent_num]
            
            # 确保起点和终点不同
            valid = True
            for i in range(self.agent_num):
                if new_agents_pos[i] == new_goals_pos[i]:
                    valid = False
                    break
            if not valid:
                continue
            
            # 检查可达性
            paths = []
            reachable = True
            for i in range(self.agent_num):
                path = self._astar_path(new_agents_pos[i], new_goals_pos[i])
                if path is None or len(path) < 2:
                    reachable = False
                    break
                paths.append(path)
            
            if reachable:
                # 更新位置
                self.agents_pos = list(new_agents_pos)
                self.goals_pos = list(new_goals_pos)
                self.start_pos = list(new_agents_pos)
                self.fixed_goals = list(new_goals_pos)
                
                # 更新 A* 参考路径
                self.ref_paths = paths
                self.ref_remaining = [len(p) - 1 for p in paths]
                return True
        
        # 如果多次尝试失败（不应该发生），打印警告
        print(f"[警告] _regenerate_start_goal 失败了 {max_attempts} 次")
        # 最后尝试：使用当前位置作为起点，随机选择一个可达的终点
        for i in range(self.agent_num):
            for pos in free_positions:
                if pos != self.agents_pos[i]:
                    path = self._astar_path(self.agents_pos[i], pos)
                    if path and len(path) >= 2:
                        self.goals_pos[i] = pos
                        self.fixed_goals[i] = pos
                        self.ref_paths[i] = path
                        self.ref_remaining[i] = len(path) - 1
                        break
        
        self.start_pos = list(self.agents_pos)
        return True

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
        prev_remaining = self.ref_remaining.copy() if self.ref_remaining is not None else None

        # 首个episode用A*动作覆盖
        if self.use_astar_first_episode and self.episode_count == 1:
            actions = [self._action_from_path(i) for i in range(self.agent_num)]

        # 临时存储新位置（先计算，再统一更新，避免顺序依赖）
        for i, action in enumerate(actions):
            if isinstance(action, (list, tuple, np.ndarray)):
                action = int(np.argmax(action))
            else:
                action = int(action)
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
                rewards[i] = -5
            elif self.agents_pos[i] == self.goals_pos[i]:
                rewards[i] = 10
            else:
                # 可选：加入距离奖励（例如负曼哈顿距离）
                gx, gy = self.goals_pos[i]
                ax, ay = self.agents_pos[i]
                rewards[i] = -0.01 * (abs(ax - gx) + abs(ay - gy))

            # A*路径进度奖励
            if self.use_astar_shaping:
                path = self.ref_paths[i] if self.ref_paths is not None else None
                if path is None:
                    pass
                else:
                    try:
                        idx = path.index(self.agents_pos[i])
                        remaining = len(path) - 1 - idx
                    except ValueError:
                        remaining = prev_remaining[i]

                    if prev_remaining[i] is not None and remaining is not None:
                        if remaining < prev_remaining[i]:
                            rewards[i] += self.progress_reward
                        else:
                            rewards[i] += self.no_progress_penalty
                    self.ref_remaining[i] = remaining

            if self.agents_pos[i] != self.goals_pos[i]:
                done_all = False

        truncated = (self.steps >= self.max_episode_steps)
        looped_to_start = False

        # 如果本回合已全部到达终点但还未到达最大步数，则重新生成新的起点终点
        if done_all and not truncated:
            # 重新随机生成新的起点和终点
            self._regenerate_start_goal()
            done_all = False
            looped_to_start = True
            self._stuck_counter = 0  # 重置卡住计数器
        
        # 检测智能体被阻塞的情况（使用简单的连续无移动步数检测）
        if not done_all and not truncated and not looped_to_start:
            # 检查是否有智能体已到达终点但其他智能体被阻塞
            agents_at_goal = [self.agents_pos[i] == self.goals_pos[i] for i in range(self.agent_num)]
            if any(agents_at_goal) and not all(agents_at_goal):
                # 有部分智能体到达终点
                # 检查位置是否与上一步相同
                if hasattr(self, '_prev_positions') and self.agents_pos == self._prev_positions:
                    self._stuck_counter = getattr(self, '_stuck_counter', 0) + 1
                else:
                    self._stuck_counter = 0
                
                # 如果连续 3 步没有移动且有智能体被阻塞，触发重置
                if self._stuck_counter >= 3:
                    self._regenerate_start_goal()
                    done_all = False
                    looped_to_start = True
                    self._stuck_counter = 0
            else:
                self._stuck_counter = 0
            
            self._prev_positions = list(self.agents_pos)

        terminated = done_all
        dones = [terminated or truncated] * self.agent_num

        # 构建观测
        obs = np.stack([self._get_local_obs(i) for i in range(self.agent_num)])
        share_obs = np.stack([self._get_share_obs()] * self.agent_num)

        info = {
            "collision": collision,
            "episode_step": self.steps,
            "episode_count": self.episode_count,
            "looped_to_start": looped_to_start
        }

        return obs, share_obs, rewards, dones, info

    def render(self, mode="human", save_path=None):
        """
        渲染当前环境状态（栅格化）
        - 灰色方块：障碍物
        - 实心圆：智能体起点
        - 空心圆：智能体终点
        同一智能体的起点与终点使用相同颜色

        Args:
            mode (str): "human" 或 "rgb_array"
            save_path (str or None): 如果提供路径，则保存图像到该路径
        Returns:
            如果 mode 为 "rgb_array"，返回图像数组 (H, W, 3)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            raise ImportError("Rendering requires matplotlib. Please install it via 'pip install matplotlib'")

        # 颜色映射：每个智能体一个颜色
        cmap = plt.get_cmap('tab10')
        agent_colors = [cmap(i % 10) for i in range(self.agent_num)]

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # 使 (0,0) 在左上角，符合矩阵索引习惯
        ax.set_xticks([])
        ax.set_yticks([])

        # 绘制网格与障碍物
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                facecolor = 'white'
                if self.map[i, j] == 1:
                    facecolor = 'lightgray'
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=0.5, edgecolor='gray', facecolor=facecolor)
                ax.add_patch(rect)

        # 绘制起点（实心圆）与终点（空心圆）
        if self.start_pos is not None:
            for idx, (sx, sy) in enumerate(self.start_pos):
                color = agent_colors[idx]
                start_circle = patches.Circle((sy, sx), radius=0.35, facecolor=color, edgecolor='black', linewidth=1.0)
                ax.add_patch(start_circle)
                ax.text(sy, sx, str(idx + 1), ha='center', va='center', fontsize=10, color='white', weight='bold')

        # 绘制当前智能体位置（较小实心圆）
        if self.agents_pos is not None:
            for idx, (ax_pos, ay_pos) in enumerate(self.agents_pos):
                color = agent_colors[idx]
                current_circle = patches.Circle((ay_pos, ax_pos), radius=0.22, facecolor=color, edgecolor='black', linewidth=0.8)
                ax.add_patch(current_circle)

        if self.goals_pos is not None:
            for idx, (gx, gy) in enumerate(self.goals_pos):
                color = agent_colors[idx]
                goal_circle = patches.Circle((gy, gx), radius=0.35, facecolor='none', edgecolor=color, linewidth=2.0)
                ax.add_patch(goal_circle)

        plt.title(f"Episode {self.episode_count}, Step {self.steps}")

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)

        if mode == "rgb_array":
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape((height, width, 3))
            plt.close(fig)
            return image
        elif mode == "human":
            if save_path is None:
                plt.show()
            plt.close(fig)
            return None
        else:
            plt.close(fig)
            raise NotImplementedError(f"不支持的渲染模式: {mode}")

    def render_ascii(self):
        """
        控制台可视化地图：
        - 1 表示障碍物
        - 0 表示可通行区域
        - A1/A2... 表示智能体起始/当前位置
        - G1/G2... 表示目标点
        """
        grid = [["0" for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.map[i, j] == 1:
                    grid[i][j] = "1"

        for idx, (x, y) in enumerate(self.agents_pos, start=1):
            grid[x][y] = f"A{idx}"

        for idx, (x, y) in enumerate(self.goals_pos, start=1):
            if grid[x][y].startswith("A"):
                grid[x][y] = f"A{idx}/G{idx}"
            else:
                grid[x][y] = f"G{idx}"

        for row in grid:
            print(" ".join(f"{cell:>3}" for cell in row))

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def close(self):
        return None

    # ============================================================================
    #                    Segment 相关方法（Phase A 使用）
    # ============================================================================
    
    def _is_free_cell(self, x, y):
        """检查坐标 (x, y) 是否为可通行格子（非障碍物且在边界内）"""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        return self.map[x, y] == 0

    def set_start_goal(self, starts, goals):
        """
        手动设置所有智能体的起点和终点（用于 Phase A 策略对比）
        校验起点和终点必须在可通行格子上，避免终点在障碍物中的问题。
        
        Args:
            starts: list of tuples, 每个智能体的起点坐标 [(x1,y1), (x2,y2), ...]
            goals: list of tuples, 每个智能体的终点坐标 [(x1,y1), (x2,y2), ...]
        """
        assert len(starts) == self.agent_num, f"starts 长度 {len(starts)} != agent_num {self.agent_num}"
        assert len(goals) == self.agent_num, f"goals 长度 {len(goals)} != agent_num {self.agent_num}"
        
        # 校验起点和终点不在障碍物中
        for i, (s, g) in enumerate(zip(starts, goals)):
            sx, sy = int(s[0]), int(s[1])
            gx, gy = int(g[0]), int(g[1])
            if not self._is_free_cell(sx, sy):
                raise ValueError(
                    f"Agent {i} 起点 ({sx}, {sy}) 在障碍物或越界，当前地图该格 map[{sx},{sy}]={self.map[sx, sy] if (0 <= sx < self.grid_size and 0 <= sy < self.grid_size) else 'OOB'}"
                )
            if not self._is_free_cell(gx, gy):
                raise ValueError(
                    f"Agent {i} 终点 ({gx}, {gy}) 在障碍物或越界，当前地图该格 map[{gx},{gy}]={self.map[gx, gy] if (0 <= gx < self.grid_size and 0 <= gy < self.grid_size) else 'OOB'}"
                )
        
        self.agents_pos = [tuple(s) for s in starts]
        self.goals_pos = [tuple(g) for g in goals]
        self.start_pos = list(self.agents_pos)
        self.fixed_goals = list(self.goals_pos)
        
        # 重算 A* 路径
        self._recalculate_astar_paths()
    
    def _recalculate_astar_paths(self):
        """重新计算所有智能体的 A* 参考路径"""
        self.ref_paths = []
        self.ref_remaining = []
        
        for i in range(self.agent_num):
            path = self._astar_path(self.agents_pos[i], self.goals_pos[i])
            self.ref_paths.append(path)
            self.ref_remaining.append(len(path) - 1 if path else None)
    
    def reset_to_segment(self, segment_info):
        """
        根据 segment 配置重置环境（不改变地图，只重置位置）
        用于 Phase A 策略对比时，将环境重置到指定场景
        
        Args:
            segment_info: dict, 包含 "agents" 列表，每个元素有 "start" 和 "goal"
            
        Returns:
            obs: 观测数组 [num_agents, obs_dim]
        """
        starts = [tuple(a["start"]) for a in segment_info["agents"]]
        goals = [tuple(a["goal"]) for a in segment_info["agents"]]
        
        self.set_start_goal(starts, goals)
        self.steps = 0  # 重置步数计数
        
        # 重建观测
        obs = np.stack([self._get_local_obs(i) for i in range(self.agent_num)])
        share_obs = np.stack([self._get_share_obs()] * self.agent_num)
        
        return obs
    
    def get_current_scene(self):
        """
        获取当前场景配置（用于保存 scene_config）
        
        Returns:
            dict: 包含 map_size 和 agents 信息
        """
        return {
            "map_size": self.grid_size,
            "agents": [
                {
                    "agent_id": i,
                    "start": list(self.start_pos[i]) if self.start_pos else list(self.agents_pos[i]),
                    "goal": list(self.goals_pos[i])
                }
                for i in range(self.agent_num)
            ]
        }
    
    def get_astar_steps_to_goal(self):
        """
        获取当前场景下各智能体到目标的 A* 步数
        
        Returns:
            list: 每个智能体的 A* 步数列表
        """
        steps = []
        for i in range(self.agent_num):
            if self.ref_paths and self.ref_paths[i]:
                steps.append(len(self.ref_paths[i]) - 1)
            else:
                path = self._astar_path(self.agents_pos[i], self.goals_pos[i])
                steps.append(len(path) - 1 if path else None)
        return steps


if __name__ == "__main__":
    # 创建环境实例
    env = Env(agent_num=2, size_range=(10, 10), obstacle_density=0.2)

    # 首先调用 reset() 初始化环境
    obs, share_obs = env.reset()
    print(f"Reset successful. Obs shape: {obs.shape}, Share obs shape: {share_obs.shape}")
    print("Initial map:")
    env.render_ascii()

    # 使用随机动作进行测试
    import numpy as np

    actions = [np.random.choice(5) for _ in range(env.agent_num)]  # 生成有效的动作列表
    print(f"Actions: {actions}")

    # 执行一步
    obs, share_obs, rewards, dones, info = env.step(actions)
    print(f"Step successful. Rewards: {rewards}, Dones: {dones}")
    print(f"Info: {info}")
    print("Map after one step:")
    env.render_ascii()

