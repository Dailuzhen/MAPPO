"""
Plan D 最佳模型详细验证脚本（11 项诊断指标）

对 best_model 在 8 种地图尺寸上各运行 N 个场景，
每个场景逐步记录动作概率、A* 对比、距离变化、撞墙检测、熵、
奖励分量等信息，并为每个场景生成慢速 GIF（到达终点定格）。

每个尺寸生成独立日志文件，最后生成汇总日志。

用法:
  python scripts/detailed_validate.py
  python scripts/detailed_validate.py --model_dir <path> --scenarios 15
"""
import sys, os, argparse
from datetime import datetime
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.env import Env
from config import get_config
from algorithms.algorithm.MAPPOPolicy import MAPPOPolicy
from algorithms.utils.util import check
from gym import spaces

try:
    import imageio
except ImportError:
    imageio = None

try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

ACTION_NAMES = ["stay", "up", "down", "left", "right"]
ACTION_DELTAS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

DEFAULT_MODEL_DIR = "results/MyEnv/MyEnv/v8_planD/run1/phase_b_ppo/best_model"
DEFAULT_MAP_SIZES = [8, 10, 12, 14, 16, 18, 20, 22]
DEFAULT_SCENARIOS = 20
DEFAULT_OBS_RADIUS = 3
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_LAYER_N = 4
GIF_FREEZE_FRAMES = 6


def parse_args():
    p = argparse.ArgumentParser(description="详细模型验证（11 项诊断）")
    p.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    p.add_argument("--map_sizes", type=int, nargs="+", default=DEFAULT_MAP_SIZES)
    p.add_argument("--scenarios", type=int, default=DEFAULT_SCENARIOS)
    p.add_argument("--obs_radius", type=int, default=DEFAULT_OBS_RADIUS)
    p.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE)
    p.add_argument("--layer_N", type=int, default=DEFAULT_LAYER_N)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--gif_duration", type=float, default=0.8,
                    help="GIF 每帧持续时间（秒），到达终点后定格")
    p.add_argument("--out_dir", type=str, default=None,
                    help="输出目录，默认 logs/detailed_val")
    p.add_argument("--use_first_reach_reward", action="store_true", default=True,
                    help="与 planE 训练一致：首次到达+10，stay+0.1，全体+5")
    p.add_argument("--no_first_reach_reward", action="store_false", dest="use_first_reach_reward",
                    help="禁用首次到达制，每步到达都+10（旧行为）")
    p.add_argument("--use_wall_penalty", action="store_true", default=True,
                    help="与 planE 训练一致：撞墙-0.1")
    p.add_argument("--use_pbrs", action="store_true", default=True,
                    help="与 planE 训练一致：势能差分奖励塑形，距离靠近+正/远离+负")
    p.add_argument("--no_pbrs", action="store_false", dest="use_pbrs",
                    help="禁用 PBRS")
    return p.parse_args()


def create_policy(env, device, hidden_size=256, layer_n=4):
    parser = get_config()
    args = parser.parse_args(["--hidden_size", str(hidden_size),
                              "--layer_N", str(layer_n)])
    obs_dim = env.observation_space[0].shape[0]
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    share_obs_dim = obs_dim * env.agent_num
    share_obs_space = spaces.Box(low=-np.inf, high=np.inf,
                                  shape=(share_obs_dim,), dtype=np.float32)
    act_space = env.action_space[0]
    return MAPPOPolicy(args, obs_space, share_obs_space, act_space, device)


def load_best_model(policy, model_dir, device):
    actor_path = os.path.join(model_dir, "actor.pt")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"Actor not found: {actor_path}")
    policy.actor.load_state_dict(torch.load(actor_path, map_location=device))
    policy.actor.eval()
    return policy


def get_astar_paths(env):
    paths = []
    for i in range(env.agent_num):
        s = tuple(env.agents_pos[i])
        g = tuple(env.goals_pos[i])
        if s == g:
            paths.append([s])
        else:
            p = env._astar_path(s, g)
            paths.append(p if p else [s])
    return paths


def astar_recommended_action(env, agent_id):
    pos = tuple(env.agents_pos[agent_id])
    goal = tuple(env.goals_pos[agent_id])
    if pos == goal:
        return 0
    path = env._astar_path(pos, goal)
    if path is None or len(path) < 2:
        return 0
    nx, ny = path[1]
    cx, cy = pos
    dx, dy = nx - cx, ny - cy
    for act_id, (adx, ady) in enumerate(ACTION_DELTAS):
        if (adx, ady) == (dx, dy):
            return act_id
    return 0


def get_action_probs(policy, obs_tensor):
    with torch.no_grad():
        obs_t = check(obs_tensor).to(**policy.actor.tpdv)
        features = policy.actor.base(obs_t)
        probs = policy.actor.act.get_probs(features)
    return probs.cpu().numpy()


def compute_entropy(probs_vec):
    p = probs_vec[probs_vec > 0]
    return float(-np.sum(p * np.log(p)))


def classify_no_move(action, prev_pos, new_pos, collision_flags, agent_id):
    if prev_pos != new_pos:
        return None
    if action == 0:
        return "stay"
    if collision_flags[agent_id]:
        return "collision"
    return "wall"


def render_ascii_map(env):
    lines = []
    gs = env.grid_size
    label_map = {}
    if env.start_pos:
        for idx, pos in enumerate(env.start_pos):
            label_map[tuple(pos)] = f"S{idx}"
    if env.goals_pos:
        for idx, pos in enumerate(env.goals_pos):
            key = tuple(pos)
            if key in label_map:
                label_map[key] += f"/G{idx}"
            else:
                label_map[key] = f"G{idx}"
    for r in range(gs):
        row_parts = []
        for c in range(gs):
            key = (r, c)
            if key in label_map:
                row_parts.append(f"{label_map[key]:>4s}")
            elif env.map[r, c] == 1:
                row_parts.append("   #")
            else:
                row_parts.append("   .")
        lines.append("    " + "".join(row_parts))
    return "\n".join(lines)


def format_path(path, max_show=12):
    if not path:
        return "(无路径)"
    coords = [f"({p[0]},{p[1]})" for p in path]
    if len(coords) <= max_show:
        return "->".join(coords)
    half = max_show // 2
    return "->".join(coords[:half]) + "->..." + "->".join(coords[-half:])


class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.fh = open(filepath, "w", encoding="utf-8")

    def log(self, msg=""):
        print(msg)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def save_gif_with_freeze(frames, gif_path, duration, freeze_frames):
    """保存 GIF，最后一帧重复 freeze_frames 次实现定格效果"""
    if not frames or len(frames) < 2:
        return
    final_frames = list(frames)
    last = final_frames[-1]
    for _ in range(freeze_frames):
        final_frames.append(last)
    imageio.mimsave(gif_path, final_frames, duration=duration, loop=0)


def run_one_size(map_size, args, env, policy, device, out_dir, ts):
    """运行单个地图尺寸的所有场景，生成独立日志 + GIF"""
    size_key = f"{map_size}x{map_size}"
    log_path = os.path.join(out_dir, f"{size_key}_{ts}.log")
    gif_subdir = os.path.join(out_dir, size_key)
    logger = Logger(log_path)

    record_gif = imageio is not None
    if record_gif:
        os.makedirs(gif_subdir, exist_ok=True)

    logger.log(f"{'=' * 70}")
    logger.log(f"  {size_key} 详细验证 | {ts}")
    logger.log(f"{'=' * 70}")
    logger.log(f"  模型: {args.model_dir}")
    logger.log(f"  场景数: {args.scenarios} | 障碍物密度: {env.obstacle_density*100:.0f}%")
    logger.log(f"  Device: {device} | A* 完全禁用 | 确定性策略")
    logger.log(f"  奖励分量: distance=距离惩罚 | pbrs=势能差分(靠近+远离-) | wall=撞墙")
    logger.log(f"  GIF: duration={args.gif_duration}s, 终点定格={GIF_FREEZE_FRAMES}帧")
    logger.log()

    size_successes = 0
    size_stay_ratios = []
    size_loop_rates = []
    size_wall_bumps = []
    size_astar_deviate = []
    size_entropies = []
    size_collisions = 0
    size_timeout_dists = []
    failure_reasons = {"timeout": 0, "collision": 0}

    for sc_idx in range(args.scenarios):
        env.rebuild_map_new_size(map_size)
        env.start_pos = None
        env.fixed_goals = None
        env.episode_count = 999
        env.reset()
        env.use_astar_first_episode = False
        env.use_astar_shaping = False
        env._stuck_counter = 0

        start_positions = [tuple(p) for p in env.agents_pos]
        target_goals = [tuple(g) for g in env.goals_pos]
        astar_paths = get_astar_paths(env)
        astar_steps = max(1, max(len(p) - 1 for p in astar_paths))
        max_steps = max(200, int(astar_steps * 5))

        logger.log(f"{'=' * 70}")
        logger.log(f"场景 {sc_idx + 1}/{args.scenarios}")
        logger.log(f"{'=' * 70}")
        logger.log(f"  障碍物密度: {env.obstacle_density*100:.0f}% | "
                    f"A*最优步数: {astar_steps} | 最大步数: {max_steps}")
        for i in range(env.agent_num):
            a_steps = len(astar_paths[i]) - 1
            logger.log(f"  Agent {i}: 起点{start_positions[i]} -> 终点{target_goals[i]} | "
                        f"A*路径({a_steps}步): {format_path(astar_paths[i])}")
        logger.log()
        logger.log(f"  地图:")
        logger.log(render_ascii_map(env))
        logger.log()
        logger.log(f"  {'─' * 64}")

        obs = np.stack([env._get_local_obs(i) for i in range(env.agent_num)])
        cumulative_rewards = [0.0] * env.agent_num
        stay_count = 0
        wall_bump_count = 0
        astar_deviate_count = [0] * env.agent_num
        visit_counter = [Counter() for _ in range(env.agent_num)]
        trajectory = [[] for _ in range(env.agent_num)]
        step_entropies = []
        total_steps = 0
        reached = False
        collision_total = 0
        frames = []
        combined_pos_history = []

        for i in range(env.agent_num):
            visit_counter[i][start_positions[i]] += 1
            trajectory[i].append((start_positions[i], None, None))

        if record_gif:
            try:
                frames.append(env.render(mode="rgb_array"))
            except Exception:
                record_gif = False

        for step in range(max_steps):
            prev_positions = [tuple(p) for p in env.agents_pos]
            obs_tensor = torch.tensor(
                obs.reshape(-1, obs.shape[-1]),
                dtype=torch.float32, device=device
            )
            probs = get_action_probs(policy, obs_tensor)
            with torch.no_grad():
                action = policy.act(obs_tensor, deterministic=True)
            actions = [int(a) for a in action.cpu().numpy().flatten().tolist()]

            astar_actions = [astar_recommended_action(env, i)
                             for i in range(env.agent_num)]
            prev_dists = [abs(prev_positions[i][0] - target_goals[i][0]) +
                          abs(prev_positions[i][1] - target_goals[i][1])
                          for i in range(env.agent_num)]

            obs, _, rewards, dones, info = env.step(actions)
            total_steps += 1

            goals_reached_this_step = info.get("all_goals_reached", False)
            looped_to_start = info.get("looped_to_start", False)
            if goals_reached_this_step:
                new_positions = [tuple(g) for g in target_goals]
            elif looped_to_start:
                new_positions = [tuple(p) for p in prev_positions]
            else:
                new_positions = [tuple(p) for p in env.agents_pos]
            new_dists = [abs(new_positions[i][0] - target_goals[i][0]) +
                         abs(new_positions[i][1] - target_goals[i][1])
                         for i in range(env.agent_num)]

            collision_flags = info.get("collision", [False] * env.agent_num)
            reward_components = info.get("reward_components",
                                         [{}] * env.agent_num)

            logger.log(f"  Step {step + 1:03d}:")
            for i in range(env.agent_num):
                act_name = ACTION_NAMES[actions[i]]
                p_vec = probs[i]
                entropy = compute_entropy(p_vec)
                step_entropies.append(entropy)
                max_prob_action = int(np.argmax(p_vec))
                is_max = (actions[i] == max_prob_action)

                no_move_reason = classify_no_move(
                    actions[i], prev_positions[i], new_positions[i],
                    collision_flags, i
                )

                move_tag = ""
                if no_move_reason == "stay":
                    move_tag = "  [主动stay]"
                elif no_move_reason == "wall":
                    move_tag = "  [撞墙!]"
                    wall_bump_count += 1
                elif no_move_reason == "collision":
                    move_tag = "  [碰撞弹回!]"

                if actions[i] == 0:
                    stay_count += 1

                astar_match = (actions[i] == astar_actions[i])
                if not astar_match:
                    astar_deviate_count[i] += 1
                astar_tag = "[一致]" if astar_match else "[偏离A*!]"

                delta_d = new_dists[i] - prev_dists[i]
                if delta_d < 0:
                    dist_tag = "靠近"
                elif delta_d > 0:
                    dist_tag = "远离!"
                else:
                    dist_tag = "无变化"

                rc = reward_components[i] if i < len(reward_components) else {}
                step_reward = rewards[i] if i < len(rewards) else 0.0
                cumulative_rewards[i] += step_reward

                vc = visit_counter[i].get(new_positions[i], 0)
                visit_counter[i][new_positions[i]] = vc + 1
                revisit_tag = f"  [第{vc + 1}次访问]" if vc > 0 else ""

                trajectory[i].append((new_positions[i], act_name, no_move_reason))

                if collision_flags[i]:
                    collision_total += 1

                logger.log(f"    Agent {i}: {prev_positions[i]} -> 动作: {act_name} -> "
                            f"{new_positions[i]}{move_tag}{revisit_tag}")
                prob_str = "  ".join(f"{ACTION_NAMES[k]}={p_vec[k]*100:.1f}%"
                                    for k in range(5))
                logger.log(f"      概率: {prob_str}  熵={entropy:.2f}")
                logger.log(f"      选择: {act_name} {'是' if is_max else '非'}最大概率动作 "
                            f"[{'YES' if is_max else 'NO'}] | "
                            f"A*推荐: {ACTION_NAMES[astar_actions[i]]} {astar_tag}")
                logger.log(f"      距目标: d={prev_dists[i]} -> d={new_dists[i]} "
                            f"(Δ={delta_d:+d} {dist_tag})")

                rc_parts = []
                for key in ["goal", "collision", "distance", "astar",
                            "anti_stuck", "stay", "pbrs", "wall"]:
                    v = rc.get(key, 0.0)
                    if v != 0.0:
                        rc_parts.append(f"{key}={v:+.3f}")
                rc_str = "  ".join(rc_parts) if rc_parts else "无分量"
                logger.log(f"      奖励: {rc_str}  总计={step_reward:.3f} | "
                            f"累计={cumulative_rewards[i]:.3f}")

            combined_pos_history.append(tuple(new_positions))

            if record_gif and not goals_reached_this_step:
                try:
                    frames.append(env.render(mode="rgb_array"))
                except Exception:
                    pass

            if goals_reached_this_step:
                reached = True
                break

            all_at_goal = all(
                new_positions[i] == target_goals[i]
                for i in range(env.agent_num)
            )
            if all_at_goal:
                reached = True
                break

            if (info.get("looped_to_start", False)
                    and not info.get("all_goals_reached", False)):
                break

        logger.log(f"  {'─' * 64}")

        sc_tag = "ok" if reached else "timeout"
        if collision_total > 0 and not reached:
            sc_tag = "col"
            failure_reasons["collision"] += 1
        elif not reached:
            failure_reasons["timeout"] += 1

        total_actions = total_steps * env.agent_num
        sc_stay_ratio = stay_count / total_actions if total_actions > 0 else 0

        last_n = (combined_pos_history[-20:]
                  if len(combined_pos_history) >= 20
                  else combined_pos_history)
        if len(last_n) > 1:
            loop_rate = 1.0 - len(set(last_n)) / len(last_n)
        else:
            loop_rate = 0.0

        result_str = "成功" if reached else "超时"
        logger.log(f"  结果: {result_str} | 步数: {total_steps} | A*最优: {astar_steps} | "
                    f"步数比: {total_steps / astar_steps:.2f}x | 碰撞: {collision_total}")
        cum_str = "  ".join(f"Agent{i}={cumulative_rewards[i]:.2f}"
                            for i in range(env.agent_num))
        logger.log(f"  stay比例: {sc_stay_ratio*100:.1f}% | 循环率: {loop_rate*100:.1f}% | "
                    f"撞墙: {wall_bump_count}次 | 累计奖励: {cum_str}")
        for i in range(env.agent_num):
            dev_pct = (astar_deviate_count[i] / total_steps * 100
                       if total_steps > 0 else 0)
            logger.log(f"  偏离A*: Agent{i}={astar_deviate_count[i]}/{total_steps}"
                        f"({dev_pct:.0f}%)")

        if not reached:
            td = sum(abs(env.agents_pos[i][0] - target_goals[i][0]) +
                     abs(env.agents_pos[i][1] - target_goals[i][1])
                     for i in range(env.agent_num)) / env.agent_num
            logger.log(f"  超时距离: {td:.1f}")
            size_timeout_dists.append(td)

        avg_ent = float(np.mean(step_entropies)) if step_entropies else 0.0
        logger.log(f"  平均熵: {avg_ent:.3f}")
        logger.log()

        for i in range(env.agent_num):
            traj_parts = []
            for idx, (pos, act, reason) in enumerate(trajectory[i]):
                tag_str = ""
                if reason == "stay":
                    tag_str = "[stay]"
                elif reason == "wall":
                    tag_str = "[wall]"
                elif reason == "collision":
                    tag_str = "[col]"
                vc = visit_counter[i].get(pos, 0)
                if vc > 1 and idx > 0:
                    tag_str += f"[x{vc}]"
                traj_parts.append(f"({pos[0]},{pos[1]}){tag_str}")

            if len(traj_parts) > 24:
                shown = traj_parts[:10] + ["..."] + traj_parts[-10:]
            else:
                shown = traj_parts
            at_goal = trajectory[i][-1][0] == target_goals[i]
            mark = "OK" if at_goal else "FAIL"
            logger.log(f"  Agent {i} 轨迹: {'->'.join(shown)} [{mark}]")

            most_visited = visit_counter[i].most_common(3)
            revisits = sum(1 for _, c in visit_counter[i].items() if c > 1)
            mv_str = ", ".join(f"({p[0]},{p[1]})x{c}" for p, c in most_visited)
            logger.log(f"    重访位置: {revisits}个 | 最多: {mv_str}")

        if record_gif and len(frames) >= 2:
            gif_path = os.path.join(gif_subdir,
                                    f"scenario_{sc_idx + 1:02d}_{sc_tag}.gif")
            try:
                save_gif_with_freeze(frames, gif_path, args.gif_duration,
                                     GIF_FREEZE_FRAMES)
                logger.log(f"  GIF: {gif_path}")
            except Exception as e:
                logger.log(f"  GIF 保存失败: {e}")

        logger.log()

        if reached:
            size_successes += 1
        size_stay_ratios.append(sc_stay_ratio)
        size_loop_rates.append(loop_rate)
        size_wall_bumps.append(wall_bump_count)
        dev_rates = [astar_deviate_count[i] / total_steps
                     if total_steps > 0 else 0
                     for i in range(env.agent_num)]
        size_astar_deviate.append(float(np.mean(dev_rates)))
        size_entropies.append(avg_ent)
        size_collisions += collision_total

    sr = size_successes / args.scenarios * 100
    stats = {
        "success_rate": sr,
        "successes": size_successes,
        "total": args.scenarios,
        "avg_stay": float(np.mean(size_stay_ratios)),
        "avg_loop": float(np.mean(size_loop_rates)),
        "avg_wall_bumps": float(np.mean(size_wall_bumps)),
        "avg_astar_deviate": float(np.mean(size_astar_deviate)),
        "avg_entropy": float(np.mean(size_entropies)),
        "collisions": size_collisions,
        "avg_timeout_dist": (float(np.mean(size_timeout_dists))
                             if size_timeout_dists else 0),
        "failure_reasons": dict(failure_reasons),
    }

    logger.log(f"{'=' * 70}")
    logger.log(f"  {size_key} 汇总")
    logger.log(f"{'=' * 70}")
    logger.log(f"  成功率: {sr:.1f}% ({size_successes}/{args.scenarios})")
    logger.log(f"  stay%: {stats['avg_stay']*100:.1f}%")
    logger.log(f"  loop%: {stats['avg_loop']*100:.1f}%")
    logger.log(f"  撞墙: {stats['avg_wall_bumps']:.1f}/场景")
    logger.log(f"  偏离A*: {stats['avg_astar_deviate']*100:.1f}%")
    logger.log(f"  平均熵: {stats['avg_entropy']:.3f}")
    logger.log(f"  碰撞: {stats['collisions']}")
    if stats['avg_timeout_dist'] > 0:
        logger.log(f"  平均超时距离: {stats['avg_timeout_dist']:.1f}")
    fr = failure_reasons
    if fr['timeout'] + fr['collision'] > 0:
        logger.log(f"  失败原因: timeout={fr['timeout']}  collision={fr['collision']}")
    logger.log()
    logger.close()

    return stats


def run_detailed_validation():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir if args.out_dir else "logs/detailed_val"
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"  详细模型验证 | {ts}")
    print(f"{'=' * 70}")
    print(f"  模型: {args.model_dir}")
    print(f"  地图尺寸: {args.map_sizes}")
    print(f"  每尺寸场景数: {args.scenarios}")
    print(f"  Device: {device}")
    print(f"  A* 完全禁用 | 确定性策略")
    use_frr = getattr(args, "use_first_reach_reward", True)
    use_wp = getattr(args, "use_wall_penalty", True)
    use_pbrs = getattr(args, "use_pbrs", True)
    print(f"  奖励: first_reach={use_frr} | wall_penalty={use_wp} | pbrs={use_pbrs}")
    print(f"  GIF: duration={args.gif_duration}s, 终点定格={GIF_FREEZE_FRAMES}帧")
    print()

    env = Env(
        agent_num=2, max_episode_steps=1000,
        obs_radius=args.obs_radius,
        use_astar_shaping=False, use_astar_first_episode=False,
        obs_include_goal_direction=True, obs_include_position=True,
        use_first_reach_reward=getattr(args, "use_first_reach_reward", True),
        use_wall_penalty=getattr(args, "use_wall_penalty", True),
        use_pbrs=getattr(args, "use_pbrs", True),
    )
    policy = create_policy(env, device, args.hidden_size, args.layer_N)
    load_best_model(policy, args.model_dir, device)

    all_size_stats = {}

    for map_size in args.map_sizes:
        size_key = f"{map_size}x{map_size}"
        print(f"\n>>> 开始 {size_key} ({args.scenarios} 场景) ...")
        stats = run_one_size(map_size, args, env, policy, device, out_dir, ts)
        all_size_stats[map_size] = stats
        print(f"<<< {size_key} 完成: {stats['success_rate']:.1f}%\n")

    summary_path = os.path.join(out_dir, f"summary_{ts}.log")
    slog = Logger(summary_path)
    slog.log(f"{'=' * 70}")
    slog.log(f"  详细验证汇总 | {ts}")
    slog.log(f"{'=' * 70}")
    slog.log(f"  模型: {args.model_dir}")
    slog.log(f"  每尺寸场景: {args.scenarios} | Device: {device}")
    slog.log(f"  奖励配置: first_reach={use_frr} | wall_penalty={use_wp} | pbrs={use_pbrs}")
    slog.log()

    total_succ = sum(s["successes"] for s in all_size_stats.values())
    total_sc = sum(s["total"] for s in all_size_stats.values())
    overall_sr = total_succ / total_sc * 100 if total_sc > 0 else 0

    header = (f"{'尺寸':>6s}  {'成功率':>6s}  {'stay%':>6s}  {'loop%':>6s}  "
              f"{'撞墙':>5s}  {'偏离A*':>7s}  {'熵':>5s}  {'碰撞':>4s}  {'超时距':>6s}")
    slog.log(header)
    slog.log("-" * len(header))
    for ms in args.map_sizes:
        s = all_size_stats[ms]
        slog.log(
            f"{ms:>2d}x{ms:<2d}  {s['success_rate']:5.1f}%  "
            f"{s['avg_stay']*100:5.1f}%  "
            f"{s['avg_loop']*100:5.1f}%  "
            f"{s['avg_wall_bumps']:5.1f}  "
            f"{s['avg_astar_deviate']*100:6.1f}%  "
            f"{s['avg_entropy']:5.2f}  "
            f"{s['collisions']:4d}  "
            f"{s['avg_timeout_dist']:5.1f}"
        )
    slog.log(f"{'综合':>5s}  {overall_sr:5.1f}% ({total_succ}/{total_sc})")
    slog.log()

    slog.log(f"失败原因分类:")
    for ms in args.map_sizes:
        fr = all_size_stats[ms]["failure_reasons"]
        if fr["timeout"] + fr["collision"] > 0:
            slog.log(f"  {ms:>2d}x{ms}: timeout={fr['timeout']}  "
                     f"collision={fr['collision']}")
    slog.log()

    slog.log(f"各尺寸日志文件:")
    for ms in args.map_sizes:
        slog.log(f"  {ms}x{ms}: {out_dir}/{ms}x{ms}_{ts}.log")
    slog.log(f"GIF 目录: {out_dir}/<size>/")
    slog.log(f"{'=' * 70}")
    slog.close()

    print(f"\n汇总日志: {summary_path}")


if __name__ == "__main__":
    run_detailed_validation()
