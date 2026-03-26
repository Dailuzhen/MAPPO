"""
Phase B 模型大样本验证脚本

对指定的 checkpoint 在不同地图尺寸上进行纯策略评估（完全禁用 A*）。
每个模型 × 每种地图尺寸测试 N 个随机场景，输出分地图成功率。

用法:
  python scripts/validate_models.py                                  # 默认验证 V8
  python scripts/validate_models.py --phase_b_dir <path> --checkpoints 50 100 200 --obs_radius 3
"""
import sys, os, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.env import Env
from config import get_config
from algorithms.algorithm.MAPPOPolicy import MAPPOPolicy
from gym import spaces

# 默认值（V8 配置，向后兼容直接运行）
DEFAULT_CHECKPOINTS = [250, 500, 700, 1350, 1650, 1800]
DEFAULT_MAP_SIZES = [8, 10, 12, 14, 16, 18, 20, 22]
DEFAULT_SCENARIOS_PER_SIZE = 20
DEFAULT_MAX_STEP_MULTIPLIER = 5
DEFAULT_MIN_MAX_STEPS = 200
DEFAULT_OBS_RADIUS = 3
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_LAYER_N = 4
DEFAULT_PHASE_B_DIR = "results/MyEnv/MyEnv/full_v8/run1/phase_b_ppo"


def parse_validate_args():
    parser = argparse.ArgumentParser(description="Phase B 模型验证")
    parser.add_argument("--phase_b_dir", type=str, default=DEFAULT_PHASE_B_DIR)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--map_sizes", type=int, nargs="+", default=DEFAULT_MAP_SIZES)
    parser.add_argument("--scenarios_per_size", type=int, default=DEFAULT_SCENARIOS_PER_SIZE)
    parser.add_argument("--obs_radius", type=int, default=DEFAULT_OBS_RADIUS)
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--layer_N", type=int, default=DEFAULT_LAYER_N)
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()


def create_policy(env, device, hidden_size=256, layer_n=4):
    """根据环境的观测/动作空间创建策略网络"""
    parser = get_config()
    args = parser.parse_args([
        "--hidden_size", str(hidden_size),
        "--layer_N", str(layer_n),
    ])
    obs_dim = env.observation_space[0].shape[0]
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    share_obs_dim = obs_dim * env.agent_num
    share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32)
    act_space = env.action_space[0]
    policy = MAPPOPolicy(args, obs_space, share_obs_space, act_space, device)
    return policy


def load_checkpoint(policy, checkpoint_ep, device, phase_b_dir):
    """加载指定 checkpoint 的 actor 权重"""
    ckpt_dir = os.path.join(phase_b_dir, f"checkpoint_ep{checkpoint_ep:03d}", "models")
    actor_path = os.path.join(ckpt_dir, "actor.pt")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"Actor not found: {actor_path}")
    policy.actor.load_state_dict(torch.load(actor_path, map_location=device))
    policy.actor.eval()
    return policy


def get_astar_steps(env):
    """获取 A* 最优步数（取最慢智能体，与 Phase B 评估一致）"""
    agent_steps = []
    for i in range(env.agent_num):
        start = tuple(env.agents_pos[i])
        goal = tuple(env.goals_pos[i])
        if start == goal:
            agent_steps.append(0)
            continue
        path = env._astar_path(start, goal)
        if path:
            agent_steps.append(len(path) - 1)
        else:
            agent_steps.append(abs(start[0] - goal[0]) + abs(start[1] - goal[1]))
    return max(1, max(agent_steps) if agent_steps else 1)


def evaluate_scenario(env, policy, device, min_max_steps=200, max_step_multiplier=5):
    """
    在当前环境状态上运行策略。
    返回 dict: reached, steps, astar_steps, collisions, stay_count, stay_ratio, loop_rate, timeout_dist
    完全禁用 A*。
    """
    env.use_astar_first_episode = False
    env.use_astar_shaping = False
    env._stuck_counter = 0

    target_goals = [tuple(g) for g in env.goals_pos]
    astar_steps = get_astar_steps(env)
    max_steps = max(min_max_steps, int(astar_steps * max_step_multiplier))

    obs = np.stack([env._get_local_obs(i) for i in range(env.agent_num)])
    collisions = 0
    stay_count = 0
    position_history = []
    total_steps = 0
    reached = False

    for step in range(max_steps):
        obs_tensor = torch.tensor(
            obs.reshape(-1, obs.shape[-1]),
            dtype=torch.float32, device=device
        )
        with torch.no_grad():
            action = policy.act(obs_tensor, deterministic=True)
        actions = action.cpu().numpy().flatten().tolist()

        for a in actions:
            if int(a) == 0:
                stay_count += 1

        obs, _, _, _, info = env.step(actions)
        total_steps += 1
        position_history.append(tuple(tuple(p) for p in env.agents_pos))

        all_at_goal = all(
            env.agents_pos[i] == target_goals[i] for i in range(env.agent_num)
        )
        if info.get("all_goals_reached", False) or all_at_goal:
            reached = True
            break

        if info.get("looped_to_start", False) and not info.get("all_goals_reached", False):
            break

        col = info.get("collision", [])
        if isinstance(col, list) and any(col):
            collisions += 1

    total_actions = total_steps * env.agent_num
    stay_ratio = stay_count / total_actions if total_actions > 0 else 0

    last_n = position_history[-20:] if len(position_history) >= 20 else position_history
    if len(last_n) > 1:
        unique_pos = len(set(last_n))
        loop_rate = 1.0 - unique_pos / len(last_n)
    else:
        loop_rate = 0.0

    timeout_dist = 0.0
    if not reached:
        timeout_dist = sum(
            abs(env.agents_pos[i][0] - target_goals[i][0]) + abs(env.agents_pos[i][1] - target_goals[i][1])
            for i in range(env.agent_num)
        ) / env.agent_num

    return {
        "reached": reached,
        "steps": total_steps,
        "astar_steps": astar_steps,
        "collisions": collisions,
        "stay_count": stay_count,
        "stay_ratio": stay_ratio,
        "loop_rate": loop_rate,
        "timeout_dist": timeout_dist,
    }


def run_validation():
    vargs = parse_validate_args()
    device = torch.device(f"cuda:{vargs.gpu_id}" if torch.cuda.is_available() else "cpu")

    CHECKPOINTS = vargs.checkpoints
    MAP_SIZES = vargs.map_sizes
    SCENARIOS_PER_SIZE = vargs.scenarios_per_size
    PHASE_B_DIR = vargs.phase_b_dir

    print(f"Device: {device}")
    print(f"Phase B dir: {PHASE_B_DIR}")
    print(f"Checkpoints: {CHECKPOINTS}")
    print(f"Map sizes: {MAP_SIZES}")
    print(f"Scenarios per size: {SCENARIOS_PER_SIZE}")
    print(f"Total scenarios per model: {len(MAP_SIZES) * SCENARIOS_PER_SIZE}")
    print(f"obs_radius: {vargs.obs_radius}")
    print(f"hidden_size: {vargs.hidden_size}, layer_N: {vargs.layer_N}")
    print(f"A* disabled: use_astar_first_episode=False, use_astar_shaping=False")
    print()

    env = Env(
        agent_num=2,
        max_episode_steps=1000,
        obs_radius=vargs.obs_radius,
        use_astar_shaping=False,
        use_astar_first_episode=False,
        obs_include_goal_direction=True,
        obs_include_position=True,
    )
    policy = create_policy(env, device, hidden_size=vargs.hidden_size, layer_n=vargs.layer_N)

    all_results = {}

    for ckpt_ep in CHECKPOINTS:
        print(f"{'='*70}")
        print(f"  Checkpoint ep{ckpt_ep}")
        print(f"{'='*70}")
        load_checkpoint(policy, ckpt_ep, device, PHASE_B_DIR)

        model_results = {}
        for map_size in MAP_SIZES:
            successes = 0
            total_steps = 0
            total_astar = 0
            total_collisions = 0
            timeouts = 0
            stay_ratios = []
            loop_rates = []
            timeout_dists = []

            for scenario in range(SCENARIOS_PER_SIZE):
                env.rebuild_map_new_size(map_size)
                env.start_pos = None
                env.fixed_goals = None
                env.episode_count = 999
                env.reset()

                result = evaluate_scenario(env, policy, device)

                stay_ratios.append(result["stay_ratio"])
                loop_rates.append(result["loop_rate"])

                if result["reached"]:
                    successes += 1
                    total_steps += result["steps"]
                    total_astar += result["astar_steps"]
                else:
                    timeouts += 1
                    timeout_dists.append(result["timeout_dist"])
                total_collisions += result["collisions"]

            sr = successes / SCENARIOS_PER_SIZE * 100
            avg_ratio = (total_steps / total_astar) if total_astar > 0 else 0
            avg_steps_val = total_steps / successes if successes > 0 else 0
            avg_stay = float(np.mean(stay_ratios)) if stay_ratios else 0
            avg_loop = float(np.mean(loop_rates)) if loop_rates else 0
            avg_tdist = float(np.mean(timeout_dists)) if timeout_dists else 0

            model_results[map_size] = {
                "success_rate": sr,
                "successes": successes,
                "total": SCENARIOS_PER_SIZE,
                "timeouts": timeouts,
                "collisions": total_collisions,
                "avg_step_ratio": round(avg_ratio, 2),
                "avg_steps": round(avg_steps_val, 1),
                "avg_stay_ratio": round(avg_stay, 3),
                "avg_loop_rate": round(avg_loop, 3),
                "avg_timeout_dist": round(avg_tdist, 1),
            }
            marker = " ***" if sr >= 85 else ""
            diag_str = f"  stay={avg_stay*100:.0f}%  loop={avg_loop*100:.0f}%"
            if timeouts > 0:
                diag_str += f"  tdist={avg_tdist:.1f}"
            print(f"  {map_size:>2d}x{map_size:<2d}: {sr:5.1f}% ({successes}/{SCENARIOS_PER_SIZE})  "
                  f"ratio={avg_ratio:.2f}x  timeout={timeouts}  collision={total_collisions}"
                  f"{diag_str}{marker}")

        overall_succ = sum(r["successes"] for r in model_results.values())
        overall_total = sum(r["total"] for r in model_results.values())
        overall_sr = overall_succ / overall_total * 100
        model_results["overall"] = {
            "success_rate": overall_sr,
            "successes": overall_succ,
            "total": overall_total,
        }
        print(f"  {'综合':>5s}: {overall_sr:5.1f}% ({overall_succ}/{overall_total})")
        print()

        all_results[f"ep{ckpt_ep}"] = model_results

    output_path = os.path.join(PHASE_B_DIR, "validation_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {output_path}")

    print(f"\n{'='*70}")
    print("  汇总对比表（成功率）")
    print(f"{'='*70}")
    header = f"{'Model':>8s}"
    for ms in MAP_SIZES:
        header += f"  {ms}x{ms:>2d}"
    header += "    综合"
    print(header)
    print("-" * len(header))
    for ckpt_ep in CHECKPOINTS:
        key = f"ep{ckpt_ep}"
        r = all_results[key]
        line = f"ep{ckpt_ep:>5d}"
        for ms in MAP_SIZES:
            sr = r[ms]["success_rate"]
            line += f"  {sr:5.1f}%"
        line += f"  {r['overall']['success_rate']:5.1f}%"
        print(line)
    print()

    print(f"{'='*70}")
    print("  诊断汇总（stay% / loop% / 超时距离）")
    print(f"{'='*70}")
    for ckpt_ep in CHECKPOINTS:
        key = f"ep{ckpt_ep}"
        r = all_results[key]
        print(f"  ep{ckpt_ep}:")
        for ms in MAP_SIZES:
            m = r[ms]
            stay_str = f"stay={m.get('avg_stay_ratio', 0)*100:.0f}%"
            loop_str = f"loop={m.get('avg_loop_rate', 0)*100:.0f}%"
            tdist_str = f"tdist={m.get('avg_timeout_dist', 0):.1f}" if m["timeouts"] > 0 else ""
            print(f"    {ms:>2d}x{ms}: {stay_str}  {loop_str}  {tdist_str}")
    print()


if __name__ == "__main__":
    run_validation()
