"""
# MAPPO训练脚本
# 用于训练多智能体近端策略优化算法
"""

#!/usr/bin/env python
import sys
import os
import setproctitle
import copy
import numpy as np
from pathlib import Path
import torch

# 添加项目根目录到Python路径
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)

from config import get_config
from envs.env_wrappers import DummyVecEnv



def make_env(all_args):
    """创建向量环境（方案A：所有并行环境使用同一张地图，保证训练与对比评估地图一致）
    
    Args:
        all_args: 配置参数
    
    Returns:
        向量环境实例
    """
    # v8: 获取观测增强参数
    obs_include_goal_direction = getattr(all_args, 'obs_include_goal_direction', True)
    obs_include_position = getattr(all_args, 'obs_include_position', True)
    
    # 先创建单个环境得到一张随机地图，再复制给所有并行环境，避免各环境地图不一致导致对比评估失败
    from envs.env import Env
    _temp_env = Env(
        max_episode_steps=all_args.episode_length,
        obs_include_goal_direction=obs_include_goal_direction,
        obs_include_position=obs_include_position,
    )
    _temp_env.seed(all_args.seed)
    shared_map = _temp_env.map.copy()
    del _temp_env

    def get_env_fn(rank):
        """获取环境构造函数（每个环境使用同一张 shared_map）"""
        def init_env():
            from envs.env import Env
            env = Env(
                max_episode_steps=all_args.episode_length,
                shared_map=shared_map,
                obs_include_goal_direction=obs_include_goal_direction,
                obs_include_position=obs_include_position,
            )
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    # 创建向量环境，包含 n_rollout_threads 个环境实例，地图一致
    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])



def parse_args(args, parser):
    """解析命令行参数
    
    Args:
        args: 命令行参数列表
        parser: 参数解析器
    
    Returns:
        解析后的参数对象
    """
    # 添加MPE环境特定参数
    parser.add_argument("--scenario_name", type=str, default="MyEnv", help="运行的场景名称")
    parser.add_argument("--num_landmarks", type=int, default=3, help="地标数量")
    parser.add_argument("--num_agents", type=int, default=2, help="智能体数量")

    # 解析参数
    #TODO
    all_args = parser.parse_known_args(args)[0]

    return all_args



def main(args):
    """主函数，执行训练流程
    
    Args:
        args: 命令行参数列表
    """
    # 获取配置参数
    parser = get_config()
    all_args = parse_args(args, parser)

    # 设置设备：优先使用GPU，如果不可用则使用CPU
    if torch.cuda.is_available():
        print("使用GPU进行训练...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        # 设置CUDA确定性模式，确保实验可复现
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("使用CPU进行训练...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 创建运行目录
    #TODO
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # 确定当前运行编号
    if all_args.use_render and all_args.render_run is not None:
        curr_run = all_args.render_run
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # 设置进程名称
    setproctitle.setproctitle(
        "mappo"
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
    )

    # 设置随机种子，确保实验可复现
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # 初始化环境
    envs = make_env(all_args)
    # 初始化评估环境（如果需要）
    if all_args.use_eval:
        eval_args = copy.deepcopy(all_args)
        eval_args.n_rollout_threads = all_args.n_eval_rollout_threads
        eval_envs = make_env(eval_args)
    else:
        eval_envs = None
    # 获取智能体数量
    num_agents = all_args.num_agents

    # 配置字典
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # 导入并初始化Runner
    from runner.runner import Runner
    runner = Runner(config)
    
    # 打印训练配置
    training_mode = getattr(all_args, 'training_mode', 'legacy')
    print("\n" + "=" * 70)
    print(f"  实验名称: {all_args.experiment_name}")
    print(f"  运行目录: {run_dir}")
    print(f"  训练模式: {training_mode}")
    print(f"  设备: {device}")
    print("=" * 70 + "\n")
    
    # 开始训练或渲染
    if all_args.use_render:
        runner.render()
    else:
        runner.run()

    # 训练结束，关闭环境
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    # 导出标量数据到JSON文件（仅在非 phase_a_only 模式下）
    try:
        if hasattr(runner, 'writter') and runner.writter is not None:
            runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
            runner.writter.close()
    except Exception as e:
        print(f"[警告] 导出日志时出错: {e}")



if __name__ == "__main__":
    # 执行主函数
    main(sys.argv[1:])
