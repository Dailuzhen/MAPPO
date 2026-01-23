"""
# MAPPO训练脚本
# 用于训练多智能体近端策略优化算法
"""

#!/usr/bin/env python
import sys
import os
import setproctitle
import numpy as np
from pathlib import Path
import torch

# 添加项目根目录到Python路径
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)

from config import get_config
from envs.env_wrappers import DummyVecEnv



def make_env(all_args):
    """创建向量环境
    
    Args:
        all_args: 配置参数
    
    Returns:
        向量环境实例
    """
    def get_env_fn(rank):
        """获取环境构造函数
        
        Args:
            rank: 环境编号
        
        Returns:
            环境构造函数
        """
        def init_env():
            """初始化环境
            
            Returns:
                环境实例
            """
            from envs.env import Env
            env = Env()
            # 设置种子，确保实验可复现
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    # 创建向量环境，包含n_rollout_threads个环境实例
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
    if all_args.cuda and torch.cuda.is_available():
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
    eval_envs = make_env(all_args) if all_args.use_eval else None
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
    # 开始训练
    runner.run()

    # 训练结束，关闭环境
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    # 导出标量数据到JSON文件
    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()



if __name__ == "__main__":
    # 执行主函数
    main(sys.argv[1:])
