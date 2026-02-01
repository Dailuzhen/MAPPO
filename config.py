import argparse

def get_config():
    """
    所有环境通用超参数的配置解析器。
    请查看每个 `scripts/train/<env>_runner.py` 文件以找到仅在特定环境中使用的私有超参数。

    基础参数:
        --algorithm_name <algorithm_name>
            指定算法，当前仅支持 `["mappo"]`
        --experiment_name <str>
            区分不同实验的标识符
        --seed <int>
            设置numpy和torch的随机种子
        --cuda
            默认True，使用GPU训练；否则使用CPU
        --cuda_deterministic
            默认确保随机种子有效；如果设置，则绕过此功能
        --n_training_threads <int>
            并行训练线程数，默认1
        --n_rollout_threads <int>
            用于训练rollout的并行环境数，默认32
        --n_eval_rollout_threads <int>
            用于评估rollout的并行环境数，默认1
        --n_render_rollout_threads <int>
            用于渲染的并行环境数，某些环境只能设置为1
        --num_env_steps <int>
            训练的环境步数（默认：10e6）
        --user_name <str>
            [wandb使用]，指定用户名以简单收集训练数据

    环境参数:
        --env_name <str>
            指定环境名称
        --use_obs_instead_of_state
            [仅适用于某些环境]默认False，使用全局状态；否则使用拼接的局部观察

    回放缓冲区参数:
        --episode_length <int>
            缓冲区中 episode 的最大长度

    网络参数:
        --use_centralized_V
            默认True，使用集中式训练模式；否则使用分散式训练模式
        --stacked_frames <int>
            堆叠的输入帧数量
        --use_stacked_frames
            是否使用堆叠帧
        --hidden_size <int>
            演员/评论家网络的隐藏层维度
        --layer_N <int>
            演员/评论家网络的层数
        --use_ReLU
            默认True，使用ReLU；否则使用Tanh
        --use_popart
            默认False，使用PopArt归一化奖励
        --use_valuenorm
            默认True，使用运行均值和标准差归一化奖励
        --use_feature_normalization
            默认True，对输入应用层归一化
        --use_orthogonal
            默认True，使用正交初始化权重和零初始化偏置；否则使用xavier均匀初始化
        --gain
            默认0.01，最后一个动作层的增益

    优化器参数:
        --lr <float>
            学习率参数（默认：5e-4）
        --critic_lr <float>
            评论家学习率（默认：5e-4）
        --opti_eps <float>
            RMSprop优化器的epsilon（默认：1e-5）
        --weight_decay <float>
            权重衰减系数（默认：0）

    PPO参数:
        --ppo_epoch <int>
            ppo训练轮数（默认：15）
        --use_clipped_value_loss
            默认裁剪价值损失；如果设置，则不裁剪
        --clip_param <float>
            ppo裁剪参数（默认：0.2）
        --num_mini_batch <int>
            ppo的批次数（默认：1）
        --entropy_coef <float>
            熵项系数（默认：0.01）
        --value_loss_coef <float>
            价值损失系数（默认：1）
        --use_max_grad_norm
            默认使用梯度最大范数；如果设置，则不使用
        --max_grad_norm <float>
            梯度最大范数（默认：10.0）
        --use_gae
            默认使用广义优势估计；如果设置，则不使用
        --gamma <float>
            奖励折扣因子（默认：0.99）
        --gae_lambda <float>
            gae lambda参数（默认：0.95）
        --use_proper_time_limits
            默认不考虑时间限制计算回报；如果设置，则考虑时间限制
        --use_huber_loss
            默认使用huber损失；如果设置，则不使用
        --use_value_active_masks
            默认True，是否在价值损失中屏蔽无用数据
        --use_policy_active_masks
            默认True，是否在策略损失中屏蔽无用数据
        --huber_delta <float>
            huber损失的系数

    运行参数:
        --use_linear_lr_decay
            默认不应用线性学习率衰减；如果设置，则使用线性学习率调度

    保存和日志参数:
        --save_interval <int>
            连续两次模型保存之间的时间间隔
        --log_interval <int>
            连续两次日志打印之间的时间间隔

    评估参数:
        --use_eval
            默认不启动评估；如果设置，则在训练的同时启动评估
        --eval_interval <int>
            连续两次评估之间的时间间隔
        --eval_episodes <int>
            单次评估的episode数量

    渲染参数:
        --save_gifs
            默认不保存渲染视频；如果设置，则保存视频
        --use_render
            默认不在训练期间渲染环境；如果设置，则开始渲染
        --render_episodes <int>
            渲染给定环境的episode数量
        --ifi <float>
            保存视频中每个渲染图像的播放间隔

    预训练参数:
        --model_dir <str>
            默认None，设置预训练模型的路径
    """
    parser = argparse.ArgumentParser(description="onpolicy", formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # 基础参数
    parser.add_argument("--experiment_name", type=str, default="check", help="区分不同实验的标识符")
    parser.add_argument("--seed", type=int, default=1, help="numpy/torch的随机种子")
    parser.add_argument("--cuda", action="store_false", default=True, help="默认True，使用GPU训练；否则使用CPU")
    parser.add_argument("--cuda_deterministic", action="store_false", default=True, help="默认确保随机种子有效；如果设置，则绕过此功能")
    parser.add_argument("--n_training_threads", type=int, default=8, help="训练的torch线程数")
    parser.add_argument("--n_rollout_threads", type=int, default=1, help="用于训练rollout的并行环境数")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=2, help="用于评估rollout的并行环境数")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1, help="用于渲染rollout的并行环境数")
    parser.add_argument("--num_env_steps", type=int, default=10e6, help="训练的环境步数（默认：10e6）")
    
    # 环境参数
    parser.add_argument("--env_name", type=str, default="MyEnv", help="指定环境名称")
    parser.add_argument("--use_obs_instead_of_state", action="store_true", default=False, help="是否使用全局状态或拼接的观察")
    
    # 回放缓冲区参数
    parser.add_argument("--episode_length", type=int, default=1000, help="任意episode的最大长度")
    
    # 网络参数
    parser.add_argument("--use_centralized_V", action="store_false", default=True, help="是否使用集中式V函数")
    parser.add_argument("--stacked_frames", type=int, default=1, help="堆叠的输入帧数量")
    parser.add_argument("--use_stacked_frames", action="store_true", default=False, help="是否使用堆叠帧")
    parser.add_argument("--hidden_size", type=int, default=128, help="演员/评论家网络的隐藏层维度")
    parser.add_argument("--layer_N", type=int, default=4, help="演员/评论家网络的层数")
    parser.add_argument("--use_ReLU", action="store_false", default=True, help="是否使用ReLU")
    parser.add_argument("--use_popart", action="store_true", default=False, help="默认False，使用PopArt归一化奖励")
    parser.add_argument("--use_valuenorm", action="store_false", default=True, help="默认True，使用运行均值和标准差归一化奖励")
    parser.add_argument("--use_feature_normalization", action="store_false", default=True, help="是否对输入应用层归一化")
    parser.add_argument("--use_orthogonal", action="store_false", default=True, help="是否使用正交初始化权重和零初始化偏置")
    parser.add_argument("--gain", type=float, default=0.01, help="最后一个动作层的增益")
    
    # 优化器参数
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率（默认：5e-4）")
    parser.add_argument("--critic_lr", type=float, default=5e-4, help="评论家学习率（默认：5e-4）")
    parser.add_argument("--opti_eps", type=float, default=1e-5, help="RMSprop优化器的epsilon（默认：1e-5）")
    parser.add_argument("--weight_decay", type=float, default=0, help="权重衰减系数（默认：0）")
    
    # PPO参数
    parser.add_argument("--ppo_epoch", type=int, default=20, help="ppo训练轮数（默认：15）")
    parser.add_argument("--use_clipped_value_loss", action="store_false", default=True, help="默认裁剪价值损失；如果设置，则不裁剪")
    parser.add_argument("--clip_param", type=float, default=0.2, help="ppo裁剪参数（默认：0.2）")
    parser.add_argument("--num_mini_batch", type=int, default=128, help="ppo的批次数（默认：1）")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="熵项系数（默认：0.01）")
    parser.add_argument("--value_loss_coef", type=float, default=1, help="价值损失系数（默认：1）")
    parser.add_argument("--use_max_grad_norm", action="store_false", default=True, help="默认使用梯度最大范数；如果设置，则不使用")
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="梯度最大范数（默认：10.0）")
    parser.add_argument("--use_gae", action="store_false", default=True, help="使用广义优势估计")
    parser.add_argument("--gamma", type=float, default=0.99, help="奖励折扣因子（默认：0.99）")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="gae lambda参数（默认：0.95）")
    parser.add_argument("--use_proper_time_limits", action="store_true", default=False, help="计算回报时考虑时间限制")
    parser.add_argument("--use_huber_loss", action="store_false", default=True, help="默认使用huber损失；如果设置，则不使用")
    parser.add_argument("--use_value_active_masks", action="store_false", default=True, help="默认True，是否在价值损失中屏蔽无用数据")
    parser.add_argument("--use_policy_active_masks", action="store_false", default=True, help="默认True，是否在策略损失中屏蔽无用数据")
    parser.add_argument("--huber_delta", type=float, default=10.0, help="huber损失的系数")
    
    # 运行参数
    parser.add_argument("--use_linear_lr_decay", action="store_true", default=False, help="使用线性学习率调度")
    
    # 保存参数
    parser.add_argument("--save_interval", type=int, default=20, help="连续两次模型保存之间的时间间隔")
    
    # 日志参数
    parser.add_argument("--log_interval", type=int, default=10, help="连续两次日志打印之间的时间间隔")
    
    # 评估参数
    parser.add_argument("--use_eval", action="store_true", default=True, help="默认启动评估；如果设置，则在训练的同时启动评估")
    parser.add_argument("--eval_interval", type=int, default=50, help="连续两次评估之间的时间间隔")
    parser.add_argument("--eval_episodes", type=int, default=50, help="单次评估的episode数量")

    # A* 行为克隆参数（旧版，保留兼容）
    parser.add_argument("--use_astar_bc", action="store_true", default=True, help="是否在PPO训练前使用A*轨迹进行行为克隆初始化")
    parser.add_argument("--astar_bc_updates", type=int, default=20, help="A*行为克隆的优化步数")
    
    # ============= 两阶段训练参数 =============
    # 训练模式选择
    parser.add_argument("--training_mode", type=str, default="two_phase", 
                        choices=["two_phase", "phase_a_only", "phase_b_only", "legacy"],
                        help="训练模式: two_phase(完整两阶段), phase_a_only(仅Phase A), phase_b_only(仅Phase B), legacy(旧版)")
    
    # Phase A 参数（A* 在线学习）
    parser.add_argument("--phase_a_episodes", type=int, default=100,
                        help="Phase A: A*在线学习的episode数")
    parser.add_argument("--phase_a_checkpoint_interval", type=int, default=10,
                        help="Phase A: checkpoint保存和对比间隔")
    parser.add_argument("--phase_a_bc_lr", type=float, default=3e-4,
                        help="Phase A: BC学习率（偏低更稳，避免后期崩溃）")
    parser.add_argument("--bc_updates_per_episode", type=int, default=10,
                        help="Phase A: 每批次最少BC梯度更新次数")
    parser.add_argument("--bc_max_updates_per_batch", type=int, default=30,
                        help="Phase A: 每批次BC更新次数上限（防灾难性遗忘）")
    parser.add_argument("--bc_batch_size", type=int, default=256,
                        help="Phase A: BC mini-batch大小")
    parser.add_argument("--max_replay_buffer_size", type=int, default=50000,
                        help="Phase A: Replay buffer 上限，超出 FIFO 丢弃（防分布偏移）")
    parser.add_argument("--bc_entropy_coef", type=float, default=0.01,
                        help="Phase A: BC 熵正则系数（防策略塌缩）")
    parser.add_argument("--bc_weight_decay", type=float, default=1e-4,
                        help="Phase A: BC 优化器 L2 正则")
    
    # Phase B 参数（纯策略 PPO 训练）
    parser.add_argument("--phase_b_episodes", type=int, default=900,
                        help="Phase B: 纯策略PPO训练的episode数")
    parser.add_argument("--phase_b_render_interval", type=int, default=100,
                        help="Phase B: 渲染gif的间隔")
    
    # 对比评估参数
    parser.add_argument("--comparison_max_steps", type=int, default=100,
                        help="策略推理单个segment的最大步数")
    parser.add_argument("--comparison_segments", type=int, default=20,
                        help="checkpoint对比时随机选择的segment数量")
    
    # 渲染参数
    parser.add_argument("--save_gifs", action="store_true", default=False, help="默认不保存渲染视频；如果设置，则保存视频")
    parser.add_argument("--use_render", action="store_true", default=False, help="默认不在训练期间渲染环境；如果设置，则开始渲染")
    parser.add_argument("--render_episodes", type=int, default=5, help="渲染给定环境的episode数量")
    parser.add_argument("--ifi", type=float, default=0.1, help="保存视频中每个渲染图像的播放间隔")
    parser.add_argument("--render_run", type=str, default=None, help="渲染时使用的run目录（例如 run2）")
    parser.add_argument("--gif_dir", type=str, default="renders/gifs", help="GIF保存目录（相对于run_dir或绝对路径）")
    parser.add_argument("--gif_fps", type=float, default=10.0, help="GIF帧率（每秒帧数）")
    parser.add_argument("--save_frames", action="store_true", default=False, help="是否逐帧保存渲染图像")
    parser.add_argument("--frames_dir", type=str, default="renders/frames", help="逐帧图片保存目录（相对于run_dir或绝对路径）")
    
    # 预训练参数
    parser.add_argument("--model_dir", type=str, default=None, help="默认None，设置预训练模型的路径")
    
    return parser
