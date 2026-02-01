"""
Phase A 工具函数模块

包含 Phase A（A* 在线学习）所需的各种工具函数：
- JSON 文件读写
- GIF 生成
- 对比图生成
- 统计计算
"""

import os
import json
import numpy as np
from pathlib import Path

try:
    import imageio.v2 as imageio
except ImportError:
    try:
        import imageio
    except ImportError:
        imageio = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


# ============================================================================
#                              JSON 文件操作
# ============================================================================

def save_json(data, path):
    """
    保存数据到 JSON 文件
    
    Args:
        data: 要保存的数据（dict 或 list）
        path: 文件路径（str 或 Path）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    """
    从 JSON 文件加载数据
    
    Args:
        path: 文件路径（str 或 Path）
        
    Returns:
        加载的数据
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
#                              GIF 生成
# ============================================================================

def make_gif_from_frames(frames, output_path, fps=10):
    """
    从帧列表生成 GIF
    
    Args:
        frames: 帧图像列表（numpy 数组）
        output_path: 输出 GIF 路径
        fps: 帧率
    """
    if imageio is None:
        raise ImportError("生成 GIF 需要 imageio 库，请运行: pip install imageio")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    duration = 1.0 / fps
    imageio.mimsave(str(output_path), frames, duration=duration)


def save_frames_to_dir(frames, frames_dir):
    """
    将帧图像保存到目录
    
    Args:
        frames: 帧图像列表
        frames_dir: 保存目录
    """
    if imageio is None:
        raise ImportError("保存帧需要 imageio 库")
    
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(frames):
        imageio.imwrite(str(frames_dir / f"frame_{i:04d}.png"), frame)


# ============================================================================
#                              对比图生成
# ============================================================================

def create_comparison_frame(astar_frame, policy_frame, 
                            episode, segment_id, current_step,
                            astar_steps, policy_steps, 
                            is_final=False, reached=False,
                            starts=None, goals=None):
    """
    创建单帧对比图（左 A*，右策略）
    
    Args:
        astar_frame: A* 轨迹帧 (numpy 数组)
        policy_frame: 策略轨迹帧 (numpy 数组)
        episode: Episode 编号
        segment_id: Segment 编号
        current_step: 当前步数
        astar_steps: A* 总步数
        policy_steps: 策略总步数
        is_final: 是否为最终帧
        reached: 策略是否到达目标
        starts: 起始点列表 [(x1,y1), (x2,y2), ...]
        goals: 终点列表 [(x1,y1), (x2,y2), ...]
        
    Returns:
        对比图帧 (numpy 数组)
    """
    if Image is None:
        # 如果没有 PIL，返回简单拼接
        return _simple_concat_frames(astar_frame, policy_frame)
    
    # 确保帧尺寸一致
    target_size = (300, 300)
    a_img = Image.fromarray(astar_frame).resize(target_size)
    p_img = Image.fromarray(policy_frame).resize(target_size)
    
    a_frame = np.array(a_img)
    p_frame = np.array(p_img)
    
    # 画布尺寸
    h, w = target_size
    header_h = 60  # 增加头部高度以显示起点终点
    footer_h = 50
    gap = 10
    
    # 创建白色画布
    canvas_h = header_h + h + footer_h
    canvas_w = w * 2 + gap
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # 放置两张图
    canvas[header_h:header_h+h, :w] = a_frame[:, :, :3] if a_frame.shape[-1] >= 3 else np.stack([a_frame]*3, axis=-1)
    canvas[header_h:header_h+h, w+gap:] = p_frame[:, :, :3] if p_frame.shape[-1] >= 3 else np.stack([p_frame]*3, axis=-1)
    
    # 添加文字
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    
    # 尝试加载字体
    font = _get_font(14)
    font_small = _get_font(11)
    font_tiny = _get_font(10)
    
    # 标题
    title = f"Episode {episode} - Segment {segment_id}"
    draw.text((canvas_w // 2 - 80, 5), title, fill=(0, 0, 0), font=font)
    
    # 显示起始点和终点信息
    if starts and goals:
        start_str = "Start: " + ", ".join([f"A{i}:{s}" for i, s in enumerate(starts)])
        goal_str = "Goal: " + ", ".join([f"A{i}:{g}" for i, g in enumerate(goals)])
        draw.text((10, 22), start_str, fill=(0, 100, 0), font=font_tiny)
        draw.text((10, 36), goal_str, fill=(200, 0, 0), font=font_tiny)
    
    # A* 步数信息
    astar_info = f"A* optimal: {astar_steps} steps"
    draw.text((canvas_w // 2 + 50, 36), astar_info, fill=(0, 0, 128), font=font_tiny)
    
    # 列标题
    draw.text((w // 2 - 25, header_h + h + 5), "A* Path", fill=(0, 0, 128), font=font)
    draw.text((w + gap + w // 2 - 25, header_h + h + 5), "Policy", fill=(128, 0, 0), font=font)
    
    # 状态信息
    if is_final:
        a_status = f"Done: {astar_steps} steps"
        if reached:
            extra = policy_steps - astar_steps
            if extra == 0:
                p_status = f"Reached! {policy_steps} steps (optimal!)"
                p_color = (0, 180, 0)  # 深绿色
            else:
                p_status = f"Reached! {policy_steps} steps (+{extra})"
                p_color = (0, 128, 0)  # 绿色
        else:
            p_status = f"Failed! {policy_steps} steps (timeout)"
            p_color = (200, 0, 0)  # 红色
    else:
        a_status = f"Step {min(current_step, astar_steps)}/{astar_steps}"
        p_status = f"Step {current_step}"
        p_color = (0, 0, 0)
    
    draw.text((w // 2 - 45, header_h + h + 25), a_status, fill=(0, 0, 0), font=font_small)
    draw.text((w + gap + w // 2 - 65, header_h + h + 25), p_status, fill=p_color, font=font_small)
    
    return np.array(img)


def generate_comparison_gif(astar_frames, policy_frames, output_path,
                            episode, segment_id, astar_steps, policy_steps,
                            reached, fps=10, starts=None, goals=None):
    """
    生成对比 GIF
    
    Args:
        astar_frames: A* 轨迹帧列表
        policy_frames: 策略轨迹帧列表
        output_path: 输出路径
        episode: Episode 编号
        segment_id: Segment 编号
        astar_steps: A* 步数
        policy_steps: 策略步数
        reached: 是否到达
        fps: 帧率
        starts: 起始点列表 [(x1,y1), (x2,y2), ...]
        goals: 终点列表 [(x1,y1), (x2,y2), ...]
    """
    # 对齐帧数
    max_frames = max(len(astar_frames), len(policy_frames))
    
    # 扩展较短的序列
    while len(astar_frames) < max_frames:
        astar_frames.append(astar_frames[-1])
    while len(policy_frames) < max_frames:
        policy_frames.append(policy_frames[-1])
    
    # 生成对比帧
    comparison_frames = []
    for i in range(max_frames):
        is_final = (i == max_frames - 1)
        frame = create_comparison_frame(
            astar_frames[i], policy_frames[i],
            episode, segment_id, i,
            astar_steps, policy_steps,
            is_final=is_final, reached=reached,
            starts=starts, goals=goals
        )
        comparison_frames.append(frame)
    
    # 保存 GIF
    make_gif_from_frames(comparison_frames, output_path, fps=fps)


def _simple_concat_frames(frame1, frame2):
    """简单水平拼接两帧（无 PIL 时使用）"""
    gap = np.ones((frame1.shape[0], 10, 3), dtype=np.uint8) * 255
    return np.concatenate([frame1[:, :, :3], gap, frame2[:, :, :3]], axis=1)


def _get_font(size):
    """获取字体，优先使用系统字体"""
    if ImageFont is None:
        return None
    
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    
    return ImageFont.load_default()


# ============================================================================
#                              统计计算
# ============================================================================

def calculate_bc_accuracy(pred_actions, expert_actions):
    """
    计算行为克隆预测准确率
    
    Args:
        pred_actions: 预测的动作 (numpy 数组)
        expert_actions: 专家动作 (numpy 数组)
        
    Returns:
        准确率 (0-1)
    """
    pred_actions = np.array(pred_actions).flatten()
    expert_actions = np.array(expert_actions).flatten()
    return np.mean(pred_actions == expert_actions)


def generate_checkpoint_summary(checkpoint_ep, start_ep, end_ep, 
                                comparison_stats, bc_losses):
    """
    生成 checkpoint 统计报告
    
    Args:
        checkpoint_ep: Checkpoint 的 episode 编号
        start_ep: 起始 episode
        end_ep: 结束 episode
        comparison_stats: 对比统计数据
        bc_losses: BC 损失列表
        
    Returns:
        summary dict
    """
    total = comparison_stats["total_segments"]
    reached = comparison_stats["reached"]
    
    # 计算策略统计
    if reached > 0:
        avg_policy_steps = comparison_stats["total_policy_steps"] / reached
        avg_astar_steps = comparison_stats["total_astar_steps"] / total
        avg_extra_steps = avg_policy_steps - avg_astar_steps
        avg_extra_ratio = avg_extra_steps / avg_astar_steps if avg_astar_steps > 0 else 0
    else:
        avg_policy_steps = 0
        avg_astar_steps = comparison_stats["total_astar_steps"] / max(1, total)
        avg_extra_steps = 0
        avg_extra_ratio = 0
    
    summary = {
        "checkpoint": f"ep{checkpoint_ep:03d}",
        "episodes_range": [start_ep, end_ep],
        "total_segments": total,
        
        "astar_statistics": {
            "avg_steps_per_segment": round(avg_astar_steps, 2)
        },
        
        "policy_statistics": {
            "reach_rate": round(reached / max(1, total), 4),
            "reached_count": reached,
            "timeout_count": comparison_stats["timeout"],
            "collision_count": comparison_stats.get("collision", 0),
            "timeout_rate": round(comparison_stats["timeout"] / max(1, total), 4),
            "avg_policy_steps": round(avg_policy_steps, 2),
            "avg_extra_steps": round(avg_extra_steps, 2),
            "avg_extra_steps_ratio": round(avg_extra_ratio, 4),
        },
        
        "bc_statistics": {
            "final_bc_loss": round(bc_losses[-1], 4) if bc_losses else None,
            "avg_bc_loss": round(float(np.mean(bc_losses)), 4) if bc_losses else None,
            "bc_loss_trend": [round(x, 4) for x in bc_losses] if bc_losses else []
        },
        
        "per_episode": comparison_stats.get("per_episode", [])
    }
    
    return summary


def create_scene_config(episode_num, map_size, total_steps, segments_data):
    """
    创建 scene_config 数据结构
    
    Args:
        episode_num: Episode 编号
        map_size: 地图尺寸
        total_steps: 总步数
        segments_data: Segment 数据列表
        
    Returns:
        scene_config dict
    """
    scene_config = {
        "episode": episode_num,
        "map_size": map_size,
        "total_steps": total_steps,
        "total_segments": len(segments_data),
        "segments": []
    }
    
    for seg in segments_data:
        scene_config["segments"].append({
            "segment_id": seg["segment_id"],
            "agents": seg["agents"],
            "astar_steps": seg["astar_steps"],
            "start_step": seg["start_step"],
            "end_step": seg["end_step"]
        })
    
    return scene_config


# ============================================================================
#                              目录管理
# ============================================================================

def setup_phase_a_directories(run_dir):
    """
    创建 Phase A 所需的目录结构
    
    Args:
        run_dir: 运行根目录
        
    Returns:
        dict 包含各目录路径
    """
    run_dir = Path(run_dir)
    
    dirs = {
        "phase_a_root": run_dir / "phase_a_astar",
        "episodes": run_dir / "phase_a_astar" / "episodes",
        "logs": run_dir / "phase_a_astar" / "logs",
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    return dirs


def get_checkpoint_dir(phase_a_root, checkpoint_ep):
    """
    获取 checkpoint 目录路径
    
    Args:
        phase_a_root: Phase A 根目录
        checkpoint_ep: Checkpoint 的 episode 编号
        
    Returns:
        checkpoint 目录 Path
    """
    return Path(phase_a_root) / f"checkpoint_ep{checkpoint_ep:03d}"


def get_episode_dir(episodes_dir, episode_num):
    """
    获取 episode 目录路径
    
    Args:
        episodes_dir: Episodes 根目录
        episode_num: Episode 编号
        
    Returns:
        episode 目录 Path
    """
    return Path(episodes_dir) / f"ep_{episode_num:03d}"


# ============================================================================
#                              打印工具
# ============================================================================

def print_phase_a_header():
    """打印 Phase A 开始标题"""
    print("\n" + "=" * 70)
    print("                    Phase A: A* 在线学习")
    print("=" * 70)


def print_episode_header(ep, total_eps):
    """打印 Episode 开始标题"""
    print(f"\n{'─' * 60}")
    print(f"[Phase A] Episode {ep}/{total_eps}")
    print(f"{'─' * 60}")


def print_checkpoint_summary(checkpoint_ep, summary):
    """打印 Checkpoint 统计摘要"""
    stats = summary["policy_statistics"]
    bc_stats = summary["bc_statistics"]
    
    print(f"\n{'═' * 60}")
    print(f"  Checkpoint ep{checkpoint_ep:03d} 统计")
    print(f"{'═' * 60}")
    print(f"  总 Segments: {summary['total_segments']}")
    print(f"  到达率: {stats['reach_rate']:.2%} ({stats['reached_count']}/{summary['total_segments']})")
    print(f"  超时率: {stats['timeout_rate']:.2%}")
    print(f"  平均额外步数: {stats['avg_extra_steps']:.1f} ({stats['avg_extra_steps_ratio']:.1%})")
    if bc_stats['final_bc_loss'] is not None:
        print(f"  BC Loss (最近): {bc_stats['final_bc_loss']:.4f}")
    print(f"{'═' * 60}\n")
