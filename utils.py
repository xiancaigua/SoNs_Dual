import pygame
import math
from parameters import *
import os
from datetime import datetime
import json
import numpy as np

# -----------------------------
# 工具函数
# -----------------------------
def load_font():
    # 尝试加载常见中文字体，否则回退系统字体
    try:
        return pygame.font.Font("SimHei.ttf", 16)
    except Exception:
        try:
            return pygame.font.SysFont("SimHei", 16)
        except Exception:
            return pygame.font.SysFont("Arial", 16)

def clamp(x, a, b):
    return max(a, min(b, x))

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def vec_len(v):
    return math.hypot(v[0], v[1])

def normalize(v):
    l = vec_len(v)
    if l == 0:
        return (0,0)
    return (v[0]/l, v[1]/l)

def add(v, w):
    return (v[0]+w[0], v[1]+w[1])

def mul(v, s):
    return (v[0]*s, v[1]*s)

def cell_of_pos(pos):
    return int(pos[0] // GRID_CELL), int(pos[1] // GRID_CELL)

def pos_of_cell(i, j):
    return (i*GRID_CELL + GRID_CELL/2.0, j*GRID_CELL + GRID_CELL/2.0)

def save_simulation_screenshot(screen, world, sim_time, result):
    """保存仿真结果截图"""
    # 创建截图保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_dir = "simulation_screenshots"
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
    
    # 生成文件名
    filename = f"{screenshot_dir}/sim_{timestamp}_{result}.png"
    
    try:
        # 保存当前屏幕内容
        pygame.image.save(screen, filename)
        print(f"截图已保存: {filename}")
        return filename
    except Exception as e:
        print(f"保存截图失败: {e}")
        return None

def create_summary_image(screen, world, sim_time, result, font):
    """创建包含统计信息的总结图像"""
    # 创建稍大的表面来容纳更多信息
    summary_surface = pygame.Surface((SCREEN_W, SCREEN_H))
    # summary_surface = pygame.Surface((SCREEN_W, SCREEN_H + 200))
    summary_surface.fill((240, 240, 240))  # 浅灰色背景
    
    # 复制当前屏幕内容
    summary_surface.blit(screen, (0, 0))
    
    # # 添加详细的统计信息区域
    # info_y = SCREEN_H + 10
    
    # # 标题
    # title_text = f"仿真总结 - {result}"
    # title_surf = font.render(title_text, True, (0, 0, 0))
    # summary_surface.blit(title_surf, (20, info_y))
    
    # # 基本信息
    # info_lines = [
    #     f"仿真时间: {sim_time:.2f}秒",
    #     f"地图覆盖率: {world.coverage_percentage():.2f}%",
    #     f"幸存小型机器人: {sum(1 for a in world.agents if a.alive)}/{len(world.agents)}",
    #     f"幸存大型机器人: {sum(1 for la in world.large_agents if la.alive)}/{len(world.large_agents)}",
    #     f"受害者状态: {'已救援' if world.victim.rescued else '未找到'}",
    #     f"障碍物数量: {len(world.obstacles)}",
    #     f"危险区域: {len(world.danger_zones)}"
    # ]
    
    # for i, line in enumerate(info_lines):
    #     text_surf = font.render(line, True, (0, 0, 0))
    #     summary_surface.blit(text_surf, (20, info_y + 30 + i * 25))
    
    # # 机器人轨迹信息
    # traj_y = info_y + 30 + len(info_lines) * 25 + 10
    # traj_title = font.render("机器人轨迹长度:", True, (0, 0, 0))
    # summary_surface.blit(traj_title, (20, traj_y))
    
    # for i, agent in enumerate(world.agents[:5]):  # 显示前5个机器人的信息
    #     traj_info = f"机器人 {agent.id}: 轨迹点{len(agent.hist)}个"
    #     traj_surf = font.render(traj_info, True, (0, 0, 0))
    #     summary_surface.blit(traj_surf, (40, traj_y + 25 + i * 20))
    
    return summary_surface

def save_simulation_summary(world, sim_time, simulation_result, screenshot_path=None, map_id=None):
    """
    将仿真结果保存为JSON文件
    
    参数:
    - world: 仿真世界对象
    - sim_time: 仿真时间
    - simulation_result: 仿真结果（success/failure/timeout）
    - screenshot_path: 截图文件路径（可选）
    """
    
    # 创建结果目录
    results_dir = "simulation_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 生成时间戳和文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results_dir}/sim_summary_{timestamp}.json"
    
    # 收集基本统计信息
    alive_small = len(world.agents)
    alive_large = len(world.large_agents)
    
    # 构建JSON数据结构
    summary_data = {
        "metadata": {
            "timestamp": timestamp,
            "simulation_time": f"{sim_time:.2f}s",
            "result": simulation_result,
            "screenshot_path": screenshot_path
        },
        "statistics": {
            "simulation_duration": sim_time,
            "alive_small_agents": alive_small,
            "total_small_agents": NUM_AGENTS,
            "alive_large_agents": alive_large,
            "total_large_agents": NUM_LARGE,
            "map_id": map_id,
            "coverage_percentage": world.coverage_percentage(),
            "victim_rescued": world.victim.rescued,
            "obstacles_count": len(world.obstacles),
            "danger_zones_count": len(world.danger_zones)
        },
        "agent_details": {
            "small_agents": [],
            "large_agents": []
        }
    }
    
    # 收集小型机器人详细信息
    for agent in world.agents + world.wasted_agents:
        agent_info = {
            "id": agent.id,
            "alive": agent.alive,
            "final_position": [int(a) for a in agent.pos],
            "trajectory_length": len(agent.hist),
            "explored_cells": len(agent.get_local_explored_cells()),
            "has_goal": agent.has_goal,
            "goal_position": [int(a) for a in agent.goal] if agent.goal is not None else None
        }
        summary_data["agent_details"]["small_agents"].append(agent_info)
    
    # 收集大型机器人详细信息
    for large_agent in world.large_agents + world.wasted_large_agents:
        large_agent_info = {
            "id": large_agent.id,
            "alive": large_agent.alive,
            "final_position": [int(a) for a in large_agent.pos],
            "trajectory_length": len(large_agent.hist),
            "known_cells": int(np.sum(large_agent.known_map != UNKNOWN)),
            "total_cells": large_agent.known_map.size,
            "known_percentage": (np.sum(large_agent.known_map != UNKNOWN) / large_agent.known_map.size) * 100,
            "son_agents": large_agent.son_ids
        }
        summary_data["agent_details"]["large_agents"].append(large_agent_info)
    
    # 保存为JSON文件
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"仿真结果已保存至: {filename}")
        return filename
    except Exception as e:
        print(f"保存JSON文件失败: {e}")
        return None

def print_simulation_summary(world, sim_time, simulation_result):
    """
    在控制台打印仿真总结（保持原有功能）
    """
    print("\n=== Simulation summary ===")
    print(f"Sim time: {sim_time:.2f}s")
    alive_small = sum(1 for a in world.agents if a.alive)
    alive_large = sum(1 for la in world.large_agents if la.alive)
    print(f"Alive: small={alive_small}/{len(world.agents)}, large={alive_large}/{len(world.large_agents)}")
    print(f"Coverage: {world.coverage_percentage():.2f}%")
    print(f"Victim rescued: {world.victim.rescued}")
    
    # Agent trajectories
    for a in world.agents:
        print(f"Agent {a.id} alive={a.alive} traj_len={len(a.hist)} last={a.pos}")
    for la in world.large_agents:
        print(f"LargeAgent {la.id} alive={la.alive} known_cells={np.sum(la.known_map != UNKNOWN)}")

