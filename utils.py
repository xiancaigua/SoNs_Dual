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
    
    return summary_surface

def save_simulation_summary(world, sim_time, simulation_result, screenshot_path=None, map_id=None,results_dir = "simulation_results"):
    """
    将仿真结果保存为JSON文件
    
    参数:
    - world: 仿真世界对象
    - sim_time: 仿真时间
    - simulation_result: 仿真结果（success/failure/timeout）
    - screenshot_path: 截图文件路径（可选）
    """
    
    # 创建结果目录
    
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
            "spawn_times":world.spawn_times,
            "dead_agents":(len(world.wasted_agents) + 10*len(world.wasted_large_agents)),
            "coverage_percentage": world.coverage_percentage(),
            "victim_rescued": world.victim.rescued,
            "obstacles_count": len(world.obstacles),
            "danger_zones_count": len(world.danger_zones),
            "explored_safe_count":int(np.sum(world.visited_grid == FREE))
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
            "energycost":agent.energy_cost,
            "final_position": [int(a) for a in agent.pos],
            "trajectory_length": len(agent.hist),
            "hist_traj": [[int(a) for a in pos] for pos in agent.hist],
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
            "energycost":large_agent.energy_cost,
            "final_position": [int(a) for a in large_agent.pos],
            "trajectory_length": len(large_agent.hist),
            "hist_traj": [[int(a) for a in pos] for pos in large_agent.hist],
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

def get_map_files(directory_path="map"):
    """
    打开指定的文件夹，获取其中所有文件的文件名和完整路径。
    
    参数:
        directory_path (str): 地图文件夹的相对路径，默认为 'map'。
        
    返回:
        list: 包含 (文件名, 完整路径) 元组的列表。
    """
    map_data = []
    
    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        print(f"❌ 错误: 目录 '{directory_path}' 不存在。")
        return map_data

    # 遍历目录下的所有文件和文件夹
    for filename in os.listdir(directory_path):
        # 构造文件的完整路径
        full_path = os.path.join(directory_path, filename)
        
        # 检查是否是文件 (排除子文件夹)
        if os.path.isfile(full_path):
            map_data.append((filename, full_path))
            
    return map_data




import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

def debug_plot_matplotlib(agent, planned_cells=None, title="Path Debug (Matplotlib)"):
    """
    使用 Matplotlib 可视化当前地图和路径。
    程序会在此处暂停，关闭窗口后继续运行。
    """
    # 1. 定义颜色映射 (根据你的 parameters.py 中的常量定义)
    # 假设: UNKNOWN=0, FREE=1, OBSTACLE=2, DANGER=3 (请根据实际修改)
    # 建立一个与地图状态对应的颜色列表
    color_list = ['gray', 'white', 'black', 'red', 'gold'] 
    # 映射顺序需对应: [UNKNOWN, FREE, OBSTACLE, DANGER, VICTIM]
    cmap = colors.ListedColormap(color_list)
    bounds = [UNKNOWN, FREE, OBSTACLE, DANGER, VICTIM, VICTIM + 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 2. 绘制地图 (注意: imshow 默认 y轴是 row, x轴是 col)
    # 我们使用 agent.local_map
    ax.imshow(agent.local_map, cmap=cmap, norm=norm, origin='upper')

    # 3. 绘制规划路径
    if planned_cells is not None and len(planned_cells) > 0:
        planned_cells = np.array(planned_cells)
        # 注意: planned_cells 存储的是 (row, col)
        # 在 plot 中，x 是 col (index 1), y 是 row (index 0)
        ax.plot(planned_cells[:, 1], planned_cells[:, 0], 
                color='lime', linewidth=2, label='Planned Path', marker='.')

        # 标记起点和终点
        ax.scatter(planned_cells[0, 1], planned_cells[0, 0], color='blue', s=100, label='Start', zorder=5)
        ax.scatter(planned_cells[-1, 1], planned_cells[-1, 0], color='orange', s=100, label='Goal', zorder=5)

    # 4. 绘制机器人当前位置
    cur_c, cur_r = cell_of_pos(agent.pos)
    ax.scatter(cur_c, cur_r, color='cyan', edgecolors='black', s=150, marker='^', label='Agent')

    # 5. 图表修饰
    ax.set_title(title)
    ax.set_xlabel("Grid Column (X)")
    ax.set_ylabel("Grid Row (Y)")
    ax.grid(True, which='both', color='lightgray', linewidth=0.5)
    ax.legend(loc='upper right')

    plt.tight_layout()
    print(f">>> 调试窗口已打开: {title}。请关闭窗口以继续程序...")
    plt.show() # 此处会阻塞程序运行

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

def visualize_ground_grid(world, title="Ground Truth Map"):
    """
    可视化 World 实例中的 ground_grid (真实地图)。
    """
    # 1. 定义颜色映射（需确保与 parameters.py 中的常量数值对应）
    # 假设：UNKNOWN=0, FREE=1, OBSTACLE=2, DANGER=3, VICTIM=4
    # 如果你的常量定义不同，请调整这里的顺序
    color_map = {
        UNKNOWN:  '#A0A0A0',  # 灰色
        FREE:     '#FFFFFF',  # 白色
        OBSTACLE: '#2C3E50',  # 深灰色/黑色
        DANGER:   '#E74C3C',  # 红色
        VICTIM:   '#F1C40F'   # 金色
    }
    
    # 提取数组
    grid_data = world.ground_grid
    h, w = grid_data.shape

    # 创建自定义 Colormap
    # 按数值顺序排列：0, 1, 2, 3, 4
    ordered_colors = [color_map[UNKNOWN], color_map[FREE], color_map[OBSTACLE], color_map[DANGER], color_map[VICTIM]]
    cmap = colors.ListedColormap(ordered_colors)
    bounds = [0, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 2. 绘制网格
    # origin='upper' 确保 (0,0) 在左上角，符合数组索引 [row, col]
    im = ax.imshow(grid_data, cmap=cmap, norm=norm, origin='upper')

    # 3. 绘制装饰元素
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Column Index (X)", fontsize=12)
    ax.set_ylabel("Row Index (Y)", fontsize=12)

    # 添加辅助网格线（可选，如果地图很大建议关闭）
    if w < 100 and h < 100:
        ax.set_xticks(np.arange(-.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-.5, h, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.2)

    # 4. 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map[FREE], edgecolor='gray', label='Free Space'),
        Patch(facecolor=color_map[OBSTACLE], edgecolor='black', label='Obstacle'),
        Patch(facecolor=color_map[DANGER], edgecolor='red', label='Danger Zone'),
        Patch(facecolor=color_map[VICTIM], edgecolor='black', label='Victim')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    # 5. (可选) 如果想同时看到机器人的当前位置
    for agent in world.agents + world.large_agents:
        if agent.alive:
            cx, cy = cell_of_pos(agent.pos)
            color = 'blue' if not agent.is_large else 'green'
            ax.scatter(cx, cy, c=color, s=40, edgecolors='white', marker='o')

    plt.tight_layout()
    plt.show()