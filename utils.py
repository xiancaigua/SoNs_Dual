import pygame
import math
from parameters import *
import os
from datetime import datetime
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
    summary_surface = pygame.Surface((SCREEN_W, SCREEN_H + 200))
    summary_surface.fill((240, 240, 240))  # 浅灰色背景
    
    # 复制当前屏幕内容
    summary_surface.blit(screen, (0, 0))
    
    # 添加详细的统计信息区域
    info_y = SCREEN_H + 10
    
    # 标题
    title_text = f"仿真总结 - {result}"
    title_surf = font.render(title_text, True, (0, 0, 0))
    summary_surface.blit(title_surf, (20, info_y))
    
    # 基本信息
    info_lines = [
        f"仿真时间: {sim_time:.2f}秒",
        f"地图覆盖率: {world.coverage_percentage():.2f}%",
        f"幸存小型机器人: {sum(1 for a in world.agents if a.alive)}/{len(world.agents)}",
        f"幸存大型机器人: {sum(1 for la in world.large_agents if la.alive)}/{len(world.large_agents)}",
        f"受害者状态: {'已救援' if world.victim.rescued else '未找到'}",
        f"障碍物数量: {len(world.obstacles)}",
        f"危险区域: {len(world.danger_zones)}"
    ]
    
    for i, line in enumerate(info_lines):
        text_surf = font.render(line, True, (0, 0, 0))
        summary_surface.blit(text_surf, (20, info_y + 30 + i * 25))
    
    # 机器人轨迹信息
    traj_y = info_y + 30 + len(info_lines) * 25 + 10
    traj_title = font.render("机器人轨迹长度:", True, (0, 0, 0))
    summary_surface.blit(traj_title, (20, traj_y))
    
    for i, agent in enumerate(world.agents[:5]):  # 显示前5个机器人的信息
        traj_info = f"机器人 {agent.id}: 轨迹点{len(agent.hist)}个"
        traj_surf = font.render(traj_info, True, (0, 0, 0))
        summary_surface.blit(traj_surf, (40, traj_y + 25 + i * 20))
    
    return summary_surface
