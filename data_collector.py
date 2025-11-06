"""
data_collector.py
批量采集多机器人仿真数据，用于学习高层策略（子目标序列生成）
依赖：
 - World, AgentBase, LargeAgent 等定义（导入自你的主程序）
 - Communication 类（消息传递模拟）
输出：
 - ./dataset/sim_data_xxx.json  文件
"""

import os
import json
import numpy as np
import random
import time
from tqdm import tqdm
from env import World        # 假设你的World定义在world.py中
from communicate import Communication
import heapq
import matplotlib.pyplot as plt
# from utils import cell_of_pos, pos_of_cell

# ========== 超参数 ==========
NUM_RUNS = 2           # 采样多少个不同环境
MAX_STEPS = 800         # 每次仿真最长时间步
DT = 0.1                # 每步时间长度
SAVE_DIR = "./dataset"  # 输出文件夹
os.makedirs(SAVE_DIR, exist_ok=True)

# ====== 可选：matplotlib 图形配置 ======
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 12

def visualize_subgoals(world, subgoals):
    """
    绘制：
    - 障碍物 (灰)
    - 危险区 (红)
    - 大节点 (蓝)
    - 被困者 (金色星形)
    - 子目标 (按模式区分颜色)
    """
    fig, ax = plt.subplots()

    # --- 画障碍物 ---
    for obs in world.obstacles:
        rect = plt.Rectangle((obs.rect.left, obs.rect.top),
                             obs.rect.width, obs.rect.height,
                             color='gray', alpha=0.5)
        ax.add_patch(rect)

    # --- 画危险区 ---
    for dz in world.danger_zones:
        circ = plt.Circle(dz.pos, dz.r, color='red', alpha=0.2)
        ax.add_patch(circ)

    # --- 画大节点 / 小节点 / 被困者 ---
    for la in world.large_agents:
        ax.scatter(la.pos[0], la.pos[1], c='blue', s=100, label='LargeAgent' if la.id == 0 else None)
    for a in world.agents:
        ax.scatter(a.pos[0], a.pos[1], c='cyan', s=30, label='Agent' if a.id == world.agents[0].id else None)
    if world.victim:
        ax.scatter(world.victim.pos[0], world.victim.pos[1], c='gold', marker='*', s=200, label='Victim')

    # --- 画路径 ---
    colors = ['green', 'lime', 'blue', 'deepskyblue', 'orange', 'magenta']
    markers = ['o', 's', 'd', '^', 'v', '*']
    for i, (x, y) in enumerate(subgoals):
        ax.scatter(x, y, s=90, c=colors[i % len(colors)], marker=markers[i % len(markers)],
                   label=f'Subgoal {i+1}')

    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title("Generated Expert Subgoals (Safe/Fast/Hybrid)")
    plt.show()


# ========== 启发式子目标生成器 ==========
def heuristic_subgoal_generator(world, num_subgoals_total=6):
    """
    基于全局地图的多目标A*专家路径生成器
    - 每种模式 ('safe', 'fast', 'hybrid') 生成两条路径
    - 每条路径提取若干关键子目标点
    """
    
    grid = world.ground_grid
    victim = world.victim
    if victim is None:
        return []

    # ============= Helper functions =============
    def cell_of_pos(pos):
        return int(pos[0] // 10), int(pos[1] // 10)

    def pos_of_cell(i, j):
        return i * 10.0 + 5.0, j * 10.0 + 5.0

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def cost_fn(x, y):
        val = grid[x, y]
        if val == 1:  # OBSTACLE
            return 9999.0
        elif val == 2:  # DANGER
            return 20.0
        else:
            return 1.0

    # ================== 改进 A* ==================
    def astar_multicost(start, goal, mode='hybrid', jitter=0.0):
        """多目标A*搜索，可调整安全/效率权重"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: heuristic(start, goal)}
        w, h = grid.shape

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),
                           (-1,-1),(1,-1),(-1,1),(1,1)]:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < w and 0 <= ny < h:
                    step_cost = cost_fn(nx, ny)
                    if step_cost >= 9999:
                        continue
                    # 加入一点随机扰动，避免路径重叠
                    rand_penalty = random.uniform(0, jitter)
                    new_g = g_score[current] + step_cost + rand_penalty
                    if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = new_g
                        h_cost = heuristic((nx, ny), goal)
                        if mode == 'safe':
                            f = new_g * 1.0 + h_cost * 0.8
                        elif mode == 'fast':
                            f = new_g * 0.5 + h_cost * 2.0
                        else:  # hybrid
                            f = new_g * 0.7 + h_cost * 1.3
                        heapq.heappush(open_set, (f, (nx, ny)))
        return None

    # ============= 选择起点与目标 =============
    if world.large_agents:
        base_start = min(world.large_agents, key=lambda la: la.id).pos
    else:
        base_start = world.random_free_pos()

    start_cell = cell_of_pos(base_start)
    goal_cell = cell_of_pos(victim.pos)

    # ============= 生成多路径 =============
    modes = ['safe', 'fast', 'hybrid']
    all_subgoals = []

    for mode in modes:
        for variant in range(2):  # 每种模式两条
            # 起点扰动以产生不同路径
            sx = start_cell[0] + random.randint(-3, 3)
            sy = start_cell[1] + random.randint(-3, 3)
            sx = int(np.clip(sx, 0, grid.shape[0]-1))
            sy = int(np.clip(sy, 0, grid.shape[1]-1))
            path = astar_multicost((sx, sy), goal_cell, mode=mode, jitter=0.3)
            if not path:
                continue

            # 从路径上提取关键点（均匀抽样）
            n_sub = max(2, num_subgoals_total // (len(modes)*2))
            idxs = np.linspace(0, len(path)-1, n_sub, dtype=int)
            key_cells = [path[i] for i in idxs]
            for c in key_cells:
                px, py = pos_of_cell(*c)
                all_subgoals.append((float(px), float(py)))

    # 确保总数限制
    all_subgoals = all_subgoals[:num_subgoals_total]
    return all_subgoals



# ========== 状态特征提取函数 ==========
def extract_state_features(world: World):
    """构造一个全局抽象状态向量"""
    coverage = len(world.grid_visited_union) / (world.width * world.height / 100)  # 简化覆盖率指标
    alive_small = sum(1 for a in world.agents if a.alive)
    alive_large = sum(1 for a in world.large_agents if a.alive)
    avg_energy = np.mean([getattr(a, 'energy', 1.0) for a in world.agents + world.large_agents])
    # 计算 victim 相对最近大节点的距离
    if world.victim is not None:
        victim_dists = [np.hypot(world.victim.pos[0] - la.pos[0], world.victim.pos[1] - la.pos[1]) for la in world.large_agents]
        victim_dist = min(victim_dists) if victim_dists else 0.0
    else:
        victim_dist = 0.0
    # 通信连通性估计（小节点与大节点之间平均距离）
    if world.agents and world.large_agents:
        avg_conn = np.mean([
            min(np.hypot(a.pos[0]-la.pos[0], a.pos[1]-la.pos[1]) for la in world.large_agents)
            for a in world.agents
        ])
    else:
        avg_conn = 0.0
    return np.array([coverage, alive_small, alive_large, avg_energy, victim_dist, avg_conn], dtype=np.float32)


# ========== 主采集流程 ==========
def run_single_sim(seed, idx):
    random.seed(seed)
    np.random.seed(seed)
    world = World(seed=seed)
    comms = Communication()

    step = 0
    found_victim = False
    subgoal_seq = heuristic_subgoal_generator(world)

    while step < MAX_STEPS:
        now_time = step * DT
        world.update_baseline(DT, comms, now_time)
        step += 1

        # 检查victim发现
        for a in world.agents:
            if world.victim and np.hypot(a.pos[0]-world.victim.pos[0], a.pos[1]-world.victim.pos[1]) < 10:
                found_victim = True
                break
        if found_victim:
            break

    # 提取状态和回报
    state_vec = extract_state_features(world)
    reward = 1.0 if found_victim else state_vec[0] * 0.5  # 奖励为发现victim或覆盖率加权

    # 保存数据
    record = {
        "seed": seed,
        "state_vector": state_vec.tolist(),
        "subgoals": subgoal_seq,
        "reward": float(reward),
        "found_victim": found_victim
    }
    out_path = os.path.join(SAVE_DIR, f"sim_data_{idx:03d}.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return out_path


def main():
    print(f"开始采集 {NUM_RUNS} 组仿真数据...")
    paths = []
    for i in tqdm(range(NUM_RUNS)):
        seed = random.randint(0, 100000)
        p = run_single_sim(seed, i)
        paths.append(p)
    print(f"数据采集完成，共生成 {len(paths)} 个样本。数据保存在 {SAVE_DIR}/")

if __name__ == "__main__":
    # main()

    # ====== 初始化随机环境 ======
    world = World(seed=42)
    comms = Communication()

    # ====== 生成子目标 ======
    subgoals = heuristic_subgoal_generator(world, num_subgoals_total=6)

    # ====== 绘图 ======
    visualize_subgoals(world, subgoals)