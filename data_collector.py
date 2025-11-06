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

# ========== 超参数 ==========
NUM_RUNS = 5000           # 采样多少个不同环境
MAX_STEPS = 800         # 每次仿真最长时间步
DT = 0.1                # 每步时间长度
SAVE_DIR = "./dataset"  # 输出文件夹
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== 启发式子目标生成器 ==========
def heuristic_subgoal_generator(world: World, num_subgoals=3):
    """
    简单启发式：在未探索区采样若干子目标，倾向于远离已访问区域
    """
    visited = world.grid_visited_union
    subgoals = []
    for _ in range(num_subgoals):
        for _ in range(500):
            x, y = world.random_free_pos()
            ci, cj = int(x // 10), int(y // 10)
            if (ci, cj) not in visited:
                subgoals.append((float(x), float(y)))
                break
    return subgoals


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
    print(f"📡 开始采集 {NUM_RUNS} 组仿真数据...")
    paths = []
    for i in tqdm(range(NUM_RUNS)):
        seed = random.randint(0, 100000)
        p = run_single_sim(seed, i)
        paths.append(p)
    print(f"✅ 数据采集完成，共生成 {len(paths)} 个样本。数据保存在 {SAVE_DIR}/")

if __name__ == "__main__":
    main()