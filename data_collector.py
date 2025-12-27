import os
import json
import numpy as np
import random
import time
from tqdm import tqdm
from env import World
from communicate import Communication
from parameters import *
from utils import cell_of_pos, pos_of_cell

# ========== 采集配置 ==========
NUM_RUNS = 10           # 运行的环境（种子）数量
MAX_STEPS = 1500        # 每个环境的最大仿真步数
DT = 0.1                # 仿真时间步长
DOWNSAMPLE_FACTOR = 2   # 地图降采样倍数 (100x70 -> 50x35)
SAVE_DIR = "./brain_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

def downsample_map(grid, factor=2):
    """
    对 known_map 进行降采样，以减小数据体积并保留关键特征 。
    """
    h, w = grid.shape
    new_h, new_w = h // factor, w // factor
    # 截取对齐后的区域
    resizing_grid = grid[:new_h * factor, :new_w * factor]
    # 使用 max-pooling 降采样：保留障碍物(1)和危险区(2)特征
    downsampled = resizing_grid.reshape(new_h, factor, new_w, factor).max(axis=(1, 3))
    return downsampled.astype(np.int8)

def extract_brain_state(world):
    """
    提取 BrainAgent 决策时的动态状态向量 (6维) 。
    """
    coverage = world.coverage_percentage()
    alive_large = sum(1 for la in world.large_agents if la.alive)
    return np.array([
        coverage,                       # 当前地图覆盖率
        float(alive_large),             # 存活的领队数量
        float(world.spawn_times),       # 历史增援次数
        float(len(world.obstacles)),     # 环境障碍物总数
        float(len(world.danger_zones)),  # 环境危险区总数
        float(world.time)               # 当前仿真时刻
    ], dtype=np.float32)

def run_brain_collection(seed, idx):
    """单次仿真采集函数"""
    random.seed(seed)
    np.random.seed(seed)
    world = World(seed=seed)
    comms = Communication()
    
    samples = []
    step = 0
    
    while step < MAX_STEPS and not world.victim.rescued:
        now_time = step * DT
        world.update(DT, comms, now_time) # 运行专家启发式逻辑
        
        # 仅在 BrainAgent 的决策周期进行检查 
        if step % int(BRAIN_REASON_INTERVAL / DT) == 0:
            middles = [la for la in world.large_agents if la.alive]
            # 获取专家分配结果
            assignments = world.brain.assign_region_to_middle(middles)
            
            # 1. 粒度要求：仅在有实际分配产生时记录
            has_valid_assignment = any(len(m_list) > 0 for m_list in assignments.items())
            
            if has_valid_assignment:
                # 2. 采集 known_map 的降采样版本 
                sampled_map = downsample_map(world.known_grid, factor=DOWNSAMPLE_FACTOR)
                
                # 3. 构造样本
                sample = {
                    "seed": seed,                   # 静态特征：种子号
                    "sim_step": step,
                    "state_vector": extract_brain_state(world).tolist(),
                    "known_map_downsampled": sampled_map.tolist(), # 降采样地图
                    "subgoals": world.brain.get_assigned_centers(assignments), # 决策标签
                    "map_factor": DOWNSAMPLE_FACTOR
                }
                samples.append(sample)
        
        step += 1
        if len(world.large_agents) == 0: break # 领队全灭则停止

    # 保存该种子产生的所有样本
    if samples:
        out_path = os.path.join(SAVE_DIR, f"brain_data_seed_{seed}.json")
        with open(out_path, "w") as f:
            json.dump(samples, f, indent=2)
        return len(samples)
    return 0

def main():
    print(f"开始 BrainAgent 专家数据采集...")
    total_saved = 0
    for i in tqdm(range(NUM_RUNS)):
        seed = random.randint(0, 1000000)
        count = run_brain_collection(seed, i)
        total_saved += count
    
    print(f"\n✅ 采集完成！总计获得 {total_saved} 个有效决策时刻。")
    print(f"数据已存入: {SAVE_DIR}")

if __name__ == "__main__":
    main()