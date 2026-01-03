import os
import json
import numpy as np
import glob
from tqdm import tqdm

def load_and_aggregate_brain_data(folder_path):
    """
    读取文件夹内所有的 JSON 文件并汇总数据。
    """
    all_maps = []
    all_states = []
    all_goals = []
    
    # 获取文件夹下所有 json 文件路径
    file_pattern = os.path.join(folder_path, "*.json")
    json_files = glob.glob(file_pattern)
    
    print(f"🔍 找到 {len(json_files)} 个数据文件，开始解析...")

    for file_path in tqdm(json_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # 每个文件是一个列表，包含多个 sim_step
                for entry in data:
                    # 1. 提取状态向量 (state_vector, 长度为 6)
                    state = np.array(entry['state_vector'], dtype=np.float32)
                    
                    # 2. 提取并预处理地图 (known_map_downsampled, 50x50)
                    # 增加一个通道维度 (1, 50, 50) 以适配 CNN
                    grid_map = np.array(entry['known_map_downsampled'], dtype=np.float32)
                    grid_map = np.expand_dims(grid_map, axis=0) 
                    
                    # 3. 提取目标点 (subgoals)
                    # 假设我们只训练预测第一个子目标点 [x, y]
                    if len(entry['subgoals']) > 0:
                        goal = np.array(entry['subgoals'], dtype=np.float32)
                        
                        all_states.append(state)
                        all_maps.append(grid_map)
                        all_goals.append(goal)
                        
            except Exception as e:
                print(f"❌ 读取文件 {file_path} 出错: {e}")

    # 转换为 NumPy 数组
    all_states = np.array(all_states)
    all_maps = np.array(all_maps)
    all_goals = np.array(all_goals)

    print("\n✅ 数据汇总完成:")
    print(f"   - 状态特征形状: {all_states.shape}") # (N, 6)
    print(f"   - 地图特征形状: {all_maps.shape}")   # (N, 1, 50, 50)
    print(f"   - 目标标签形状: {all_goals.shape}")   # (N, 2)
    
    return all_states, all_maps, all_goals

# --- 使用示例 ---
data_folder = "./neural/brain_dataset"
states, maps, goals = load_and_aggregate_brain_data(data_folder)

# 如果你想保存为更快的格式（如训练前直接加载）
np.savez_compressed("training_data.npz", states=states, maps=maps, goals=goals)