import json
import os
import glob
import numpy as np

# ----------------------------
# 配置路径
# ----------------------------
json_folder = "./base1json/未知危险环境base图"  # JSON 文件所在文件夹
# json_folder = "./simulation_results"  # JSON 文件所在文件夹
json_files = glob.glob(os.path.join(json_folder, "*.json"))

# ----------------------------
# 初始化统计量
# ----------------------------
total_runs = 0
success_count = 0
simulation_times = []
small_agent_death_rates = []
coverage_percentages = []
small_agent_traj_lengths = []
small_agent_explored_cells = []

# ----------------------------
# 处理每个文件
# ----------------------------
for fpath in json_files:
    with open(fpath, "r") as f:
        data = json.load(f)
    
    total_runs += 1
    
    stats = data.get("statistics", {})
    agents = data.get("agent_details", {})
    
    # 成功与否
    if stats.get("victim_rescued", False):
        success_count += 1
    
    # 仿真时间
    simulation_times.append(stats.get("simulation_duration", 0))
    
    # 小机器人死亡率
    small_agents = agents.get("small_agents", [])
    if small_agents:
        alive_count = sum(1 for a in small_agents if a.get("alive", False))
        death_rate = 1 - alive_count / len(small_agents)
        small_agent_death_rates.append(death_rate)
        
        # 轨迹长度和探索格子数
        traj_lengths = [a.get("trajectory_length", 0) for a in small_agents]
        explored_cells = [a.get("explored_cells", 0) for a in small_agents]
        small_agent_traj_lengths.append(np.mean(traj_lengths))
        small_agent_explored_cells.append(np.mean(explored_cells))
    
    # 覆盖率
    coverage_percentages.append(stats.get("coverage_percentage", 0))

# ----------------------------
# 输出统计结果
# ----------------------------
print("Total runs:", total_runs)
print("Success rate: {:.2f}%".format(100 * success_count / total_runs))
print("Average simulation time: {:.2f}s".format(np.mean(simulation_times)))
print("Average small agent death rate: {:.2f}%".format(100 * np.mean(small_agent_death_rates)))
print("Average coverage: {:.2f}%".format(np.mean(coverage_percentages)))
print("Average small agent trajectory length: {:.2f}".format(np.mean(small_agent_traj_lengths)))
print("Average small agent explored cells: {:.2f}".format(np.mean(small_agent_explored_cells)))
