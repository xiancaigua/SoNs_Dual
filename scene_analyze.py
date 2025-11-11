import os
import json
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 配置参数
# -------------------------------
# RESULTS_DIR = "simulation_results"   # 存放JSON的目录
RESULTS_DIR = "base1json/未知危险环境base图"   # 存放JSON的目录
NUM_SCENES = 10                      # 场景数量
FILES_PER_SCENE = 10                 # 每个场景的文件数

# -------------------------------
# 工具函数
# -------------------------------
def load_json_files(folder):
    """加载指定文件夹下的前100个JSON文件"""
    files = sorted([f for f in os.listdir(folder) if f.endswith(".json")])
    files = files[:NUM_SCENES * FILES_PER_SCENE]
    data = []
    for fname in files:
        path = os.path.join(folder, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data.append(json.load(f))
        except Exception as e:
            print(f"⚠️ 无法读取 {fname}: {e}")
    print(f"✅ 成功加载 {len(data)} 个文件")
    return data


# -------------------------------
# 主分析逻辑
# -------------------------------
def analyze_overall(data):
    """总体概览统计"""
    all_alive_rates, all_traj_lengths = [], []

    for record in data:
        stat = record["statistics"]
        agents = record["agent_details"]["small_agents"]

        # ✅ 每个实验的存活率
        if stat["total_small_agents"] > 0:
            alive_rate = stat["alive_small_agents"] / 6
            # alive_rate = stat["alive_small_agents"] / stat["total_small_agents"]
            all_alive_rates.append(alive_rate)

        # 所有小机器人的轨迹长度
        traj_lengths = [a["trajectory_length"] for a in agents]
        all_traj_lengths.extend(traj_lengths)

    print("\n=== 🌍 总体统计 ===")
    print(f"平均存活率: {np.mean(all_alive_rates):.3f} (方差={np.var(all_alive_rates):.4f})")
    print(f"轨迹长度平均: {np.mean(all_traj_lengths):.1f}, 方差: {np.var(all_traj_lengths):.1f}")
    return all_alive_rates, all_traj_lengths


def analyze_by_scene(data):
    """按场景统计 (平均存活率 + 方差, 平均轨迹长度 + 方差)"""
    scene_stats = []

    for i in range(NUM_SCENES):
        scene_data = data[i*FILES_PER_SCENE:(i+1)*FILES_PER_SCENE]

        # ✅ 每个实验一个存活率
        exp_alive_rates = []
        traj_lengths_all = []

        for record in scene_data:
            stat = record["statistics"]
            agents = record["agent_details"]["small_agents"]

            # 每次实验一个存活率
            if stat["total_small_agents"] > 0:
                exp_alive_rates.append(stat["alive_small_agents"] / 6)
                # exp_alive_rates.append(stat["alive_small_agents"] / stat["total_small_agents"])

            # 所有小机器人轨迹长度
            traj_lengths_all += [a["trajectory_length"] for a in agents]

        mean_alive = np.mean(exp_alive_rates)
        var_alive = np.var(exp_alive_rates)
        mean_traj = np.mean(traj_lengths_all)
        var_traj = np.var(traj_lengths_all)

        scene_stats.append((mean_alive, var_alive, mean_traj, var_traj))

        print(f"\n--- 场景 {i+1} ---")
        print(f"平均存活率: {mean_alive:.3f} (方差={var_alive:.4f})")
        print(f"平均轨迹长度: {mean_traj:.1f} (方差={var_traj:.1f})")

    return scene_stats


# -------------------------------
# 绘图逻辑
# -------------------------------
def plot_scene_stats(scene_stats, save_dir="analysis_results"):
    """绘制每个场景的统计图（包含方差标注）"""
    os.makedirs(save_dir, exist_ok=True)
    scenes = np.arange(1, NUM_SCENES + 1)
    mean_alive = [s[0] for s in scene_stats]
    var_alive = [s[1] for s in scene_stats]
    mean_traj = [s[2] for s in scene_stats]
    var_traj = [s[3] for s in scene_stats]

    # ========== 图1：平均存活率 + 方差 ==========
    plt.figure(figsize=(8, 4))
    bars = plt.bar(scenes, mean_alive, color="#4CAF50", alpha=0.8,
                   yerr=np.sqrt(var_alive), capsize=5, ecolor='black')
    plt.title("Average Survival Rate per Scene (with Variance)", fontsize=14)
    plt.xlabel("Scene ID", fontsize=12)
    plt.ylabel("Mean Survival Rate", fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(scenes)

    # for bar, mean, var in zip(bars, mean_alive, var_alive):
    #     plt.text(bar.get_x() + bar.get_width()/2, mean + np.sqrt(var) + 0.02,
    #              f"{var:.2f}", ha="center", va="bottom", fontsize=9, color='darkred')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scene_survival_rate_variance.png"), dpi=200)
    plt.close()

    # ========== 图2：平均轨迹长度 + 方差 ==========
    plt.figure(figsize=(8, 4))
    bars = plt.bar(scenes, mean_traj, color="#2196F3", alpha=0.8,
                   yerr=np.sqrt(var_traj), capsize=5, ecolor='black')
    plt.title("Average Trajectory Length per Scene (with Variance)", fontsize=14)
    plt.xlabel("Scene ID", fontsize=12)
    plt.ylabel("Mean Trajectory Length", fontsize=12)
    plt.xticks(scenes)

    # for bar, mean, var in zip(bars, mean_traj, var_traj):
    #     plt.text(bar.get_x() + bar.get_width()/2, mean + np.sqrt(var) + 5,
    #              f"{var:.1f}", ha="center", va="bottom", fontsize=9, color='darkred')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scene_traj_length_variance.png"), dpi=200)
    plt.close()

    print(f"\n📊 图表已保存到: {save_dir}")


# -------------------------------
# 主执行
# -------------------------------
if __name__ == "__main__":
    data = load_json_files(RESULTS_DIR)
    analyze_overall(data)
    scene_stats = analyze_by_scene(data)
    plot_scene_stats(scene_stats)
