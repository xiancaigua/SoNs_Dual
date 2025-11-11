import os
import json
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
#                参数配置
# =====================================================
# RESULTS_DIR = "simulation_results"   # 存放JSON的目录
RESULTS_DIR = "experiment_results/base重测"   # 存放JSON的目录
NUM_SCENES = 10                      # 场景数量
FILES_PER_SCENE = 10                 # 每个场景的实验次数

# =====================================================
#                工具函数
# =====================================================
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

# =====================================================
#                总体分析
# =====================================================
def analyze_overall(data):
    """总体数据统计"""
    alive_rates, traj_lengths, coverages, success_flags, durations = [], [], [], [], []

    for record in data:
        stat = record["statistics"]
        agents = record["agent_details"]["small_agents"]

        # ✅ 综合生存率（大小机器人加权）
        total_small = stat["total_small_agents"]
        total_large = stat["total_large_agents"]
        alive_small = stat["alive_small_agents"]
        alive_large = stat["alive_large_agents"]

        total_equiv = total_small + 2 * total_large
        alive_equiv = alive_small + 2 * alive_large
        if total_equiv > 0:
            alive_rates.append(alive_equiv / total_equiv)

        # ✅ 覆盖率
        coverages.append(stat["coverage_percentage"])

        # ✅ 轨迹长度
        traj_lengths.extend([a["trajectory_length"] for a in agents])

        # ✅ 成功率
        success_flags.append(1 if stat["victim_rescued"] else 0)

        # ✅ 耗时
        durations.append(stat["simulation_duration"])

    print("\n=== 🌍 总体统计 ===")
    print(f"平均生存率: {np.mean(alive_rates):.3f} (方差={np.var(alive_rates):.4f})")
    print(f"平均探索覆盖率: {np.mean(coverages):.2f}% (方差={np.var(coverages):.2f})")
    print(f"平均轨迹长度: {np.mean(traj_lengths):.1f} (方差={np.var(traj_lengths):.1f})")
    print(f"总体成功率: {np.mean(success_flags):.3f}")
    print(f"平均耗时: {np.mean(durations):.2f}s (方差={np.var(durations):.2f})")

    return alive_rates, traj_lengths, coverages, success_flags, durations

# =====================================================
#                场景分析
# =====================================================
def analyze_by_scene(data):
    """逐场景分析：平均生存率、轨迹长度、覆盖率、成功率、耗时 + 方差"""
    scene_stats = []

    for i in range(NUM_SCENES):
        scene_data = data[i*FILES_PER_SCENE:(i+1)*FILES_PER_SCENE]

        alive_rates, traj_lengths, coverages, success_flags, durations = [], [], [], [], []

        for record in scene_data:
            stat = record["statistics"]
            agents = record["agent_details"]["small_agents"]

            # ✅ 生存率（大小机器人加权）
            total_small = stat["total_small_agents"]
            total_large = stat["total_large_agents"]
            alive_small = stat["alive_small_agents"]
            alive_large = stat["alive_large_agents"]

            total_equiv = total_small + 2 * total_large
            alive_equiv = alive_small + 2 * alive_large
            if total_equiv > 0:
                alive_rates.append(alive_equiv / total_equiv)

            # ✅ 覆盖率
            coverages.append(stat["coverage_percentage"])

            # ✅ 轨迹长度
            traj_lengths += [a["trajectory_length"] for a in agents]

            # ✅ 成功率
            success_flags.append(1 if stat["victim_rescued"] else 0)

            # ✅ 耗时
            durations.append(stat["simulation_duration"])

        # 汇总
        mean_alive, var_alive = np.mean(alive_rates), np.var(alive_rates)
        mean_traj, var_traj = np.mean(traj_lengths), np.var(traj_lengths)
        mean_cov, var_cov = np.mean(coverages), np.var(coverages)
        success_rate = np.mean(success_flags)
        mean_time, var_time = np.mean(durations), np.var(durations)

        scene_stats.append((mean_alive, var_alive,
                            mean_traj, var_traj,
                            mean_cov, var_cov,
                            success_rate,
                            mean_time, var_time))

        print(f"\n--- 场景 {i+1} ---")
        print(f"平均生存率: {mean_alive:.3f} (方差={var_alive:.4f})")
        print(f"平均轨迹长度: {mean_traj:.1f} (方差={var_traj:.1f})")
        print(f"平均覆盖率: {mean_cov:.2f}% (方差={var_cov:.2f})")
        print(f"成功率: {success_rate:.2f}")
        print(f"平均耗时: {mean_time:.2f}s (方差={var_time:.2f})")

    return scene_stats

# =====================================================
#                绘图函数
# =====================================================
def plot_scene_stats(scene_stats, save_dir="analysis_results"):
    os.makedirs(save_dir, exist_ok=True)
    scenes = np.arange(1, NUM_SCENES + 1)

    mean_alive = [s[0] for s in scene_stats]
    var_alive = [s[1] for s in scene_stats]
    mean_traj = [s[2] for s in scene_stats]
    var_traj = [s[3] for s in scene_stats]
    mean_cov = [s[4] for s in scene_stats]
    var_cov = [s[5] for s in scene_stats]
    success_rate = [s[6] for s in scene_stats]
    mean_time = [s[7] for s in scene_stats]
    var_time = [s[8] for s in scene_stats]

    def _plot_bar_with_var(values, variances, title, ylabel, color, filename, unit=""):
        plt.figure(figsize=(8, 4))
        bars = plt.bar(scenes, values, yerr=np.sqrt(variances),
                       color=color, alpha=0.8, capsize=5, ecolor='black')
        plt.title(title, fontsize=14)
        plt.xlabel("Scene ID")
        plt.ylabel(ylabel)
        plt.xticks(scenes)
        for bar, mean, var in zip(bars, values, variances):
            plt.text(bar.get_x() + bar.get_width()/2, mean + np.sqrt(var) + 0.02*max(values),
                     f"σ²={var:.2f}{unit}", ha="center", va="bottom", fontsize=9, color='darkred')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=200)
        plt.close()

    # 各类统计图
    _plot_bar_with_var(mean_alive, var_alive, "Average Survival Rate per Scene", "Survival Rate", "#4CAF50", "scene_survival_rate.png")
    _plot_bar_with_var(mean_cov, var_cov, "Average Coverage per Scene (%)", "Coverage (%)", "#FF9800", "scene_coverage.png")
    _plot_bar_with_var(mean_traj, var_traj, "Average Trajectory Length per Scene", "Trajectory Length", "#2196F3", "scene_trajectory.png")
    _plot_bar_with_var(mean_time, var_time, "Average Simulation Time per Scene", "Time (s)", "#9E9E9E", "scene_simulation_time.png", unit="s²")

    # 成功率
    plt.figure(figsize=(8, 4))
    bars = plt.bar(scenes, success_rate, color="#9C27B0", alpha=0.8)
    plt.title("Rescue Success Rate per Scene", fontsize=14)
    plt.xlabel("Scene ID")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.1)
    plt.xticks(scenes)
    for bar, val in zip(bars, success_rate):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scene_success_rate.png"), dpi=200)
    plt.close()

    print(f"\n📊 图表已保存到: {save_dir}")

# =====================================================
#                主执行逻辑
# =====================================================
if __name__ == "__main__":
    data = load_json_files(RESULTS_DIR)
    analyze_overall(data)
    scene_stats = analyze_by_scene(data)
    plot_scene_stats(scene_stats)
