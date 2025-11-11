import os
import json
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
#                参数配置
# =====================================================
METHOD_DIRS = {
    "Ours": "experiment_results/mine_result",
    "ERRT": "experiment_results/base重测"
}

NUM_SCENES = 10
FILES_PER_SCENE = 10
SAVE_DIR = "comparison_results"

# =====================================================
#                数据加载与解析
# =====================================================
def load_json_data(folder):
    """加载一个方法的所有JSON文件"""
    files = sorted([f for f in os.listdir(folder) if f.endswith(".json")])
    data = []
    for fname in files[:NUM_SCENES * FILES_PER_SCENE]:
        path = os.path.join(folder, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data.append(json.load(f))
        except Exception as e:
            print(f"⚠️ 读取失败 {fname}: {e}")
    print(f"✅ {folder}: 读取 {len(data)} 个文件")
    return data

# =====================================================
#                场景统计计算
# =====================================================
def compute_scene_stats(data):
    """计算每个场景的均值和方差"""
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

            total_equiv = total_small + 0 * total_large
            alive_equiv = alive_small + 0 * alive_large
            if total_equiv > 0:
                alive_rates.append(alive_equiv / total_equiv)

            # ✅ 轨迹长度
            traj_lengths += [a["trajectory_length"] for a in agents]

            # ✅ 覆盖率
            coverages.append(stat["coverage_percentage"])

            # ✅ 成功率
            success_flags.append(1 if stat["victim_rescued"] else 0)

            # ✅ 耗时
            durations.append(stat["simulation_duration"])

        # 汇总
        stats = {
            "survival_mean": np.mean(alive_rates),
            "survival_var": np.var(alive_rates),
            "traj_mean": np.mean(traj_lengths),
            "traj_var": np.var(traj_lengths),
            "coverage_mean": np.mean(coverages),
            "coverage_var": np.var(coverages),
            "success_mean": np.mean(success_flags),
            "time_mean": np.mean(durations),
            "time_var": np.var(durations)
        }
        scene_stats.append(stats)
    return scene_stats

# =====================================================
#                可视化
# =====================================================
def plot_comparison(stats_A, stats_B, metric_key, ylabel, title, colorA, colorB, filename, unit=""):
    """绘制两个方法的对比柱状图"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    scenes = np.arange(1, NUM_SCENES + 1)
    mean_A = [s[f"{metric_key}_mean"] for s in stats_A]
    var_A = [s.get(f"{metric_key}_var", 0) for s in stats_A]
    mean_B = [s[f"{metric_key}_mean"] for s in stats_B]
    var_B = [s.get(f"{metric_key}_var", 0) for s in stats_B]

    width = 0.35
    plt.figure(figsize=(9, 4))
    bars_A = plt.bar(scenes - width/2, mean_A, width, yerr=np.sqrt(var_A), label="Ours",
                     color=colorA, capsize=4, alpha=0.8)
    bars_B = plt.bar(scenes + width/2, mean_B, width, yerr=np.sqrt(var_B), label="ERRT",
                     color=colorB, capsize=4, alpha=0.8)
    plt.xlabel("Scene ID", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(scenes)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 标注平均值 ± 方差
    # for bars, means, vars in [(bars_A, mean_A, var_A), (bars_B, mean_B, var_B)]:
    #     for bar, mean, var in zip(bars, means, vars):
    #         plt.text(bar.get_x() + bar.get_width()/2, mean + 0.02 * max(mean_A + mean_B),
    #                  f"{mean:.2f}\nσ²={var:.2f}{unit}", ha="center", va="bottom", fontsize=8, color="black")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=200)
    plt.close()

# =====================================================
#                主执行逻辑
# =====================================================
if __name__ == "__main__":
    # 读取两个方法的数据
    data_A = load_json_data(METHOD_DIRS["Ours"])
    data_B = load_json_data(METHOD_DIRS["ERRT"])

    # 计算每个场景的统计结果
    stats_A = compute_scene_stats(data_A)
    stats_B = compute_scene_stats(data_B)

    # 绘制五个指标的对比图
    plot_comparison(stats_A, stats_B, "success", "Success Rate",
                    "Rescue Success Rate Comparison", "#9C27B0", "#E1BEE7", "compare_success_rate.png")

    plot_comparison(stats_A, stats_B, "survival", "Survival Rate",
                    "Robot Survival Rate Comparison", "#4CAF50", "#A5D6A7", "compare_survival_rate.png")

    plot_comparison(stats_A, stats_B, "coverage", "Coverage (%)",
                    "Exploration Coverage Comparison", "#FF9800", "#FFE0B2", "compare_coverage.png")

    plot_comparison(stats_A, stats_B, "traj", "Trajectory Length",
                    "Trajectory Length Comparison", "#2196F3", "#90CAF9", "compare_trajectory.png")

    plot_comparison(stats_A, stats_B, "time", "Simulation Time (s)",
                    "Simulation Duration Comparison", "#9E9E9E", "#BDBDBD", "compare_time.png", unit="s²")

    print(f"\n📊 对比图已全部保存到: {SAVE_DIR}")
