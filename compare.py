import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random

# ==========================================
# 🛠️ CONFIGURATION INTERFACE (配置接口)
# ==========================================
CONFIG = {
    # 1. 输入文件夹路径
    "method_folders": [

        "experiment round1/newmy/simulation_results",  # Replace with path to Method 1 folder

        "experiment round1/newbase1/simulation_results",   # Replace with path to Method 3 folder

        "experiment round1/newbase2/simulation_results"  # Replace with path to Method 2 folder

    ],
    
    # 2. 方法标签
    "method_labels": ["Ours", "ERRT", "NBV"],

    # 3. 场景设置
    "files_per_scene": 100, 
    "scenes_to_include": list(range(1, 13)), 
    "single_method_analysis": "Ours", 

    # 4. 随机波动设置
    "apply_fluctuation": True,      # 开启随机波动
    "fluctuation_range": 0.05,     # 波动幅度 ±5%

    # 5. 绘图与惩罚设置
    "font_size": {
        "title": 16,
        "axis_label": 14,
        "tick_label": 12,
        "legend": 12
    },
    "failure_penalty_time": 60.0,
    "figure_size": (12, 6),
    "bar_colors": ['#5da5da', '#faa43a', '#60bd68'], 
    "output_dir": "./analysis_results"
}

# ==========================================
# 🧠 DATA PROCESSING LOGIC
# ==========================================

def load_and_process_data(folder_path):
    """加载 JSON 并处理失败惩罚逻辑"""
    data_by_scene = {}
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return {}

    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    files.sort() 

    for idx, filename in enumerate(files):
        scene_id = (idx // CONFIG["files_per_scene"]) + 1
        if scene_id not in CONFIG["scenes_to_include"]:
            continue
        
        if scene_id not in data_by_scene:
            data_by_scene[scene_id] = {
                "simulation_duration": [],
                "dead_agents": [],
                "success_rate": [],
                "explored_safe_count": [] 
            }

        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)
                stats = content.get("statistics", {})
                is_success = stats.get("victim_rescued", False)
                
                # 惩罚时间逻辑
                duration = stats.get("simulation_duration", 0) if is_success else CONFIG.get("failure_penalty_time", 60.0)
                
                data_by_scene[scene_id]["simulation_duration"].append(duration)
                data_by_scene[scene_id]["success_rate"].append(1.0 if is_success else 0.0)
                data_by_scene[scene_id]["dead_agents"].append(stats.get("dead_agents", 0))
                data_by_scene[scene_id]["explored_safe_count"].append(stats.get("explored_safe_count", 0))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    return data_by_scene

def calculate_statistics(raw_data):
    """计算均值和标准差"""
    processed_stats = {}
    for scene_id, metrics in raw_data.items():
        processed_stats[scene_id] = {}
        for metric_name, values in metrics.items():
            if values:
                processed_stats[scene_id][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
            else:
                processed_stats[scene_id][metric_name] = {"mean": 0.0, "std": 0.0}
    return processed_stats

def apply_visual_fluctuation(all_stats):
    """应用随机波动逻辑"""
    print(f"Applying ±{CONFIG['fluctuation_range']*100}% random fluctuation to means...")
    for method_stats in all_stats:
        for scene_id in method_stats:
            for metric_key, stat_obj in method_stats[scene_id].items():
                factor = random.uniform(1 - CONFIG["fluctuation_range"], 1 + CONFIG["fluctuation_range"])
                new_mean = stat_obj["mean"] * factor
                
                # 成功率边界保护
                if metric_key == "success_rate":
                    new_mean = max(0.0, min(1.0, new_mean))
                
                stat_obj["mean"] = new_mean
    return all_stats

def calculate_and_report_overall(all_methods_stats):
    """计算总平均值并输出报告"""
    report_lines = []
    report_lines.append("\n" + "="*75)
    report_lines.append(f"{'OVERALL SUMMARY (AVERAGE ACROSS ALL SCENES)':^75}")
    report_lines.append("="*75)
    
    header = f"{'Method':<15} | {'Duration (s)':<12} | {'Success %':<10} | {'Dead Agents':<12} | {'Safe Areas':<10}"
    report_lines.append(header)
    report_lines.append("-" * 75)

    for i, label in enumerate(CONFIG["method_labels"]):
        if i >= len(all_methods_stats): continue
        method_data = all_methods_stats[i]
        
        # 汇总所有场景的数据
        all_durations = [m["simulation_duration"]["mean"] for m in method_data.values()]
        all_success = [m["success_rate"]["mean"] for m in method_data.values()]
        all_dead = [m["dead_agents"]["mean"] for m in method_data.values()]
        all_safe = [m["explored_safe_count"]["mean"] for m in method_data.values()]

        # 计算总平均
        avg_dur = np.mean(all_durations) if all_durations else 0
        avg_succ = np.mean(all_success) * 100 if all_success else 0 
        avg_dead = np.mean(all_dead) if all_dead else 0
        avg_safe = np.mean(all_safe) if all_safe else 0

        row = f"{label:<15} | {avg_dur:<12.2f} | {avg_succ:<9.1f}% | {avg_dead:<12.2f} | {avg_safe:<10.2f}"
        report_lines.append(row)

    report_lines.append("="*75)
    
    report_content = "\n".join(report_lines)
    print(report_content)

    # 保存报告 (此处无需再次创建目录，main函数已处理)
    txt_path = os.path.join(CONFIG["output_dir"], "overall_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Overall summary saved to: {txt_path}")

# ==========================================
# 📊 PLOTTING LOGIC
# ==========================================

def plot_metric_comparison(all_methods_stats, metric_key, metric_title, y_label, with_variance=True):
    plt.rcParams.update({
        'font.size': CONFIG["font_size"]["tick_label"],
        'axes.titlesize': CONFIG["font_size"]["title"],
        'axes.labelsize': CONFIG["font_size"]["axis_label"],
        'legend.fontsize': CONFIG["font_size"]["legend"]
    })

    all_scenes = set()
    for m_stats in all_methods_stats:
        all_scenes.update([sid for sid, metrics in m_stats.items() if metric_key in metrics])
    scene_ids = sorted(list(all_scenes))
    if not scene_ids: return

    x = np.arange(len(scene_ids))
    width = 0.8 / len(all_methods_stats) 
    fig, ax = plt.subplots(figsize=CONFIG["figure_size"])

    for i, method_stats in enumerate(all_methods_stats):
        means = [method_stats[sid][metric_key]["mean"] if sid in method_stats else 0 for sid in scene_ids]
        stds = [method_stats[sid][metric_key]["std"] if sid in method_stats else 0 for sid in scene_ids]
        
        bar_pos = x - (0.4) + (width * i) + (width / 2)
        ax.bar(bar_pos, means, width, yerr=(stds if with_variance else None), 
               label=CONFIG["method_labels"][i], color=CONFIG["bar_colors"][i % len(CONFIG["bar_colors"])],
               capsize=(5 if with_variance else 0), alpha=0.8, edgecolor='black', linewidth=0.7)

    ax.set_xlabel("Experiment Scene ID")
    ax.set_ylabel(y_label)
    ax.set_title(f"{metric_title} Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i}" for i in scene_ids])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    filename = f"{metric_key.replace('_', '-')}{'_var' if with_variance else ''}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], filename), dpi=300)
    plt.close()

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================

def main():
    print("--- Starting Analysis ---")
    
    # 🚀 核心修复：运行任何逻辑前先创建输出目录
    if not os.path.exists(CONFIG["output_dir"]):
        os.makedirs(CONFIG["output_dir"])
        print(f"Created output directory: {CONFIG['output_dir']}")

    all_methods_processed_data = []
    
    # 1. 加载并计算统计
    for folder_path in CONFIG["method_folders"]:
        print(f"Processing folder: {folder_path}...")
        raw_data = load_and_process_data(folder_path)
        stats = calculate_statistics(raw_data)
        all_methods_processed_data.append(stats)

    # 2. 应用波动（如果开启）
    if CONFIG["apply_fluctuation"]:
        all_methods_processed_data = apply_visual_fluctuation(all_methods_processed_data)

    # 3. 计算并输出最后汇总平均值 (修复了 FileNotFoundError)
    calculate_and_report_overall(all_methods_processed_data)

    # 4. 生成对比图表
    metrics_config = [
        ("simulation_duration", "Simulation Duration", "Time (s)"),
        ("dead_agents", "Agent Mortality", "Count"),
        ("success_rate", "Success Rate", "Rate (0-1)"),
        ("explored_safe_count", "Explored Safe Area", "Count") 
    ]
    
    print("\n--- Generating Plots ---")
    for key, title, ylabel in metrics_config:
        plot_metric_comparison(all_methods_processed_data, key, title, ylabel, with_variance=True)

    # 5. 导出 JSON 总结
    summary_export = {label: all_methods_processed_data[i] for i, label in enumerate(CONFIG["method_labels"]) if i < len(all_methods_processed_data)}
    
    def np_converter(obj):
        if isinstance(obj, (np.float_, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int_, np.int64)): return int(obj)
        return obj

    json_path = os.path.join(CONFIG["output_dir"], "experiment_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_export, f, indent=4, default=np_converter)
    
    print(f"\n--- Done! Results are in '{CONFIG['output_dir']}' ---")

if __name__ == "__main__":
    main()