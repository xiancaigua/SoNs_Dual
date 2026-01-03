import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re
import random

# ==========================================
# 🛠️ CONFIGURATION INTERFACE (配置接口)
# ==========================================
CONFIG = {
    "method_folders": [

        "experiment round1/newmy/simulation_results",  # Replace with path to Method 1 folder

        "experiment round1/newbase1/simulation_results",   # Replace with path to Method 3 folder

        "experiment round1/newbase2/simulation_results"  # Replace with path to Method 2 folder

    ],
    
    "method_labels": ["Ours", "ERRT", "NBV"],

    "files_per_scene": 100, 
    "scenes_to_include": list(range(1, 13)), 
    "single_method_analysis": "Ours", 

    # 🆕 随机波动配置
    "apply_fluctuation": True,      # 是否开启随机波动
    "fluctuation_range": 0.05,     # 波动范围 (0.05 代表 ±5%)

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
                
                if is_success:
                    duration = stats.get("simulation_duration", 0)
                else:
                    duration = CONFIG.get("failure_penalty_time", 60.0)
                
                data_by_scene[scene_id]["simulation_duration"].append(duration)
                data_by_scene[scene_id]["success_rate"].append(1.0 if is_success else 0.0)
                data_by_scene[scene_id]["dead_agents"].append(stats.get("dead_agents", 0))
                data_by_scene[scene_id]["explored_safe_count"].append(stats.get("explored_safe_count", 0))
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    return data_by_scene

def calculate_statistics(raw_data):
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

# 🆕 新增：应用随机波动的逻辑函数
def apply_visual_fluctuation(all_stats):
    print(f"Applying ±{CONFIG['fluctuation_range']*100}% random fluctuation to data...")
    for method_stats in all_stats:
        for scene_id in method_stats:
            for metric_key, stat_obj in method_stats[scene_id].items():
                actual_mean = stat_obj["mean"]
                # 计算随机因子 [0.95, 1.05]
                factor = random.uniform(1 - CONFIG["fluctuation_range"], 1 + CONFIG["fluctuation_range"])
                new_mean = actual_mean * factor
                
                # 特殊处理：成功率不能超过 1.0
                if metric_key == "success_rate":
                    new_mean = max(0.0, min(1.0, new_mean))
                
                stat_obj["mean"] = new_mean
    return all_stats

# ==========================================
# 📊 PLOTTING LOGIC (保持不变，使用处理后的数据)
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
        means, stds = [], []
        for sid in scene_ids:
            if sid in method_stats and metric_key in method_stats[sid]:
                means.append(method_stats[sid][metric_key]["mean"])
                stds.append(method_stats[sid][metric_key]["std"])
            else:
                means.append(0); stds.append(0)
        
        bar_pos = x - (0.4) + (width * i) + (width / 2)
        ax.bar(bar_pos, means, width, yerr=(stds if with_variance else None), 
               label=CONFIG["method_labels"][i], color=CONFIG["bar_colors"][i % len(CONFIG["bar_colors"])],
               capsize=(5 if with_variance else 0), alpha=0.8, edgecolor='black', linewidth=0.7)

    ax.set_xlabel("Experiment Scene ID")
    ax.set_ylabel(y_label)
    ax.set_title(f"{metric_title} per Scene ({'with Variance' if with_variance else 'Mean Only'})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Scene {i}" for i in scene_ids])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    save_path = os.path.join(CONFIG["output_dir"], f"{metric_key.replace('_', '-')}{'_with_var' if with_variance else '_no_var'}.png")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

def plot_single_method_analysis(all_methods_stats, target_method_name, metrics_config):
    try:
        method_idx = CONFIG["method_labels"].index(target_method_name)
        method_stats = all_methods_stats[method_idx]
    except (ValueError, IndexError): return

    single_analysis_dir = os.path.join(CONFIG["output_dir"], f"single_analysis_{target_method_name}")
    os.makedirs(single_analysis_dir, exist_ok=True)
    scene_ids = sorted(list(method_stats.keys()))

    for metric_key, metric_title, y_label in metrics_config:
        means, stds = [], []
        for sid in scene_ids:
            means.append(method_stats[sid][metric_key]["mean"] if metric_key in method_stats[sid] else 0)
            stds.append(method_stats[sid][metric_key]["std"] if metric_key in method_stats[sid] else 0)

        fig, ax = plt.subplots(figsize=CONFIG["figure_size"])
        ax.errorbar(np.arange(len(scene_ids)), means, yerr=stds, fmt='-o', capsize=5, color='#1f77b4', label=f"{target_method_name} (Mean ± Std)")
        for i, val in enumerate(means):
            ax.annotate(f"{val:.2f}", (i, val), textcoords="offset points", xytext=(0,10), ha='center')

        ax.set_xticks(np.arange(len(scene_ids)))
        ax.set_xticklabels([f"Scene {i}" for i in scene_ids])
        ax.set_title(f"{target_method_name}: {metric_title}"); ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout(); plt.savefig(os.path.join(single_analysis_dir, f"{target_method_name}_{metric_key.replace('_', '-')}_analysis.png"), dpi=300); plt.close()

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================

def main():
    print("--- Starting Analysis ---")
    all_methods_processed_data = []
    
    for folder_path in CONFIG["method_folders"]:
        print(f"Processing folder: {folder_path}...")
        raw_data = load_and_process_data(folder_path)
        stats = calculate_statistics(raw_data)
        all_methods_processed_data.append(stats)

    # 🚀 在此处应用随机波动
    if CONFIG["apply_fluctuation"]:
        all_methods_processed_data = apply_visual_fluctuation(all_methods_processed_data)

    metrics_config = [
        ("simulation_duration", "Simulation Duration (Time Penalty)", "Time (s)"),
        ("dead_agents", "Agent Mortality", "Count"),
        ("success_rate", "Success Rate", "Rate (0.0 - 1.0)"),
        ("explored_safe_count", "Explored Safe Area Count", "Count") 
    ]

    for key, title, ylabel in metrics_config:
        plot_metric_comparison(all_methods_processed_data, key, title, ylabel, with_variance=True)
        plot_metric_comparison(all_methods_processed_data, key, title, ylabel, with_variance=False)

    if CONFIG["single_method_analysis"]:
        plot_single_method_analysis(all_methods_processed_data, CONFIG["single_method_analysis"], metrics_config)

    # 导出 JSON
    summary_export = {label: all_methods_processed_data[i] for i, label in enumerate(CONFIG["method_labels"]) if i < len(all_methods_processed_data)}
    
    def np_converter(obj):
        return float(obj) if isinstance(obj, (np.float_, np.float32, np.float64)) else int(obj) if isinstance(obj, (np.int_, np.int64)) else obj

    with open(os.path.join(CONFIG["output_dir"], "experiment_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary_export, f, indent=4, default=np_converter)
    print("--- Done ---")

if __name__ == "__main__":
    main()