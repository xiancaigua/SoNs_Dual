import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re

# ==========================================
# 🛠️ CONFIGURATION INTERFACE (配置接口)
# ==========================================
CONFIG = {
    # 1. Input Folders: Change these to your actual folder paths
    "method_folders": [
        "simulation_results_BASE1",  # Replace with path to Method 1 folder
        "./newbase1/simulation_results",  # Replace with path to Method 2 folder
        "./newbase2/simulation_results"   # Replace with path to Method 3 folder
    ],
    
    # 2. Method Labels: Names to appear in the legend
    "method_labels": [
        "Ours", 
        "ERRT", 
        "NBV"
    ],

    # 3. Scene Settings
    "files_per_scene": 100,  # How many files constitute one scene
    
    # 🆕 Scene Selection: List of scene IDs to include in analysis (1-based)
    # Example: [1, 2, 3, 5] to include only scenes 1, 2, 3, and 5.
    # Use list(range(1, 13)) for all 12 scenes.
    "scenes_to_include": list(range(1, 13)), 

    # 🆕 Single Method Analysis: Name of the method to analyze individually across scenes
    # Must match one of the names in "method_labels". Set to None to disable.
    "single_method_analysis": "Ours", 

    # 4. Plotting Settings
    "font_size": {
        "title": 16,
        "axis_label": 14,
        "tick_label": 12,
        "legend": 12
    },
    # 失败惩罚时间必须配置在主配置中，而不是 font_size
    "failure_penalty_time": 60.0,
    "figure_size": (12, 6),
    "bar_colors": ['#5da5da', '#faa43a', '#60bd68'], # Blue, Orange, Green
    "output_dir": "./analysis_results"
}

# ==========================================
# 🧠 DATA PROCESSING LOGIC
# ==========================================

def load_and_process_data(folder_path):
    """
    Loads JSON files from a folder, groups them by scene (100 files/scene),
    and calculates raw lists of metrics with failure penalty logic.
    Only includes scenes specified in CONFIG["scenes_to_include"].
    """
    data_by_scene = {} # {scene_id: {'simulation_duration': [], 'dead_agents': [], 'success_rate': [], 'explored_safe_count': []}}
    
    # Get all json files and sort them to ensure 1-100 are Scene 1
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return {}

    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    files.sort() # Critical: Ensure correct order based on timestamp/name

    for idx, filename in enumerate(files):
        # Calculate Scene ID (1-based)
        scene_id = (idx // CONFIG["files_per_scene"]) + 1
        
        # 🚨 Filter: Skip scenes not in the include list
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
                
                # 1. Determine Success First
                is_success = stats.get("victim_rescued", False)
                
                # 2. Apply Duration Logic (Modified Rule)
                if is_success:
                    # If success, use actual duration
                    duration = stats.get("simulation_duration", 0)
                else:
                    # If failed, use the standardized penalty time (e.g., 60)
                    duration = CONFIG.get("failure_penalty_time", 60.0)
                
                # Append processed duration
                data_by_scene[scene_id]["simulation_duration"].append(duration)
                
                # Append success rate (1.0 or 0.0)
                data_by_scene[scene_id]["success_rate"].append(
                    1.0 if is_success else 0.0
                )
                
                # Append dead agents
                data_by_scene[scene_id]["dead_agents"].append(
                    stats.get("dead_agents", 0)
                )

                # 3. Append new metric: Explored Safe Count
                data_by_scene[scene_id]["explored_safe_count"].append(
                    stats.get("explored_safe_count", 0)
                )
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return data_by_scene

def calculate_statistics(raw_data):
    """
    Calculates Mean and Standard Deviation for the raw data.
    """
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

# ==========================================
# 📊 PLOTTING LOGIC
# ==========================================

def plot_metric_comparison(all_methods_stats, metric_key, metric_title, y_label, with_variance=True):
    """
    Generates a grouped bar chart for a specific metric comparing all methods.
    """
    # Setup fonts
    plt.rcParams.update({
        'font.size': CONFIG["font_size"]["tick_label"],
        'axes.titlesize': CONFIG["font_size"]["title"],
        'axes.labelsize': CONFIG["font_size"]["axis_label"],
        'legend.fontsize': CONFIG["font_size"]["legend"]
    })

    # Identify all unique scenes present across methods (and filtered by config)
    all_scenes = set()
    for m_stats in all_methods_stats:
        all_scenes.update([sid for sid, metrics in m_stats.items() if metric_key in metrics])
        
    # Sort scenes to ensure order matches CONFIG["scenes_to_include"] if possible, or just numerical
    scene_ids = sorted(list(all_scenes))
    
    if not scene_ids:
        print(f"No data found for plotting {metric_key}")
        return

    x = np.arange(len(scene_ids))
    width = 0.8 / len(all_methods_stats)  # Dynamic bar width

    fig, ax = plt.subplots(figsize=CONFIG["figure_size"])

    for i, method_stats in enumerate(all_methods_stats):
        means = []
        stds = []
        for sid in scene_ids:
            if sid in method_stats and metric_key in method_stats[sid]:
                means.append(method_stats[sid][metric_key]["mean"])
                stds.append(method_stats[sid][metric_key]["std"])
            else:
                means.append(0)
                stds.append(0)
        
        # Calculate bar position
        bar_pos = x - (0.4) + (width * i) + (width / 2)
        
        # Error bars config
        yerr = stds if with_variance else None
        capsize = 5 if with_variance else 0
        
        ax.bar(
            bar_pos, 
            means, 
            width, 
            yerr=yerr, 
            label=CONFIG["method_labels"][i], 
            color=CONFIG["bar_colors"][i % len(CONFIG["bar_colors"])],
            capsize=capsize,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.7
        )

    # Chart formatting
    ax.set_xlabel("Experiment Scene ID")
    ax.set_ylabel(y_label)
    variance_text = "with Variance" if with_variance else "Mean Only"
    ax.set_title(f"{metric_title} per Scene ({variance_text})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Scene {i}" for i in scene_ids])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Save plot
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    suffix = "_with_var" if with_variance else "_no_var"
    key_clean = metric_key.replace('_', '-')
    filename = f"{key_clean}{suffix}.png"
    save_path = os.path.join(CONFIG["output_dir"], filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot: {save_path}")

def plot_single_method_analysis(all_methods_stats, target_method_name, metrics_config):
    """
    Generates detailed per-scene analysis plots for a single specified method.
    """
    if target_method_name is None:
        return

    # Find the index of the target method
    try:
        method_idx = CONFIG["method_labels"].index(target_method_name)
        method_stats = all_methods_stats[method_idx]
    except ValueError:
        print(f"Error: Method '{target_method_name}' not found in labels.")
        return
    except IndexError:
        print(f"Error: No data available for method '{target_method_name}'.")
        return

    print(f"\n--- Generating Single Method Analysis for '{target_method_name}' ---")

    # Ensure directory exists
    single_analysis_dir = os.path.join(CONFIG["output_dir"], f"single_analysis_{target_method_name}")
    os.makedirs(single_analysis_dir, exist_ok=True)

    scene_ids = sorted(list(method_stats.keys()))
    if not scene_ids:
        print(f"No data found for method {target_method_name}")
        return

    # Setup fonts
    plt.rcParams.update({
        'font.size': CONFIG["font_size"]["tick_label"],
        'axes.titlesize': CONFIG["font_size"]["title"],
        'axes.labelsize': CONFIG["font_size"]["axis_label"]
    })

    x = np.arange(len(scene_ids))

    for metric_key, metric_title, y_label in metrics_config:
        means = []
        stds = []
        
        for sid in scene_ids:
            if metric_key in method_stats[sid]:
                means.append(method_stats[sid][metric_key]["mean"])
                stds.append(method_stats[sid][metric_key]["std"])
            else:
                means.append(0)
                stds.append(0)

        fig, ax = plt.subplots(figsize=CONFIG["figure_size"])
        
        # Plot line with markers
        # Using error bars to show variance
        ax.errorbar(x, means, yerr=stds, fmt='-o', capsize=5, 
                    color='#1f77b4', ecolor='lightgray', elinewidth=2, markeredgewidth=2,
                    label=f"{target_method_name} (Mean ± Std)")

        # Annotate values
        for i, val in enumerate(means):
            ax.annotate(f"{val:.2f}", (x[i], val), textcoords="offset points", xytext=(0,10), ha='center')

        ax.set_xlabel("Experiment Scene ID")
        ax.set_ylabel(y_label)
        ax.set_title(f"{target_method_name}: {metric_title} Analysis across Scenes")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Scene {i}" for i in scene_ids])
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.grid(axis='x', linestyle=':', alpha=0.3)

        # Save plot
        key_clean = metric_key.replace('_', '-')
        filename = f"{target_method_name}_{key_clean}_analysis.png"
        save_path = os.path.join(single_analysis_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved single analysis plot: {save_path}")

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================

def main():
    print("--- Starting Analysis ---")
    print(f"Scenes to analyze: {CONFIG['scenes_to_include']}")
    
    all_methods_processed_data = []
    
    # 1. Process each method's folder
    for folder_path in CONFIG["method_folders"]:
        print(f"Processing folder: {folder_path}...")
        raw_data = load_and_process_data(folder_path)
        stats = calculate_statistics(raw_data)
        all_methods_processed_data.append(stats)

    # 2. Define Metrics Configuration
    metrics_config = [
        ("simulation_duration", "Simulation Duration (Time Penalty)", "Time (s)"),
        ("dead_agents", "Agent Mortality", "Count"),
        ("success_rate", "Success Rate", "Rate (0.0 - 1.0)"),
        ("explored_safe_count", "Explored Safe Area Count", "Count") 
    ]

    # 3. Generate Comparison Plots (Multi-method)
    print("\n--- Generating Comparison Plots ---")
    for key, title, ylabel in metrics_config:
        # Version 1: With Variance
        plot_metric_comparison(all_methods_processed_data, key, title, ylabel, with_variance=True)
        # Version 2: Without Variance
        plot_metric_comparison(all_methods_processed_data, key, title, ylabel, with_variance=False)

    # 4. Generate Single Method Analysis (if configured)
    if CONFIG["single_method_analysis"]:
        plot_single_method_analysis(all_methods_processed_data, CONFIG["single_method_analysis"], metrics_config)

    # 5. Export Summary JSON
    print("\n--- Exporting Summary JSON ---")
    summary_export = {}
    for i, label in enumerate(CONFIG["method_labels"]):
        if i < len(all_methods_processed_data):
            summary_export[label] = all_methods_processed_data[i]
            
    json_path = os.path.join(CONFIG["output_dir"], "experiment_summary.json")
    
    # Helper to convert numpy types for JSON serialization
    def np_converter(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        return obj

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_export, f, indent=4, default=np_converter)
    
    print(f"Summary saved to: {json_path}")
    print("--- Done ---")

if __name__ == "__main__":
    main()