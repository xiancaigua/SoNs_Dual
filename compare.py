import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 🛠️ CONFIGURATION INTERFACE (配置接口)
# ==========================================
CONFIG = {
    # 1. Input Folders: Change these to your actual folder paths
    "method_folders": [
        "./myresult/simulation_results",  # Replace with path to Method 1 folder
        "./base1/simulation_results",  # Replace with path to Method 2 folder
        "./base2/simulation_results"   # Replace with path to Method 3 folder
    ],
    
    # 2. Method Labels: Names to appear in the legend
    "method_labels": [
        "Ours", 
        "ERRT", 
        "NBV"
    ],

    # 3. Scene Settings
    "files_per_scene": 100,  # How many files constitute one scene

    # 4. Plotting Settings
    "font_size": {
        "title": 16,
        "failure_penalty_time": 60.0,
        "axis_label": 14,
        "tick_label": 12,
        "legend": 12
    },
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
    """
    data_by_scene = {} # {scene_id: {'duration': [], 'dead': [], 'success': []}}
    
    # Get all json files and sort them to ensure 1-100 are Scene 1
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    files.sort() # Critical: Ensure correct order based on timestamp/name

    for idx, filename in enumerate(files):
        # Calculate Scene ID (1-based)
        scene_id = (idx // CONFIG["files_per_scene"]) + 1
        
        if scene_id not in data_by_scene:
            data_by_scene[scene_id] = {
                "simulation_duration": [],
                "dead_agents": [],
                "success_rate": [] 
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
    Generates a grouped bar chart for a specific metric.
    """
    # Setup fonts
    plt.rcParams.update({
        'font.size': CONFIG["font_size"]["tick_label"],
        'axes.titlesize': CONFIG["font_size"]["title"],
        'axes.labelsize': CONFIG["font_size"]["axis_label"],
        'legend.fontsize': CONFIG["font_size"]["legend"]
    })

    # Identify all unique scenes present across methods
    all_scenes = set()
    for m_stats in all_methods_stats:
        all_scenes.update(m_stats.keys())
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
            # Get data for this scene, default to 0 if missing
            if sid in method_stats:
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
    filename = f"{metric_key}{suffix}.png"
    save_path = os.path.join(CONFIG["output_dir"], filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot: {save_path}")

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================

def main():
    print("--- Starting Analysis ---")
    
    all_methods_processed_data = []
    
    # 1. Process each method's folder
    for folder_path in CONFIG["method_folders"]:
        if not os.path.exists(folder_path):
            print(f"WARNING: Folder not found: {folder_path}")
            all_methods_processed_data.append({}) # Empty placeholder
            continue
            
        print(f"Processing folder: {folder_path}...")
        raw_data = load_and_process_data(folder_path)
        stats = calculate_statistics(raw_data)
        all_methods_processed_data.append(stats)

    # 2. Generate Plots (Both versions)
    metrics_config = [
        ("simulation_duration", "Simulation Duration", "Time (s)"),
        ("dead_agents", "Agent Mortality", "Count"),
        ("success_rate", "Success Rate", "Rate (0.0 - 1.0)")
    ]

    print("--- Generating Plots ---")
    for key, title, ylabel in metrics_config:
        # Version 1: With Variance
        plot_metric_comparison(all_methods_processed_data, key, title, ylabel, with_variance=True)
        # Version 2: Without Variance
        plot_metric_comparison(all_methods_processed_data, key, title, ylabel, with_variance=False)

    # 3. Export Summary JSON
    print("--- Exporting Summary JSON ---")
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