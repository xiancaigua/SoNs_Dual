import json
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import re

# =======================================================
# 📌 配置区
# =======================================================
# 实验文件所在的目录 (请根据实际情况修改)
DATA_DIRECTORY = "newmy/simulation_results" 
TOTAL_EXPERIMENTS = 1200
EXPERIMENTS_PER_SCENE = 100
SCENE_COUNT = TOTAL_EXPERIMENTS // EXPERIMENTS_PER_SCENE

# 失败惩罚时间 (如果任务失败，仿真时间记为此值)
FAILURE_PENALTY_TIME = 60.0

# 绘图配置
PLOT_CONFIG = {
    "font_size": 12,
    "bar_color": 'skyblue',
    "error_color": 'dimgray',
    "error_capsize": 5,
    "alpha": 0.7
}

# =======================================================
# ⚙️ 核心数据处理函数
# =======================================================

def calculate_single_experiment_metrics(data):
    """计算单个实验的关键指标."""
    stats = data['statistics']
    details = data['agent_details']
    
    # 1. 成功率
    is_success = stats.get('victim_rescued', False)
    success_rate = 1.0 if is_success else 0.0

    # 2. 仿真时间 (应用失败惩罚)
    raw_duration = stats.get('simulation_duration', 0)
    sim_duration = raw_duration if is_success else FAILURE_PENALTY_TIME

    # 3. 轨迹长度总和
    total_trajectory_length = sum(
        a['trajectory_length'] for a in details['small_agents'] if 'trajectory_length' in a
    ) + sum(
        a['trajectory_length'] for a in details['large_agents'] if 'trajectory_length' in a
    )

    # 4. 能量消耗总和
    total_energy_cost = sum(
        a['energycost'] for a in details['small_agents'] if 'energycost' in a
    ) + sum(
        a['energycost'] for a in details['large_agents'] if 'energycost' in a
    )
    
    # 5. 个体能量消耗列表
    small_agent_costs = [a['energycost'] for a in details['small_agents'] if 'energycost' in a]
    large_agent_costs = [a['energycost'] for a in details['large_agents'] if 'energycost' in a]

    # 6. 🆕 新增指标：已探索的安全区域数量
    explored_safe_count = stats.get('explored_safe_count', 0)

    return {
        'simulation_duration': sim_duration, # 使用处理后的时间
        'success_rate': success_rate,
        'dead_agents': stats.get('dead_agents', 0),
        'coverage_percentage': stats.get('coverage_percentage', 0),
        'explored_safe_count': explored_safe_count, # 🆕
        'total_trajectory_length': total_trajectory_length,
        'total_energy_cost': total_energy_cost,
        'small_agent_costs': small_agent_costs,
        'large_agent_costs': large_agent_costs
    }

def aggregate_statistics():
    """
    对所有实验数据进行分组和统计计算。
    """
    
    # --- 第一步：获取文件列表并排序 ---
    if not os.path.exists(DATA_DIRECTORY):
        print(f"错误: 找不到目录 {DATA_DIRECTORY}")
        return {}, {}

    all_files = []
    for filename in os.listdir(DATA_DIRECTORY):
        if filename.endswith('.json') and not filename.startswith('analysis_report'):
            all_files.append(filename)

    # 🚨 关键：按文件名排序
    all_files.sort()
    
    if len(all_files) < TOTAL_EXPERIMENTS:
        print(f"警告: 目录下只找到 {len(all_files)} 个 JSON 文件，但预期为 {TOTAL_EXPERIMENTS} 个。将使用找到的文件进行分析。")
        
    files_to_process = all_files[:TOTAL_EXPERIMENTS]
    
    # --- 第二步：分组和计算 ---
    scene_results = defaultdict(lambda: defaultdict(list))
    
    for i, file_name in enumerate(files_to_process):
        scene_id = i // EXPERIMENTS_PER_SCENE + 1
        
        full_path = os.path.join(DATA_DIRECTORY, file_name)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"警告: 文件 {file_name} JSON 解析失败，跳过。")
            continue
        except Exception as e:
            print(f"警告: 读取文件 {file_name} 时发生错误: {e}")
            continue

        metrics = calculate_single_experiment_metrics(data)

        # 记录每项指标
        scene_results[scene_id]['simulation_duration'].append(metrics['simulation_duration'])
        scene_results[scene_id]['success_rate'].append(metrics['success_rate'])
        scene_results[scene_id]['dead_agents'].append(metrics['dead_agents'])
        scene_results[scene_id]['coverage_percentage'].append(metrics['coverage_percentage'])
        # 🆕 记录新指标
        scene_results[scene_id]['explored_safe_count'].append(metrics['explored_safe_count'])
        
        scene_results[scene_id]['total_trajectory_length'].append(metrics['total_trajectory_length'])
        scene_results[scene_id]['total_energy_cost'].append(metrics['total_energy_cost'])
        scene_results[scene_id]['small_agent_costs_flat'].extend(metrics['small_agent_costs'])
        scene_results[scene_id]['large_agent_costs_flat'].extend(metrics['large_agent_costs'])
        
        # 记录每次实验的机器人个体消耗总和
        scene_results[scene_id]['exp_small_cost_avg'].append(np.mean(metrics['small_agent_costs']) if metrics['small_agent_costs'] else 0)
        scene_results[scene_id]['exp_large_cost_avg'].append(np.mean(metrics['large_agent_costs']) if metrics['large_agent_costs'] else 0)

    # --- 第三步：最终汇总计算 ---
    final_global_summary = {}
    final_scene_summary = {}
    
    all_metrics = defaultdict(list)
    
    # 场景 ID 列表
    actual_scene_ids = sorted(scene_results.keys())

    for scene_id in actual_scene_ids:
        results = scene_results[scene_id]
        scene_summary = {}
        
        # 1. 核心指标 (平均值，用于全局和场景汇总)
        core_metrics = {
            'simulation_duration': np.array(results['simulation_duration']),
            'success_rate': np.array(results['success_rate']),
            'dead_agents': np.array(results['dead_agents']),
            'coverage_percentage': np.array(results['coverage_percentage']),
            'explored_safe_count': np.array(results['explored_safe_count']), # 🆕 加入核心计算
            'total_trajectory_length': np.array(results['total_trajectory_length']),
            'total_energy_cost': np.array(results['total_energy_cost']),
        }

        # 场景汇总 (包含平均值和标准差)
        for name, values in core_metrics.items():
            if values.size == 0: continue
            scene_summary[f'avg_{name}'] = np.mean(values)
            scene_summary[f'std_{name}'] = np.std(values)
            all_metrics[name].extend(values) # 收集所有场景数据用于全局计算
            
        # 2. 机器人个体能耗详细分析 (保持不变)
        small_costs = np.array(results['exp_small_cost_avg'])
        scene_summary['small_cost_avg_of_exp_avg'] = np.mean(small_costs) if small_costs.size > 0 else 0
        scene_summary['small_cost_std_of_exp_avg'] = np.std(small_costs) if small_costs.size > 0 else 0
        
        flat_small_costs = np.array(results['small_agent_costs_flat'])
        if flat_small_costs.size > 0:
            scene_summary['small_cost_min'] = np.min(flat_small_costs)
            scene_summary['small_cost_max'] = np.max(flat_small_costs)
            scene_summary['small_cost_variance'] = np.var(flat_small_costs)
        else:
            scene_summary['small_cost_min'] = scene_summary['small_cost_max'] = scene_summary['small_cost_variance'] = 0

        large_costs = np.array(results['exp_large_cost_avg'])
        scene_summary['large_cost_avg_of_exp_avg'] = np.mean(large_costs) if large_costs.size > 0 else 0
        scene_summary['large_cost_std_of_exp_avg'] = np.std(large_costs) if large_costs.size > 0 else 0
        
        flat_large_costs = np.array(results['large_agent_costs_flat'])
        if flat_large_costs.size > 0:
            scene_summary['large_cost_min'] = np.min(flat_large_costs)
            scene_summary['large_cost_max'] = np.max(flat_large_costs)
            scene_summary['large_cost_variance'] = np.var(flat_large_costs)
        else:
             scene_summary['large_cost_min'] = scene_summary['large_cost_max'] = scene_summary['large_cost_variance'] = 0
        
        final_scene_summary[f'scene_{scene_id}'] = scene_summary

    # 3. 全局汇总
    for name, values in all_metrics.items():
        values_array = np.array(values)
        if values_array.size > 0:
            final_global_summary[f'global_avg_{name}'] = np.mean(values_array)
            final_global_summary[f'global_std_{name}'] = np.std(values_array)
        else:
            final_global_summary[f'global_avg_{name}'] = 0
            final_global_summary[f'global_std_{name}'] = 0
            
    return final_global_summary, final_scene_summary

def save_json_report(data, filename):
    """保存JSON报告到文件."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    print(f"✅ 报告已保存: {filename}")

# =======================================================
# 📈 可视化函数 (plot_scene_comparisons)
# =======================================================

def plot_scene_comparisons(scene_summary, output_dir="analysis_plots"):
    """
    Generates comparison charts for metrics across different scenes (With Error Bars).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    scene_ids_str = sorted(scene_summary.keys())
    if not scene_ids_str:
        print("No scenes found for plotting.")
        return
    
    scene_ids_int = [int(s.split('_')[-1]) for s in scene_ids_str]
    
    # 🆕 在这里添加了 'explored_safe_count'
    metrics_to_plot = {
        'simulation_duration': 'Avg. Simulation Duration (s)',
        'success_rate': 'Success Rate',
        'dead_agents': 'Avg. Robot Deaths',
        'coverage_percentage': 'Avg. Map Coverage (%)',
        'explored_safe_count': 'Avg. Explored Safe Areas (Count)', # 新增
        'total_trajectory_length': 'Avg. Total Trajectory Length',
        'total_energy_cost': 'Avg. Total Energy Consumption',
        'small_cost_avg_of_exp_avg': 'Avg. Small Agent Energy Cost',
        'large_cost_avg_of_exp_avg': 'Avg. Large Agent Energy Cost',
    }
    
    for metric, title in metrics_to_plot.items():
        if metric.startswith(('small', 'large')):
            avg_key = metric
            std_key = metric.replace('avg', 'std')
        else:
            avg_key = f'avg_{metric}'
            std_key = f'std_{metric}'

        averages = [scene_summary[s].get(avg_key, 0.0) for s in scene_ids_str]
        stds = [scene_summary[s].get(std_key, 0.0) for s in scene_ids_str]
        
        stds_cleaned = np.nan_to_num(stds, nan=0.0, posinf=0.0, neginf=0.0)
        
        if not any(averages):
             print(f"Skipping plot for '{metric}': All average values are zero.")
             continue
        
        plt.rcParams.update({'font.size': PLOT_CONFIG["font_size"]})

        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(scene_ids_int, averages, 
               yerr=stds_cleaned, 
               capsize=PLOT_CONFIG["error_capsize"], 
               color=PLOT_CONFIG["bar_color"], 
               alpha=PLOT_CONFIG["alpha"], 
               edgecolor='black', 
               error_kw={'ecolor': PLOT_CONFIG["error_color"], 'linewidth': 1}) 

        ax.set_xlabel("Scene ID")
        ax.set_ylabel(title)
        ax.set_title(f"Comparison of {title} Across Scenes")
        ax.set_xticks(scene_ids_int)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        max_avg = max(averages) if averages else 1.0
        
        for i, avg in enumerate(averages):
            std_val = stds_cleaned[i]
            offset = max_avg * 0.015
            label_pos = avg + std_val + offset
            ax.text(scene_ids_int[i], label_pos, 
                    f'{avg:.2f} ± {std_val:.2f}', 
                    ha='center', va='bottom', fontsize=PLOT_CONFIG["font_size"] - 2)

        plt.tight_layout()
        safe_metric_name = re.sub(r'[^\w\-_\. ]', '_', metric)
        plot_filename = os.path.join(output_dir, f'{safe_metric_name}_comparison.png')
        plt.savefig(plot_filename)
        plt.close(fig)
        print(f"✅ Plot saved: {plot_filename}")


def plot_scene_comparisons_no_error(scene_summary, output_dir="analysis_plots"):
    """
    Generates comparison bar charts WITHOUT error bars.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    scene_ids_str = sorted(scene_summary.keys())
    if not scene_ids_str:
        return
    
    scene_ids_int = [int(s.split('_')[-1]) for s in scene_ids_str]
    
    # 🆕 同样添加 'explored_safe_count'
    metrics_to_plot = {
        'simulation_duration': 'Avg. Simulation Duration (s)',
        'success_rate': 'Success Rate',
        'dead_agents': 'Avg. Robot Deaths',
        'coverage_percentage': 'Avg. Map Coverage (%)',
        'explored_safe_count': 'Avg. Explored Safe Areas (Count)', # 新增
        'total_trajectory_length': 'Avg. Total Trajectory Length',
        'total_energy_cost': 'Avg. Total Energy Consumption',
        'small_cost_avg_of_exp_avg': 'Avg. Small Agent Energy Cost',
        'large_cost_avg_of_exp_avg': 'Avg. Large Agent Energy Cost',
    }
    
    for metric, title in metrics_to_plot.items():
        if metric.startswith(('small', 'large')):
            avg_key = metric
        else:
            avg_key = f'avg_{metric}'

        averages = [scene_summary[s].get(avg_key, 0.0) for s in scene_ids_str]
        
        if not any(averages):
             continue
        
        plt.rcParams.update({'font.size': PLOT_CONFIG["font_size"]})

        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(scene_ids_int, averages, 
               color=PLOT_CONFIG["bar_color"], 
               alpha=PLOT_CONFIG["alpha"], 
               edgecolor='black') 

        ax.set_xlabel("Scene ID")
        ax.set_ylabel(title)
        ax.set_title(f"Comparison of {title} Across Scenes (No Error Bar)") 
        ax.set_xticks(scene_ids_int)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        max_avg = max(averages) if averages else 1.0 
        
        for i, avg in enumerate(averages):
            offset = max_avg * 0.015
            label_pos = avg + offset
            ax.text(scene_ids_int[i], label_pos, 
                    f'{avg:.2f}', 
                    ha='center', va='bottom', fontsize=PLOT_CONFIG["font_size"] - 2)

        plt.tight_layout()
        safe_metric_name = re.sub(r'[^\w\-_\. ]', '_', metric)
        plot_filename = os.path.join(output_dir, f'{safe_metric_name}_no_error_comparison.png')
        plt.savefig(plot_filename)
        plt.close(fig)
        print(f"✅ Plot saved (No Error Bar): {plot_filename}")

if __name__ == "__main__":
    print(f"--- 启动实验结果分析 (预期 {TOTAL_EXPERIMENTS} 次实验, 每 {EXPERIMENTS_PER_SCENE} 次为一组) ---")
    
    # 1. 聚合统计数据
    global_summary, scene_summary = aggregate_statistics() 
    
    if not scene_summary:
        print("致命错误: 未能成功处理任何实验文件。请检查 DATA_DIRECTORY 是否正确。")
    else:
        # 2. 输出 (1) 全局汇总 JSON
        print("\n--- 正在生成全局汇总报告 ---")
        save_json_report(global_summary, "analysis_report_global_summary.json")

        # 3. 输出 (2) 场景详细 JSON 报告
        print("\n--- 正在生成场景详细报告 ---")
        save_json_report(scene_summary, "analysis_report_scene_details.json")

        # 4. 生成可视化图表
        print("\n--- 正在生成可视化图表 ---")
        plot_scene_comparisons(scene_summary)
        plot_scene_comparisons_no_error(scene_summary)
        
        print("\n--- 分析完成 ---")