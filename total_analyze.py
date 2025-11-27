import json
import os
from collections import defaultdict
import sys # 用于更友好的错误输出

# =======================================================
# 📌 配置区 (CONFIGURATION)
# =======================================================
CONFIG = {
    # 实验结果汇总 JSON 文件的路径
    "json_file_path": "analysis_results1\experiment_summary.json",

    # 您想要计算整体平均值的指标列表
    "metrics_to_analyze": [
        "simulation_duration",
        "dead_agents",
        "success_rate",
        "explored_safe_count"
    ]
}

# =======================================================
# 📊 分析函数
# =======================================================

def load_data(file_path):
    """加载 JSON 数据并检查文件是否存在。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON 文件未找到: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_grand_averages(data, metrics):
    """
    计算所有地图 ID 下，指定指标的整体平均值 (Grand Mean)。
    即：计算所有地图的 'mean' 值的平均值。
    """
    aggregated_results = defaultdict(lambda: defaultdict(list))
    final_analysis = {}

    # 1. 聚合所有 'mean' 值，按方法和指标分组
    for method_name, map_data in data.items():
        # map_data 是 { "1": { metrics... }, "2": { metrics... }, ... }
        for map_id, metric_data in map_data.items():
            for metric in metrics:
                if metric in metric_data and 'mean' in metric_data[metric]:
                    # 将该地图下的 mean 值添加到列表中
                    aggregated_results[method_name][metric].append(metric_data[metric]['mean'])

    # 2. 计算整体平均值 (Grand Mean)
    for method_name, metric_lists in aggregated_results.items():
        final_analysis[method_name] = {}
        for metric, means in metric_lists.items():
            if means:
                grand_mean = sum(means) / len(means)
                final_analysis[method_name][metric] = grand_mean
            else:
                final_analysis[method_name][metric] = None
                
    return final_analysis

def print_results(analysis):
    """以表格形式打印比较结果。"""
    print("\n" + "=" * 60)
    print("           实验结果整体平均值分析 (Grand Average Analysis)")
    print("=" * 60)
    
    # 确定所有方法和指标
    methods = sorted(list(analysis.keys()))
    metrics = sorted(list(set(m for res in analysis.values() for m in res.keys())))

    # 定义指标的中文显示名称
    metric_name_map = {
        "simulation_duration": "仿真时长 (s)",
        "dead_agents": "死亡机器人数",
        "success_rate": "成功率",
        "explored_safe_count": "探索安全区域数",
    }
    
    # 确定列宽以确保对齐
    method_col_width = max(len(m) for m in methods) if methods else 10
    
    # 打印表头
    header = f"{'指标 (Metric)':<30} | "
    for method in methods:
        header += f"{method:>{method_col_width}} | "
    print(header)
    print("-" * 60)

    # 打印数据行
    for metric in metrics:
        metric_display_name = metric_name_map.get(metric, metric)
        row = f"{metric_display_name:<30} | "
        
        for method in methods:
            mean_value = analysis[method].get(metric)
            if mean_value is not None:
                # 格式化输出，时长和比率保留两位小数，计数保留一位
                if 'rate' in metric or 'duration' in metric:
                    row += f"{mean_value:>{method_col_width}.2f} | "
                else:
                    row += f"{mean_value:>{method_col_width}.1f} | "
            else:
                row += f"{'N/A':>{method_col_width}} | "
        
        print(row)
        
    print("=" * 60)
    print("\n说明:")
    print("  - 结果为所有地图 ID 下，对应指标 'mean' 值的平均值 (即 Grand Mean)。")
    print("  - 成功率越高越好，死亡机器人数越低越好。")


# =======================================================
# 🚀 主程序执行 (MAIN EXECUTION)
# =======================================================
if __name__ == "__main__":
    print(f"正在加载实验汇总文件: {CONFIG['json_file_path']}")
    try:
        data = load_data(CONFIG["json_file_path"])
        
        # 计算每个方法和指标的整体平均值
        grand_averages = calculate_grand_averages(data, CONFIG["metrics_to_analyze"])
        
        # 打印汇总分析结果
        print_results(grand_averages)

    except FileNotFoundError as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        print("请确保 JSON 文件路径配置正确，且文件与脚本位于同一目录或路径设置无误。", file=sys.stderr)
    except json.JSONDecodeError:
        print("❌ 错误: JSON 文件解析失败。请检查文件内容是否为有效的 JSON 格式。", file=sys.stderr)
    except Exception as e:
        print(f"❌ 发生未知错误: {e}", file=sys.stderr)