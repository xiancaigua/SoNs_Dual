import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.signal import savgol_filter
import json

# =============================================================
# 工具函数：平滑轨迹
# =============================================================
def smooth_traj(traj, window=7, poly=3):
    traj = np.array(traj)
    if len(traj) < window:
        return traj
    x = savgol_filter(traj[:, 0], window, poly)
    y = savgol_filter(traj[:, 1], window, poly)
    return np.vstack([x, y]).T

# =============================================================
# 将背景图绘制到 matplotlib
# =============================================================
def draw_background(img_path, width=1000, height=700):
    img = mpimg.imread(img_path)
    plt.imshow(img, extent=[0, width, height, 0])   # extent 保持 pygame 坐标系
    plt.xlim(0, width)
    plt.ylim(height, 0)

# =============================================================
# 绘制轨迹
# =============================================================
def draw_traj(traj, lw = 2,label=None):
    traj = smooth_traj(traj)
    plt.plot(traj[:, 0], traj[:, 1], linewidth=lw, label=label)
    plt.scatter(traj[0, 0], traj[0, 1], s=40, label=f"{label}-start")
    plt.scatter(traj[-1, 0], traj[-1, 1], s=40, marker="x", label=f"{label}-end")

# =============================================================
# 主函数：加载 JSON + 背景图 + 轨迹
# =============================================================
def visualize_simulation_with_background(
        json_path,
        bg_image_path,
        plot_small_agents=True,
        plot_large_agents=True,
        custom_select_small=None,
        custom_select_large=None,
        map_w=1000,
        map_h=700
    ):

    with open(json_path, "r") as f:
        data = json.load(f)

    small_agents = data["agent_details"]["small_agents"]
    large_agents = data["agent_details"]["large_agents"]

    plt.figure(figsize=(12, 8))

    # ① 先画背景
    draw_background(bg_image_path, map_w, map_h)

    # ② 再叠加轨迹
    if plot_small_agents:
        for ag in small_agents:
            aid = ag["id"]
            if custom_select_small is not None and aid not in custom_select_small:
                continue
            draw_traj(ag["hist_traj"], 1.5, label=f"small-{aid}")

    if plot_large_agents:
        for ag in large_agents:
            aid = ag["id"]
            if custom_select_large is not None and aid not in custom_select_large:
                continue
            draw_traj(ag["hist_traj"], 2, label=f"large-{aid}")

    # plt.legend()
    plt.show()


# ========== 使用示例 ==========
visualize_simulation_with_background(
    json_path="newbase1\simulation_results\sim_summary_20251124_225536.json",
    bg_image_path="maps\initial_map_seed110716_mapindex0_20251125_151131.png",   # ⬅️ 修改为你上传的背景图片路径
    plot_small_agents=True,
    plot_large_agents=True,
)
