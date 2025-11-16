import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import make_interp_spline  # ✅ 用于轨迹平滑

# ======== 1️⃣ 读取 JSON 文件 ========
json_path = "./simulation_results/sim_summary_20251113_112212.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

stats = data["statistics"]
smalls = data["agent_details"]["small_agents"]
larges = data["agent_details"]["large_agents"]

# ======== 2️⃣ 地图绘制函数 ========
def draw_base_map(ax, map_id):
    ax.set_facecolor("white")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 700)
    ax.set_aspect('equal')
    ax.set_title(f"Map {map_id} Visualization (Result: {stats['result']})", fontsize=14)

    if map_id == 1:
        # 主障碍
        rect = patches.Rectangle((300, 300), 200, 150, linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
        ax.add_patch(rect)

        # 危险区域
        for cx, cy, r in [(100, 100, 60), (700, 100, 60), (100, 500, 60), (700, 500, 60)]:
            danger = patches.Circle((cx, cy), r, color='red', alpha=0.3)
            ax.add_patch(danger)

        # 受害者
        ax.plot(300, 500, marker='*', color='lime', markersize=18)

        # 出生点
        ax.plot(300, 100, marker='o', color='blue', markersize=10)

# ======== 3️⃣ 平滑函数 ========
def smooth_path(x, y, smooth_factor=1000):
    """使用三次样条插值平滑路径"""
    if len(x) < 4:
        return x, y  # 太短无法平滑
    t = np.arange(len(x))
    t_new = np.linspace(t.min(), t.max(), smooth_factor)
    spline_x = make_interp_spline(t, x, k=3)(t_new)
    spline_y = make_interp_spline(t, y, k=3)(t_new)
    return spline_x, spline_y

# ======== 4️⃣ 绘制机器人轨迹 ========
def draw_agents(ax, smalls, larges):
    # ---------- 小机器人 ----------
    for sa in smalls:
        # if sa["alive"]:
        #     continue
        traj = sa["hist_traj"]
        traj_x = [p[0] for p in traj]
        traj_y = [p[1] for p in traj]
        color = 'green' if sa["alive"] else 'gray'

        # ✅ 平滑化轨迹
        sx, sy = smooth_path(traj_x, traj_y, smooth_factor=1500)
        ax.plot(sx, sy, color=color, linewidth=1.5, alpha=0.7)

        # 当前末尾位置
        ax.scatter(traj_x[-1], traj_y[-1], color=color, s=40)

        # # 目标点与连线
        # if sa["goal_position"]:
        #     gx, gy = sa["goal_position"]
        #     ax.plot([traj_x[-1], gx], [traj_y[-1], gy], '--', color=color, alpha=0.4)
        #     ax.scatter(gx, gy, color='orange', s=30, marker='x')

    # ---------- 大机器人 ----------
    # for la in larges:
    #     traj = la["hist_traj"]
    #     if traj and len(traj) > 1:
    #         traj_x = [p[0] for p in traj]
    #         traj_y = [p[1] for p in traj]
    #         # ✅ 平滑大节点轨迹
    #         sx, sy = smooth_path(traj_x, traj_y, smooth_factor=600)
    #         ax.plot(sx, sy, color='blue', linewidth=2.5, alpha=0.8)
    #     # 当前位置
    #     ax.scatter(*la["final_position"], color='blue', s=80, marker='s', edgecolor='black')
        # ax.text(la["final_position"][0]+10, la["final_position"][1], f"L{la['id']}", color='blue', fontsize=9)

# ======== 5️⃣ 主绘制逻辑 ========
fig, ax = plt.subplots(figsize=(10, 7))
draw_base_map(ax, stats["map_id"])
draw_agents(ax, smalls, larges)

ax.legend(loc="upper right", fontsize=8)
ax.set_xlabel("X (world units)")
ax.set_ylabel("Y (world units)")
ax.invert_yaxis()  # pygame坐标一致

plt.tight_layout()
plt.show()
