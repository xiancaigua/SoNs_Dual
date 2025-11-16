import matplotlib.pyplot as plt
import numpy as np
import random
import os
from datetime import datetime

# ===========================================================
# 基础参数
# ===========================================================
WORLD_W, WORLD_H = 800, 600
FREE, OBSTACLE, DANGER = 0, 1, 2

# ===========================================================
# 基本类
# ===========================================================
class Victim:
    def __init__(self, x, y):
        self.pos = (x, y)

class DummyRect:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

class DummyDangerZone:
    def __init__(self, x, y, r):
        self.x, self.y, self.r = x, y, r

# ===========================================================
# 随机地图生成类
# ===========================================================
class RandomWorld:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.obstacles = []
        self.danger_zones = []
        self.victim = None
        self.spawn_center = (0, 0)

    def generate_random_map(self, num_obstacles=8, num_dangers=5):
        """生成一个随机地图"""
        self.obstacles.clear()
        self.danger_zones.clear()

        # ---------- 障碍物 ----------
        for _ in range(num_obstacles):
            w = random.randint(40, 160)
            h = random.randint(30, 100)
            x = random.randint(20, WORLD_W - w - 20)
            y = random.randint(20, WORLD_H - h - 20)
            self.obstacles.append(DummyRect(x, y, w, h))

        # ---------- 危险区 ----------
        for _ in range(num_dangers):
            while True:
                x = random.randint(80, WORLD_W - 80)
                y = random.randint(80, WORLD_H - 80)
                r = random.randint(40, 90)
                # 确保危险区不与障碍物严重重叠
                overlap = False
                for obs in self.obstacles:
                    if (obs.x - r < x < obs.x + obs.w + r) and (obs.y - r < y < obs.y + obs.h + r):
                        overlap = True
                        break
                if not overlap:
                    self.danger_zones.append(DummyDangerZone(x, y, r))
                    break

        # ---------- Victim ----------
        while True:
            vx = random.randint(50, WORLD_W - 50)
            vy = random.randint(50, WORLD_H - 50)
            if not any(obs.x <= vx <= obs.x + obs.w and obs.y <= vy <= obs.y + obs.h for obs in self.obstacles):
                self.victim = Victim(vx, vy)
                break

        # ---------- 出生点 ----------
        while True:
            sx = random.randint(50, WORLD_W - 50)
            sy = random.randint(50, WORLD_H - 50)
            if not any(obs.x <= sx <= obs.x + obs.w and obs.y <= sy <= obs.y + obs.h for obs in self.obstacles):
                self.spawn_center = (sx, sy)
                break

# ===========================================================
# ✅ 可视化函数（风格完全一致）
# ===========================================================
def visualize_random_map(world: RandomWorld, map_id:int, save_dir="saved_random_maps", show=False):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"random_map_{map_id}.png")

    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.set_ylim(WORLD_H, 0)
    ax.set_xlim(0, WORLD_W)
    ax.set_aspect('equal')
    ax.set_facecolor("#eeeeee")

    # 去掉坐标轴与边框
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---------- 绘制障碍物 ----------
    for obs in world.obstacles:
        rect = plt.Rectangle((obs.x, obs.y), obs.w, obs.h, color='gray', alpha=0.8)
        ax.add_patch(rect)

    # ---------- 绘制危险区 ----------
    for dz in world.danger_zones:
        circ = plt.Circle((dz.x, dz.y), dz.r, color='red', alpha=0.3, edgecolor='darkred')
        ax.add_patch(circ)

    # ---------- 绘制 victim & 出生点 ----------
    if world.victim:
        plt.scatter(*world.victim.pos, c='gold', s=200, marker='*', edgecolors='black')
    plt.scatter(*world.spawn_center, c='blue', s=100, marker='o')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Random Map {map_id} saved to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

# ===========================================================
# 主入口：批量生成随机地图
# ===========================================================
if __name__ == "__main__":
    num_maps = 10
    seed = int(datetime.now().timestamp()) % 10000
    print(f"🌱 Using random seed: {seed}")
    random.seed(seed)

    for i in range(1, num_maps + 1):
        rw = RandomWorld()
        rw.generate_random_map()
        visualize_random_map(rw, i)
