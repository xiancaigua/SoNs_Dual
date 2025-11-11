import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.image as mpimg
import os

FREE, OBSTACLE, DANGER = 0, 1, 2
WORLD_W, WORLD_H = 800, 600

class Victim:
    def __init__(self, x, y):
        self.pos = (x, y)

class DummyRect:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

class DummyDangerZone:
    def __init__(self, x, y, r):
        self.x, self.y, self.r = x, y, r

class World:
    def __init__(self):
        self.obstacles = []
        self.danger_zones = []
        self.victim = None
        self.spawn_center = (0, 0)
        self.ground_grid = np.zeros((WORLD_H // 10, WORLD_W // 10), dtype=int)

    def _add_rect_obstacle(self, x, y, w, h):
        self.obstacles.append(DummyRect(x, y, w, h))

    def _add_danger(self, x, y, r):
        self.danger_zones.append(DummyDangerZone(x, y, r))

    def _update_ground_grid(self):
        pass

    def generate_fixed_map(self, map_id:int):
        self.obstacles.clear()
        self.danger_zones.clear()
        self.ground_grid[:] = FREE

        if map_id == 1:
            self._add_rect_obstacle(300, 300, 200, 150)
            self._add_danger(100, 100, 60)
            self._add_danger(700, 100, 60)
            self._add_danger(100, 500, 60)
            self._add_danger(700, 500, 60)
            self.victim = Victim(300, 500)
            self.spawn_center = (300, 100)

        elif map_id == 2:
            for x in range(100, 800, 150):
                self._add_rect_obstacle(x, 200, 80, 20)
                self._add_rect_obstacle(x - 50, 400, 80, 20)
            self._add_danger(400, 300, 100)
            self.victim = Victim(600, 50)
            self.spawn_center = (150, 500)

        elif map_id == 3:
            self._add_rect_obstacle(300, 100, 500, 40)
            self._add_rect_obstacle(300, 460, 500, 40)
            self._add_danger(500, 300, 70)
            self.victim = Victim(750, 300)
            self.spawn_center = (100, 300)

        elif map_id == 4:
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                x = 400 + 200 * np.cos(angle)
                y = 300 + 150 * np.sin(angle)
                self._add_rect_obstacle(x-20, y-20, 40, 40)
            self._add_danger(400, 300, 50)
            self.victim = Victim(700, 500)
            self.spawn_center = (150, 150)

        elif map_id == 5:
            for y in range(50, 550, 100):
                self._add_rect_obstacle(400, y, 40, 60)
            self._add_danger(500, 250, 80)
            self.victim = Victim(100, 300)
            self.spawn_center = (700, 100)

        elif map_id == 6:
            self._add_rect_obstacle(200, 150, 100, 200)
            self._add_rect_obstacle(600, 150, 100, 200)
            self._add_danger(400, 300, 90)
            self.victim = Victim(400, 500)
            self.spawn_center = (400, 100)

        elif map_id == 7:
            self._add_rect_obstacle(400, 0, 40, 250)
            self._add_rect_obstacle(400, 350, 40, 250)
            self._add_danger(300, 300, 40)
            self.victim = Victim(750, 300)
            self.spawn_center = (150, 300)

        elif map_id == 8:
            for _ in range(10):
                x = random.randint(100, 700)
                y = random.randint(100, 500)
                self._add_rect_obstacle(x, y, 30, 30)
            self._add_danger(100, 350, 60)
            self.victim = Victim(100, 100)
            self.spawn_center = (100, 500)

        elif map_id == 9:
            self._add_rect_obstacle(300, 0, 40, 300)
            self._add_rect_obstacle(300, 300, 400, 40)
            self._add_danger(650, 200, 75)
            self.victim = Victim(450, 150)
            self.spawn_center = (150, 150)

        elif map_id == 10:
            self._add_rect_obstacle(250, 200, 400, 40)
            self._add_rect_obstacle(250, 400, 400, 40)
            self._add_danger(400, 300, 60)
            self.victim = Victim(750, 550)
            self.spawn_center = (100, 100)

        else:
            raise ValueError("Invalid map_id, must be 1~10")

# ===========================================================
# ✅ 单地图可视化（无坐标轴、无图注）
# ===========================================================
def visualize_map(world: World, map_id: int, save_dir="saved_maps", show=False):
    world.generate_fixed_map(map_id)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"map_{map_id}.png")

    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.set_ylim(WORLD_H, 0)
    ax.set_xlim(0, WORLD_W)
    ax.set_aspect('equal')
    ax.set_facecolor("#eeeeee")

    # 🚫 去除坐标轴刻度数字与边框
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 绘制障碍物
    for obs in world.obstacles:
        rect = plt.Rectangle((obs.x, obs.y), obs.w, obs.h, color='gray', alpha=0.8)
        ax.add_patch(rect)

    # 绘制危险区
    for dz in world.danger_zones:
        circ = plt.Circle((dz.x, dz.y), dz.r, color='red', alpha=0.3, edgecolor='darkred')
        ax.add_patch(circ)

    # 绘制受害者与出生点（无图例）
    if world.victim:
        plt.scatter(*world.victim.pos, c='gold', s=200, marker='*', edgecolors='black')
    plt.scatter(*world.spawn_center, c='blue', s=100, marker='o')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Map {map_id} saved to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

# ===========================================================
# ✅ 合并图（在总图上添加图例）
# ===========================================================
def merge_maps_grid(image_dir="saved_maps", rows=2, cols=5, save_name="merged_maps.png"):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith(".png")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    for i, ax in enumerate(axes):
        if i < len(image_files):
            img_path = os.path.join(image_dir, image_files[i])
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Map {i+1}", fontsize=14)
        else:
            ax.axis('off')

    # ✅ 统一图例（在合成总图上添加）
    handles = [
        plt.Line2D([0], [0], color='gray', lw=6, label='Obstacle'),
        plt.Line2D([0], [0], color='red', lw=6, alpha=0.4, label='Danger Zone'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Spawn Center'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=12, label='Victim')
    ]
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=14, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 给图例留空间
    save_path = os.path.join(image_dir, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 拼接完成，已保存到：{save_path}")

# ===========================================================
# 主入口
# ===========================================================
if __name__ == "__main__":
    world = World()
    for mid in range(1, 11):
        visualize_map(world, mid)
    merge_maps_grid("saved_maps")
