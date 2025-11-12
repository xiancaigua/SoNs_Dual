import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import *
import heapq

# ======== 全局参数 ========
GRID_W, GRID_H = 40, 30
FREE, OBSTACLE, DANGER, VICTIM, UNKNOWN = 0, 1, 2, 3, -1
AGENT_COLORS = {
    "small": "#2a6df4",
    "large": "#e2b000",
    "dead":  "#888888"
}

REALTIME = True
FPS = 5


# ======== 工具函数 ========
def heuristic(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
def in_bounds(x, y): return 0 <= x < GRID_W and 0 <= y < GRID_H


# ======== A*可达性检测 ========
def a_star(start, goal, world_grid):
    if not in_bounds(*goal) or world_grid[goal[1], goal[0]] == OBSTACLE:
        return False
    open_heap = [(0, start)]
    g = {start: 0}
    while open_heap:
        f, cur = heapq.heappop(open_heap)
        if cur == goal:
            return True
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nb = (cur[0]+dx, cur[1]+dy)
            if not in_bounds(*nb) or world_grid[nb[1], nb[0]] == OBSTACLE:
                continue
            tentative = g[cur] + 1
            if tentative < g.get(nb, float('inf')):
                g[nb] = tentative
                heapq.heappush(open_heap, (tentative + heuristic(nb, goal), nb))
    return False


# ======== 世界类 ========
class DiscreteWorld:
    def __init__(self, map_mode="random", map_id=1, seed=42):
        if seed is not None:
            random.seed(seed)
        self.true_grid = np.zeros((GRID_H, GRID_W), dtype=np.int8)
        self.agents = []
        self.large_agents = []
        self.wasted_agents = []
        self.wasted_large_agents = []
        self.victim = None
        self.brain_id = None
        self.t = 0

        if map_mode == "fixed":
            self.generate_fixed_map(map_id)
        else:
            self.generate_random_world()

        # 全局已知地图
        self.global_known = np.full((GRID_H, GRID_W), UNKNOWN, dtype=np.int8)

    # ---------- 随机世界 ----------
    def generate_random_world(self, num_obstacles=15, num_dangers=6):
        for _ in range(num_obstacles):
            x, y = random.randint(0, GRID_W-4), random.randint(0, GRID_H-4)
            w, h = random.randint(2, 4), random.randint(2, 4)
            self.true_grid[y:y+h, x:x+w] = OBSTACLE
        for _ in range(num_dangers):
            x, y = random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)
            if self.true_grid[y, x] == FREE:
                self.true_grid[y, x] = DANGER

        # 生成受害者
        self.victim = (random.randint(5, GRID_W-5), random.randint(5, GRID_H-5))
        self.true_grid[self.victim[1], self.victim[0]] = VICTIM

        # 初始化agent（一个大节点+多个小节点）
        self.spawn_agents([(5, 5)], [(7, 5), (6, 6), (8, 6), (7, 7)])

    # ---------- 固定世界 ----------
    def generate_fixed_map(self, map_id=1):
        self.true_grid[:] = FREE
        if map_id == 1:
            self.true_grid[10:15, 10:25] = OBSTACLE
            self.true_grid[20:23, 30:35] = DANGER
            victim_pos = (35, 25)
            small_starts = [(5, 5), (7, 5), (6, 7), (8, 6)]
            large_starts = [(4, 4)]
        elif map_id == 2:
            self.true_grid[8:22, 15] = OBSTACLE
            self.true_grid[10, 10:15] = DANGER
            victim_pos = (38, 20)
            small_starts = [(3, 25), (4, 25), (5, 25), (6, 25)]
            large_starts = [(2, 24)]
        else:
            raise ValueError("Unsupported map_id")

        self.victim = victim_pos
        self.true_grid[victim_pos[1], victim_pos[0]] = VICTIM
        self.spawn_agents(large_starts, small_starts)

    # ---------- 生成机器人 ----------
    def spawn_agents(self, large_positions, small_positions):
        for i, pos in enumerate(large_positions):
            self.large_agents.append(LargeAgent(i, pos))
        for j, pos in enumerate(small_positions):
            self.agents.append(SmallAgent(j, pos))

    # ---------- 仿真步 ----------
    def step(self):
        self.t += 1

        # ---- 0. 早期退出条件 ----
        if (not self.large_agents) and (not self.agents):
            return
        
        # ---- 1. 处理死亡节点（小/大）并归档 ----
        dead_small = [a for a in self.agents if not a.alive]
        if dead_small:
            self.wasted_agents.extend(dead_small)
        self.agents = [a for a in self.agents if a.alive]

        # 对大节点：保留原来 is_brain 标识用于迁移（如果需要）
        dead_large = [la for la in self.large_agents if not la.alive]
        if dead_large:
            # 注意：不要把脑节点自动丢进 wasted_large_agents（下面会专门处理brain替换）
            for la in dead_large:
                if not getattr(la, "is_brain", False):
                    self.wasted_large_agents.append(la)
        alive_large_agents = [la for la in self.large_agents if la.alive]

        # ---- 2. 选举/更新 brain 节点（如果当前 brain 不存在或死亡则选新脑） ----
        old_brain_idx = getattr(self, "brain_id", None)
        old_brain = None
        if old_brain_idx is not None and 0 <= old_brain_idx < len(self.large_agents):
            old_brain = self.large_agents[old_brain_idx]

        # If current brain is missing or dead -> elect new brain among alive_large_agents
        if (old_brain is None) or (not getattr(old_brain, "alive", False)):
            if alive_large_agents:
                # simple policy: choose lowest id (你可以改成其它策略)
                new_brain = min(alive_large_agents, key=lambda a: a.id)
                # carry over some state from previous brain if exists
                if old_brain is not None:
                    # archive old brain
                    self.wasted_large_agents.append(old_brain)
                    try:
                        # 把老脑的 known_map/son_ids/history 交接到新脑（浅/深拷贝按需）
                        new_brain.local_map = np.copy(old_brain.local_map)
                        new_brain.son_ids = list(getattr(old_brain, "son_ids", []))
                        # last reason time 保持
                        new_brain.last_reason_time = getattr(old_brain, "last_reason_time", new_brain.last_reason_time)
                    except Exception:
                        pass
                # 标记新脑
                for la in alive_large_agents:
                    la.is_brain = False
                new_brain.is_brain = True
                # 将 large_agents 列表替换为当前活着的集合，并设置 brain_id 为 index
                self.large_agents = alive_large_agents
                self.brain_id = self.large_agents.index(new_brain)
            else:
                # 没有任何大节点存活：清空并退出
                self.large_agents = []
                self.brain_id = None
                return
        else:
            # brain 还存活：更新 large_agents 为 alive 集合（保留 brain 的索引）
            # 但需要确保 brain 在新的 alive_large_agents 中找到其新索引
            self.large_agents = alive_large_agents
            # 更新 brain_id
            for idx, la in enumerate(self.large_agents):
                if getattr(la, "is_brain", False):
                    self.brain_id = idx
                    break
        
        if len(self.large_agents) == 0:
            return  # No large agents left to proceed
        
        brain = None
        if getattr(self, "brain_id", None) is not None and 0 <= self.brain_id < len(self.large_agents):
            brain = self.large_agents[self.brain_id]
        
        # ---- 3. 更新层级关系：为每个小节点分配最近的大节点为 father（考虑切换阈值） ----
        for sa in self.agents:
            # find the nearest alive large agent
            if not self.large_agents:
                sa.father_id = None
                continue
            candidates = sorted(self.large_agents, key=lambda la: distance(la.pos, sa.pos))
            nearest = candidates[0]
            old_father = self.find_agent_by_id(getattr(sa, "father_id", None))
            if old_father is not None:
                dist_old = distance(sa.pos, old_father.pos)
                dist_new = distance(sa.pos, nearest.pos)
                # hysteresis: 小于阈值则不切换
                if abs(dist_new - dist_old) < 50:
                    continue
            sa.father_id = nearest.id
            # maintain son's list on the chosen large agent (we will rebuild later)
        # rebuild son lists for large agents
        for la in self.large_agents:
            la.son_ids = []
        for sa in self.agents:
            parent = self.find_agent_by_id(getattr(sa, "father_id", None))
            if parent is not None:
                parent.son_ids.append(sa.id)  

        # ---- 4. Agents update sensing -> 更新各自 local_map, large 节点把自己的 sensing 融合到 known_map ----
        for a in self.agents:
            a.sense(self)
        for a in self.large_agents:
            a.sense(self)
            if getattr(a, "is_large", False):
                # large 节点把自己的 local map 融入 known_map（优先级可调整）
                a.fuse_own_sensing(self.agents)
        

        for a in self.large_agents + self.agents:
            if a.alive:
                a.step(self)

        # 更新全局已知
        for a in self.large_agents + self.agents:
            for (x, y) in a.explored:
                if in_bounds(x, y):
                    self.global_known[y, x] = self.true_grid[y, x]

    def find_agent_by_id(self, id_):
        for la in self.large_agents:
            if la.id == id_:
                return la
        for sa in self.agents:
            if sa.id == id_:
                return sa
        return None


from behaviors_disc import *


# ======== Agent基类 ========
class BaseAgent:
    def __init__(self, id_, pos, sensor_range, comm_range):
        self.id = id_
        self.pos = pos
        self.sensor_range = sensor_range
        self.comm_range = comm_range
        self.alive = True
        self.explored = set([pos])
        self.local_map = np.full((GRID_H, GRID_W), UNKNOWN, dtype=np.int8)

        self.sons_ids = []  # 仅大节点使用
        self.father_id = None  # 仅小节点使用

    def sense(self, world):
        r = self.sensor_range
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                nx, ny = self.pos[0]+dx, self.pos[1]+dy
                if in_bounds(nx, ny):
                    val = world.true_grid[ny, nx]
                    if val != UNKNOWN:
                        if val == DANGER and random.random() < 0.3:
                            # 30% 概率未能识别到危险
                            self.local_map[ny, nx] = val
                        elif val == VICTIM or val == OBSTACLE:
                            self.local_map[ny, nx] = val
                        else:
                            self.local_map[ny, nx] = FREE
                    self.explored.add((nx, ny))

    def step(self, world):
        self.sense(world)
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = self.pos[0]+dx, self.pos[1]+dy
            if not in_bounds(nx, ny):
                continue
            val = world.true_grid[ny, nx]
            if val == OBSTACLE:
                continue
            elif val == DANGER:
                self.alive = False
                return
            elif val == VICTIM:
                print(f"✅ Agent {self.id} rescued victim at {nx,ny}")
                world.true_grid[ny, nx] = FREE
                self.pos = (nx, ny)
                return
            elif val == FREE:
                self.pos = (nx, ny)
                self.explored.add((nx, ny))
                return

    def get_local_explored_cells(self):
        """返回该agent已探索（非UNKNOWN）的格子集合（i,j）"""
        inds = np.where(self.local_map != UNKNOWN)
        return set(zip(inds[0].tolist(), inds[1].tolist()))
    


# ======== 小/大节点定义 ========
class SmallAgent(BaseAgent):
    def __init__(self, id_, pos):
        super().__init__(id_, pos, sensor_range=2, comm_range=4)
        self.type = "small"

class LargeAgent(BaseAgent):
    def __init__(self, id_, pos):
        super().__init__(id_, pos, sensor_range=4, comm_range=8)
        self.type = "large"
        self.is_large = True
        # self.local_map = np.full((GRID_H, GRID_W), UNKNOWN, dtype=np)
        self.is_brain = False
        self.commanded_agents = []
        self.multi_behavior = ERRTFrontierAssignmentBehavior()
    def fuse_own_sensing(self, small_agents):
        for sa in small_agents:
            if sa.father_id == self.id:
                for (x, y) in sa.explored:
                    patch = sa.get_local_explored_cells()
                    """将收到的patch应用到自己的known_map"""
                    for (i,j) in patch:
                        # val is occupancy code
                        val = sa.local_map[i,j]
                        if 0 <= i < GRID_H and 0 <= j < GRID_W:
                            # overwrite unknowns or keep obstacle/danger priority
                            if self.local_map[i,j] == UNKNOWN and val != UNKNOWN:
                                self.local_map[i,j] = val
                            else:
                                # 若已有UNKNOWN则覆盖，否则保持原先（或根据优先级更新）
                                # 优先级： DANGER/OBSTACLE > VICTIM > FREE
                                cur = self.local_map[i,j]
                                if val == DANGER or val == OBSTACLE:
                                    self.local_map[i,j] = val
                                elif val == VICTIM:
                                    self.local_map[i,j] = val
                                elif cur == UNKNOWN:
                                    self.local_map[i,j] = val
    def reason(self, world):
        
        # 大节点可以作为智能体决策的“区域指挥官”
        assignments = self.multi_behavior.decide(self, self.commanded_agents)
        return assignments

# ======== 绘图函数 ========
def draw_world(world, fig, main_ax, mini_axes):
    """
    可视化规则：
    1. 未探索格（UNKNOWN）→ 深灰
    2. 已探索格 → 鲜明本色
    3. 真实物体未探到 → 半透灰蒙覆盖，可隐约看见轮廓
    """
    # ---------- 颜色表 ----------
    bright = {
        FREE:     mcolors.to_rgba("#ffffff"),   # 探索后自由格
        OBSTACLE: mcolors.to_rgba("#444444"),   # 障碍
        DANGER:   mcolors.to_rgba("#cc3333"),   # 危险
        VICTIM:   mcolors.to_rgba("#44cc44"),   # 受害者
    }
    unknown_gray = mcolors.to_rgba("#404040")   # 未探索底色
    hazy_alpha = 0.35                           # 灰蒙透明度

    # ---------- 主图 RGBA ----------
    img = np.full((GRID_H, GRID_W, 4), unknown_gray)   # 先整张铺深灰

    for y in range(GRID_H):
        for x in range(GRID_W):
            real = world.true_grid[y, x]
            if world.global_known[y, x] != UNKNOWN:
                # 已探索 → 鲜明色
                img[y, x] = bright.get(real, unknown_gray)
            else:
                # 未探索 → 在深灰基础上叠半透灰蒙的真实物体
                r, g, b, _ = bright.get(real, unknown_gray)
                img[y, x] = (r * 0.6, g * 0.6, b * 0.6, hazy_alpha)

    # ---------- 绘制 ----------
    main_ax.clear()
    main_ax.imshow(img, origin="upper")
    main_ax.set_xticks([])
    main_ax.set_yticks([])
    main_ax.set_title(f"Global View - Step {world.t}")

    # ---------- 机器人 ----------
    for a in world.large_agents + world.agents:
        color = AGENT_COLORS["dead"] if not a.alive else AGENT_COLORS[a.type]
        main_ax.scatter(a.pos[0], a.pos[1],
                        color=color, s=60, edgecolors='black', linewidth=0.8, zorder=5)

    # ---------- 局部小图 ----------
    for ax, agent in zip(mini_axes, world.large_agents + world.agents):
        ax.clear()
        local_img = np.full((GRID_H, GRID_W, 4), unknown_gray)
        for y in range(GRID_H):
            for x in range(GRID_W):
                v = agent.local_map[y, x]
                if v != UNKNOWN:
                    local_img[y, x] = bright.get(v, unknown_gray)
                else:
                    r, g, b, _ = bright.get(world.true_grid[y, x], unknown_gray)
                    local_img[y, x] = (r * 0.6, g * 0.6, b * 0.6, hazy_alpha)
        ax.imshow(local_img, origin="upper")
        ax.set_title(f"A{agent.id}-{agent.type[0]}{'(dead)' if not agent.alive else ''}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.canvas.draw()
    plt.pause(1.0 / FPS)


# ======== 主程序 ========
def main():
    world = DiscreteWorld(map_mode="fixed", map_id=1)
    steps = 80

    plt.ion()
    fig = plt.figure(figsize=(12, 6))
    grid = fig.add_gridspec(2, 3)
    main_ax = fig.add_subplot(grid[:, 0:2])
    mini_axes = [fig.add_subplot(grid[i, 2]) for i in range(2)]

    for _ in range(steps):
        world.step()
        main_ax.clear()
        for ax in mini_axes: ax.clear()
        draw_world(world, fig, main_ax, mini_axes)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
