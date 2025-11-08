# main.py
"""
异构多机器人在危险环境中的探索系统（单文件实现）
- 使用 pygame 可视化
- Brain nodes（脑节点）和 Agent nodes（小节点）
- 静态障碍 / 随机危险区域 / 被困者（目标点）
- 协同搜索、断联独立决策、脑节点周期性推理与下发指令
- 输出覆盖率、轨迹与最终统计

运行:
    pip install pygame
    python main.py
"""
import heapq
import pygame
import random
import math
import sys
import time
import numpy as np
from collections import deque, defaultdict

from parameters import *

# -----------------------------
# 工具函数
# -----------------------------
def load_font():
    # 尝试加载常见中文字体，否则回退系统字体
    try:
        return pygame.font.Font("SimHei.ttf", 16)
    except Exception:
        try:
            return pygame.font.SysFont("SimHei", 16)
        except Exception:
            return pygame.font.SysFont("Arial", 16)

def clamp(x, a, b):
    return max(a, min(b, x))

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def vec_len(v):
    return math.hypot(v[0], v[1])

def normalize(v):
    l = vec_len(v)
    if l == 0:
        return (0,0)
    return (v[0]/l, v[1]/l)

def add(v, w):
    return (v[0]+w[0], v[1]+w[1])

def mul(v, s):
    return (v[0]*s, v[1]*s)

def cell_of_pos(pos):
    return int(pos[0] // GRID_CELL), int(pos[1] // GRID_CELL)

def pos_of_cell(i, j):
    return (i*GRID_CELL + GRID_CELL/2.0, j*GRID_CELL + GRID_CELL/2.0)



# -----------------------------
# 环境元素类
# -----------------------------
class Obstacle:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(int(x), int(y), int(w), int(h))
        self.center = (x + w/2, y + h/2)
        self.width = w
        self.height = h
    def draw(self, screen):
        pygame.draw.rect(screen, (80,80,80), self.rect)
class DangerZone:
    def __init__(self, x, y, r=DANGER_ZONE_RADIUS):
        self.pos = (x,y)
        self.r = r
    def contains(self, p):
        return distance(self.pos, p) <= self.r
    def draw(self, screen):
        # 创建一个临时的 Surface
        temp_surface = pygame.Surface((self.r * 2, self.r * 2), pygame.SRCALPHA)
        # 填充浅红色（带透明度）
        transparent_red = (200, 40, 40, 100)  # RGBA，最后一个值是透明度
        pygame.draw.circle(temp_surface, transparent_red, (self.r, self.r), self.r)
        # 将透明 Surface 绘制到屏幕上
        screen.blit(temp_surface, (int(self.pos[0] - self.r), int(self.pos[1] - self.r)))

        # 绘制边框
        pygame.draw.circle(screen, (120, 20, 20), (int(self.pos[0]), int(self.pos[1])), self.r, 2)
class Victim:
    def __init__(self, x, y):
        self.pos = (x,y)
        self.rescued = False
    def draw(self, screen):
        color = (30,200,30) if not self.rescued else (100,200,100)
        pygame.draw.circle(screen, color, (int(self.pos[0]), int(self.pos[1])), 8)

# -----------------------------
# 通信系统
# -----------------------------

class Communication:
    """简单的点对点基于距离的通信系统，带可配置丢包与延迟（延迟目前同步模拟）"""
    def __init__(self, packet_loss=COMM_PACKET_LOSS, delay=COMM_DELAY):
        self.packet_loss = packet_loss
        self.delay = delay
        self.queue = deque()  # (deliver_time, sender_id, receiver_id, message)

    def send(self, sender, receiver, message, now_time):
        """发送消息到单个接收者；不保证送达（丢包）"""
        if random.random() < self.packet_loss:
            return False
        deliver_time = now_time + self.delay
        self.queue.append((deliver_time, sender, receiver, message))
        return True

    def broadcast(self, sender, receivers,\
        message, now_time, \
        range_limit=None):
        sent = 0
        for r in receivers:
            if range_limit is None or distance(sender.pos, r.pos) <= range_limit:
                if self.send(sender, r, message, now_time):
                    sent += 1
        return sent

    def deliver(self, now_time):
        """将已到达时间的消息投递给目标（外部负责调用并处理）"""
        delivered = []
        while self.queue and self.queue[0][0] <= now_time:
            _, sender, receiver, message = self.queue.popleft()
            delivered.append((sender, receiver, message))
        return delivered

# -----------------------------
# 行为模块（策略模式）
# -----------------------------
class Behavior:
    def decide(self, agent, sense_data, dt):
        """return desired_velocity (vx, vy) in pixels/sec"""
        raise NotImplementedError

class ReactiveBehavior(Behavior):
    """简单的力场避障 + 向目标/随机游走"""
    def decide(self, agent, sense_data, dt):
        # desired direction towards assigned target if any, else along current vel/random
        if agent.has_goal and agent.goal is not None:
            dx = agent.goal[0] - agent.pos[0]
            dy = agent.goal[1] - agent.pos[1]
            tx, ty = normalize((dx, dy))
        else:
            # random drift
            tx, ty = normalize((agent.vel[0], agent.vel[1]))
            if tx == 0 and ty == 0:
                angle = random.uniform(0, 2*math.pi)
                tx, ty = math.cos(angle), math.sin(angle)

        # repulsion from obstacles and dangers within sense data
        rx, ry = 0.0, 0.0
        for (px, py), d in sense_data.get('near_obstacles', []):
            vx = agent.pos[0] - px
            vy = agent.pos[1] - py
            nd = math.hypot(vx, vy) + 1e-6
            mag = 800.0 / (nd*nd)
            rx += (vx/nd)*mag
            ry += (vy/nd)*mag
        for (dz_pos, d, r) in sense_data.get('near_dangers', []):
            vx = agent.pos[0] - dz_pos[0]
            vy = agent.pos[1] - dz_pos[1]
            nd = math.hypot(vx, vy) + 1e-6
            mag = 3000.0 / (nd + 1.0)
            rx += (vx/nd)*mag
            ry += (vy/nd)*mag

        steer_x = tx*40.0 + rx
        steer_y = ty*40.0 + ry
        sx, sy = normalize((steer_x, steer_y))
        vx = sx * AGENT_MAX_SPEED
        vy = sy * AGENT_MAX_SPEED
        return vx, vy

class ExploreBehavior(Behavior):
    """在没有明确目标时的探索行为（随机游走 + 轻微偏向未知）"""
    def decide(self, agent, sense_data, dt):
        # If agent has goal delegate to reactive (goal-following)
        if agent.has_goal and agent.goal is not None:
            # follow toward goal
            dx = agent.goal[0] - agent.pos[0]
            dy = agent.goal[1] - agent.pos[1]
            sx, sy = normalize((dx, dy))
            return sx*AGENT_MAX_SPEED, sy*AGENT_MAX_SPEED
        # else random with occasional direction change
        if random.random() < 0.02:
            ang = random.uniform(0, 2*math.pi)
            agent.vel = (math.cos(ang)*10.0, math.sin(ang)*10.0)
        vx = agent.vel[0]
        vy = agent.vel[1]
        sx, sy = normalize((vx, vy))
        return sx*AGENT_MAX_SPEED, sy*AGENT_MAX_SPEED

class FollowBehavior(Behavior):
    """跟随分配的 waypoint（简单）"""
    def decide(self, agent, sense_data, dt):
        if agent.has_goal and agent.goal is not None:
            dx = agent.goal[0] - agent.pos[0]
            dy = agent.goal[1] - agent.pos[1]
            sx, sy = normalize((dx, dy))
            return sx*AGENT_MAX_SPEED, sy*AGENT_MAX_SPEED
        else:
            return 0.0, 0.0


# -----------------------------
# Agent（小节点）类
# -----------------------------

class AgentBase:
    def __init__(self, id_, x, y, sensor_range=SENSOR_SMALL, is_large=False):
        self.id = id_
        self.pos = (x, y)
        angle = random.random()*2*math.pi
        self.vel = (math.cos(angle)*10.0, math.sin(angle)*10.0)  # px/sec initial drift
        self.alive = True
        self.sensor_range = sensor_range
        self.local_map = np.full((GRID_W, GRID_H), UNKNOWN, dtype=np.int8)
        self.has_goal = False
        self.goal = None  # waypoint in world coords
        self.comm_ok = True
        self.hist = [self.pos]
        self.is_large = is_large  # LargeAgent flag
        # default behavior
        self.behavior = ReactiveBehavior() if not is_large else ExploreBehavior()

    # 环境感知：返回用于行为决策的局部信息（近障碍、危险区、victim）
    def sense(self, env):
        near_obstacles = []
        near_dangers = []
        victim_seen = False
        # obstacles: use closest point distance
        for obs in env.obstacles:
            cx = clamp(self.pos[0], obs.rect.left, obs.rect.right)
            cy = clamp(self.pos[1], obs.rect.top, obs.rect.bottom)
            d = distance((cx, cy), self.pos)
            if d <= self.sensor_range:
                near_obstacles.append(((cx, cy), d))
        for dz in env.danger_zones:
            d = distance(self.pos, dz.pos)
            if d <= self.sensor_range + dz.r:
                near_dangers.append((dz.pos, d, dz.r))
        # victim
        if distance(self.pos, env.victim.pos) <= self.sensor_range:
            victim_seen = True
        return {'near_obstacles': near_obstacles, 'near_dangers': near_dangers, 'victim_seen': victim_seen}

    def update_local_map_from_sensing(self, env):
        """基于当前传感器范围写入 local_map（仅在视野内将真实世界写入local_map）"""
        cx, cy = int(self.pos[0]), int(self.pos[1])
        r = int(self.sensor_range)
        min_i = max(0, (cx - r)//GRID_CELL)
        max_i = min(GRID_W-1, (cx + r)//GRID_CELL)
        min_j = max(0, (cy - r)//GRID_CELL)
        max_j = min(GRID_H-1, (cy + r)//GRID_CELL)
        for i in range(min_i, max_i+1):
            for j in range(min_j, max_j+1):
                gx, gy = pos_of_cell(i, j)
                if math.hypot(gx - cx, gy - cy) <= r:
                    # sample world truth
                    val = env.ground_grid[i, j]
                    self.local_map[i, j] = val
    
    def integrate_map_patch(self, patch):
        """将收到的patch应用到自己的local_map"""
        for (i,j,val) in patch:
            # val is occupancy code
            if 0 <= i < GRID_W and 0 <= j < GRID_H:
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

    def get_local_explored_cells(self):
        """返回该agent已探索（非UNKNOWN）的格子集合（i,j）"""
        inds = np.where(self.local_map != UNKNOWN)
        return set(zip(inds[0].tolist(), inds[1].tolist()))

    def step_motion(self, desired_vx, desired_vy, dt, env):
        """基于期望速度更新位置并处理障碍碰撞、边界、危险区检测"""
        if not self.alive:
            return
        # clamp speed length
        speed = math.hypot(desired_vx, desired_vy)
        if speed > AGENT_MAX_SPEED:
            scale = AGENT_MAX_SPEED / (speed + 1e-9)
            desired_vx *= scale
            desired_vy *= scale
        new_x = self.pos[0] + desired_vx * dt
        new_y = self.pos[1] + desired_vy * dt

        new_pos = (clamp(new_x, 2, WORLD_W-2), clamp(new_y, 2, WORLD_H-2))

        # obstacle collision simple handling: if new point inside obstacle, attempt axis-projection
        collided = False
        for obs in env.obstacles:
            if obs.rect.collidepoint(new_pos):
                collided = True
                alt1 = (new_pos[0], self.pos[1])
                alt2 = (self.pos[0], new_pos[1])
                if not obs.rect.collidepoint(alt1):
                    new_pos = alt1
                elif not obs.rect.collidepoint(alt2):
                    new_pos = alt2
                else:
                    # stay
                    new_pos = self.pos
                break

        self.pos = new_pos
        self.vel = (desired_vx, desired_vy)
        self.hist.append(self.pos)

        # check danger
        for dz in env.danger_zones:
            if dz.contains(self.pos):
                self.alive = False
                print(f"[{time.time():.2f}] Agent {self.id} 被危险区摧毁 at {self.pos}")
                break

        # check victim rescue
        if distance(self.pos, env.victim.pos) <= GRID_CELL and not env.victim.rescued:
            env.victim.rescued = True
            print(f"[{time.time():.2f}] Agent {self.id} 找到并救援了被困者！")

    def send_map_patch(self, comms, targets, now_time, radius=None):
        """向 targets 发送本地地图的补丁（稀疏表示）"""
        # pack as list of (i,j,val) for explored cells near agent to limit bandwidth
        explored = self.get_local_explored_cells()
        patch = []
        # limit patch to cells within a radius (in cells)
        if radius is None:
            radius = int(self.sensor_range // GRID_CELL) + 2
        ci, cj = cell_of_pos(self.pos)
        for (i,j) in list(explored):
            if abs(i-ci) <= radius and abs(j-cj) <= radius:
                patch.append((i,j,int(self.local_map[i,j])))
        msg = {'type':'map_patch', 'from':self.id, 'patch':patch, 'pos':self.pos}
        for t in targets:
            comms.send(self, t, msg, now_time)

    def draw_hist(self, screen, color=(100,100,255)):
        if len(self.hist) >= 2:
            pts = [(int(p[0]), int(p[1])) for p in self.hist]
            pygame.draw.lines(screen, color, False, pts, 1)

    def draw_self(self, screen, color=(40,120,220)):
        if not self.alive:
            pygame.draw.circle(screen, (90,90,90), (int(self.pos[0]), int(self.pos[1])), AGENT_RADIUS)
            x,y = int(self.pos[0]), int(self.pos[1])
            pygame.draw.line(screen, (160,160,160), (x-4,y-4),(x+4,y+4),2)
            pygame.draw.line(screen, (160,160,160), (x-4,y+4),(x+4,y-4),2)
        else:
            pygame.draw.circle(screen, color, (int(self.pos[0]), int(self.pos[1])), AGENT_RADIUS)
            # sensor range (subtle)
            surf = pygame.Surface((self.sensor_range*2, self.sensor_range*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (50,50,50,8), (int(self.sensor_range), int(self.sensor_range)), int(self.sensor_range))
            screen.blit(surf, (int(self.pos[0]-self.sensor_range), int(self.pos[1]-self.sensor_range)))

class LargeAgent(AgentBase):
    def __init__(self, id_, x, y):
        super().__init__(id_, x, y, sensor_range=SENSOR_LARGE, is_large=True)
        self.behavior = ExploreBehavior()
        self.last_reason_time = time.time()
        self.known_map = np.full((GRID_W, GRID_H), UNKNOWN, dtype=np.int8)  # 脑节点的地图副本
        self.assigned = {}  # agent_id -> waypoint

    def request_map_patch(self, comms, agent, now_time):
        pass

    def integrate_map_patch(self, patch):
        """将收到的patch应用到自己的known_map"""
        for (i,j,val) in patch:
            # val is occupancy code
            if 0 <= i < GRID_W and 0 <= j < GRID_H:
                # overwrite unknowns or keep obstacle/danger priority
                if self.known_map[i,j] == UNKNOWN and val != UNKNOWN:
                    self.known_map[i,j] = val
                else:
                    # 若已有UNKNOWN则覆盖，否则保持原先（或根据优先级更新）
                    # 优先级： DANGER/OBSTACLE > VICTIM > FREE
                    cur = self.known_map[i,j]
                    if val == DANGER or val == OBSTACLE:
                        self.known_map[i,j] = val
                    elif val == VICTIM:
                        self.known_map[i,j] = val
                    elif cur == UNKNOWN:
                        self.known_map[i,j] = val
    
    def get_local_explored_cells(self):
        """返回该agent已探索（非UNKNOWN）的格子集合（i,j）"""
        inds = np.where(self.known_map != UNKNOWN)
        return set(zip(inds[0].tolist(), inds[1].tolist()))
    
    def send_map_patch(self, comms, targets, now_time, radius=None):
        """向 targets 发送本地地图的补丁（稀疏表示）"""
        # pack as list of (i,j,val) for explored cells near agent to limit bandwidth
        explored = self.get_local_explored_cells()
        patch = []
        # limit patch to cells within a radius (in cells)
        if radius is None:
            radius = int(self.sensor_range // GRID_CELL) + 2
        ci, cj = cell_of_pos(self.pos)
        for (i,j) in list(explored):
            if abs(i-ci) <= radius and abs(j-cj) <= radius:
                patch.append((i,j,int(self.local_map[i,j])))
        msg = {'type':'map_patch', 'from':self.id, 'patch':patch, 'pos':self.pos}
        for t in targets:
            comms.send(self, t, msg, now_time)

    def fuse_own_sensing(self):
        """将自己感知到的地图写入 known_map"""
        inds = np.where(self.local_map != UNKNOWN)
        for i,j in zip(inds[0].tolist(), inds[1].tolist()):
            self.known_map[i,j] = self.local_map[i,j]

    def find_frontiers(self):
        """非常简单的frontier: 找known_map中 UNKNOWN 邻接 FREE 的格子"""
        frontiers = []
        for i in range(1, GRID_W-1):
            for j in range(1, GRID_H-1):
                if self.known_map[i,j] == UNKNOWN:
                    # check neighbors for known free
                    neigh_known = False
                    for di in (-1,0,1):
                        for dj in (-1,0,1):
                            if di==0 and dj==0: continue
                            ni, nj = i+di, j+dj
                            if 0<=ni<GRID_W and 0<=nj<GRID_H:
                                if self.known_map[ni,nj] != UNKNOWN and self.known_map[ni,nj] != OBSTACLE:
                                    neigh_known = True
                                    break
                        if neigh_known: break
                    if neigh_known:
                        frontiers.append((i,j))
        random.shuffle(frontiers)
        return frontiers

    def assign_frontiers(self, agents):
        frontiers = self.find_frontiers()
        if not frontiers:
            return
        free_frontiers = frontiers.copy()
        for a in agents:
            if not a.alive: continue
            # only assign if agent within comm range
            if distance(self.pos, a.pos) <= AGENT_COMM_RANGE:
                # find nearest frontier
                best = None
                bestd = 1e9
                for f in free_frontiers:
                    fx, fy = pos_of_cell(f[0], f[1])
                    d = distance(a.pos, (fx, fy))
                    if d < bestd:
                        best = (fx, fy)
                        bestd = d
                if best:
                    a.has_goal = True
                    a.goal = best
                    self.assigned[a.id] = best
                    try:
                        free_frontiers.remove((int(best[0]//GRID_CELL), int(best[1]//GRID_CELL)))
                    except ValueError:
                        pass

    def reason_and_assign(self, agents, now_time):
        if now_time - self.last_reason_time < BRAIN_REASON_INTERVAL:
            return
        self.last_reason_time = now_time
        # fuse own sensing
        self.fuse_own_sensing()
        # assign frontiers
        self.assign_frontiers(agents)

    def draw_self(self, screen):
        # draw communication range
        x,y = int(self.pos[0]), int(self.pos[1])
        pygame.draw.circle(screen, (200,160,60), (x,y), LARGE_RADIUS)
        surf = pygame.Surface((AGENT_COMM_RANGE*2, AGENT_COMM_RANGE*2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (200,160,60,30), (AGENT_COMM_RANGE, AGENT_COMM_RANGE), AGENT_COMM_RANGE)
        screen.blit(surf, (x-AGENT_COMM_RANGE, y-AGENT_COMM_RANGE))

# -----------------------------
# 环境（World）类
# -----------------------------
# -----------------------------
# 环境（World）类
# -----------------------------
class World:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.width = WORLD_W
        self.height = WORLD_H
        self.time = 0.0
        self.obstacles = []
        self.danger_zones = []
        self.agents = []
        self.large_agents = []
        self.victim = None
        self.grid_visited_union = set()

        # 初始化地面栅格
        self.ground_grid = np.full((GRID_W, GRID_H), FREE, dtype=np.int8)

        # 环境生成顺序
        self.generate_static_obstacles(NUM_OBSTACLES)
        self.generate_danger_zones(NUM_DANGER_ZONES)
        self.spawn_large_agents(NUM_LARGE)
        self.spawn_agents(NUM_AGENTS)
        self.place_victim()   # victim放在最后，以确保存在brain node

    # ========== 基础工具函数 ==========
    def is_in_obstacle(self, x, y):
        """判断坐标是否在障碍物内部"""
        for obs in self.obstacles:
            if obs.rect.collidepoint(x, y):
                return True
        return False

    def is_in_danger(self, x, y):
        """判断坐标是否在危险区内"""
        for dz in self.danger_zones:
            if distance((x, y), dz.pos) < dz.r:
                return True
        return False

    def random_free_pos(self, margin=30):
        for _ in range(5000):
            x = random.uniform(margin, WORLD_W - margin)
            y = random.uniform(margin, WORLD_H - margin)
            if self.is_in_obstacle(x, y) or self.is_in_danger(x, y):
                continue
            return (x, y)
        return (WORLD_W / 2, WORLD_H / 2)

    # ========== 环境生成 ==========
    def generate_static_obstacles(self, n=NUM_OBSTACLES):
        # whatever
        for _ in range(n):
            w = random.randint(30, 120)
            h = random.randint(20, 120)
            x = random.uniform(0, WORLD_W - w)
            y = random.uniform(0, WORLD_H - h)
            obs = Obstacle(x, y, w, h)
            self.obstacles.append(obs)
            # 更新地面占据网格
            left = int(x // GRID_CELL)
            top = int(y // GRID_CELL)
            right = int((x + w) // GRID_CELL)
            bottom = int((y + h) // GRID_CELL)
            for i in range(left, min(GRID_W, right + 1)):
                for j in range(top, min(GRID_H, bottom + 1)):
                    self.ground_grid[i, j] = OBSTACLE

    def generate_danger_zones(self, num=NUM_DANGER_ZONES, max_attempts=200):
        # no overlap with obstacles or existing danger zones
        for _ in range(num):
            attempts = 0
            while attempts < max_attempts:
                x = random.randint(40, WORLD_W - 40)
                y = random.randint(40, WORLD_H - 40)
                r = random.randint(15, 60)
                # 检查与障碍物和已有危险区的重叠
                overlap = False
                for obs in self.obstacles:
                    cx = clamp(x, obs.rect.left, obs.rect.right)
                    cy = clamp(y, obs.rect.top, obs.rect.bottom)
                    if math.hypot(cx - x, cy - y) < r + max(obs.width, obs.height) * 0.5:
                        overlap = True
                        break
                for dz in self.danger_zones:
                    if math.hypot(dz.pos[0] - x, dz.pos[1] - y) < (dz.r + r) * 0.8:
                        overlap = True
                        break
                if not overlap:
                    self.danger_zones.append(DangerZone(x, y, r))
                    # 更新地面栅格
                    left = max(0, int((x - r) // GRID_CELL))
                    right = min(GRID_W - 1, int((x + r) // GRID_CELL))
                    top = max(0, int((y - r) // GRID_CELL))
                    bottom = min(GRID_H - 1, int((y + r) // GRID_CELL))
                    for i in range(left, right + 1):
                        for j in range(top, bottom + 1):
                            gx, gy = pos_of_cell(i, j)
                            if math.hypot(gx - x, gy - y) <= r:
                                self.ground_grid[i, j] = DANGER
                    break
                attempts += 1

    def spawn_large_agents(self, num=2):
        # free space
        self.large_agents = []
        base_pos = None
        attempts = 0

        while attempts < 20:
            # 随机一个基础点
            x, y = self.random_free_pos(margin=60)
            if not self.is_in_obstacle(x, y) and not self.is_in_danger(x, y):
                base_pos = (x, y)
                break
            attempts += 1
        if base_pos is None:
            # fallback
            base_pos = (WORLD_W/2, WORLD_H/2)
            print("Warning: failed to find suitable base pos for large agents, using center.")

        # 生成多个相互接近的大节点
        for i in range(num):
            for _ in range(300):
                angle = random.uniform(0, 2 * math.pi)
                r = random.uniform(0, 30)
                x = base_pos[0] + r * math.cos(angle)
                y = base_pos[1] + r * math.sin(angle)

                # 边界约束
                x = clamp(x, 0, WORLD_W)
                y = clamp(y, 0, WORLD_H)

                if not self.is_in_obstacle(x, y) and not self.is_in_danger(x, y):
                    self.large_agents.append(LargeAgent(i, x, y))
                    break
            else:
                # 如果找不到合适位置，则直接使用base_pos附近
                self.large_agents.append(LargeAgent(i, base_pos[0] + random.uniform(-10, 10),
                                                    base_pos[1] + random.uniform(-10, 10)))


    def spawn_agents(self, num=5):
        # free space
        # near large agents
        if not self.large_agents:
            raise ValueError("No large agents available to spawn agents around.")

        self.agents = []
        for i in range(num):
            attempts = 0
            while attempts < 100:  # 最多尝试 100 次
                # 随机选择一个大节点
                large_agent = random.choice(self.large_agents)
                # 在大节点周围生成随机偏移
                offset_x = random.uniform(-50, 50)  # 偏移范围可根据需求调整
                offset_y = random.uniform(-50, 50)
                x = clamp(large_agent.pos[0] + offset_x, 0, WORLD_W)
                y = clamp(large_agent.pos[1] + offset_y, 0, WORLD_H)
                # 检查是否在自由空间中
                if not self.is_in_obstacle(x, y) and not self.is_in_danger(x, y):
                    # 创建小节点
                    self.agents.append(AgentBase(i + 1000, x, y))
                    break
                attempts += 1
            else:
                print(f"Failed to generate agent {i} in free space after 100 attempts.")

    def place_victim(self):
        """生成距离大脑节点最远且A*可达的victim"""
        if not self.large_agents:
            print("Warning: No brain nodes, skipping victim placement.")
            return None

        brain_cells = [cell_of_pos((b.pos[0], b.pos[1])) for b in self.large_agents]
        best_point, best_dist = None, -1

        # 随机搜索远点并用A*验证
        for _ in range(10):
            x = random.randint(50, SCREEN_W - 50)
            y = random.randint(50, SCREEN_H - 50)
            cell = cell_of_pos((x, y))

            # 检查是否在自由空间中
            if self.ground_grid[cell[0], cell[1]] != FREE:
                continue

            for bc in brain_cells:
                path = self.astar(self.ground_grid, bc, cell)
                if path is not None:
                    dist = math.hypot(bc[0] - cell[0], bc[1] - cell[1])
                    if dist > best_dist:
                        best_point, best_dist = (x, y), dist
                    break

        if best_point is None:
            best_point = self.random_free_pos()

        self.victim = Victim(best_point[0], best_point[1])
        self.ground_grid[best_point[0]//GRID_CELL, best_point[1]//GRID_CELL] = VICTIM
        return self.victim

    def astar(self, grid, start, goal):
        """基于栅格地图的A*"""
        w, h = len(grid), len(grid[0])
        open_set = [(0, start)]
        came_from = {}
        gscore = {start: 0}
        fscore = {start: abs(start[0] - goal[0]) + abs(start[1] - goal[1])}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return True  # 存在路径即可
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < w and 0 <= ny < h and grid[nx][ny] == FREE:
                    ng = gscore[current] + 1
                    if (nx, ny) not in gscore or ng < gscore[(nx, ny)]:
                        came_from[(nx, ny)] = current
                        gscore[(nx, ny)] = ng
                        fscore[(nx, ny)] = ng + abs(nx - goal[0]) + abs(ny - goal[1])
                        heapq.heappush(open_set, (fscore[(nx, ny)], (nx, ny)))
        return None

    def update(self, dt, comms:Communication, now_time):
        self.time += dt
        # 1) 更新每个代理的感知信息 -> 更新 local_map
        for a in self.agents + self.large_agents:
            if not a.alive:
                continue
            try:
                a.update_local_map_from_sensing(self)
                if a.is_large:
                    a.fuse_own_sensing()
            except Exception as e:
                print(f"Error updating local map for agent {a.id}: {e}")

        # 2) deliver communications queued
        # TODO: multi-step communication process
        """
        1. 脑节点给大节点分配任务组
        2. 大节点给小节点分配逐个任务
        3. 小节点发送地图补丁给大节点
        4. 大节点给脑节点发消息补全信息
        5. 小节点发现victim后发送救援警报给大节点和脑节点
        """
        deliveries = comms.deliver(now_time)
        for sender, receiver, msg in deliveries:
            # msg dispatch
            if msg.get('type') == 'map_patch':
                # receiver should integrate patch
                receiver.integrate_map_patch(msg['patch'])
            elif msg.get('type') == 'rescue_alert':
                # receiver may prioritize moving to victim
                if isinstance(receiver, AgentBase):
                    receiver.has_goal = True
                    receiver.goal = msg['pos']

        # 3) Large agents perform reasoning (System 2)
        # for la in self.large_agents:
        #     if not la.alive:
        #         continue
        #     try:
        #         # 限制推理频率
        #         if now_time - la.last_reason_time >= BRAIN_REASON_INTERVAL:
        #             la.reason_and_assign(self.agents, now_time)
        #             la.last_reason_time = now_time

        #         # 主动请求附近小节点的地图补丁
        #         for a in self.agents:
        #             if a.alive and distance(la.pos, a.pos) <= AGENT_COMM_RANGE:
        #                 la.request_map_patch(comms, a, now_time)
        #     except Exception as e:
        #         print(f"Error in reasoning or task assignment for large agent {la.id}: {e}")

        # 4) Agents decide & move
        for a in self.agents + self.large_agents:
            if not a.alive:
                continue
            try:
                sense = a.sense(self)
                desired_vx, desired_vy = a.behavior.decide(a, sense, dt)
                # 限制速度范围
                speed = math.hypot(desired_vx, desired_vy)
                if speed > AGENT_MAX_SPEED:
                    scale = AGENT_MAX_SPEED / (speed + 1e-9)
                    desired_vx *= scale
                    desired_vy *= scale
                a.step_motion(desired_vx, desired_vy, dt, self)

                # 更新已访问的网格单元
                ci, cj = cell_of_pos(a.pos)
                self.grid_visited_union.add((ci, cj))
            except Exception as e:
                print(f"Error in decision or motion for agent {a.id}: {e}")

        # 5) periodic communications: small agents send map patches to large agents if within range
        for a in self.agents + self.large_agents:
            if not a.alive:
                continue
            try:
                for aa in self.agents + self.large_agents:
                    if not aa.alive or aa.id == a.id:
                        continue
                    if distance(a.pos, aa.pos) <= AGENT_COMM_RANGE:
                        a.send_map_patch(comms, [aa], now_time)
            except Exception as e:
                print(f"Error in periodic communication for agent {a.id}: {e}")

    def draw(self, screen):
        # draw explored overlay: union of all agents' explored cells => white, else dark gray
        explored_union = np.full((GRID_W, GRID_H), False, dtype=bool)
        for a in (self.agents + self.large_agents):
            explored = (a.local_map != UNKNOWN)
            explored_union = np.logical_or(explored_union, explored)

        # draw cells
        for i in range(GRID_W):
            for j in range(GRID_H):
                rect = pygame.Rect(i*GRID_CELL, j*GRID_CELL, GRID_CELL, GRID_CELL)
                if not explored_union[i,j]:
                    color = (60,60,60)  # unknown gray
                else:
                    color = (245,245,245)  # explored white
                pygame.draw.rect(screen, color, rect)

        # draw obstacles (on top)
        for obs in self.obstacles:
            obs.draw(screen)
        # draw danger zones
        for dz in self.danger_zones:
            dz.draw(screen)
        # draw victim
        if self.victim:
            self.victim.draw(screen)
        # draw agents' traces and bodies
        for la in self.large_agents:
            la.draw_hist(screen, color=(200,160,60))
            la.draw_self(screen)
        for a in self.agents:
            a.draw_hist(screen)
            a.draw_self(screen)

    def coverage_percentage(self):
        total = GRID_W * GRID_H
        # count union of explored cells
        explored = 0
        union_set = set()
        for a in (self.agents + self.large_agents):
            union_set |= a.get_local_explored_cells()
        explored = len(union_set)
        return (explored / total) * 100.0
# -----------------------------
# 主循环与UI
# -----------------------------
def main():
    global SEED
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
    if VISUALIZE:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("exploration simulation")
        clock = pygame.time.Clock()
        font = load_font()

    # communication system
    comms = Communication(packet_loss=COMM_PACKET_LOSS, delay=COMM_DELAY)

    world = World(seed=SEED)

    running = True
    paused = False
    sim_time = 0.0

    while running:
        dt = clock.tick(FPS) / 1000.0
        sim_time += 0 if paused else dt
        now_time = time.time()

        # 事件处理（仅在可视化模式下启用）
        if VISUALIZE:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
        if not paused:
            world.update(dt, comms, now_time)

        # 绘制（仅在可视化模式下启用）
        if VISUALIZE:
            world.draw(screen)

            # HUD
            alive = sum(1 for a in world.agents if a.alive) + sum(1 for la in world.large_agents if la.alive)
            coverage = world.coverage_percentage()
            hud1 = f"Time: {sim_time:.1f}s  Alive agents: {alive}/{len(world.agents)+len(world.large_agents)}  Coverage: {coverage:.2f}%"
            hud2 = f"Obstacles: {len(world.obstacles)}  Dangers: {len(world.danger_zones)}  Victim: {'rescued' if world.victim.rescued else 'missing'}"
            screen.blit(font.render(hud1, True, (10, 10, 10)), (8, 6))
            screen.blit(font.render(hud2, True, (10, 10, 10)), (8, 24))

            # legend
            pygame.draw.circle(screen, (40, 120, 220), (SCREEN_W - 120, 30), AGENT_RADIUS)
            screen.blit(font.render("Small Agent", True, (0, 0, 0)), (SCREEN_W - 96, 22))
            pygame.draw.circle(screen, (200, 160, 60), (SCREEN_W - 120, 60), LARGE_RADIUS)
            screen.blit(font.render("Large Agent", True, (0, 0, 0)), (SCREEN_W - 96, 52))
            pygame.draw.circle(screen, (200, 40, 40), (SCREEN_W - 120, 90), 8)
            screen.blit(font.render("Danger", True, (0, 0, 0)), (SCREEN_W - 96, 82))

            pygame.display.flip()

        # end conditions
        if world.victim.rescued and coverage >= 50.0:
            print("Mission success: victim rescued and coverage >= 50%")
            paused = True
            running = False
        if sum(1 for a in world.agents if a.alive) == 0:
            print("All small agents destroyed. Mission failed.")
            paused = True
            running = False
        if sim_time > 300.0:
            print("Max sim time reached.")
            paused = True
            running = False

    # summary
    print("\n=== Simulation summary ===")
    print(f"Sim time: {sim_time:.2f}s")
    alive_small = sum(1 for a in world.agents if a.alive)
    alive_large = sum(1 for la in world.large_agents if la.alive)
    print(f"Alive: small={alive_small}/{len(world.agents)}, large={alive_large}/{len(world.large_agents)}")
    print(f"Coverage: {world.coverage_percentage():.2f}%")
    print(f"Victim rescued: {world.victim.rescued}")
    # Agent trajectories
    for a in world.agents:
        print(f"Agent {a.id} alive={a.alive} traj_len={len(a.hist)} last={a.pos}")
    for la in world.large_agents:
        print(f"LargeAgent {la.id} alive={la.alive} known_cells={np.sum(la.known_map != UNKNOWN)}")

    if VISUALIZE:
        pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
