import pygame
import random
import math
import time
import numpy as np

from behaviors import *
from utils import *
# -----------------------------
# Agent（小节点）类
# -----------------------------

class AgentBase:
    def __init__(self, id_, x, y, sensor_range=SENSOR_SMALL, is_large=False, behavior=None):
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
        self.father_id = None  # LargeAgent id
        self.son_ids = []  # assigned small Agent ids
        self.hist = [self.pos]
        self.is_large = is_large  # LargeAgent flag
        # default behavior
        if behavior is not None:
            self.behavior = behavior
        else:
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

    def draw_goal(self, screen):
        if not self.has_goal or self.goal is None:
            return
        """绘制目标点及其连接线"""
        goal_x, goal_y = int(self.goal[0]), int(self.goal[1])
        
        # 1. 绘制从机器人到目标点的连线
        pygame.draw.line(screen, (200, 200, 50), 
                         (int(self.pos[0]), int(self.pos[1])), 
                         (goal_x, goal_y), 2)
        
        # 2. 绘制目标点标记（不同形状表示不同类型的目标）
        if self.is_large:
            # 大型机器人目标点：带圆圈的十字
            pygame.draw.circle(screen, self.current_goal_color, (goal_x, goal_y), 8, 2)
            pygame.draw.line(screen, self.current_goal_color, (goal_x-6, goal_y), (goal_x+6, goal_y), 2)
            pygame.draw.line(screen, self.current_goal_color, (goal_x, goal_y-6), (goal_x, goal_y+6), 2)
        else:
            # 小型机器人目标点：实心三角形
            points = [
                (goal_x, goal_y-8),
                (goal_x-6, goal_y+6),
                (goal_x+6, goal_y+6)
            ]
            pygame.draw.polygon(screen, (200, 200, 50), points)
        
        # 3. 绘制距离文本（可选）
        dist = distance(self.pos, self.goal)
        if dist > 50:  # 只在距离较远时显示距离，避免遮挡
            font = pygame.font.SysFont('Arial', 10)
            dist_text = f"{int(dist)}px"
            text_surf = font.render(dist_text, True, (200, 200, 50))
            # 在连线中点显示距离
            mid_x = (self.pos[0] + goal_x) // 2
            mid_y = (self.pos[1] + goal_y) // 2
            screen.blit(text_surf, (mid_x, mid_y))

class LargeAgent(AgentBase):
    def __init__(self, id_, x, y, is_brain=False, behavior=None ,multi_behavior=None):
        super().__init__(id_, x, y, sensor_range=SENSOR_LARGE, is_large=True, behavior=behavior)
        self.last_reason_time = time.time()
        self.known_map = np.full((GRID_W, GRID_H), UNKNOWN, dtype=np.int8)  # 脑节点的地图副本
        self.assigned = {}  # agent_id -> waypoint
        self.is_brain = is_brain  # LargeAgent作为脑节点
        if multi_behavior is not None:
            self.multi_behavior = multi_behavior
        else:
            self.multi_behavior = easyFrontierAssignmentBehavior()

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

    def fuse_own_sensing(self):
        """将自己感知到的地图写入 known_map"""
        inds = np.where(self.local_map != UNKNOWN)
        for i,j in zip(inds[0].tolist(), inds[1].tolist()):
            self.known_map[i,j] = self.local_map[i,j]

    def reason_and_assign(self, agents, now_time):
        if now_time - self.last_reason_time < BRAIN_REASON_INTERVAL:
            return
        self.last_reason_time = now_time
        # fuse own sensing
        self.fuse_own_sensing()
        # assign frontiers
        assigns = self.multi_behavior.decide(self, agents)
        return assigns

    def draw_self(self, screen):
        # draw communication range
        x,y = int(self.pos[0]), int(self.pos[1])
        pygame.draw.circle(screen, (200,160,60), (x,y), LARGE_RADIUS)
        surf = pygame.Surface((AGENT_COMM_RANGE*2, AGENT_COMM_RANGE*2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (200,160,60,30), (AGENT_COMM_RANGE, AGENT_COMM_RANGE), AGENT_COMM_RANGE)
        screen.blit(surf, (x-AGENT_COMM_RANGE, y-AGENT_COMM_RANGE))
