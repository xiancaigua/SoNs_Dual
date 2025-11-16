import pygame
import random
import math
import time
import numpy as np

from behaviors import *
from utils import *
# -----------------------------
# Agent（小节点）类
# 全知的AgentBase，LargeAgent继承自此类
# -----------------------------

class oldAgentBase:
    def __init__(self, id_, x, y, sensor_range=SENSOR_SMALL, is_large=False, behavior=None):
        self.id = id_
        self.pos = (x, y)
        angle = random.random()*2*math.pi
        self.vel = (math.cos(angle)*10.0, math.sin(angle)*10.0)  # px/sec initial drift
        self.alive = True
        self.sensor_range = sensor_range
        self.local_map = np.full((GRID_H, GRID_W), UNKNOWN, dtype=np.int8)
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
        cx, cy = int(self.pos[0]), int(self.pos[1])
        r = int(self.sensor_range)
        min_i = max(0, (cx - r)//GRID_CELL)
        max_i = min(GRID_W-1, (cx + r)//GRID_CELL)
        min_j = max(0, (cy - r)//GRID_CELL)
        max_j = min(GRID_H-1, (cy + r)//GRID_CELL)
        for i in range(min_i, max_i+1):
            for j in range(min_j, max_j+1):
                gx, gy = pos_of_cell(i, j)
                if self.local_map[j, i] != UNKNOWN: # 只更新未知区域
                    continue
                if math.hypot(gx - cx, gy - cy) <= r:
                    val = env.ground_grid[j, i]
                    # ✅ 不写入危险区，只写障碍物或可通行
                    if val == OBSTACLE:
                        self.local_map[j, i] = OBSTACLE
                    elif val == FREE or val == VICTIM:
                        self.local_map[j, i] = val
                    elif val == DANGER:
                        # 如果是 DANGER → 留FREE，没法成功探测出来
                        self.local_map[j, i] = FREE

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
            pygame.draw.circle(screen, (100, 120, 10), (goal_x, goal_y), 8, 2)
            pygame.draw.line(screen, (100, 120, 10), (goal_x-6, goal_y), (goal_x+6, goal_y), 2)
            pygame.draw.line(screen, (100, 120, 10), (goal_x, goal_y-6), (goal_x, goal_y+6), 2)
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

# ===============================
# Agent（小节点）类
# ===============================
class AgentBase:
    def __init__(self, id_, x, y, sensor_range=SENSOR_SMALL, is_large=False, behavior=None):
        self.id = id_
        self.pos = (x, y)
        angle = random.random() * 2 * math.pi
        self.vel = (math.cos(angle)*10.0, math.sin(angle)*10.0)
        self.alive = True
        self.sensor_range = sensor_range
        self.local_map = np.full((GRID_H, GRID_W), UNKNOWN, dtype=np.int8)
        self.has_goal = False
        self.goal = None
        self.comm_ok = True
        self.father_id = None
        self.son_ids = []
        self.hist = [self.pos]
        self.is_large = is_large
        self.death_prob = 0.8  # ✅ 死亡概率参数（可调）

        if behavior is not None:
            self.behavior = behavior
        else:
            self.behavior = ReactiveBehavior() if not is_large else ExploreBehavior()

    # =====================================================
    # 感知函数：不再能直接探测到危险区
    # =====================================================
    def sense(self, env):
        near_obstacles = []
        victim_seen = False

        # 感知障碍物
        for obs in env.obstacles:
            cx = clamp(self.pos[0], obs.rect.left, obs.rect.right)
            cy = clamp(self.pos[1], obs.rect.top, obs.rect.bottom)
            d = distance((cx, cy), self.pos)
            if d <= self.sensor_range:
                near_obstacles.append(((cx, cy), d))

        # ✅ 不再感知 env.danger_zones
        # 机器人不知道危险区，只能事后推断

        # 感知victim
        if distance(self.pos, env.victim.pos) <= self.sensor_range:
            victim_seen = True

        return {'near_obstacles': near_obstacles, 'victim_seen': victim_seen}

    # =====================================================
    # 基于真实世界更新local_map（不包含危险区信息）
    # =====================================================
    def update_local_map_from_sensing(self, env):
        cx, cy = int(self.pos[0]), int(self.pos[1])
        r = int(self.sensor_range)
        min_i = max(0, (cx - r)//GRID_CELL)
        max_i = min(GRID_W-1, (cx + r)//GRID_CELL)
        min_j = max(0, (cy - r)//GRID_CELL)
        max_j = min(GRID_H-1, (cy + r)//GRID_CELL)
        for i in range(min_i, max_i+1):
            for j in range(min_j, max_j+1):
                gx, gy = pos_of_cell(i, j)
                if self.local_map[j, i] != UNKNOWN: # 只更新未知区域
                    continue
                if math.hypot(gx - cx, gy - cy) <= r:
                    val = env.ground_grid[j, i]
                    # ✅ 不写入危险区，只写障碍物或可通行
                    if val == OBSTACLE:
                        self.local_map[j, i] = OBSTACLE
                    elif val == FREE or val == VICTIM:
                        self.local_map[j, i] = val
                    elif val == DANGER:
                        # 如果是 DANGER → 留FREE，没法成功探测出来
                        self.local_map[j, i] = FREE

    def get_local_explored_cells(self):
        """返回该agent已探索（非UNKNOWN）的格子集合（i,j）"""
        inds = np.where(self.local_map != UNKNOWN)
        return set(zip(inds[0].tolist(), inds[1].tolist()))

    # =====================================================
    # 移动函数 + 概率死亡机制 + 死亡通报
    # =====================================================
    def step_motion(self, desired_vx, desired_vy, dt, env):
        if not self.alive:
            return

        # 限速
        speed = math.hypot(desired_vx, desired_vy)
        if speed > AGENT_MAX_SPEED:
            scale = AGENT_MAX_SPEED / (speed + 1e-9)
            desired_vx *= scale
            desired_vy *= scale

        parent = env.find_agent_by_id(self.father_id)
        if parent is not None and parent.alive:
            px, py = parent.pos
            cx, cy = self.pos
            parent_vec = np.array([px - cx, py - cy])
            dist_to_parent = np.linalg.norm(parent_vec)

            # 如果距离超过安全范围，加入吸引力（让它更靠近父节点）
            if dist_to_parent > AGENT_COMM_RANGE * 0.8:
                dir_to_parent = parent_vec / (dist_to_parent + 1e-6)
                # 吸引分量，越远吸引越强
                attraction = 0.25 * (dist_to_parent - AGENT_COMM_RANGE * 0.8)
                desired_vx += attraction * dir_to_parent[0]
                desired_vy += attraction * dir_to_parent[1]

            # 如果超出最大距离 -> 强制回归（忽略原目标）
            if dist_to_parent > AGENT_COMM_RANGE * 1.2:
                dir_to_parent = parent_vec / (dist_to_parent + 1e-6)
                desired_vx = AGENT_MAX_SPEED * dir_to_parent[0]
                desired_vy = AGENT_MAX_SPEED * dir_to_parent[1]
                # 可以标记状态（进入“reconnect模式”）
                self.state = "RETURN_TO_PARENT"
            else:
                self.state = "NORMAL"

        new_x = clamp(self.pos[0] + desired_vx * dt, 2, WORLD_W-2)
        new_y = clamp(self.pos[1] + desired_vy * dt, 2, WORLD_H-2)
        new_pos = (new_x, new_y)

        # 碰撞检测
        for obs in env.obstacles:
            if obs.rect.collidepoint(new_pos):
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

        # ✅ 危险区概率死亡机制
        for dz in env.danger_zones:
            if dz.contains(self.pos):
                if random.random() < self.death_prob:  # 例如 40% 几率死亡
                    self.alive = False
                    print(f"[{time.time():.2f}] ⚠ Agent {self.id} 在危险区死亡 at {self.pos}")
                    # 向附近机器人广播死亡警报
                    self.broadcast_death_alert(env)
                    return

        # 救援检测
        if distance(self.pos, env.victim.pos) <= GRID_CELL and not env.victim.rescued:
            env.victim.rescued = True
            print(f"[{time.time():.2f}] ✅ Agent {self.id} 找到并救援了被困者！")

    # =====================================================
    # 死亡消息广播
    # =====================================================
    def broadcast_death_alert(self, env):
        """机器人死亡后通报其位置，使其他机器人认为此区域危险"""
        if self.is_large:
            for other in env.large_agents:
                other.receive_death_alert(self.pos)
            for other in env.agents:
                if other.alive and distance(self.pos, other.pos) < AGENT_COMM_RANGE:
                    other.receive_death_alert(self.pos)
        else:
            for other in env.agents + env.large_agents:
                if other.alive and distance(self.pos, other.pos) < AGENT_COMM_RANGE:
                    other.receive_death_alert(self.pos)

    # =====================================================
    # 接收死亡通报
    # =====================================================
    def receive_death_alert(self, danger_pos):
        """接收同伴死亡信息后，在local_map上划定经验危险区"""
        cx, cy = cell_of_pos(danger_pos)
        r = 15  # 经验危险半径（格）
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    if math.hypot(dx, dy) <= r:
                        self.local_map[ny, nx] = DANGER  # ✅ 主观危险区域

    # ------------------------------
    # 辅助：小节点/大节点的默认行为扩展
    # ------------------------------
    def _agent_accept_and_adjust_goal(agent, world, search_radius_cells=3):
        """
        接收目标点后在本地地图附近寻找最近的安全格（不在 OBSTACLE）
        返回 True if adjusted/accepted, False otherwise
        """
        if agent.goal is None:
            return False
        gx, gy = agent.goal
        gi, gj = cell_of_pos((gx, gy))
        H, W = agent.local_map.shape
        # BFS-like search around target cell for FREE or VICTIM (note agent.local_map uses j,i indexing)
        best = None
        for r in range(0, search_radius_cells+1):
            for di in range(-r, r+1):
                for dj in range(-r, r+1):
                    i = gi + di
                    j = gj + dj
                    if 0 <= i < W and 0 <= j < H:
                        val = agent.local_map[j, i]
                        if val == FREE or val == VICTIM:
                            # convert back to world pos
                            px, py = pos_of_cell(i, j)
                            # safety check vs known brain map/danger if available
                            best = (px, py)
                            return_agent_goal = best
                            agent.goal = best
                            agent.has_goal = True
                            return True
        # cannot find safe local point
        return False

    def _agent_execute_retreat(agent, world, retreat_distance=30.0):
        """
        简单避险：沿速度反向或指向所属大节点退回
        """
        # choose vector away from nearest danger center if known (brain)
        # fallback: reverse current velocity
        if (agent.vel[0] != 0 or agent.vel[1] != 0):
            vx, vy = agent.vel
            nx, ny = -vx, -vy
        else:
            nx, ny = random.uniform(-1,1), random.uniform(-1,1)

        nx, ny = normalize((nx, ny))
        new_x = clamp(agent.pos[0] + nx * retreat_distance, 0, WORLD_W)
        new_y = clamp(agent.pos[1] + ny * retreat_distance, 0, WORLD_H)
        agent.pos = (new_x, new_y)
        # optionally mark local_map around as DANGER in agent's belief
        try:
            ci, cj = cell_of_pos(agent.pos)
            agent.local_map[cj, ci] = DANGER
        except Exception:
            pass
        return


    def send_map_patch(self, comms, targets, now_time, radius=SENSOR_SMALL):
        """向 targets 发送本地地图的补丁（稀疏表示）"""
        # pack as list of (i,j,val) for explored cells near agent to limit bandwidth
        explored = self.get_local_explored_cells()
        patch = []
        # limit patch to cells within a radius (in cells)
        if self.is_large:
            radius = SENSOR_LARGE
        ci, cj = cell_of_pos(self.pos)
        for (i,j) in list(explored):
            if self.local_map[i, j] == VICTIM:
                # always include victim cells
                patch.append((i,j,int(self.local_map[i,j])))
                continue
            if abs(i-ci) <= radius and abs(j-cj) <= radius:
                patch.append((i,j,int(self.local_map[i,j])))
        msg = {'type':'map_patch', 'from':self.id, 'patch':patch, 'pos':self.pos}
        for t in targets:
            comms.send(self, t, msg, now_time)

    def execute_retreat(self):
        """简单的避险策略：反方向退后一定距离"""
        vx, vy = -self.vel[0], -self.vel[1]
        self.pos = (self.pos[0] + vx*5, self.pos[1] + vy*5)

    def request_aid(self, env):
        """上报死亡/求援信息"""
        msg = {'type':'death_report', 'from': self.id, 'pos': self.pos}
        for a in env.agents:
            if a != self and distance(a.pos, self.pos) < 200:
                a.handle_death_report(msg)




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
            pygame.draw.circle(screen, (100, 120, 10), (goal_x, goal_y), 8, 2)
            pygame.draw.line(screen, (100, 120, 10), (goal_x-6, goal_y), (goal_x+6, goal_y), 2)
            pygame.draw.line(screen, (100, 120, 10), (goal_x, goal_y-6), (goal_x, goal_y+6), 2)
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
        self.last_reason_time = -10
        self.brain_reason_time = -10
        self.known_map = np.full((GRID_H, GRID_W), UNKNOWN, dtype=np.int8)  # 脑节点的地图副本
        # self.local_map = np.full((GRID_H, GRID_W), UNKNOWN, dtype=np.int8)
        # self.assigned = {}  # agent_id -> waypoint
        self.is_brain = is_brain  # LargeAgent作为脑节点
        if multi_behavior is not None:
            self.multi_behavior = multi_behavior
        else:
            self.multi_behavior = easyFrontierAssignmentBehavior()
        self.brain_planner = BrainGlobalPlanner()

    def integrate_map_patch(self, patch):
        """将收到的patch应用到自己的known_map"""
        for (i,j,val) in patch:
            # val is occupancy code
            if 0 <= i < GRID_H and 0 <= j < GRID_W:
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
        self.local_map = self.known_map.copy()

    def fuse_own_sensing(self):
        """将自己感知到的地图写入 known_map"""
        inds = np.where(self.local_map != UNKNOWN)
        for i,j in zip(inds[0].tolist(), inds[1].tolist()):
            self.known_map[i,j] = self.local_map[i,j]

    def brain_reason_and_assign(self, agents, now_time):
        # brain node: global planning
        if now_time - self.brain_reason_time < BRAIN_REASON_INTERVAL:
            return
        self.brain_reason_time = now_time
        self.fuse_own_sensing()
        assigns = self.brain_planner.decide(self, agents)
        return assigns

    def reason_and_assign(self, agents, now_time):
        if now_time - self.last_reason_time < BRAIN_REASON_INTERVAL:
            return
        self.last_reason_time = now_time
        # fuse own sensing
        self.fuse_own_sensing()
        # assign frontiers
        assigns = self.multi_behavior.decide(self, agents)
        return assigns

    
# ===============================

    def _agent_request_aid(agent, world, emergency_type='aid'):
        """
        小大节点请求援助的默认实现：发送 message 给其所属 brain node (if known) or broadcast
        msg format: {'type': 'AID_REQUEST', 'from': agent.id, 'pos': agent.pos, 'urgency': value, 'reason': ...}
        """
        msg = {'type':'AID_REQUEST', 'from': agent.id, 'pos': agent.pos, 'urgency': 1.0, 'reason': emergency_type}
        # if agent has father_id / known brain: try to send to brain
        target = None
        if getattr(agent, 'father_id', None) is not None:
            # find brain in world.large_agents list
            for la in world.large_agents:
                if la.id == agent.father_id:
                    target = la
                    break
        if target is None:
            # fallback: send to brain id in world if available
            if getattr(world, 'brain_id', None) is not None:
                try:
                    target = world.large_agents[world.brain_id]
                except Exception:
                    target = None
        # use world.communication if provided; else directly call handler
        if target is not None:
            try:
                world_comms = getattr(world, 'comms', None)
                if world_comms is not None:
                    world_comms.send(agent, target, msg, world.time)
            except Exception:
                pass
        else:
            # broadcast to all large agents
            for la in world.large_agents:
                try:
                    if getattr(world, 'comms', None):
                        world.comms.send(agent, la, msg, world.time)
                except Exception:
                    pass
        return

#  ===============================

    def draw_self(self, screen):
        if self.is_brain:
            color = (220, 100, 40)
        else:
            color = (200, 160, 60)
        # draw communication range
        x,y = int(self.pos[0]), int(self.pos[1])
        if not self.alive:
            pygame.draw.circle(screen, (90,90,90), (int(self.pos[0]), int(self.pos[1])), LARGE_RADIUS)
            x,y = int(self.pos[0]), int(self.pos[1])
            pygame.draw.line(screen, (160,160,160), (x-4,y-4),(x+4,y+4),2)
            pygame.draw.line(screen, (160,160,160), (x-4,y+4),(x+4,y-4),2)
        else:
            pygame.draw.circle(screen, color, (x,y), LARGE_RADIUS)
            surf = pygame.Surface((AGENT_COMM_RANGE*2, AGENT_COMM_RANGE*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (200,160,60,30), (AGENT_COMM_RANGE, AGENT_COMM_RANGE), AGENT_COMM_RANGE)
            screen.blit(surf, (x-AGENT_COMM_RANGE, y-AGENT_COMM_RANGE))









