import pygame
import random
import math
import time
import numpy as np
import heapq

from behaviors import *
from utils import *
from collections import deque

# ===============================
# Agent（小节点）类
# ===============================
class AgentBase:
    def __init__(self, id, x, y, sensor_range=SENSOR_SMALL, is_large=False, behavior=None):
        self.id = id
        self.is_large = is_large
        self.alive = True
        self.energy_cost = 0.0  # 初始能量（可扩展）
        self.energy_cost_rate = 0.1 if not is_large else 0.2  # 能量消耗速率（可调）

        self.sensor_range = sensor_range
        self.local_map = np.full((GRID_H, GRID_W), UNKNOWN, dtype=np.int8)

        self.pos = (x, y)
        angle = random.random() * 2 * math.pi
        self.vel = (math.cos(angle)*10.0, math.sin(angle)*10.0)
        self.speed = AGENT_MAX_SPEED if not is_large else LARGE_AGENT_MAX_SPEED

        self.has_goal = False
        self.goal = None
        self.planned_path = None
        self.task_seq = None
        self.hold_state = False

        self.father_id = None
        self.son_ids = []
        self.hist = [self.pos]
        
        self.death_prob = 0.8  # ✅ 死亡概率参数（可调）

        if behavior is not None:
            self.behavior = behavior
        else:
            self.behavior = PathPlanningBehavior()

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
    def clear_seq(self):
        self.has_goal = False
        self.goal = None
        self.planned_path = None
        self.task_seq = None
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

    def plan_path_sequence(self, unknown_cost=5.0, danger_proximity_penalty=4.0, proximity_radius=2):
        """
        为接收到的 task_seq（[(x,y), ...] 世界坐标点序列）规划一条安全且尽量快速的栅格路径序列。
        - 允许通过 FREE 和 UNKNOWN（但 UNKNOWN 代价更高）。
        - 禁止通过 OBSTACLE 和 已知 DANGER 格子。
        - 如果某个目标不可达，会向父节点发送 path_fail / support_request（触发重分配），并返回 False。
        - 返回: (True, planned_world_path) 或 (False, None)
        ---
        依赖/约定：
        - env.ground_grid[row, col] 表示真实世界栅格（但小机器人只能看到 self.local_map）
        - self.local_map[row, col] 为 agent 的主观栅格认知（UNKNOWN/FREE/OBSTACLE/DANGER/VICTIM）
        - cell_of_pos(pos) -> (row, col)
        - pos_of_cell(col, row) 或 pos_of_cell(x_cell, y_cell) -> (x_world, y_world)
        （此处使用 pos_of_cell(col, row) 风格会导致混淆；为保证兼容，使用 pos_of_cell(col, row) 的结果前请确保你的项目实现。
        上文 AgentBase 中使用 pos_of_cell(i, j) 返回 (x,y) 且 i 为列(x)、j 为行(y)。因此下面保持同样调用顺序：pos_of_cell(col, row)）
        """
        # neighbor offsets (8-connected)
        neighs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]

        def cell_traversal_cost(r, c):
            if not (0 <= r < GRID_H and 0 <= c < GRID_W):
                return float('inf')
            val = int(self.local_map[r, c])
            if val == OBSTACLE or val == DANGER:
                return float('inf')
            base = unknown_cost if val == UNKNOWN else 1.0

            # danger proximity penalty (保持原逻辑)
            prox_pen = 0.0
            if danger_proximity_penalty > 0 and proximity_radius > 0:
                for dr in range(-proximity_radius, proximity_radius + 1):
                    for dc in range(-proximity_radius, proximity_radius + 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < GRID_H and 0 <= cc < GRID_W:
                            if self.local_map[rr, cc] == DANGER:
                                d = math.hypot(dr, dc)
                                if d == 0:
                                    prox_pen += danger_proximity_penalty * 2.0
                                else:
                                    prox_pen += danger_proximity_penalty / (d + 1.0)
            return base + prox_pen

        def _is_safe_area(r, c, world_map):
            """
            检查 (r, c) 及其周围 3x3 区域是否包含 OBSTACLE 或 DANGER 区域。
            这与机器人的暴毙判定保持一致。
            """
            GRID_H, GRID_W = world_map.shape # 假设 world_map 是一个 NumPy 数组或类似结构

            # 定义要检查的相对偏移量 (3x3 区域)
            # 这里的 dx, dy 应该是 [-1, 0, 1] 的所有组合
            shifts = [[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]]
            for shift in shifts:
                # 检查邻居点的坐标
                dr, dc = shift
                nr, nc = r + dr, c + dc
                
                # 1. 边界检查
                if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                    # 即使出界也算是安全问题，但通常寻路不会允许走到边缘。
                    # 如果出界，则视为不可行 (取决于你的地图边界处理逻辑)。
                    return False 
                
                # 2. 地图内容检查
                # 假设你的 A* 寻路是基于 Small Agent 自身的 local_map，这里使用传入的地图。
                if world_map[nr, nc] in (OBSTACLE, DANGER):
                    return False  # 发现危险区或障碍物，不安全
            
            return True # 3x3 区域完全安全

        def astar_multi(start_rc, waypoints_rc):
            """
            Multi-goal A* (方案一 state = (r,c,k))
            start_rc: (row,col)
            waypoints_rc: [(row,col), ...]
            """
            # no waypoints → no path
            if not waypoints_rc:
                return None

            # prepare
            N = len(waypoints_rc)
            sr, sc = start_rc

            # A* state: (r, c, k)
            start_state = (sr, sc, 0)

            import heapq
            open_heap = []
            heapq.heappush(open_heap, (0, start_state))

            gscore = { start_state: 0.0 }
            came = {}

            def heuristic(r, c, k):
                if k >= N:
                    return 0
                tr, tc = waypoints_rc[k]
                return math.hypot(tr - r, tc - c)

            while open_heap:
                _, (cr, cc, k) = heapq.heappop(open_heap)

                # finish (k==N means all waypoints reached)
                if k == N:
                    # reconstruct using came[]
                    path = []
                    node = (cr, cc, k)
                    while node in came:
                        path.append(node)
                        node = came[node]
                    path.append((sr, sc, 0))
                    path.reverse()

                    # drop k dimension → list[(r,c)]
                    final_path = [(r, c) for (r, c, _) in path]
                    return final_path

                for dr, dc in neighs:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                        continue
                    # if not _is_safe_area(nr, nc, self.local_map):
                    if self.local_map[nr, nc] in (OBSTACLE, DANGER):
                        continue
                    base_cost = 1
                    # base_cost = cell_traversal_cost(nr, nc)
                    step_cost = math.hypot(dr, dc) * base_cost

                    nk = k
                    if k < N and (nr, nc) == waypoints_rc[k]:
                        nk = k + 1

                    ns = (nr, nc, nk)
                    tentative = gscore[(cr, cc, k)] + step_cost

                    if ns not in gscore or tentative < gscore[ns] - 1e-9:
                        gscore[ns] = tentative
                        f = tentative + heuristic(nr, nc, nk)
                        came[ns] = (cr, cc, k)
                        heapq.heappush(open_heap, (f, ns))

            return None

        # plan from current pos to each waypoint sequentially
        planned_cells = []
        cur_pos = self.pos
        cur_c, cur_r = cell_of_pos(cur_pos)
        way_rc = [cell_of_pos((t[1],t[0])) for t in self.task_seq]
        # 使用 multi-goal A* 替换原来逐段 A*
        planned_cells = astar_multi((cur_r, cur_c), way_rc)

        if planned_cells is None or len(planned_cells) == 0:

            frontiers = []
            for r in range(GRID_H):
                for c in range(GRID_W):
                    if self.local_map[r, c] != FREE:
                        continue
                    # if any neighbor is UNKNOWN → frontier
                    is_frontier = False
                    for dr, dc in neighs:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < GRID_H and 0 <= cc < GRID_W:
                            if self.local_map[rr, cc] == UNKNOWN:
                                is_frontier = True
                                break
                    if is_frontier:
                        frontiers.append((r, c))

            if len(frontiers) == 0:
                return False, None

            cur_r, cur_c = cell_of_pos(self.pos)
            frontiers.sort(key=lambda rc: math.hypot(rc[0]-cur_r, rc[1]-cur_c))

            selected_path = None
            for fr, fc in frontiers:
                # 原来的 astar 保留，但这里 multi-goal 不适用
                path_rc = astar_multi((cur_r, cur_c), [(fr, fc)])  # ★ MODIFIED: 用 multi 兼容单目标
                if path_rc is not None:
                    selected_path = path_rc
                    break
            planned_cells = selected_path

        # -------------------
        # 原路径转换逻辑保持不变
        # -------------------
        planned_world = []
        for (r, c) in planned_cells:
            px, py = pos_of_cell(c, r)
            planned_world.append((px, py))

        self.planned_path = planned_world
        self.has_goal = True

        if len(planned_world) > 0:
            self.goal = planned_world[0]

        return True, planned_world

    def step_motion(self):
        """
        小智能体的运动逻辑（纯路径跟踪版本）
        输入路径 = self.planned_path，由外部规划器负责生成。
        
        执行顺序：
        1. 是否有路径
        2. 生存状态检查（能量）
        3. 轨迹跟踪（向planned_path[0]移动）
        """

        # ------------------------------
        # 无路径：无需移动
        # ------------------------------
        if self.hold_state or (not self.has_goal) or self.planned_path is None or len(self.planned_path) == 0:
            return


        # ------------------------------
        # 跟踪当前 waypoint
        # ------------------------------
        target = self.planned_path[0]
        tx, ty = target
        x, y = self.pos

        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)

        # 计算移动
        if dist <= self.speed:
            # 可以直接到达 waypoint
            new_x, new_y = tx, ty
        else:
            # 按比例移动 speed 距离
            new_x = x + self.speed * dx / dist
            new_y = y + self.speed * dy / dist

        self.pos = (new_x, new_y)

        # self.energy -= self.energy_decay_rate * math.hypot(new_x - x, new_y - y)
        # if self.energy <= 0:
            # self.energy = 0
            # return
        self.hist.append(self.pos)
        self.energy_cost += self.energy_cost_rate * math.hypot(new_x - x, new_y - y)

        # ------------------------------
        # 是否到达 waypoint
        # ------------------------------
        if math.hypot(new_x - tx, new_y - ty) < 1e-3:
            # 移除已完成的点
            self.planned_path.pop(0)

            # 是否完成整个任务轨迹
            if len(self.planned_path) == 0:
                self.has_goal = False
                self.goal = None
                return
            self.goal = self.planned_path[0]

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
    def __init__(self, id, x, y, is_brain=False, behavior=None ,multi_behavior=None):
        super().__init__(id, x, y, sensor_range=SENSOR_LARGE, is_large=True, behavior=behavior)
        self.last_reason_time = -10
        self.brain_reason_time = -10
        self.known_map = np.full((GRID_H, GRID_W), UNKNOWN, dtype=np.int8)  # 脑节点的地图副本
        self.is_brain = is_brain  # LargeAgent作为脑节点
        self.multi_behavior = ERRTFrontierAssignmentBehavior()
        self.brain_planner = BrainGlobalPlanner()

        self.assignments = None
        self.region = None

        # === 救援相关属性 ===
        self.death_queue = []
        self.rescue_target = None

    def large_reason(self, children):
        if len(children) == 0:
            return None
        return self.multi_behavior.decide(self,children)

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
        self.fuse_own_sensing()
        self.local_map = self.known_map.copy()

    def fuse_own_sensing(self):
        """将自己感知到的地图写入 known_map"""
        inds = np.where(self.local_map != UNKNOWN)
        for i,j in zip(inds[0].tolist(), inds[1].tolist()):
            if self.known_map[i,j] == DANGER:
                continue
            self.known_map[i,j] = self.local_map[i,j]

    def nav_to_centroid(self, my_children):
        # 只有在 rescue_target 为 None 时才会执行到这里        
        if my_children:
            # 计算重心
            xs = [c.pos[0] for c in my_children]
            ys = [c.pos[1] for c in my_children]
            centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
            
            # 规划去重心
            # 注意：为了避免每一帧都疯狂重规划导致抖动/性能下降，可以加一个距离阈值
            # 只有当当前目标距离新重心太远，或者当前没有目标时才重规划
            should_update_centroid = False
            if not self.has_goal:
                should_update_centroid = True
            else:
                # 如果当前目标和现在的重心偏差超过 30 像素，则更新
                curr_goal_dist = math.hypot(self.goal[0] - centroid[0], self.goal[1] - centroid[1])
                if curr_goal_dist > 30.0:
                    should_update_centroid = True
            
            if should_update_centroid:
                self.planned_path = self.base_astar_path(self.pos, centroid)
                self.has_goal = True
                self.goal = centroid
        else:
            # 没有子节点？原地待命或者随机漫步
            pass
    def base_astar_path(self, start_pos, goal_pos):
        sx, sy = cell_of_pos((start_pos[0], start_pos[1]))
        gx, gy = cell_of_pos((goal_pos[0], goal_pos[1]))

        open_set = []
        heapq.heappush(open_set, (0, (sx, sy)))
        came = {}
        g_score = { (sx, sy): 0 }

        def h(x, y):  # 曼哈顿启发
            return abs(x - gx) + abs(y - gy)

        while open_set:
            _, (x, y) = heapq.heappop(open_set)

            if (x, y) == (gx, gy):
                # reconstruct
                path = []
                cur = (x, y)
                while cur in came:
                    path.append(pos_of_cell(*cur))
                    cur = came[cur]
                path.reverse()
                return path

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x + dx, y + dy

                # 越界 or 危险 or 障碍
                if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                    continue
                if not self.known_map[ny, nx] == FREE:
                    continue

                tentative = g_score[(x, y)] + 1
                if (nx, ny) not in g_score or tentative < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative
                    priority = tentative + h(nx, ny)
                    heapq.heappush(open_set, (priority, (nx, ny)))
                    came[(nx, ny)] = (x, y)

        return None

    def update_strategy(self, world, my_children):
        """
        每帧调用一次。根据优先级决定当前的导航目标。
        优先级 1: 救援模式 (处理 death_queue -> 前往事故点 -> 扫描 & 增援)
        优先级 2: 跟随模式 (计算子节点重心 -> 跟随)
        """
        if not self.alive:
            return

        # --- 阶段 1: 救援任务判定 ---
        
        # 1.1 如果闲置且队列有死亡事件，提取任务
        if self.rescue_target is None and self.death_queue:
            event = self.death_queue.pop(0)
            self.rescue_target = event['loc']
            print(f"[{self.id}] ⚠️ 收到求救信号，中断当前任务，前往: {self.rescue_target}")
            # 立即清空旧路径，确立救援优先级
            self.planned_path = []
            self.has_goal = False

        # 1.2 如果处于救援模式 (有 rescue_target)
        if self.rescue_target:
            for child in my_children:
                child.hold_state = True
                child.clear_seq()
            dist = math.hypot(self.pos[0] - self.rescue_target[0], self.pos[1] - self.rescue_target[1])
            arrival_threshold = self.sensor_range * 0.3

            if dist < arrival_threshold:
                # A. 到达现场 -> 执行救援动作
                print(f"[{self.id}] ✅ 到达事故现场。扫描危险区并呼叫增援。")
                self.recognize_danger_area(world)
                
                # 调用世界的增援接口
                if hasattr(world, 'spawn_reinforcement_agent'):
                    world.spawn_reinforcement_agent(self.id)
                sons = [a for a in world.agents if a.id == self.id]
                assigns = self.generate_and_dispatch_tasks(
                    child_agents=sons,
                    comm=None,
                    region_mask=self.region,
                    frontier_cells=None,
                    num_points=30,
                    min_dist=4.0,
                    now_time=None
                )
                for son in sons:
                    seq_new = assigns.get(son.id, [])
                    son.task_seq = seq_new[:]
                    son.has_goal = len(seq_new) > 0
                    son.local_map = self.known_map.copy()
                    if len(son.task_seq)>0:
                        son.plan_path_sequence()
                # B. 救援完成 -> 清除目标
                self.rescue_target = None
                self.has_goal = False
                
                # C. 立即进入阶段 2 (本帧不发呆，马上转为跟随模式)
                pass 
            else:
                # D. 未到达 -> 规划/执行前往救援点的路径
                # 如果当前没有路径，或者路径终点不是救援点，重新规划
                needs_replan = False
                if not self.has_goal or self.goal is None:
                    needs_replan = True
                # elif math.hypot(self.goal[0]-self.rescue_target[0], self.goal[1]-self.rescue_target[1]) > 1.0:
                #     needs_replan = True
                
                if needs_replan:
                    success = self.navigate_to_point(self.rescue_target, world)
                    if not success:
                        print(f"[{self.id}] ❌ 救援路径不可达，跳过此任务。")
                        self.rescue_target = None # 放弃，防止卡死
                        self.has_goal = False
                
                # 如果正在救援，直接返回，不再执行下面的跟随逻辑
                return 

        # --- 阶段 2: 日常跟随模式 (Centroid Following) ---
        # 只有在 rescue_target 为 None 时才会执行到这里        
        if my_children:
            for child in my_children:
                child.hold_state = False
            # 计算重心
            xs = [c.pos[0] for c in my_children]
            ys = [c.pos[1] for c in my_children]
            centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
            
            # 规划去重心
            # 注意：为了避免每一帧都疯狂重规划导致抖动/性能下降，可以加一个距离阈值
            # 只有当当前目标距离新重心太远，或者当前没有目标时才重规划
            should_update_centroid = False
            if not self.has_goal:
                should_update_centroid = True
            else:
                # 如果当前目标和现在的重心偏差超过 30 像素，则更新
                curr_goal_dist = math.hypot(self.goal[0] - centroid[0], self.goal[1] - centroid[1])
                if curr_goal_dist > 30.0:
                    should_update_centroid = True
            
            if should_update_centroid:
                self.navigate_to_point(centroid, world)
        else:
            # 没有子节点？原地待命或者随机漫步
            pass

    def recognize_danger_area(self, world):
        """到达现场后，将传感器范围内的真实 DANGER 区域同步到 known_map"""
        cx, cy = int(self.pos[0]), int(self.pos[1])
        r = 150
        min_i = max(0, (cx - r)//GRID_CELL)
        max_i = min(GRID_W-1, (cx + r)//GRID_CELL)
        min_j = max(0, (cy - r)//GRID_CELL)
        max_j = min(GRID_H-1, (cy + r)//GRID_CELL)
        for i in range(min_i, max_i+1):
            for j in range(min_j, max_j+1):
                gx, gy = pos_of_cell(i, j)
                if math.hypot(gx - cx, gy - cy) <= r:
                    val = world.ground_grid[j, i]
                    self.known_map[j, i] = val

    def sample_candidate_points(self, region_mask, frontier_cells=None, num_points=20, min_dist=5.0):
        """
        高级组合采样：
        - frontier 优先
        - 基于信息熵/未知概率加权
        - Poisson Disk 最小间距保证均匀分布

        region_mask: 2D bool array，未探索区域
        frontier_cells: [(i,j), ...] 前沿格子，可为空
        num_points: 最终采样点数量
        min_dist: Poisson Disk 最小间距（格子坐标单位）
        """
        H, W = region_mask.shape
        # --------------------------
        # 1. 生成候选格子
        candidate_cells = [(i,j) for i in range(H) for j in range(W) 
                        if region_mask[i,j] and self.known_map[i,j]==UNKNOWN]

        if not candidate_cells:
            return []

        # --------------------------
        # 2. 计算权重
        weights = np.zeros(len(candidate_cells))
        
        # 定义邻域搜索半径
        search_radius = 3  # 可以调整这个参数
        
        for idx, (i, j) in enumerate(candidate_cells):
            # 统计周围邻域内的未知点数量
            unknown_count = 0
            total_cells = 0
            
            # 搜索周围的网格
            for di in range(-search_radius, search_radius + 1):
                for dj in range(-search_radius, search_radius + 1):
                    ni, nj = i + di, j + dj
                    
                    # 检查边界
                    if 0 <= ni < H and 0 <= nj < W:
                        total_cells += 1
                        # 计算欧氏距离，只考虑圆形邻域
                        distance = np.sqrt(di*di + dj*dj)
                        if distance <= search_radius:
                            # 如果是未知区域，增加计数
                            if self.known_map[ni, nj] == UNKNOWN:
                                unknown_count += 1
            
            # 计算未知点密度（可选：使用高斯权重衰减）
            if total_cells > 0:
                # 方法1: 简单密度计算
                density = unknown_count / total_cells
                
                
                weights[idx] = density

        # 归一化
        weights /= weights.sum()

        # --------------------------
        # 3. 加权随机选择初步候选点
        num_candidates = min(len(candidate_cells), num_points*3)  # 先挑更多候选
        chosen_idx = np.random.choice(len(candidate_cells), size=num_candidates, replace=False, p=weights)
        preliminary_points = [pos_of_cell(j,i) for i,j in np.array(candidate_cells)[chosen_idx]]

        # --------------------------
        # 4. Poisson Disk 最小间距筛选
        final_points = []
        random.shuffle(preliminary_points)
        for p in preliminary_points:
            if all(np.hypot(p[0]-q[0], p[1]-q[1]) >= min_dist for q in final_points):
                final_points.append(p)
            if len(final_points) >= num_points:
                break

        return final_points

    def solve_mtsp_and_assign(self, points, child_agents):
        """
        修正版 multi-TSP 分配（更均衡）：
        - 当点数量较少或 agent 很靠近时，使用负载感知的最近分配（避免所有点都给同一个 agent）
        - 否则使用简单的 k-means 聚类（k = len(child_agents)）把点分为 k 个簇，再把簇分配给离簇心最近的 agent
        - 每个 agent 内部使用 nearest-neighbor 做路径顺序
        返回：{child_id: [p1, p2, ...]}
        """
        if not child_agents or not points:
            return {}

        num_agents = len(child_agents)
        num_points = len(points)

        # helper: nearest-neighbor sort
        def nn_sort(start, pts):
            path = []
            cur = start
            pts = pts.copy()
            while pts:
                nxt = min(pts, key=lambda p: (p[0]-cur[0])**2 + (p[1]-cur[1])**2)
                path.append(nxt)
                pts.remove(nxt)
                cur = nxt
            return path

        if num_points <= max(8, 2 * num_agents):
        # If very few points, use load-balanced greedy assignment
            # initialize
            assign = {c.id: [] for c in child_agents}
            loads = {c.id: 0 for c in child_agents}
            avg_load = max(1, num_points / num_agents)
            alpha = 0.8  # load penalty factor

            for px, py in points:
                best_c = None
                best_score = float('inf')
                for c in child_agents:
                    cx, cy = c.pos
                    d2 = (cx - px)**2 + (cy - py)**2
                    # score = distance * (1 + alpha * (current_load / avg_load))
                    score = d2 * (1.0 + alpha * (loads[c.id] / avg_load))
                    if score < best_score:
                        best_score = score
                        best_c = c
                assign[best_c.id].append((px, py))
                loads[best_c.id] += 1

            # order within each agent
            final = {}
            for cid, pts in assign.items():
                if pts:
                    child = next(c for c in child_agents if c.id == cid)
                    final[cid] = nn_sort(child.pos, pts)
                else:
                    final[cid] = []
            return final

        # Otherwise use simple k-means clustering (k = num_agents)
        # Initialize centroids by sampling distinct points
        k = num_agents
        pts = [tuple(p) for p in points]
        if len(pts) <= k:
            # fallback: assign one point per agent as possible, rest by greedy
            assign = {c.id: [] for c in child_agents}
            for i, p in enumerate(pts):
                assign[child_agents[i % k].id].append(p)
            # remaining none
            final = {}
            for cid, pts in assign.items():
                if pts:
                    child = next(c for c in child_agents if c.id == cid)
                    final[cid] = nn_sort(child.pos, pts)
                else:
                    final[cid] = []
            return final

        # random distinct init
        centroids = random.sample(pts, k)
        max_iters = 30
        for _ in range(max_iters):
            clusters = [[] for _ in range(k)]
            # assign step: nearest centroid
            for p in pts:
                dists = [ (p[0]-cx)**2 + (p[1]-cy)**2 for (cx,cy) in centroids ]
                idx = int(min(range(k), key=lambda i: dists[i]))
                clusters[idx].append(p)
            # update step
            new_centroids = []
            changed = False
            for cl in clusters:
                if not cl:
                    # keep old centroid or reinit to a random point
                    new_centroids.append(random.choice(pts))
                    continue
                mx = sum(p[0] for p in cl) / len(cl)
                my = sum(p[1] for p in cl) / len(cl)
                new_centroids.append((mx, my))
            for a,b in zip(centroids, new_centroids):
                if (a[0]-b[0])**2 + (a[1]-b[1])**2 > 1e-6:
                    changed = True
                    break
            centroids = new_centroids
            if not changed:
                break

        # Now assign clusters to agents by nearest centroid -> agent mapping
        # compute centroid->best agent
        centroid_to_agent = [-1]*k
        used_agents = set()
        for ci, cen in enumerate(centroids):
            best_agent = None
            best_d = float('inf')
            for ag in child_agents:
                if ag.id in used_agents:
                    continue
                ax, ay = ag.pos
                d = (ax - cen[0])**2 + (ay - cen[1])**2
                if d < best_d:
                    best_d = d
                    best_agent = ag
            if best_agent is None:
                # allow reuse if not enough distinct
                best_agent = min(child_agents, key=lambda ag: (ag.pos[0]-cen[0])**2 + (ag.pos[1]-cen[1])**2)
            centroid_to_agent[ci] = best_agent.id
            used_agents.add(best_agent.id)

        # If k > number of agents (shouldn't), map remaining to nearest agent
        # Build assignments
        assign = {c.id: [] for c in child_agents}
        for ci, cl in enumerate(clusters):
            aid = centroid_to_agent[ci]
            if aid not in assign:
                # find nearest agent id as fallback
                nearest = min(child_agents, key=lambda ag: (ag.pos[0]-centroids[ci][0])**2 + (ag.pos[1]-centroids[ci][1])**2)
                aid = nearest.id
            assign[aid].extend(cl)

        # If any agent ended up with 0 points, do a balancing pass: move nearest point from largest cluster
        empty_agents = [aid for aid, lst in assign.items() if len(lst) == 0]
        if empty_agents:
            # sort agents by load descending
            loads = sorted(assign.items(), key=lambda kv: -len(kv[1]))
            for empty_aid in empty_agents:
                if not loads:
                    break
                # take one point from the largest
                src_aid, src_pts = loads[0]
                if not src_pts:
                    loads.pop(0)
                    continue
                # pick the point in src_pts that is closest to the empty agent
                empty_agent = next(ag for ag in child_agents if ag.id == empty_aid)
                p_best = min(src_pts, key=lambda p: (p[0]-empty_agent.pos[0])**2 + (p[1]-empty_agent.pos[1])**2)
                assign[empty_aid].append(p_best)
                src_pts.remove(p_best)
                # resort loads
                loads = sorted(assign.items(), key=lambda kv: -len(kv[1]))

        # Final nearest-neighbor ordering per agent
        final_assign = {}
        for cid, pts in assign.items():
            if pts:
                child = next(c for c in child_agents if c.id == cid)
                final_assign[cid] = nn_sort(child.pos, pts)
            else:
                final_assign[cid] = []

        return final_assign

    def generate_and_dispatch_tasks(self, child_agents, comm, region_mask, frontier_cells=None, num_points=20, min_dist=5.0, now_time=None):
        """
        完整任务生成与分发流程：
        1) 高级组合采样 candidate points
        2) multi-TSP 分配给 child agents
        3) 通过 comm 发送 target_assign 消息

        region_mask: LargeAgent 分配的未探索区域
        child_agents: 该 LargeAgent 的子节点 (M robots)
        comm: Communication 实例
        frontier_cells: optional 前沿格子列表
        num_points: 总采样点数量
        min_dist: Poisson Disk 最小间距
        now_time: 当前时间戳
        """
        final_points = self.sample_candidate_points(region_mask, frontier_cells, num_points, min_dist)
        assignments = self.solve_mtsp_and_assign(final_points, child_agents)

        return assignments

    def navigate_to_point(self, target_pos, world):
            """
            导航到任意坐标 target_pos。
            如果直接不可达（在墙里），会尝试寻找附近的最近安全点。
            返回: Boolean (是否成功生成路径)
            """
            # 1. 检查目标点安全性，若不安全寻找替代点
            target_pos = self._ensure_safe_target(target_pos, world)
            if target_pos is None:
                return False

            # 2. A* 寻路
            path = self._astar_path(self.pos, target_pos, world)
            if path and len(path) > 0:
                self.planned_path = path
                self.goal = target_pos
                self.has_goal = True
                return True
            return False
    # ---------------------------------------------------------
    # 工具函数：检查点是否安全
    # ---------------------------------------------------------
    def _ensure_safe_target(self, pos, world):
            """如果 pos 在障碍物/已知危险中，BFS 搜索最近的可行点"""
            cx, cy = cell_of_pos(pos)
            if self._is_safe_cell(world, cx, cy):
                return pos
            
            # BFS 搜索最近安全格子
            # 注意：这里使用 visited_grid 或 known_map，LargeAgent 应该基于自己的认知避障？
            # 通常为了防止Large送死，这里可以用 world.ground_grid 或者 world.visited_grid (上帝视角辅助)
            # 或者严格点，用 self.known_map
            
            start_node = (cx, cy)
            q = deque([start_node])
            visited = {start_node}
            
            # 限制搜索范围防止卡死
            max_steps = 200 
            steps = 0

            while q and steps < max_steps:
                curr_x, curr_y = q.popleft()
                steps += 1

                if self._is_safe_cell(world, curr_x, curr_y):
                    return pos_of_cell(curr_x, curr_y)

                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nx, ny = curr_x + dx, curr_y + dy
                    if (nx, ny) not in visited and 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                        visited.add((nx, ny))
                        q.append((nx, ny))
            return None


    def _is_safe_cell(self, world, cx, cy):
        if cx < 0 or cx >= GRID_W or cy < 0 or cy >= GRID_H:
            return False
        val = world.visited_grid[cy, cx]
        return val ==FREE
    # ---------------------------------------------------------
    # 工具函数：寻找最近安全点（BFS）
    # ---------------------------------------------------------
    def _find_safe_target(self, pos, world):
        px, py = pos
        sx, sy = cell_of_pos((px, py))

        q = deque([(sx, sy)])
        visited = set()

        while q:
            x, y = q.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                if self._is_safe_cell(world, x, y):
                    return pos_of_cell(x, y)

            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                q.append((x + dx, y + dy))

        return None
    # 工具函数：A* 寻路
    def _astar_path(self, start_pos, goal_pos, world):
        sx, sy = cell_of_pos((start_pos[0], start_pos[1]))
        gx, gy = cell_of_pos((goal_pos[0], goal_pos[1]))

        open_set = []
        heapq.heappush(open_set, (0, (sx, sy)))
        came = {}
        g_score = { (sx, sy): 0 }

        def h(x, y):  # 曼哈顿启发
            return abs(x - gx) + abs(y - gy)

        while open_set:
            _, (x, y) = heapq.heappop(open_set)

            if (x, y) == (gx, gy):
                # reconstruct
                path = []
                cur = (x, y)
                while cur in came:
                    path.append(pos_of_cell(*cur))
                    cur = came[cur]
                path.reverse()
                return path

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x + dx, y + dy

                # 越界 or 危险 or 障碍
                if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                    continue
                if not self._is_safe_cell(world, nx, ny):
                    continue

                tentative = g_score[(x, y)] + 1
                if (nx, ny) not in g_score or tentative < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative
                    priority = tentative + h(nx, ny)
                    heapq.heappush(open_set, (priority, (nx, ny)))
                    came[(nx, ny)] = (x, y)

        return None
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


class BrainAgent(LargeAgent):
    def __init__(self, id, x, y, is_brain=True, behavior=None, multi_behavior=None):
        super().__init__(id, x, y, is_brain, behavior, multi_behavior)
    
    def partition_and_select_regions(self,
                                    middles, 
                                    block_rows=7,
                                    block_cols=10,
                                    unknown_thresh=10):
        """
        1) 将全图直接划分为 block_rows × block_cols 个大矩形区域
        2) 对每个区域统计 UNKNOWN 数量
        3) 若 UNKNOWN 很少，则跳过该块
        4) 计算该块中心到所有 Middle 的最小距离，作为“可达性”
        5) 综合评分 = 未知数 / (距离 + 1e-5)
        6) 返回按评分排序的区域信息（(score, mask, bbox)）
        """

        H, W = self.known_map.shape
        km = self.known_map

        # 1. 先收集中层 middle 机器人
        if len(middles) == 0:
            return []  # 没有中层就不分区

        # 2. 计算分块大小
        h_step = H // block_rows
        w_step = W // block_cols

        regions = []  # 将返回 (score, region_mask, bbox)

        m_poses = []
        for m in middles:
            x,y = cell_of_pos(m.pos)
            m_poses.append((x,y))

        # 3. 遍历所有矩形块
        for r in range(block_rows):
            for c in range(block_cols):

                # 3.1 计算块的 bbox
                r0 = r * h_step
                r1 = (r+1) * h_step if r < block_rows-1 else H

                c0 = c * w_step
                c1 = (c+1) * w_step if c < block_cols-1 else W

                block = km[r0:r1, c0:c1]

                # 3.2 统计 UNKNOWN 数量
                unknown_count = np.sum(block == UNKNOWN)
                danger_count = np.sum(block == DANGER)
                if unknown_count < unknown_thresh:
                    continue  # 未知太少不值得探索

                # 3.3 计算此区域中心
                center = np.array([(r0+r1)/2, (c0+c1)/2])

                # 3.4 求所有 middle 到该块中心的最小距离
                dists = []
                for pos in m_poses:
                    pi, pj = pos
                    d = math.sqrt((pi-center[1])**2 + (pj-center[0])**2)
                    dists.append(d)
                min_dist = min(dists) if dists else 9999

                # 3.5 评分：未知越多越高，距离越小越高
                score = (unknown_count- danger_count**1.23)/ (min_dist**0.7 + 1e-6)

                # 3.6 mask
                mask = np.zeros((H, W), dtype=bool)
                mask[r0:r1, c0:c1] = True

                regions.append({
                    "score": score,
                    "mask": mask,
                    "bbox": (r0, r1, c0, c1),
                    "unknown": unknown_count,
                    "dist": min_dist
                })

        # 4. 按评分排序
        regions_sorted = sorted(regions, key=lambda x: -x["score"])

        return regions_sorted

    def assign_region_to_middle(self, middles, max_assign_per_middle=1):
        """
        给 middle-level LargeAgent 分配探索区域。
        输入:
            middles: list of middle-level agents
            max_assign_per_middle: 每个 middle 最多分配多少个区域
        输出:
            { middle_id : [region_mask1, region_mask2, ...] }
        """

        if not middles:
            return {}

        # middle 的 ID → 位置
        middle_pos = {m.id: m.pos for m in middles}

        # 每个 middle 剩余可分配次数
        remaining_counts = {m.id: max_assign_per_middle for m in middles}

        # 初始化分配结果（每个值是 list，不是单个 mask）
        assignments = {m.id: [] for m in middles}

        # 获取区域（已按区域价值排序）
        regions = self.partition_and_select_regions(middles)
        if not regions:
            return assignments

        # 开始给区域做分配
        for region in regions:
            region_mask = region["mask"]

            # 跳过无效区域
            ys, xs = np.where(region_mask)
            if len(xs) == 0:
                continue

            # 计算区域中心
            rc_x, rc_y = pos_of_cell(np.mean(xs), np.mean(ys))
            # region_center = (rc_x, rc_y)

            best_mid = None
            best_dist = float("inf")

            # 找最近且还没分配满的 middle
            for mid in middles:
                if remaining_counts[mid.id] <= 0:
                    continue

                mx, my = middle_pos[mid.id]
                d = (mx - rc_x) ** 2 + (my - rc_y) ** 2

                if d < best_dist:
                    best_dist = d
                    best_mid = mid.id

            # 如果找到了分配对象
            if best_mid is not None:
                assignments[best_mid].append(region_mask)
                remaining_counts[best_mid] -= 1

            # 如果所有 middle 都满了 → 可以提前结束
            if all(c <= 0 for c in remaining_counts.values()):
                break

        return assignments









