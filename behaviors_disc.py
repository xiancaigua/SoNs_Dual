
# import torch
import math
import random
import numpy as np
import heapq
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import fclusterdata
from utils import *
# from model import SlowModel  # 导入你的模型定义
import time
from sklearn.cluster import DBSCAN

# -----------------------------
# 行为模块（策略模式）
# -----------------------------
class Behavior:
    def decide(self, agent, sense_data, dt):
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

class PathPlanningBehaviorDiscrete(Behavior):
    """
    离散格子版 PathPlanning:
      - astar 返回格子路径 [(x,y),...]
      - decide(agent) 返回 next_cell (x,y) 或 None
    设计假设:
      - agent.pos 是格子坐标 (x,y)（整数）
      - agent.goal 如果存在，也应为格子坐标 (x,y)
      - agent.local_map 是格子地图，shape (H,W)，用 FREE/OBSTACLE/DANGER/UNKNOWN 标记
    """

    def __init__(self, replan_interval=1, goal_tolerance=0):
        self.replan_interval = replan_interval
        self.goal_tolerance = goal_tolerance  # 对格子一般置 0 或 1
        self.last_plan_step = -999
        self.cached_path = None  # list of cells [(x,y),...]
        self.path_index = 0
        self.last_goal = None

    # ---------------- A* ----------------
    def astar(self, grid, start, goal):
        """A* 返回从 start 到 goal 的格子路径（含 start 和 goal），找不到返回 None"""
        H, W = grid.shape
        if not (0 <= goal[0] < W and 0 <= goal[1] < H):
            return None
        if grid[goal[1], goal[0]] in (OBSTACLE, DANGER):
            return None

        open_heap = []
        heapq.heappush(open_heap, (0 + self.heuristic(start, goal), 0, start))
        came_from = {}
        gscore = {start: 0}
        visited = set()

        while open_heap:
            f, g, current = heapq.heappop(open_heap)
            if current == goal:
                # reconstruct
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            if current in visited:
                continue
            visited.add(current)

            cx, cy = current
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nb = (cx + dx, cy + dy)
                nx, ny = nb
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if grid[ny, nx] in (OBSTACLE, DANGER):
                    continue
                tentative = g + 1
                if tentative < gscore.get(nb, float('inf')):
                    gscore[nb] = tentative
                    came_from[nb] = current
                    heapq.heappush(open_heap, (tentative + self.heuristic(nb, goal), tentative, nb))
        return None

    def heuristic(self, a, b):
        # Manhattan
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ---------------- decide ----------------
    def decide(self, agent, world_step_counter=None):
        """
        输入 agent（具有 .pos (x,y), .goal (x,y) 或 None, .local_map grid）
        返回: 下一个格子 (x,y) 或 None（fallback）
        world_step_counter: 可选的计时器（用于 replan_interval）
        """
        # ensure integer positions
        start = (int(agent.pos[0]), int(agent.pos[1]))

        # 若无目标 -> 随机安全探索
        if agent.target is None:
            return self.safe_random_explore(agent)

        goal = (int(agent.target[0]), int(agent.target[1]))

        # 已到达目标（格子相同或相邻）
        if start == goal or self.heuristic(start, goal) <= self.goal_tolerance:
            agent.has_goal = False
            agent.goal = None
            self.cached_path = None
            self.path_index = 0
            return None

        # 是否需要重算路径
        goal_changed = (self.last_goal is None) or (goal != self.last_goal)
        need_replan = (
            self.cached_path is None
            or self.path_index >= (len(self.cached_path) if self.cached_path else 0)
            or goal_changed
            or (world_step_counter is not None and (world_step_counter - self.last_plan_step) >= self.replan_interval)
        )

        if need_replan:
            # 使用 agent.local_map（离散格子）做 A*
            path = self.astar(agent.local_map, start, goal)
            if not path:
                # 无路径 -> 随机安全探索（离散）
                self.cached_path = None
                self.path_index = 0
                self.last_goal = None
                return self.safe_random_explore(agent)
            self.cached_path = path
            self.path_index = 1  # path[0] == start, 所以下一步是 path[1]
            self.last_plan_step = world_step_counter if world_step_counter is not None else 0
            self.last_goal = goal

        # 如果有缓存路径，取下一个格子
        if self.cached_path and self.path_index < len(self.cached_path):
            next_cell = self.cached_path[self.path_index]
            # 如果 next_cell 突然变得不可通行（agent.local_map），尝试重规划下一步
            nx, ny = next_cell
            if agent.local_map[ny, nx] in (OBSTACLE, DANGER):
                # invalidate and try replan next tick
                self.cached_path = None
                self.path_index = 0
                return self.safe_random_explore(agent)
            # advance index for next call
            self.path_index += 1
            return next_cell

        # 路径被走完
        self.cached_path = None
        self.path_index = 0
        return None

    # ---------------- safe random explore ----------------
    def safe_random_explore(self, agent):
        """
        离散随机探索：在 4 邻域中随机选择一个安全的 FREE 格子（优先未探索格）
        返回选中的格子 (x,y) 或 None（表示原地不动）
        """
        x0, y0 = int(agent.pos[0]), int(agent.pos[1])
        neighbors = []
        unseen = []
        safe = []
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x0 + dx, y0 + dy
            if not (0 <= nx < agent.local_map.shape[1] and 0 <= ny < agent.local_map.shape[0]):
                continue
            val_local = agent.local_map[ny, nx]
            # if local unknown, prefer exploring unknown (but ensure not obstacle by global info if available)
            if val_local == UNKNOWN:
                unseen.append((nx, ny))
            elif val_local == FREE:
                safe.append((nx, ny))
            # avoid OBSTACLE/DANGER
        # 优先未见过的格子
        if unseen:
            return random.choice(unseen)
        if safe:
            return random.choice(safe)
        # 没有安全格子，保持原地
        return None


# ---------------------------------------------------------------------
class Multi_Behavior:
    def decide(self, agent, agents):
        raise NotImplementedError

class easyFrontierAssignmentBehavior(Multi_Behavior):
    """
    实现原LargeAgent的 find_frontiers / assign_frontiers / reason_and_assign 三个核心函数逻辑
    """

    def __init__(self, frontier_threshold=0.5):
        self.frontier_threshold = frontier_threshold
        self.last_assignment = {}

    def find_frontiers(self, global_map):
        """识别地图前沿区域"""
        frontiers = []
        for y in range(1, global_map.shape[0] - 1):
            for x in range(1, global_map.shape[1] - 1):
                if global_map[y, x] == UNKNOWN:  # 未知区域
                    neighbors = global_map[y-1:y+2, x-1:x+2].flatten()
                    if np.any(neighbors == 1.0) or np.any(neighbors == 0.0):
                        frontiers.append((x, y))
        return frontiers

    def assign_frontiers(self, frontiers, agents):
        """根据距离将前沿点分配给每个Agent"""
        assignments = {}
        available = frontiers.copy()
        for agent in agents:
            if not available:
                assignments[agent.id] = None
                continue
            distances = [np.linalg.norm(np.array(agent.pos) - np.array(f)) for f in available]
            target = available[int(np.argmin(distances))]
            assignments[agent.id] = target
            available.remove(target)
        return assignments

    def decide(self, agent, agents):
        """为所有Agents决定前沿点分配"""
        frontiers = self.find_frontiers(agent.known_map)
        if not frontiers:
            return {a.id: None for a in agents}
        assignments = self.assign_frontiers(frontiers, agents)
        self.last_assignment = assignments

        return assignments  # {agent_id: (x, y)}

# class NeuralFrontierAssignmentBehavior(Multi_Behavior):
#     # TODO 此类方法暂时废弃
#     """
#     使用训练好的 SlowModel（Diffusion Transformer）进行神经前沿预测与分配。
#     逻辑：
#     1. 将当前脑节点的 known_map 与各 agent 的全局状态编码为状态向量；
#     2. 调用 SlowModel.sample(state_vec) 生成一组子目标序列；
#     3. 选择前若干子目标作为“前沿点”；
#     4. 根据距离或策略为每个小节点分配目标。
#     """

#     def __init__(self, model_path="models/slow_model.pth", device="cpu", n_subgoals=5):
#         super().__init__()
#         self.device = device
#         self.model = SlowModel().to(device)
#         self.model.load_state_dict(torch.load(model_path, map_location=device))
#         self.model.eval()
#         self.n_subgoals = n_subgoals
#         self.last_assignment = {}

#     # -------------------------------
#     # 状态编码函数
#     # -------------------------------
#     def encode_global_state(self, brain_agent, agents):
#         """
#         将当前全局状态编码为一个 state vector，输入给 SlowModel。
#         示例特征（可按需扩展）：
#         - 探索率（覆盖百分比）
#         - 活跃节点数
#         - 大脑节点位置
#         - victim 是否被救
#         """
#         explored = np.sum(brain_agent.known_map != -1)
#         total = brain_agent.known_map.size
#         coverage = explored / total

#         n_alive = sum([a.alive for a in agents])
#         brain_pos = np.array(brain_agent.pos, dtype=np.float32) / np.array([WORLD_W, WORLD_H])
#         victim_status = 1.0 if hasattr(brain_agent, "victim_found") and brain_agent.victim_found else 0.0

#         # 形成固定长度状态向量（可以根据模型训练时的定义调整）
#         state_vec = np.array([
#             coverage,
#             n_alive / len(agents),
#             brain_pos[0],
#             brain_pos[1],
#             victim_status,
#             0.0  # 占位符，可用于平均连接强度等
#         ], dtype=np.float32)
#         return state_vec

#     # -------------------------------
#     # 主决策函数
#     # -------------------------------
#     def decide(self, brain_agent, agents):
#         """
#         利用神经网络生成前沿目标并分配。
#         """
#         # === Step 1: 状态编码 ===
#         state_vec = self.encode_global_state(brain_agent, agents)

#         # === Step 2: 生成子目标序列 ===
#         with torch.no_grad():
#             input_tensor = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
#             subgoals = self.model.sample(input_tensor, n_samples=1)  # np.ndarray [N, 2]
#             if isinstance(subgoals, torch.Tensor):
#                 subgoals = subgoals.cpu().numpy()
#         subgoals = np.clip(subgoals, 0, 1)  # 归一化输出映射
#         subgoals[:, 0] *= WORLD_W
#         subgoals[:, 1] *= WORLD_H

#         # === Step 3: 将子目标作为“前沿” ===
#         frontiers = [tuple(map(float, g)) for g in subgoals[:self.n_subgoals]]

#         # === Step 4: 分配到各 agent ===
#         assignments = {}
#         available = frontiers.copy()
#         for a in agents:
#             if not available:
#                 assignments[a.id] = None
#                 continue
#             distances = [distance(a.pos, f) for f in available]
#             target = available[int(np.argmin(distances))]
#             assignments[a.id] = target
#             available.remove(target)

#         self.last_assignment = assignments
#         return assignments

class ERRTFrontierAssignmentBehavior(Multi_Behavior):
    """
    基于E-RRT思想的多目标前沿分配算法
    融合信息增益、距离成本和驱动成本进行优化决策
    """

    def __init__(self, frontier_threshold=0.5, sensor_range=6.0, 
                 weight_distance=1.0, weight_gain=2.0, weight_actuation=0.5):
        self.frontier_threshold = frontier_threshold
        self.sensor_range = sensor_range
        self.weights = {
            'distance': weight_distance,
            'gain': weight_gain,
            'actuation': weight_actuation
        }
        self.last_assignment = {}
        self.candidate_goals = []

    def generate_pseudo_random_goals(self, global_map, num_goals=20):
        """生成伪随机目标点（类似E-RRT的目标生成策略）"""
        frontiers = self.find_frontiers(global_map)
        if not frontiers:
            return []

        # 从前沿点中随机选择候选目标
        indices = np.random.choice(len(frontiers), 
                                 min(num_goals, len(frontiers)), 
                                 replace=False)
        return [frontiers[i] for i in indices]

    def calculate_information_gain(self, goal, global_map, agent_pos):
        """计算目标点的信息增益（类似E-RRT的探索收益评估）"""
        x, y = goal
        info_gain = 0
        
        # 模拟传感器视野（简化版）
        for dy in range(-int(self.sensor_range), int(self.sensor_range)+1):
            for dx in range(-int(self.sensor_range), int(self.sensor_range)+1):
                dist = math.sqrt(dx**2 + dy**2)
                if dist <= self.sensor_range:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < global_map.shape[1] and 
                        0 <= ny < global_map.shape[0]):
                        try:
                            if global_map[ny, nx] == UNKNOWN:  # 未知区域
                                # 检查视线是否被阻挡（简化版碰撞检测）
                                if self.line_of_sight_clear((x, y), (nx, ny), global_map):
                                    info_gain += 1
                        except:
                            pass
        return info_gain

    def line_of_sight_clear(self, start, end, global_map):
        """检查两点之间视线是否清晰（简化版碰撞检测）"""
        x1, y1 = start
        x2, y2 = end
        
        # Bresenham直线算法检查路径上的点
        points = self.bresenham_line(x1, y1, x2, y2)
        for px, py in points:
            if (0 <= px < global_map.shape[1] and 
                0 <= py < global_map.shape[0]):
                if global_map[py, px] == OBSTACLE:  # 障碍物
                    return False
        return True

    def bresenham_line(self, x1, y1, x2, y2):
        """Bresenham直线算法"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = -1 if x1 > x2 else 1
        sy = -1 if y1 > y2 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x2:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x, y))
        return points

    def calculate_distance_cost(self, agent_pos, goal):
        """计算距离成本"""
        return np.linalg.norm(np.array(agent_pos) - np.array(goal))

    def build_cost_matrix(self, agents, candidate_goals, global_map):
        """构建多目标优化的成本矩阵"""
        num_agents = len(agents)
        num_goals = len(candidate_goals)
        
        # 初始化成本矩阵
        cost_matrix = np.zeros((num_agents, num_goals))
        
        for i, agent in enumerate(agents):
            for j, goal in enumerate(candidate_goals):
                # 距离成本
                dist_cost = self.calculate_distance_cost(agent.pos, goal)
                
                # 信息增益（负成本，因为我们要最大化增益）
                info_gain = self.calculate_information_gain(goal, global_map, agent.pos)
                gain_cost = -info_gain  # 负号表示我们要最大化这个值
                
                
                # 加权总和（类似E-RRT的代价函数）
                total_cost = (self.weights['distance'] * dist_cost +
                            self.weights['gain'] * gain_cost)
                
                cost_matrix[i, j] = total_cost
                
        return cost_matrix

    def optimize_assignment(self, cost_matrix):
        """使用匈牙利算法进行最优分配"""
        # 匈牙利算法求解最小成本匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind

    def decide(self, agent, agents):
        """基于E-RRT思想的多目标优化决策"""
        global_map = agent.local_map
        
        # 1. 生成伪随机候选目标
        self.candidate_goals = self.generate_pseudo_random_goals(global_map)
        
        if not self.candidate_goals:
            return {a.id: None for a in agents}
        
        # 2. 构建多目标成本矩阵
        cost_matrix = self.build_cost_matrix(agents, self.candidate_goals, global_map)

        # 3. 使用优化算法进行分配
        try:
            row_ind, col_ind = self.optimize_assignment(cost_matrix)
            
            # 4. 构建分配结果
            assignments = {}
            for i, agent_id in enumerate([a.id for a in agents]):
                if i in row_ind:
                    match_idx = np.where(row_ind == i)[0][0]
                    goal_idx = col_ind[match_idx]
                    assignments[agent_id] = cell_of_pos(self.candidate_goals[goal_idx])
                else:
                    # 如果没有匹配到目标，使用fallback策略
                    assignments[agent_id] = cell_of_pos(self.fallback_assignment(agents[i], global_map))
            
            self.last_assignment = assignments

            return assignments
            
        except Exception as e:
            # 如果优化失败，使用fallback策略
            print(f"Optimization failed: {e}, using fallback strategy")
            return self.fallback_assignment_strategy(agents, global_map)

    def fallback_assignment(self, agent, global_map):
        """备选分配策略（当优化失败时使用）"""
        frontiers = self.find_frontiers(global_map)
        if not frontiers:
            return None
        
        # 简单的最远前沿策略
        distances = [np.linalg.norm(np.array(agent.pos) - np.array(f)) 
                    for f in frontiers]
        return frontiers[np.argmax(distances)]

    def fallback_assignment_strategy(self, agents, global_map):
        """全局备选分配策略"""
        frontiers = self.find_frontiers(global_map)
        if not frontiers:
            return {a.id: None for a in agents}
        
        # 使用原始的距离最近分配策略
        assignments = {}
        available = frontiers.copy()
        for agent in agents:
            if not available:
                assignments[agent.id] = None
                continue
            distances = [np.linalg.norm(np.array(agent.pos) - np.array(f)) 
                        for f in available]
            target = available[np.argmin(distances)]
            assignments[agent.id] = target
            available.remove(target)
        
        return assignments

    def find_frontiers(self, global_map):
        """识别地图前沿区域（修正版）"""
        frontiers = []
        for y in range(1, global_map.shape[0] - 1):
            for x in range(1, global_map.shape[1] - 1):
                if global_map[y, x] == UNKNOWN:
                    neighbors = global_map[y-1:y+2, x-1:x+2].flatten()
                    if np.any(neighbors == FREE):
                        # ✅ 将网格坐标转换为世界坐标
                        gx, gy = pos_of_cell(x, y)
                        frontiers.append((gx, gy))
        return frontiers

class BrainGlobalPlanner(Multi_Behavior):
    def __init__(self, frontier_threshold=0.5, sensor_range=6.0, 
                 weight_distance=1.0, weight_gain=2.0, weight_actuation=0.5):
        self.frontier_threshold = frontier_threshold
        self.sensor_range = sensor_range
        self.weights = {
            'distance': weight_distance,
            'gain': weight_gain,
            'actuation': weight_actuation
        }
        self.largents = []
        self.last_assignment = {}
        self.candidate_goals = []
        self.plan_interval = BRAIN_REASON_INTERVAL
        self.last_plan_time = 0.0
        self.cluster_eps = SENSOR_LARGE * 0.4

    def generate_pseudo_random_goals(self, global_map, num_goals=20):
        """生成伪随机目标点（类似E-RRT的目标生成策略）"""
        frontiers = self.find_frontiers(global_map)
        if not frontiers:
            return []

        # 从前沿点中随机选择候选目标
        indices = np.random.choice(len(frontiers), 
                                 min(num_goals, len(frontiers)), 
                                 replace=False)
        return [frontiers[i] for i in indices]

    def calculate_information_gain(self, goal, global_map, agent_pos):
        """计算目标点的信息增益（类似E-RRT的探索收益评估）"""
        x, y = goal
        info_gain = 0
        
        # 模拟传感器视野（简化版）
        for dy in range(-int(self.sensor_range), int(self.sensor_range)+1):
            for dx in range(-int(self.sensor_range), int(self.sensor_range)+1):
                dist = math.sqrt(dx**2 + dy**2)
                if dist <= self.sensor_range:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < global_map.shape[1] and 
                        0 <= ny < global_map.shape[0]):
                        try:
                            if global_map[ny, nx] == UNKNOWN:  # 未知区域
                                # 检查视线是否被阻挡（简化版碰撞检测）
                                if self.line_of_sight_clear((x, y), (nx, ny), global_map):
                                    info_gain += 1
                        except:
                            pass
        return info_gain

    def line_of_sight_clear(self, start, end, global_map):
        """检查两点之间视线是否清晰（简化版碰撞检测）"""
        x1, y1 = start
        x2, y2 = end
        
        # Bresenham直线算法检查路径上的点
        points = self.bresenham_line(x1, y1, x2, y2)
        for px, py in points:
            if (0 <= px < global_map.shape[1] and 
                0 <= py < global_map.shape[0]):
                if global_map[py, px] == OBSTACLE:  # 障碍物
                    return False
        return True

    def bresenham_line(self, x1, y1, x2, y2):
        """Bresenham直线算法"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = -1 if x1 > x2 else 1
        sy = -1 if y1 > y2 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x2:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x, y))
        return points

    def calculate_distance_cost(self, agent_pos, goal):
        """计算距离成本"""
        return np.linalg.norm(np.array(agent_pos) - np.array(goal))

    def build_cost_matrix(self, agents, candidate_goals, global_map):
        """构建多目标优化的成本矩阵"""
        num_agents = len(agents)
        num_goals = len(candidate_goals)
        
        # 初始化成本矩阵
        cost_matrix = np.zeros((num_agents, num_goals))
        
        for i, agent in enumerate(agents):
            for j, goal in enumerate(candidate_goals):
                # 距离成本
                dist_cost = self.calculate_distance_cost(agent.pos, goal)
                
                # 信息增益（负成本，因为我们要最大化增益）
                info_gain = self.calculate_information_gain(goal, global_map, agent.pos)
                gain_cost = -info_gain  # 负号表示我们要最大化这个值
                
                
                # 加权总和（类似E-RRT的代价函数）
                total_cost = (self.weights['distance'] * dist_cost +
                            self.weights['gain'] * gain_cost)
                
                cost_matrix[i, j] = total_cost
                
        return cost_matrix

    def optimize_assignment(self, cost_matrix):
        """使用匈牙利算法进行最优分配"""
        # 匈牙利算法求解最小成本匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind      

    def decide(self, brain_agent, large_agents):

        self.largents = large_agents
        self.min_cluster_size = max(NUM_LARGE, len(large_agents))
        global_map = brain_agent.local_map
        goals = self.compute_global_plan(global_map)

        if not goals or len(large_agents) == 0:
            return None

        # 基于距离的匈牙利分配
        cost_matrix = self.build_cost_matrix(large_agents, goals, global_map)

        # 3. 使用优化算法进行分配
        try:
            row_ind, col_ind = self.optimize_assignment(cost_matrix)

            # 4. 构建分配结果
            assignments = {}
            for i, agent_id in enumerate([a.id for a in large_agents]):
                if i in row_ind:
                    match_idx = np.where(row_ind == i)[0][0]
                    goal_idx = col_ind[match_idx]
                    assignments[agent_id] = goals[goal_idx]
                else:
                    # 如果没有匹配到目标，使用fallback策略
                    assignments[agent_id] = self.fallback_assignment(large_agents[i], global_map)
                    print(f"Agent {agent_id} has no matched goal, using fallback.")
        except Exception as e:
            print(f"Optimization failed: {e}, using fallback strategy")
            return self.fallback_assignment_strategy(large_agents, global_map)

        # row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # assignments = {large_agents[i].id: goals[j] for i, j in zip(row_ind, col_ind)}
        return assignments
    

    def fallback_assignment(self, agent, global_map):
        """备选分配策略（当优化失败时使用）"""
        frontiers = self.find_frontiers(global_map)
        if not frontiers:
            return None
        
        # 简单的最远前沿策略
        distances = [np.linalg.norm(np.array(agent.pos) - np.array(f)) 
                    for f in frontiers]
        return frontiers[np.argmax(distances)]

    def fallback_assignment_strategy(self, agents, global_map):
        """全局备选分配策略"""
        frontiers = self.find_frontiers(global_map)
        if not frontiers:
            return {a.id: None for a in agents}
        
        # 使用原始的距离最近分配策略
        assignments = {}
        available = frontiers.copy()
        for agent in agents:
            if not available:
                assignments[agent.id] = None
                continue
            distances = [np.linalg.norm(np.array(agent.pos) - np.array(f)) 
                        for f in available]
            target = available[np.argmin(distances)]
            assignments[agent.id] = target
            available.remove(target)
        
        return assignments

    # =====================================================
    # 全局任务生成
    # =====================================================
    def compute_global_plan(self, global_map):
        """根据全局地图生成若干高层目标点"""
        global_goals = []
        frontiers = self.find_frontiers(global_map)
        if frontiers is None or len(frontiers) == 0:
            return []

        clusters = self.cluster_frontiers(frontiers)
        selected_clusters = [self.cluster_centroid(c) for c in clusters if len(c) > self.min_cluster_size]
        selected_clusters.sort(reverse=True, key=lambda x: x[0])
        for cnt,centroid in enumerate(selected_clusters):
            if centroid is not None:
                global_goals.append(centroid)
        final = []
        for g in global_goals:
            gx = clamp(g[0], 0, WORLD_W)
            gy = clamp(g[1], 0, WORLD_H)
            if not any(math.hypot(gx - f[0], gy - f[1]) < 10.0 for f in final):
                x,y = cell_of_pos((gx, gy))
                final.append((x, y))
        self.global_goals = final
        return self.global_goals
    
    def find_frontiers(self, global_map, safe_margin=10):
        """
        识别地图前沿区域（带安全检查）
        ---------------------------------------------------
        逻辑：
        1. 未知单元格（UNKNOWN）
        2. 邻近至少一个FREE单元格（表示边界）
        3. 周围 safe_margin 范围内没有 OBSTACLE / DANGER
        """
        frontiers = []
        h, w = global_map.shape

        for y in range(safe_margin, h - safe_margin):
            for x in range(safe_margin, w - safe_margin):
                # ✅ Step 1: 必须是未知区
                if global_map[y, x] != UNKNOWN:
                    continue

                # ✅ Step 2: 必须邻近自由区
                # neighbors = global_map[y-1:y+2, x-1:x+2].flatten()
                # if not np.any(neighbors == FREE):
                #     continue

                # ✅ Step 3: 安全检查（周围不能有障碍或危险）
                region = global_map[
                    y - safe_margin:y + safe_margin + 1,
                    x - safe_margin:x + safe_margin + 1
                ]
                if np.any(region == OBSTACLE) or np.any(region == DANGER):
                    continue

                # ✅ Step 4（可选）: 视线检查（从大脑节点中心到该点）
                # if hasattr(self, "brain_center"):
                #     bx, by = cell_of_pos(self.brain_center)
                #     if not self.line_of_sight_clear((bx, by), (x, y), global_map):
                #         continue

                # ✅ Step 5: 转换为世界坐标后保存
                gx, gy = pos_of_cell(x, y)
                frontiers.append((gx, gy))
                # frontiers.append((x, y))
        return frontiers

    # =====================================================
    # 前沿聚类（Frontier Clustering）
    # =====================================================
    
    def cluster_frontiers(self, frontiers_world_pts):
        # 使用简单的聚类（基于距离） -> 返回 list of arrays of points
        if frontiers_world_pts is None or len(frontiers_world_pts) == 0:
            return []
        if len(frontiers_world_pts) <= self.min_cluster_size:
            # too few points -> each point is its own small cluster (but we'll filter later)
            return [frontiers_world_pts]

        # fclusterdata from scipy: Euclidean clustering; t = cluster_eps, criterion='distance'
        try:
            labels = fclusterdata(frontiers_world_pts, t=self.cluster_eps, criterion='distance')
            clusters = []
            for lab in np.unique(labels):
                members = frontiers_world_pts[labels == lab]
                if len(members) >= 1:
                    clusters.append(members)
            return clusters
        except Exception:
            # fallback: simple grid-binning clustering
            bins = {}
            for x,y in frontiers_world_pts:
                bx = int(x // self.cluster_eps)
                by = int(y // self.cluster_eps)
                bins.setdefault((bx,by),[]).append((x,y))
            return [np.array(v) for v in bins.values()]

    # =====================================================
    # 计算每个簇的中心点
    # =====================================================
    def cluster_centroid(self, cluster_points):
        """
        计算聚类中心（加权或简单平均）
        """
        if len(cluster_points) == 0:
            return None

        # 简单平均（可替换为加权中心）
        cx = np.mean(cluster_points[:, 0])
        cy = np.mean(cluster_points[:, 1])
        return (cx, cy)

class InformedLocalAssignmentBehavior(Multi_Behavior):
    """
    信息驱动的局部任务分配行为
    ----------------------------------------
    功能：
      - 在大节点目标点周围采样；
      - 加入约束（靠近本体且不过远离局部目标）；
      - 计算信息增益并选择最优采样点；
      - 使用匈牙利算法分配；
      - 自动将大节点自身目标点后移到安全位置。
    """

    def __init__(self,
                 sample_radius=40,
                 num_samples=50,
                 num_selected=12,
                 safe_margin=20,
                 sensor_range=6.0,
                 w_gain=2.0,
                 w_large_dist=0.5,
                 w_goal_dist=0.1,
                 retreat_ratio=0.85):
        self.sample_radius = sample_radius      # 采样半径
        self.num_samples = num_samples          # 采样候选数量
        self.num_selected = num_selected        # 最终选择的目标数
        self.safe_margin = safe_margin
        self.sensor_range = sensor_range
        self.w_gain = w_gain
        self.w_large_dist = w_large_dist
        self.w_goal_dist = w_goal_dist
        self.retreat_ratio = retreat_ratio
        self.last_assignment = {}

    # =====================================================
    # 主接口
    # =====================================================
    def decide(self, large_agent, sub_agents):
        if large_agent.goal is None:
            return {a.id: None for a in sub_agents}

        global_map = large_agent.known_map
        large_pos = np.array(large_agent.pos)
        goal_pos = np.array(large_agent.goal)

        # 1️⃣ 采样并计算信息增益
        sampled_points = self.sample_candidates(global_map, large_pos, goal_pos)

        if not sampled_points:
            print(f"[Warning] LargeAgent {large_agent.id}: 无有效采样点")
            return {a.id: None for a in sub_agents}

        # 2️⃣ 从采样点中选出最优若干个
        selected_points = self.select_best_points(global_map, sampled_points, large_pos, goal_pos)

        if len(selected_points) == 0:
            return {a.id: None for a in sub_agents}

        # 3️⃣ 构建代价矩阵并分配小节点
        cost_matrix = np.zeros((len(sub_agents), len(selected_points)))
        for i, ag in enumerate(sub_agents):
            for j, g in enumerate(selected_points):
                cost_matrix[i, j] = np.linalg.norm(np.array(ag.pos) - np.array(g))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments = {}
        for i, ag in enumerate(sub_agents):
            if i in row_ind:
                j = col_ind[np.where(row_ind == i)[0][0]]
                assignments[ag.id] = selected_points[j]
            else:
                assignments[ag.id] = None

        self.last_assignment = assignments

        # 4️⃣ 调整大节点自身目标（后移）
        large_agent.goal = self.recede_large_goal(large_pos, goal_pos)

        return assignments

    # =====================================================
    # 候选点采样
    # =====================================================
    def sample_candidates(self, global_map, large_pos, goal_pos):
        """
        在大节点当前位置为圆心、指向目标方向的半圆区域采样
        """
        candidates = []
        goal_dir = np.array(goal_pos) - np.array(large_pos)
        if np.linalg.norm(goal_dir) < 1e-6:
            return []  # 目标方向无效

        goal_dir = goal_dir / np.linalg.norm(goal_dir)  # 单位化方向

        for _ in range(self.num_samples * 3):  # 适当放宽采样次数，以免半圆过滤太多
            ang = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0.5 * self.sample_radius, self.sample_radius)
            dx = r * np.cos(ang)
            dy = r * np.sin(ang)
            candidate = np.array([int(goal_pos[0] + dx), int(goal_pos[1] + dy)])
            candidate_dir =  candidate - goal_pos
            candidate_dir = candidate_dir / np.linalg.norm(candidate_dir + 1e-6)
            # ✅ 判断是否在“目标方向半圆”内：点积 > 0 表示锐角
            if np.dot(candidate_dir, goal_dir) < 0:
                continue

            # ✅ 转格子坐标
            i, j = cell_of_pos(candidate)
            if not (0 <= i < global_map.shape[1] and 0 <= j < global_map.shape[0]):
                continue

            # ✅ 检查安全
            if self.is_point_safe(global_map, (i, j)):
                candidates.append(tuple(candidate))

            if len(candidates) >= self.num_samples:
                break

        return candidates

    def is_point_safe(self, global_map, point):
        """检查该点附近是否安全"""
        x, y = point
        region = global_map[max(0, y-2):min(global_map.shape[0], y+3),
                            max(0, x-2):min(global_map.shape[1], x+3)]
        if np.any(region == OBSTACLE) or np.any(region == DANGER):
            return False
        return True

    # =====================================================
    # 信息增益与多目标评分
    # =====================================================
    def select_best_points(self, global_map, candidates, large_pos, goal_pos):
        """计算信息增益 + 距离约束，选出最优采样点"""
        scored_points = []
        for pt in candidates:
            info_gain = self.calculate_information_gain(global_map, pt)
            dist_large = np.linalg.norm(np.array(pt) - large_pos)
            dist_goal = np.linalg.norm(np.array(pt) - goal_pos)

            # 奖励靠近大节点，惩罚远离目标
            score = (self.w_gain * info_gain
                     - self.w_large_dist * dist_large
                     - self.w_goal_dist * dist_goal)
            scored_points.append((score, pt))

        # 按得分排序，取前K个
        scored_points.sort(reverse=True, key=lambda x: x[0])
        best = [p for _, p in scored_points[:self.num_selected]]
        return best

    def calculate_information_gain(self, global_map, center):
        """估算某点的信息增益（统计视野范围内未知格数）"""
        cx, cy = center
        r = int(self.sensor_range)
        x0 = max(0, cx - r)
        x1 = min(global_map.shape[1], cx + r)
        y0 = max(0, cy - r)
        y1 = min(global_map.shape[0], cy + r)
        region = global_map[y0:y1, x0:x1]
        return np.sum(region == UNKNOWN)

    # =====================================================
    # 大节点目标后移
    # =====================================================
    def recede_large_goal(self, large_pos, goal_pos):
        """将大节点目标往自身方向缩回一定比例"""
        dx, dy = goal_pos[0] - large_pos[0], goal_pos[1] - large_pos[1]
        new_gx = large_pos[0] + dx * self.retreat_ratio
        new_gy = large_pos[1] + dy * self.retreat_ratio
        return (new_gx, new_gy)




class FormationAssignmentBehavior(Multi_Behavior):
    """
    编队式局部任务分配行为
    ------------------------------------------------
    功能：
      - 根据 formation_type 生成编队形状；
      - 自动在大节点目标方向上布置子节点位置；
      - 通过匈牙利算法匹配小节点；
      - 大节点自身后撤到安全位置。
    """

    def __init__(self, formation_type="vshape", spacing=30, retreat_ratio=0.85):
        self.formation_type = formation_type
        self.spacing = spacing              # 子节点间的间距
        self.retreat_ratio = retreat_ratio  # 大节点后撤比例
        self.last_assignment = {}

    # =====================================================
    # 主接口
    # =====================================================
    def decide(self, large_agent, sub_agents):
        if large_agent.goal is None:
            return {a.id: None for a in sub_agents}

        large_pos = np.array(large_agent.pos)
        goal_pos = np.array(large_agent.goal)
        direction = goal_pos - large_pos
        if np.linalg.norm(direction) < 1e-6:
            return {a.id: None for a in sub_agents}

        direction = direction / np.linalg.norm(direction)
        perpendicular = np.array([-direction[1], direction[0]])  # 垂直方向

        # 1️⃣ 生成编队目标点（相对位置）
        formation_points = self.generate_formation_points(
            center=goal_pos,
            direction=direction,
            perpendicular=perpendicular,
            num_agents=len(sub_agents)
        )

        # 2️⃣ 匈牙利分配
        cost_matrix = np.zeros((len(sub_agents), len(formation_points)))
        for i, ag in enumerate(sub_agents):
            for j, g in enumerate(formation_points):
                cost_matrix[i, j] = np.linalg.norm(np.array(ag.pos) - np.array(g))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments = {}
        for i, ag in enumerate(sub_agents):
            if i in row_ind:
                j = col_ind[np.where(row_ind == i)[0][0]]
                assignments[ag.id] = formation_points[j]
            else:
                assignments[ag.id] = None

        self.last_assignment = assignments

        # 3️⃣ 大节点目标后移（保持安全）
        large_agent.goal = self.recede_large_goal(large_pos, goal_pos)

        return assignments

    # =====================================================
    # 编队点生成逻辑
    # =====================================================
    def generate_formation_points(self, center, direction, perpendicular, num_agents):
        """根据 formation_type 生成相对布局"""
        points = []

        if self.formation_type == "line":
            # ---- 沿目标方向排成一条直线 ----
            for i in range(num_agents):
                offset = direction * (i + 1) * self.spacing
                points.append(center + offset)

        elif self.formation_type == "vshape":
            # ---- V形编队（左右分散）----
            for i in range(num_agents):
                side = 1 if i % 2 == 0 else -1
                layer = (i // 2 + 1)
                offset = direction * layer * self.spacing + perpendicular * side * layer * self.spacing * 0.7
                points.append(center + offset)

        elif self.formation_type == "circle":
            # ---- 环形编队 ----
            radius = self.spacing * max(2, num_agents / (2 * np.pi))
            for i in range(num_agents):
                angle = 2 * np.pi * i / num_agents
                dx = radius * np.cos(angle)
                dy = radius * np.sin(angle)
                points.append(center + np.array([dx, dy]))

        elif self.formation_type == "wedge":
            # ---- 扇形编队（更开阔）----
            for i in range(num_agents):
                theta = np.deg2rad(-45 + 90 * (i / max(1, num_agents - 1)))
                rot = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])
                offset = rot.dot(direction) * self.spacing * 2
                points.append(center + offset * (i + 1) / 2)

        else:
            raise ValueError(f"❌ Unknown formation type: {self.formation_type}")

        return points

    # =====================================================
    # 安全性调整（后撤）
    # =====================================================
    def recede_large_goal(self, large_pos, goal_pos):
        dx, dy = goal_pos[0] - large_pos[0], goal_pos[1] - large_pos[1]
        new_gx = large_pos[0] + dx * self.retreat_ratio
        new_gy = large_pos[1] + dy * self.retreat_ratio
        return (new_gx, new_gy)







