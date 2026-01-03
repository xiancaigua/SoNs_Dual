
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

class PathPlanningBehavior(Behavior):
    """
    路径规划 + 随机探索 + 安全约束 的智能行为类
    --------------------------------------------------------
    功能：
      - 基于 local_map 的 A* 路径规划
      - 当目标点变化或路径失效时自动重规划
      - 到达目标后自动切换为随机探索
      - 随机探索时仍考虑危险与障碍的安全避让
    """

    def __init__(self, replan_interval=1, goal_tolerance=10.0):
        super().__init__()
        self.replan_interval = replan_interval
        self.goal_tolerance = goal_tolerance
        self.last_plan_time = -999.0
        self.cached_path = None
        self.path_index = 0
        self.last_goal = None

    # ======================================================
    # 基础：A* 路径规划
    # ======================================================
    def astar(self, grid, start, goal):
        w, h = grid.shape[1], grid.shape[0]
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal)}

        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = current[0]+dx, current[1]+dy
                if 0 <= nx < w and 0 <= ny < h:
                    val = grid[ny, nx]   # ✅ 行列顺序修正
                    if val in [OBSTACLE, DANGER]:  # ✅ 安全过滤
                        continue
                    tentative_g = gscore[current] + 1
                    if (nx, ny) not in gscore or tentative_g < gscore[(nx, ny)]:
                        came_from[(nx, ny)] = current
                        gscore[(nx, ny)] = tentative_g
                        fscore[(nx, ny)] = tentative_g + self.heuristic((nx, ny), goal)
                        heapq.heappush(open_list, (fscore[(nx, ny)], (nx, ny)))
        return None

    def heuristic(self, a, b):
        """启发式函数（曼哈顿距离）"""
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def reconstruct_path(self, came_from, current):
        """重建A*路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ======================================================
    # 主行为决策逻辑
    # ======================================================
    def decide(self, agent):
        now = getattr(agent, "sim_time", 0.0)
        grid = agent.local_map

        # ---- [1] 检查是否有目标 ----
        if not agent.has_goal or agent.goal is None:
            # 无目标 → 安全随机游走
            return self.safe_random_explore(agent, grid)
            return 0, 0

        # ---- [2] 检查是否到达目标 ----
        goal_dist = distance(agent.pos, agent.goal)
        if goal_dist < self.goal_tolerance:
            agent.has_goal = False
            agent.goal = None
            return self.safe_random_explore(agent, grid)
            return 0, 0

        # ---- [3] 目标点是否变化？ ----
        goal_changed = (
            self.last_goal is None or 
            distance(self.last_goal, agent.goal) > self.goal_tolerance
        )

        start_cell = cell_of_pos(agent.pos)
        goal_cell = cell_of_pos(agent.goal)

        # ---- [4] 判断是否需要重新规划 ----
        need_replan = (
            self.cached_path is None or
            self.path_index >= len(self.cached_path) or
            goal_changed or
            now - self.last_plan_time > self.replan_interval
        )

        if need_replan:
            path = self.astar(grid, start_cell, goal_cell)
            if path is None:
                # 无路径 → 安全随机游走
                return self.safe_random_explore(agent, grid)
                return 0, 0
            self.cached_path = path
            self.path_index = 0
            self.last_plan_time = now
            self.last_goal = agent.goal

        # ---- [5] 沿路径前进 ----
        if self.path_index < len(self.cached_path):
            target_cell = self.cached_path[self.path_index]
            target_pos = pos_of_cell(*target_cell)
            dx = target_pos[0] - agent.pos[0]
            dy = target_pos[1] - agent.pos[1]
            dist = math.hypot(dx, dy)
            if dist < 5:  # 到达该节点
                self.path_index += 1
            sx, sy = normalize((dx, dy))
            vx = sx * AGENT_MAX_SPEED
            vy = sy * AGENT_MAX_SPEED
            return vx, vy

        # ---- [6] 路径执行完毕 ----
        agent.has_goal = False
        agent.goal = None
        return self.safe_random_explore(agent, grid)
        return 0, 0

    # ======================================================
    # 安全随机探索逻辑
    # ======================================================
    def safe_random_explore(self, agent, grid):
        """
        随机探索时也遵守安全约束：
        - 避开 OBSTACLE / DANGER 区域；
        - 保持平滑方向变化；
        """
        if random.random() < 0.02:
            ang = random.uniform(0, 2*math.pi)
            agent.vel = (math.cos(ang)*10.0, math.sin(ang)*10.0)

        vx, vy = agent.vel
        sx, sy = normalize((vx, vy))

        # 预测下一个位置，检查是否安全
        next_x = int(clamp(agent.pos[0] + sx * 5, 0, grid.shape[0]-1))
        next_y = int(clamp(agent.pos[1] + sy * 5, 0, grid.shape[1]-1))

        if grid[next_x, next_y] in [1, 2]:  # OBSTACLE 或 DANGER
            # 选择随机安全方向
            safe_dirs = []
            for ang in np.linspace(0, 2*math.pi, 16):
                dx, dy = math.cos(ang), math.sin(ang)
                tx = int(clamp(agent.pos[0] + dx * 5, 0, grid.shape[0]-1))
                ty = int(clamp(agent.pos[1] + dy * 5, 0, grid.shape[1]-1))
                if grid[tx, ty] not in [1, 2]:
                    safe_dirs.append((dx, dy))
            if safe_dirs:
                sx, sy = random.choice(safe_dirs)

        vx = sx * AGENT_MAX_SPEED
        vy = sy * AGENT_MAX_SPEED
        return vx, vy


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
        global_map = agent.known_map
        
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
                    assignments[agent_id] = self.candidate_goals[goal_idx]
                else:
                    # 如果没有匹配到目标，使用fallback策略
                    assignments[agent_id] = self.fallback_assignment(agents[i], global_map)
            
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
        self.cluster_eps = 10

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
        now = time.time()
        if now - self.last_plan_time < self.plan_interval:
            return None
        self.last_plan_time = now

        self.largents = large_agents
        self.min_cluster_size = 1
        # self.min_cluster_size = max(NUM_LARGE, len(large_agents))
        global_map = brain_agent.known_map
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
        selected_clusters = [self.cluster_centroid(c) for c in clusters if len(c) >= self.min_cluster_size]
        selected_clusters.sort(reverse=True, key=lambda x: x[0])
        for cnt,centroid in enumerate(selected_clusters):
            if centroid is not None:
                global_goals.append(centroid)
        final = []
        for g in global_goals:
            gx = clamp(g[0], 0, WORLD_W)
            gy = clamp(g[1], 0, WORLD_H)
            if not any(math.hypot(gx - f[0], gy - f[1]) < 10.0 for f in final):
                final.append((gx, gy))
        self.global_goals = final
        return self.global_goals
    
    def find_frontiers(self, global_map, safe_margin=2):
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
                neighbors = global_map[y-1:y+2, x-1:x+2].flatten()
                if not np.any(neighbors == FREE):
                    continue

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
                 sample_radius=100,
                 num_samples=50,
                 num_selected=12,
                 safe_margin=20,
                 sensor_range=6.0,
                 w_gain=2000.0,
                 w_large_dist=0.0,
                 w_goal_dist=0.0,
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
        if large_agent.task_seq is None:
            return {a.id: None for a in sub_agents}

        global_map = large_agent.known_map
        large_pos = np.array(large_agent.pos)
        goal_pos = np.array(large_agent.task_seq[0])

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
        large_agent.task_seq = [self.recede_large_goal(large_pos, goal_pos)]

        return assignments

    # =====================================================
    # 候选点采样
    # =====================================================
    def sample_candidates(self, global_map, large_pos, goal_pos):
        """
        在大节点当前位置为圆心、指向目标方向的半圆区域采样
        """
        candidates = []

        for _ in range(self.num_samples * 3):  # 适当放宽采样次数，以免半圆过滤太多
            ang = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0.8 * self.sample_radius, self.sample_radius)
            dx = r * np.cos(ang)
            dy = r * np.sin(ang)
            candidate = np.array([int(goal_pos[0] + dx), int(goal_pos[1] + dy)])
            candidate_dir =  candidate - goal_pos
            candidate_dir = candidate_dir / np.linalg.norm(candidate_dir + 1e-6)

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
        # region = global_map[max(0, y-2):min(global_map.shape[0], y+3),
        #                     max(0, x-2):min(global_map.shape[1], x+3)]
        # if np.any(region == OBSTACLE) or np.any(region == DANGER):
        if global_map[y,x] == OBSTACLE or global_map[y,x] == DANGER:
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

class MCTSNode:
    """MCTS 树节点"""
    def __init__(self, position, parent=None, action_to_reach=None):
        self.position = position  # 节点代表的机器人位置 (x, y)
        self.parent = parent
        self.action_to_reach = action_to_reach  # 到达此节点的动作（目标点）
        
        self.children = []
        self.visits = 0
        self.value = 0.0  # 累积奖励
        self.untried_actions = [] # 尚未扩展的动作

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=2.0): # 增大 C 参数，鼓励探索更远的目标 (解决问题 2)
        """使用 UCT 公式选择最佳子节点"""
        if not self.children:
            return None
        
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

class DMCEExplorationBehavior(Multi_Behavior):
    """
    基于 DMCE (Decentralized Monte Carlo Exploration) 论文思想的栅格化实现。
    核心特性：
    1. 使用 MCTS 进行多步路径规划。
    2. 动作空间：前沿点聚类中心 + 局部随机采样。
    3. 协作：在 Rollout 阶段扣除队友计划覆盖的区域增益。
    """

    def __init__(self, simulation_budget=10, max_depth=3, rollout_depth=5, num_best_targets=4,world=None):
            super().__init__()
            self.sensor_range = SENSOR_SMALL
            self.simulation_budget = simulation_budget  # MCTS 迭代次数 (影响速度)
            self.max_tree_depth = max_depth             # 树的最大深度
            self.rollout_depth = rollout_depth          # Rollout 往下模拟几步 (解决目标保守问题)
            self.num_best_targets = num_best_targets    # 最终输出的最佳目标数量 (新增需求)
            self.c_param = 12.0                          # 增大 UCT C 参数 (解决目标保守问题)
            self.DISTANCE_PENALTY_WEIGHT = 0.01         # 距离惩罚权重 (解决规划失败和保守问题)
            
            self.cached_frontiers = []
            self.world = world

    # =================================================
    # 1. MCTS 主流程 (decide)
    # =================================================
    def decide(self, agent, agents):
        grid = getattr(agent, "known_map", agent.local_map)
        my_pos = agent.pos
        cell_of_mypos = cell_of_pos(my_pos)

        # 1. 前沿点
        self.cached_frontiers = self._find_frontier_centroids(grid, cell_of_mypos)
        if not self.cached_frontiers:
            return {}

        # 2. 队友计划
        peer_plans = self._collect_peer_plans(agent, agents)

        # 3. MCTS
        root = MCTSNode(position=my_pos)
        root.untried_actions = self._get_feasible_actions(root.position, grid)

        for _ in range(self.simulation_budget):
            leaf = self._select(root)
            depth = self._get_depth(leaf)

            if depth < self.max_tree_depth:
                child = self._expand(leaf, grid)
                if child:
                    reward = self._simulate(child, grid, peer_plans)
                    self._backpropagate(child, reward)
            else:
                reward = self._evaluate_state(leaf.position, grid, peer_plans)
                self._backpropagate(leaf, reward)

        if not root.children:
            return {}

        # 4. 取最优目标
        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        best_targets = [
            pos_of_cell(child.action_to_reach[0],child.action_to_reach[1])
            for child in sorted_children[:self.num_best_targets]
        ]

        for bt in best_targets:
            if bt[0]>WORLD_W or bt[1]>WORLD_H or bt[0]<0 or bt[1]<0:
                best_targets.remove(bt)
                print(f"[Warning] MCTS generated out-of-bounds target: {bt}")
        # 5. 找 son agents
        son_agents = [a for a in agents if getattr(a, "father_id", None) == agent.id]

        # 6. 分配
        return self._assign_targets_to_sons(son_agents, best_targets)


    # =================================================
    # 2. MCTS 四大核心步骤
    # =================================================
    def _select(self, node:MCTSNode):
        """Selection: 递归选择 UCT 最大的节点，直到遇到未完全扩展的节点"""
        while not node.untried_actions and node.children:
            node = node.best_child(self.c_param)
        return node

    def _expand(self, node, grid):
        """Expansion: 从未尝试的动作中选一个，生成新节点"""
        if not node.untried_actions:
            return None
        
        action_target = node.untried_actions.pop()
        child_node = MCTSNode(position=action_target, parent=node, action_to_reach=action_target)
        child_node.untried_actions = self._get_rollout_actions(child_node.position, grid)
        
        node.children.append(child_node)
        return child_node

    def _simulate(self, node, grid, peer_plans):
        """Rollout: 随机游走若干步，计算累积奖励 (修正：加入距离惩罚)"""
        current_pos = node.position
        accumulated_reward = self._evaluate_state(current_pos, grid, peer_plans)
        
        gamma = 0.95 # 提高折扣因子，鼓励远距离规划 (解决问题 2)
        
        for _ in range(self.rollout_depth):
            possible_moves = self._get_rollout_actions(current_pos, grid, max_count=5)
            if not possible_moves:
                break
            
            # 1. 选择下一个动作
            next_pos = random.choice(possible_moves)
            
            # 2. 计算距离成本（欧式距离，假设是世界坐标）
            dist = math.hypot(next_pos[0] - current_pos[0], next_pos[1] - current_pos[1]) 
            
            # 3. 计算收益 (信息增益)
            step_reward = self._evaluate_state(next_pos, grid, peer_plans)
            
            # 4. 计算净奖励：(信息增益) - (距离惩罚)
            # 引入距离惩罚解决规划失败和保守问题
            net_reward = step_reward - (dist * self.DISTANCE_PENALTY_WEIGHT)
            
            # 5. 累加（带折扣因子）
            # 仅在净收益为正时累加，或允许负值以惩罚高成本低收益路径
            accumulated_reward += net_reward * gamma 
            current_pos = next_pos
            
        return accumulated_reward

    def _backpropagate(self, node, reward):
        """Backpropagation: 沿树向上传递奖励"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    # =================================================
    # 3. 辅助逻辑 (动作生成、收益评估、协作)
    # =================================================
    
    def _get_feasible_actions(self, pos, grid, max_count=10):
        """
        生成可行动作集。
        策略：优先选择前沿聚类中心，其次是附近的空闲点。
        """
        actions = []
        
        # 1. 加入全局前沿点 (Global Exploration)
        # 为了防止动作空间过大，只取距离当前 pos 最近的 N 个前沿点
        if self.cached_frontiers:
            # 按距离排序
            sorted_frontiers = sorted(
                self.cached_frontiers, 
                key=lambda p: (p[0]-pos[0])**2 + (p[1]-pos[1])**2
            )
            actions.extend(sorted_frontiers[:max_count])
            
        # 2. (可选) 加入局部随机点 (Local Exploitation)
        # 在 pos 周围随机采样几个 free point，模拟 RRT 的 extend
        for _ in range(3):
            rand_x = int(pos[0] + random.uniform(-10, 10))
            rand_y = int(pos[1] + random.uniform(-10, 10))
            cell_of_randxy = cell_of_pos((rand_x,rand_y))
            if 0<=cell_of_randxy[0]<GRID_W and 0<=cell_of_randxy[1]<GRID_H and grid[cell_of_randxy[1],cell_of_randxy[0]] == FREE:
                actions.append((rand_x, rand_y))
            
        # 去重
        return list(set(actions))

    def _get_rollout_actions(self, pos, grid, max_count=5):
        """
        Rollout专用动作：生成短距离、局部随机的动作。
        这模拟了机器人接下来几步的连续、快速移动，而不是跳跃到远处的全局目标。
        
        Args:
            pos (tuple): 当前世界坐标 (x, y)。
            grid (np.array): 地图。
            max_count (int): 返回的动作最大数量。
        
        Returns:
            list: 短距离目标点的世界坐标列表。
        """
        actions = []
        # 1. 局部随机采样 (模拟短距离行走)
        for _ in range(max_count):
            # 采样一个较小的局部区域，模拟几步行走。
            # 假设步长不超过 10-20 个世界坐标单位
            # 如果 GRID_CELL 很大，这个范围也要相应调整
            rand_x = pos[0] + random.uniform(-10, 10) 
            rand_y = pos[1] + random.uniform(-10, 10)
            cell_of_randxy = cell_of_pos((rand_x,rand_y))
            if 0<=cell_of_randxy[0]<GRID_W and 0<=cell_of_randxy[1]<GRID_H and grid[cell_of_randxy[1],cell_of_randxy[0]] == FREE:
                actions.append((rand_x, rand_y))
            # 理想情况下，这里应该检查 (rand_x, rand_y) 是否在自由空间内，
            # 为了 Rollout 速度，可以跳过复杂的碰撞检测，只做边界检查。
            
            actions.append((rand_x, rand_y))
            
        # 2. (可选但推荐) 加入相邻的栅格中心点（模拟简单步进）
        # c, r = cell_of_pos(pos)
        # for dc in [-1, 0, 1]:
        #     for dr in [-1, 0, 1]:
        #         if dc != 0 or dr != 0:
        #             try:
        #                 actions.append(pos_of_cell(c + dc, r + dr))
        #             except:
        #                 pass

        return list(set(actions))

    def _evaluate_state(self, pos, grid, peer_plans):
        """
        计算收益 (Reward Function)。
        """
        # 1. 模拟传感器视野
        visible_cells = self._get_visible_cells(pos, grid)
        
        # 2. 计算基础增益 (有多少个 UNKNOWN)
        gain = 0
        for (c, r) in visible_cells:
            if 0<=c<GRID_W and 0<=r<GRID_H:
                if grid[r, c] == UNKNOWN: # UNKNOWN
                    gain += 1.5
                elif grid[r, c] == DANGER:
                    gain -= 2
        
        # 3. 协作惩罚
        for (c, r) in visible_cells:
            for peer_traj in peer_plans:
                if self._is_covered_by_peers(c, r, peer_traj):
                    gain -= 10 # 扣除收益
                    break
                    
        return max(0, gain)

    def _collect_peer_plans(self, my_agent, all_agents):
        """
        收集其他**大机器人**的未来计划 (轨迹点列表)。
        注意：这里的 all_agents 应该是所有大型探索机器人。
        """
        plans = []
        # 遍历所有大机器人，收集它们的计划。
        for a in all_agents: 
            if a.id != my_agent.id and a.alive:
                # 假设 large agent 有 planned_path 属性 (指向它的目标点)
                # 或者 a.task_seq 存储了它当前的下一个目标点
                if hasattr(a, 'planned_path') and a.planned_path:
                    plans.append(a.planned_path)
                elif hasattr(a, 'task_seq') and a.task_seq:
                    # 假设 task_seq 列表的第一个元素是下一个目标点
                    plans.append([a.task_seq[0]]) 
                elif hasattr(a, 'target_list') and a.target_list:
                    # 收集其分配给小机器人的所有目标点，这些都应该被避让
                    plans.extend(a.target_list)
        return plans

    def _is_covered_by_peers(self, c, r, peer_traj):
        """检查某个格子是否被队友的轨迹覆盖"""
        # 简单实现：检查是否距离轨迹上的点小于传感器半径
        # 为了性能，可以只检查轨迹的终点或几个关键点
        for px, py in peer_traj[::5]: # 降采样加速
            dist_sq = (c*GRID_CELL - px)**2 + (r*GRID_CELL - py)**2 # 假设 grid_size=20 转换坐标
            if dist_sq < (self.sensor_range * GRID_CELL)**2:
                return True
        return False

    def _find_frontier_centroids(self, grid, cell_of_mypos=None, cluster_radius_cells=2):
            """
            寻找前沿点并聚类求重心。
            1. 找到所有前沿单元格 (UNKNOWN 邻近 FREE)。
            2. 使用简单的贪婪聚类方法分组。
            3. 返回聚类中心的世界坐标。
            
            Args:
                grid (np.array): 机器人的已知全局地图。
                cluster_radius_cells (int): 用于分组前沿点的聚类半径 (栅格单位)。
                
            Returns:
                list: 世界坐标列表 [(x1, y1), (x2, y2), ...]
            """
            H, W = grid.shape
            all_frontiers_cells = []
            cx, cy = cell_of_mypos

            # 1. 找到所有前沿单元格 (Grid Coordinates: c, r)
            # 遍历时跳过地图边缘
            for r in range(cy-15, cy+16):
                for c in range(cx-15, cx+16):
                    if 0<=r<H and 0<=c<W:
                        # 必须是未知区域
                        if grid[r, c] == UNKNOWN: 
                            # 检查 8 邻域
                            neighbors = grid[r-1:r+2, c-1:c+2].flatten()
                            
                            # 如果邻居中有 FREE 区域，则这是一个前沿点
                            if np.any(neighbors == FREE):
                                all_frontiers_cells.append((c, r)) # (col, row) 栅格坐标

            if not all_frontiers_cells:
                return []

            # 2. 聚类和计算重心 (Simple Greedy Clustering)
            
            centroids_world_pos = all_frontiers_cells  

            return centroids_world_pos

    def _get_visible_cells(self, pos, grid):
        """获取 pos 位置传感器覆盖的栅格坐标列表"""
        # 简单圆形区域
        cells = []
        cx, cy = cell_of_pos(pos) # 转栅格坐标
        r = int(self.sensor_range)
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                if i*i + j*j <= r*r:
                    cells.append((cx+i, cy+j))
        return cells

    def _get_depth(self, node):
        d = 0
        while node.parent:
            node = node.parent
            d += 1
        return d
    
    def _get_nearest_frontier(self, pos):
        if not self.cached_frontiers: return None
        return min(self.cached_frontiers, key=lambda p: (p[0]-pos[0])**2 + (p[1]-pos[1])**2)


    def _assign_targets_to_sons(self, son_agents, targets):
        """
        将 DMCE 生成的目标点分配给子机器人
        返回: {son_id: target_pos}
        """
        H, W = GRID_H, GRID_W
        def is_valid_cell(r, c):
            """不泄露危险信息，仅判断是否可通行"""
            if not (0 <= r < H and 0 <= c < W):
                return False
            val = self.world.ground_grid[r, c]
            return val != OBSTACLE and val != DANGER   # 不判断 DANGER，避免信息泄露


        def random_valid_cell(max_trials=200):
            """兜底：随机采样一个合法目标点"""
            print("[Info] DMCE: 目标点不足，随机分配合法位置")
            for _ in range(max_trials):
                r = np.random.randint(0, H)
                c = np.random.randint(0, W)
                if is_valid_cell(r, c):
                    return (r, c)
            return None

        if not son_agents:
            return {}
        elif not targets or len(targets) < len(son_agents):
            for _ in range(len(son_agents) - len(targets)):
                targets.append(random_valid_cell())

        assignments = {}
        remaining_targets = targets.copy()

        for son in son_agents:
            if not remaining_targets:
                break

            sx, sy = son.pos
            # 找最近目标
            best_t = min(
                remaining_targets,
                key=lambda p: (p[0]-sx)**2 + (p[1]-sy)**2
            )
            assignments[son.id] = best_t
            remaining_targets.remove(best_t)

        return assignments
