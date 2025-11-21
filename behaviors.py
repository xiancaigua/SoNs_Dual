
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
        now = time.time()
        if now - self.last_plan_time < self.plan_interval:
            return None
        self.last_plan_time = now

        self.largents = large_agents
        self.min_cluster_size = max(NUM_LARGE, len(large_agents))
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
                final.append((gx, gy))
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

    def best_child(self, c_param=1.414):
        """使用 UCT 公式选择最佳子节点"""
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

    def __init__(self, sensor_range=6.0, simulation_budget=50, max_depth=5, rollout_depth=3):
        super().__init__()
        self.sensor_range = SENSOR_SMALL
        self.simulation_budget = simulation_budget # MCTS 迭代次数
        self.max_tree_depth = max_depth            # 树的最大深度
        self.rollout_depth = rollout_depth         # Rollout 往下模拟几步
        self.c_param = 1.414                       # 探索/利用平衡参数
        
        self.cached_frontiers = [] # 缓存前沿点以加速扩展

    # =================================================
    # 1. MCTS 主流程
    # =================================================
    def decide(self, agent, agents):
        """
        主决策函数：
        输入：当前agent，所有agent列表
        输出：assignments {agent_id: target_pos}
        """
        grid = agent.known_map
        my_pos = agent.pos
        
        # 1. 识别全图前沿点 (作为 MCTS 的动作空间基础)
        self.cached_frontiers = self._find_frontier_centroids(grid)
        
        if not self.cached_frontiers:
            return {agent.id: None} # 无处可去，探索完成

        # 2. 获取队友的计划 (用于去中心化协作)
        # 假设 agents 列表里的对象有 planned_path 或 goal 属性
        peer_plans = self._collect_peer_plans(agent, agents)

        # 3. 构建 MCTS 树
        root = MCTSNode(position=my_pos)
        # 根节点的动作空间就是所有可选的前沿点
        root.untried_actions = self._get_feasible_actions(root.position, grid)

        for _ in range(self.simulation_budget):
            leaf = self._select(root)
            
            # 只有在树没达到最大深度时才扩展
            depth = self._get_depth(leaf)
            if depth < self.max_tree_depth:
                child = self._expand(leaf, grid)
                if child:
                    # 模拟 + 评估 (考虑协作)
                    reward = self._simulate(child, grid, peer_plans)
                    self._backpropagate(child, reward)
            else:
                # 如果已达最大深度，直接评估当前节点
                reward = self._evaluate_state(leaf.position, grid, peer_plans)
                self._backpropagate(leaf, reward)

        # 4. 选择最佳动作 (访问次数最多的子节点)
        if not root.children:
            # 兜底：如果没有生成子节点，去最近的前沿点
            return {agent.id: self._get_nearest_frontier(my_pos)}
        
        best_child = sorted(root.children, key=lambda c: c.visits)[-1]
        best_target = best_child.action_to_reach
        
        return {agent.id: best_target}

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
            return None # 无法扩展
        
        # 弹出一个动作 (这里动作就是下一个要去的目标点坐标)
        action_target = node.untried_actions.pop()
        
        # 创建子节点
        child_node = MCTSNode(position=action_target, parent=node, action_to_reach=action_target)
        
        # 为子节点生成它的动作空间 (从这个新位置出发能去哪)
        # 关键：子节点的动作空间应该是相对于子节点位置的
        child_node.untried_actions = self._get_feasible_actions(child_node.position, grid)
        
        node.children.append(child_node)
        return child_node

    def _simulate(self, node, grid, peer_plans):
        """Rollout: 随机游走若干步，计算累积奖励"""
        current_pos = node.position
        accumulated_reward = self._evaluate_state(current_pos, grid, peer_plans)
        
        # 简单模拟：随机选几个后续点
        for _ in range(self.rollout_depth):
            possible_moves = self._get_feasible_actions(current_pos, grid, max_count=5)
            if not possible_moves:
                break
            next_pos = random.choice(possible_moves)
            # 累加每一步的收益 (带折扣因子 gamma，论文公式 7)
            gamma = 0.9
            step_reward = self._evaluate_state(next_pos, grid, peer_plans)
            accumulated_reward += step_reward * gamma
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
            rand_x = int(pos[0] + random.uniform(-50, 50))
            rand_y = int(pos[1] + random.uniform(-50, 50))
            # 边界和碰撞检查略...
            actions.append((rand_x, rand_y))
            
        # 去重
        return list(set(actions))

    def _evaluate_state(self, pos, grid, peer_plans):
        """
        计算收益 (Reward Function) - 对应论文 Eq. (2) 和 Eq. (6)
        收益 = (该位置能看到的新未知区域) - (队友已经覆盖的区域)
        """
        # 1. 模拟传感器视野
        visible_cells = self._get_visible_cells(pos, grid)
        
        # 2. 计算基础增益 (有多少个 UNKNOWN)
        gain = 0
        for (c, r) in visible_cells:
            if grid[r, c] == UNKNOWN: # UNKNOWN
                gain += 1.5
            elif grid[r, c] == DANGER:
                gain -= 2
        
        
        # 3. 协作惩罚：检查这些格子是否在队友的计划范围内
        # 如果队友计划去附近，假设他们会把那里探开，所以我们要减去这部分收益
        for (c, r) in visible_cells:
            for peer_traj in peer_plans:
                # 简单判断：如果格子离队友轨迹上的点很近，就认为被覆盖了
                # 这里为了效率，简化为：检查格子是否被队友"预定"
                # 论文中使用 Multi-robot Rollout，这里简化为静态惩罚
                if self._is_covered_by_peers(c, r, peer_traj):
                    gain -= 1 # 扣除收益
                    break
                    
        return max(0, gain)

    def _collect_peer_plans(self, my_agent, all_agents):
        """收集其他机器人的未来计划 (轨迹点列表)"""
        plans = []
        for a in all_agents:
            if a.id != my_agent.id and a.alive:
                # 假设 agent 对象有 planned_path 属性 (A* 结果)
                if hasattr(a, 'planned_path') and a.planned_path:
                    plans.append(a.planned_path)
                elif hasattr(a, 'goal') and a.goal:
                    plans.append([a.goal]) # 如果只有目标，就只存目标
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

    def _find_frontier_centroids(self, grid, cluster_radius_cells=5):
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

            # 1. 找到所有前沿单元格 (Grid Coordinates: c, r)
            # 遍历时跳过地图边缘
            for r in range(1, H - 1):
                for c in range(1, W - 1):
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
            
            unassigned_frontiers = set(all_frontiers_cells)
            centroids_world_pos = [] 

            while unassigned_frontiers:
                # 2.1. 选取第一个未分配的点作为当前簇的起点
                start_cell = unassigned_frontiers.pop()
                
                cluster_sum_c, cluster_sum_r = start_cell
                cluster_count = 1
                
                to_remove = [] 
                
                # 2.2. 找到所有与起始点足够接近的点，并将其加入簇中
                # 转化为列表迭代，效率较低但实现简单
                temp_unassigned = list(unassigned_frontiers) 
                for next_cell in temp_unassigned:
                    # 使用平方距离避免开方运算
                    dist_sq = (next_cell[0] - start_cell[0])**2 + (next_cell[1] - start_cell[1])**2
                    
                    if dist_sq <= cluster_radius_cells**2:
                        to_remove.append(next_cell)
                        cluster_sum_c += next_cell[0]
                        cluster_sum_r += next_cell[1]
                        cluster_count += 1
                
                # 从未分配集合中移除已分组的点
                for cell in to_remove:
                    unassigned_frontiers.remove(cell)
                    
                # 2.3. 计算重心（平均值）
                centroid_c = cluster_sum_c / cluster_count
                centroid_r = cluster_sum_r / cluster_count
                
                # 3. 转换并存储（四舍五入到最近的栅格，然后转换为世界坐标）
                final_centroid_c = int(round(centroid_c))
                final_centroid_r = int(round(centroid_r))
                
                try:
                    # 假设 pos_of_cell(col, row) → (x_world, y_world)
                    gx, gy = pos_of_cell(final_centroid_c, final_centroid_r)
                    centroids_world_pos.append((gx, gy))
                except NameError:
                    # 如果 pos_of_cell 不可用，返回栅格坐标（应急处理）
                    centroids_world_pos.append((final_centroid_c, final_centroid_r)) 

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



