
import torch
import math
import random
import numpy as np
import heapq
from scipy.optimize import linear_sum_assignment
from utils import *
from model import SlowModel  # 导入你的模型定义
import matplotlib.pyplot as plt

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

    def __init__(self, replan_interval=0.5, goal_tolerance=10.0):
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
    def decide(self, agent, sense_data, dt):
        now = getattr(agent, "sim_time", 0.0)
        grid = agent.local_map

        # ---- [1] 检查是否有目标 ----
        if not agent.has_goal or agent.goal is None:
            # 无目标 → 安全随机游走
            # return self.safe_random_explore(agent, grid)
            return 0, 0

        # ---- [2] 检查是否到达目标 ----
        goal_dist = distance(agent.pos, agent.goal)
        if goal_dist < self.goal_tolerance:
            agent.has_goal = False
            agent.goal = None
            # return self.safe_random_explore(agent, grid)
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
                return 0, 0
                # return self.safe_random_explore(agent, grid)
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
        # return self.safe_random_explore(agent, grid)
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

class NeuralFrontierAssignmentBehavior(Multi_Behavior):
    # TODO 此类方法暂时废弃
    """
    使用训练好的 SlowModel（Diffusion Transformer）进行神经前沿预测与分配。
    逻辑：
    1. 将当前脑节点的 known_map 与各 agent 的全局状态编码为状态向量；
    2. 调用 SlowModel.sample(state_vec) 生成一组子目标序列；
    3. 选择前若干子目标作为“前沿点”；
    4. 根据距离或策略为每个小节点分配目标。
    """

    def __init__(self, model_path="models/slow_model.pth", device="cpu", n_subgoals=5):
        super().__init__()
        self.device = device
        self.model = SlowModel().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.n_subgoals = n_subgoals
        self.last_assignment = {}

    # -------------------------------
    # 状态编码函数
    # -------------------------------
    def encode_global_state(self, brain_agent, agents):
        """
        将当前全局状态编码为一个 state vector，输入给 SlowModel。
        示例特征（可按需扩展）：
        - 探索率（覆盖百分比）
        - 活跃节点数
        - 大脑节点位置
        - victim 是否被救
        """
        explored = np.sum(brain_agent.known_map != -1)
        total = brain_agent.known_map.size
        coverage = explored / total

        n_alive = sum([a.alive for a in agents])
        brain_pos = np.array(brain_agent.pos, dtype=np.float32) / np.array([WORLD_W, WORLD_H])
        victim_status = 1.0 if hasattr(brain_agent, "victim_found") and brain_agent.victim_found else 0.0

        # 形成固定长度状态向量（可以根据模型训练时的定义调整）
        state_vec = np.array([
            coverage,
            n_alive / len(agents),
            brain_pos[0],
            brain_pos[1],
            victim_status,
            0.0  # 占位符，可用于平均连接强度等
        ], dtype=np.float32)
        return state_vec

    # -------------------------------
    # 主决策函数
    # -------------------------------
    def decide(self, brain_agent, agents):
        """
        利用神经网络生成前沿目标并分配。
        """
        # === Step 1: 状态编码 ===
        state_vec = self.encode_global_state(brain_agent, agents)

        # === Step 2: 生成子目标序列 ===
        with torch.no_grad():
            input_tensor = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            subgoals = self.model.sample(input_tensor, n_samples=1)  # np.ndarray [N, 2]
            if isinstance(subgoals, torch.Tensor):
                subgoals = subgoals.cpu().numpy()
        subgoals = np.clip(subgoals, 0, 1)  # 归一化输出映射
        subgoals[:, 0] *= WORLD_W
        subgoals[:, 1] *= WORLD_H

        # === Step 3: 将子目标作为“前沿” ===
        frontiers = [tuple(map(float, g)) for g in subgoals[:self.n_subgoals]]

        # === Step 4: 分配到各 agent ===
        assignments = {}
        available = frontiers.copy()
        for a in agents:
            if not available:
                assignments[a.id] = None
                continue
            distances = [distance(a.pos, f) for f in available]
            target = available[int(np.argmin(distances))]
            assignments[a.id] = target
            available.remove(target)

        self.last_assignment = assignments
        return assignments

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
        
        # plt.imshow(global_map, cmap='viridis')
        # x1s = [g[0]//10 for g in self.candidate_goals]
        # y1s = [g[1]//10 for g in self.candidate_goals]
        # x2s = [a.pos[0]//10 for a in agents]
        # y2s = [a.pos[1]//10 for a in agents]
        # plt.scatter(x1s, y1s, c='red', marker='x', label='Candidate Goals')
        # plt.scatter(x2s, y2s, c='blue', marker='o', label='Agents')

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
    