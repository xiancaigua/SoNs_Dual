
import torch
import math
import random
import numpy as np

from utils import *
from model import SlowModel  # 导入你的模型定义

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
                if global_map[y, x] == 0.5:  # 未知区域
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