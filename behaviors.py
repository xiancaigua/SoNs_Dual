
import math
import random
import numpy as np

from utils import *

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

