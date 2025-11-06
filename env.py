import heapq
import pygame
import random
import math
import sys
import time
import numpy as np
from collections import deque, defaultdict

from parameters import *
from utils import *
from communicate import Communication
from agent import AgentBase, LargeAgent
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
        if self.large_agents:
            brain_node = min(self.large_agents, key=lambda a: a.id)
            brain_node.is_brain = True


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
                if isinstance(receiver, LargeAgent):
                    receiver.integrate_map_patch(msg['patch'])
            elif msg.get('type') == 'rescue_alert':
                # receiver may prioritize moving to victim
                if isinstance(receiver, AgentBase):
                    receiver.has_goal = True
                    receiver.goal = msg['pos']

        # 3) Large agents perform reasoning (System 2)
        for la in self.large_agents:
            if not la.alive:
                continue
            try:
                # 限制推理频率
                if now_time - la.last_reason_time >= BRAIN_REASON_INTERVAL:
                    la.reason_and_assign(self.agents, now_time)
                    la.last_reason_time = now_time

            except Exception as e:
                print(f"Error in reasoning or task assignment for large agent {la.id}: {e}")

        # 4) Agents decide & move
        for a in self.agents:
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
        for la in self.large_agents:
            if not la.alive:
                continue
            try:
                sense = la.sense(self)
                desired_vx, desired_vy = la.behavior.decide(a, sense, dt)
                # 限制速度范围
                speed = math.hypot(desired_vx, desired_vy)
                if speed > AGENT_MAX_SPEED:
                    scale = AGENT_MAX_SPEED / (speed + 1e-9)
                    desired_vx *= scale
                    desired_vy *= scale
                la.step_motion(desired_vx, desired_vy, dt, self)

                # 更新已访问的网格单元
                ci, cj = cell_of_pos(la.pos)
                self.grid_visited_union.add((ci, cj))
            except Exception as e:
                print(f"Error in decision or motion for large agent {la.id}: {e}")

        # 5) periodic communications: small agents send map patches to large agents if within range
        for a in self.agents:
            if not a.alive:
                continue
            try:
                for la in self.large_agents:
                    if not la.alive:
                        continue
                    if distance(a.pos, la.pos) <= AGENT_COMM_RANGE:
                        a.send_map_patch(comms, [la], now_time)
            except Exception as e:
                print(f"Error in periodic communication for agent {a.id}: {e}")

    def update_data_collect(self, dt, comms:Communication, now_time):
        self.time += dt
        # 1) 更新每个代理的感知信息 -> 更新 local_map
        for a in self.agents + self.large_agents:
            if not a.alive:
                continue
            try:
                a.update_local_map_from_sensing(self)
            except Exception as e:
                print(f"Error updating local map for agent {a.id}: {e}")

        # 2) deliver communications queued
        deliveries = comms.deliver(now_time)
        for sender, receiver, msg in deliveries:
            # msg dispatch
            if msg.get('type') == 'map_patch':
                # receiver should integrate patch
                if isinstance(receiver, LargeAgent):
                    receiver.integrate_map_patch(msg['patch'])
            elif msg.get('type') == 'rescue_alert':
                # receiver may prioritize moving to victim
                if isinstance(receiver, AgentBase):
                    receiver.has_goal = True
                    receiver.goal = msg['pos']

        # 3) Large agents perform reasoning (System 2)
        for la in self.large_agents:
            if not la.alive:
                continue
            try:
                # 限制推理频率
                if now_time - la.last_reason_time >= BRAIN_REASON_INTERVAL:
                    la.reason_and_assign(self.agents, now_time)
                    la.last_reason_time = now_time

            except Exception as e:
                print(f"Error in reasoning or task assignment for large agent {la.id}: {e}")

        # 4) Agents decide & move
        for a in self.agents:
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
        for la in self.large_agents:
            if not la.alive:
                continue
            try:
                sense = la.sense(self)
                desired_vx, desired_vy = la.behavior.decide(a, sense, dt)
                # 限制速度范围
                speed = math.hypot(desired_vx, desired_vy)
                if speed > AGENT_MAX_SPEED:
                    scale = AGENT_MAX_SPEED / (speed + 1e-9)
                    desired_vx *= scale
                    desired_vy *= scale
                la.step_motion(desired_vx, desired_vy, dt, self)

                # 更新已访问的网格单元
                ci, cj = cell_of_pos(la.pos)
                self.grid_visited_union.add((ci, cj))
            except Exception as e:
                print(f"Error in decision or motion for large agent {la.id}: {e}")

        # 5) periodic communications: small agents send map patches to large agents if within range
        for a in self.agents:
            if not a.alive:
                continue
            try:
                for la in self.large_agents:
                    if not la.alive:
                        continue
                    if distance(a.pos, la.pos) <= AGENT_COMM_RANGE:
                        a.send_map_patch(comms, [la], now_time)
            except Exception as e:
                print(f"Error in periodic communication for agent {a.id}: {e}")

    def update_baseline(self, dt, comms:Communication, now_time):
        self.time += dt
        # 1) 更新每个代理的感知信息 -> 更新 local_map
        for a in self.agents + self.large_agents:
            if not a.alive:
                continue
            try:
                a.update_local_map_from_sensing(self)
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
                if isinstance(receiver, LargeAgent):
                    receiver.integrate_map_patch(msg['patch'])
            elif msg.get('type') == 'rescue_alert':
                # receiver may prioritize moving to victim
                if isinstance(receiver, AgentBase):
                    receiver.has_goal = True
                    receiver.goal = msg['pos']

        # 3) Large agents perform reasoning (System 2)
        for la in self.large_agents:
            if not la.alive:
                continue
            try:
                # 限制推理频率
                if now_time - la.last_reason_time >= BRAIN_REASON_INTERVAL:
                    la.reason_and_assign(self.agents, now_time)
                    la.last_reason_time = now_time

            except Exception as e:
                print(f"Error in reasoning or task assignment for large agent {la.id}: {e}")

        # 4) Agents decide & move
        for a in self.agents:
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
        for la in self.large_agents:
            if not la.alive:
                continue
            try:
                sense = la.sense(self)
                desired_vx, desired_vy = la.behavior.decide(a, sense, dt)
                # 限制速度范围
                speed = math.hypot(desired_vx, desired_vy)
                if speed > AGENT_MAX_SPEED:
                    scale = AGENT_MAX_SPEED / (speed + 1e-9)
                    desired_vx *= scale
                    desired_vy *= scale
                la.step_motion(desired_vx, desired_vy, dt, self)

                # 更新已访问的网格单元
                ci, cj = cell_of_pos(la.pos)
                self.grid_visited_union.add((ci, cj))
            except Exception as e:
                print(f"Error in decision or motion for large agent {la.id}: {e}")

        # 5) periodic communications: small agents send map patches to large agents if within range
        for a in self.agents:
            if not a.alive:
                continue
            try:
                for la in self.large_agents:
                    if not la.alive:
                        continue
                    if distance(a.pos, la.pos) <= AGENT_COMM_RANGE:
                        a.send_map_patch(comms, [la], now_time)
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