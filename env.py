import heapq
import pygame
import random
import math
import copy
import numpy as np

from parameters import *
from utils import *
from communicate import Communication
from agent import AgentBase, LargeAgent
from behaviors import *
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
    def __init__(self, seed=None, world_id = -1):
        if seed is not None:
            random.seed(seed)
        self.width = WORLD_W
        self.height = WORLD_H
        self.time = 0.0
        self.obstacles = []
        self.danger_zones = []
        self.agents = []
        self.large_agents = []
        self.wasted_agents = []
        self.wasted_large_agents = []
        self.brain_id = None
        self.victim = None
        self.grid_visited_union = set()

        # 初始化地面栅格
        self.ground_grid = np.full((GRID_H, GRID_W), FREE, dtype=np.int8)

        if world_id >= 0:
            self.generate_fixed_map(world_id)  # 使用固定地图布局
            self.spawn_large_agents(NUM_LARGE)
            self.spawn_agents(NUM_AGENTS)
        else:
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
                    self.ground_grid[j, i] = OBSTACLE

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
                                self.ground_grid[j, i] = DANGER
                    break
                attempts += 1

    def spawn_large_agents(self, num=2):
        # free space
        self.large_agents = []
        base_pos = None
        attempts = 0
        if hasattr(self, "spawn_center"):
            base_x, base_y = self.spawn_center
            for i in range(num):
                # 让每个大节点稍微分开
                angle = (2 * math.pi / num) * i
                r = 30 + random.uniform(-10, 10)
                x = base_x + r * math.cos(angle)
                y = base_y + r * math.sin(angle)

                # 确保安全位置
                if self.is_in_obstacle(x, y) or self.is_in_danger(x, y):
                    x, y = self.random_free_pos(margin=60)

                la = LargeAgent(
                    i, x, y,
                    multi_behavior=ERRTFrontierAssignmentBehavior(),
                    behavior=ExploreBehavior()
                )
                self.large_agents.append(la)
        else:
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
                        self.large_agents.append(LargeAgent(i, x, y,\
                                                            multi_behavior=ERRTFrontierAssignmentBehavior(),
                                                            behavior=ExploreBehavior()))
                        break
                else:
                    # 如果找不到合适位置，则直接使用base_pos附近
                    self.large_agents.append(LargeAgent(i, base_pos[0] + random.uniform(-10, 10),
                                                        base_pos[1] + random.uniform(-10, 10),
                                                        multi_behavior=ERRTFrontierAssignmentBehavior(),
                                                        behavior=ExploreBehavior()))
        if self.large_agents:
            brain_node = min(self.large_agents, key=lambda a: a.id)
            brain_node.is_brain = True
            self.brain_id = brain_node.id

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
                    self.agents.append(AgentBase(i + 1000, x, y, behavior=ExploreBehavior()))
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
            if self.ground_grid[cell[1], cell[0]] != FREE:
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
        w, h = grid.shape[1], grid.shape[0]
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
                if 0 <= nx < w and 0 <= ny < h and grid[ny][nx] == FREE:
                    ng = gscore[current] + 1
                    if (nx, ny) not in gscore or ng < gscore[(nx, ny)]:
                        came_from[(nx, ny)] = current
                        gscore[(nx, ny)] = ng
                        fscore[(nx, ny)] = ng + abs(nx - goal[0]) + abs(ny - goal[1])
                        heapq.heappush(open_set, (fscore[(nx, ny)], (nx, ny)))
        return None

    def update_baseline(self, dt, comms:Communication, now_time):
        # Tik
        # 结束判断
        # 死亡节点确认与剔除
        # 动态脑节点选举
        # 分层通信结构：小节点向脑节点周期发送 map_patch，脑节点整合地图
        # 信息整合机制：receiver.integrate_map_patch() 更新全局地图 集体形成共享认知
        # 自组织任务分配 执行 reason_and_assign() 并广播目标 忽视层级结构所有机器人都可以收到，大小结点之间无指挥关系
        self.time += dt

        victim_pos = self.victim.pos
        victim_cell = cell_of_pos(victim_pos)
        for la in self.large_agents:
            if la.known_map[victim_cell[1], victim_cell[0]] != UNKNOWN:
                self.victim.rescued = True
            

        # 1) 检查哪些节点还活着，选举脑节点
        dead_agents = [a for a in self.agents if not a.alive]
        self.wasted_agents.extend(dead_agents)
        self.agents = [a for a in self.agents if a.alive]

        dead_large = [la for la in self.large_agents if not la.alive and not la.is_brain]
        self.wasted_large_agents.extend(dead_large)
        alive_large_agents = [la for la in self.large_agents if la.alive]
        
        if self.brain_id is None or not self.large_agents[self.brain_id].alive:
            if alive_large_agents:
                pass # 继续下面的选举流程
            else:
                self.large_agents = []
                self.brain_id = None
                return  # 没有活着的大节点，跳过本轮更新
        else:
            self.large_agents = alive_large_agents
        if len(self.large_agents) == 0 or len(self.agents) == 0:
            return  # 没有节点，跳过本轮更新

        # 2) 更新每个代理的感知信息 -> 更新 local_map
        for a in self.agents + self.large_agents:
            a.update_local_map_from_sensing(self)

        # 3) periodic communications: small agents send map patches to large agents if within range
        for a in self.agents + self.large_agents:
            for la in self.large_agents:
                if distance(a.pos, la.pos) <= AGENT_COMM_RANGE:
                    a.send_map_patch(comms, [la], now_time)

        # 4) deliver communications queued
        # 多机器人系统之间的层级化的结构与信息互通
        deliveries = comms.deliver(now_time)
        for sender, receiver, msg in deliveries:
            # msg dispatch
            if msg.get('type') == 'map_patch':
                # receiver should integrate patch
                if isinstance(receiver, LargeAgent):
                    receiver.integrate_map_patch(msg['patch'])

        # 5) Large agents perform reasoning (System 2)
        for la in self.large_agents:
            if now_time - la.last_reason_time >= BRAIN_REASON_INTERVAL:
                assigns = la.reason_and_assign([la], now_time)
                la.last_reason_time = now_time
                for aid, wp in assigns.items():
                    agent = next((a for a in self.agents if a.id == aid), None)
                    if agent is not None:
                        agent.has_goal = True
                        agent.goal = wp
                    Largeagent = next((a for a in self.large_agents if a.id == aid), None)
                    if Largeagent is not None:
                        Largeagent.has_goal = True
                        Largeagent.goal = wp
                break

        # 4) Agents decide & move
        for a in self.agents:
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
        for la in self.large_agents:
            sense = la.sense(self)
            desired_vx, desired_vy = la.behavior.decide(la, sense, dt)
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

    def draw(self, screen):
        # draw explored overlay: union of all agents' explored cells => white, else dark gray
        explored_union = np.full((GRID_H, GRID_W), False, dtype=bool)
        for a in (self.agents + self.large_agents + self.wasted_agents + self.wasted_large_agents):
            explored = (a.local_map != UNKNOWN)
            explored_union = np.logical_or(explored_union, explored)
        
        # explored_union = self.large_agents[self.brain_id].known_map.copy()

        # draw cells
        for i in range(GRID_W):
            for j in range(GRID_H):
                rect = pygame.Rect(i*GRID_CELL, j*GRID_CELL, GRID_CELL, GRID_CELL)
                if not explored_union[j,i]:
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
        for la in self.large_agents + self.wasted_large_agents:
            # la.draw_hist(screen, color=(200,160,60))
            la.draw_self(screen)
            la.draw_goal(screen)
        for a in self.agents + self.wasted_agents:
            # a.draw_hist(screen)
            a.draw_self(screen)
            a.draw_goal(screen)
        if self.brain_id is not None:
            self.draw_brain_and_agent_views(screen,[1, 2])


    def draw_brain_and_agent_views(self, screen, agents_to_show_id=None):
        """在右侧绘制脑节点 known_map + 指定 agent 的 local_map"""
        sidebar_x = SCREEN_W  # 右侧起始位置
        sidebar_w = 400
        sidebar_h = SCREEN_H
        pygame.draw.rect(screen, (30, 30, 30), (sidebar_x, 0, sidebar_w, sidebar_h))

        # ========= 找到脑节点 =========
        brain = self.large_agents[self.brain_id] if self.brain_id is not None else None
        if brain:
            self._draw_map_on_sidebar(screen, brain.known_map, sidebar_x, 20, title="Brain Known Map")

        # ========= 绘制指定 agent 视图 =========
        for i in agents_to_show_id or []:
            if i == self.brain_id:
                continue
            agent_to_show_id = i
            if agent_to_show_id is not None:
                target_agent = next((a for a in (self.agents + self.large_agents + 
                                                self.wasted_large_agents + self.wasted_agents) \
                                    if a.id == agent_to_show_id), None)
                if target_agent is not None:
                    self._draw_map_on_sidebar(screen, target_agent.local_map, \
                                              sidebar_x, (250 + 230*agents_to_show_id.index(i)), title=f"Agent {agent_to_show_id} Local Map")


    def _draw_map_on_sidebar(self, screen, grid_map, x_offset, y_offset, title="Map"):
        """通用绘制函数，用于绘制地图数组"""
        if grid_map is None:
            return

        h, w = grid_map.shape

        # 定义颜色表
        color_map = {
            UNKNOWN: (40, 40, 40),
            FREE: (230, 230, 230),
            OBSTACLE: (100, 100, 100),
            DANGER: (255, 60, 60),
            VICTIM: (255, 255, 0)
        }

        surf = pygame.Surface((w, h))
        for y in range(h):
            for x in range(w):
                val = grid_map[y, x]
                surf.set_at((x, y), color_map.get(val, (255, 255, 255)))

        # 缩放与绘制
        scaled = pygame.transform.scale(surf, (360, 200))
        screen.blit(scaled, (x_offset + 20, y_offset))

        # 边框 + 标题
        pygame.draw.rect(screen, (255, 255, 255), (x_offset + 20, y_offset, 360, 200), 2)
        font = pygame.font.SysFont(None, 22)
        text = font.render(title, True, (255, 255, 255))
        screen.blit(text, (x_offset + 30, y_offset - 20))


    def coverage_percentage(self):
        total = GRID_W * GRID_H
        # count union of explored cells
        explored = 0
        union_set = set()
        for a in (self.agents + self.large_agents):
            union_set |= a.get_local_explored_cells()
        explored = len(union_set)
        return (explored / total) * 100.0


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

    def update(self, dt, comms:Communication, now_time):
        # Tik
        # 结束判断
        # 死亡节点确认与剔除
        # 动态脑节点选举
        # 分层通信结构：小节点向脑节点周期发送 map_patch，脑节点整合地图
        # 信息整合机制：receiver.integrate_map_patch() 更新全局地图 集体形成共享认知
        # 自组织任务分配 执行 reason_and_assign() 并广播目标 忽视层级结构所有机器人都可以收到，大小结点之间无指挥关系
        # TODO：任务分配没有考虑层级关系，也没有考虑通讯距离
        # TODO：机器人的行为逻辑可能需要调整以适应新的通信和任务分配机制
        self.time += dt

        victim_pos = self.victim.pos
        victim_cell = cell_of_pos(victim_pos)
        if self.brain_id is not None \
            and self.large_agents[self.brain_id].known_map[victim_cell[1], victim_cell[0]] != UNKNOWN:
            self.victim.rescued = True

        # 1) 检查哪些节点还活着，选举脑节点
        dead_agents = [a for a in self.agents if not a.alive]
        self.wasted_agents.extend(dead_agents)
        self.agents = [a for a in self.agents if a.alive]

        dead_large = [la for la in self.large_agents if not la.alive and not la.is_brain]
        self.wasted_large_agents.extend(dead_large)
        alive_large_agents = [la for la in self.large_agents if la.alive]
        
        if self.brain_id is None or not self.large_agents[self.brain_id].alive:
            if alive_large_agents:
                # 选举成功
                brain_node = min(alive_large_agents, key=lambda a: a.id)
                id_in_list = alive_large_agents.index(brain_node)
                
                # 信息交接
                brain_node.is_brain = True
                brain_node.last_reason_time = self.large_agents[self.brain_id].last_reason_time
                brain_node.known_map = np.copy(self.large_agents[self.brain_id].known_map)
                brain_node.son_ids = list(getattr(self.large_agents[self.brain_id], "son_ids", []))
                brain_node.father_id = None

                # 确认原来脑节点去逝
                self.wasted_large_agents.append(self.large_agents[self.brain_id])
                self.large_agents = alive_large_agents
                self.large_agents[id_in_list] = brain_node
                
                # 最终交接
                self.brain_id = self.large_agents.index(brain_node)
            else:
                self.large_agents = []
                self.brain_id = None
                return  # 没有活着的大节点，跳过本轮更新
        else:
            self.large_agents = alive_large_agents
        if len(self.large_agents) == 0 or len(self.agents) == 0:
            return  # 没有节点，跳过本轮更新
        

        brain = None
        if self.brain_id is not None and 0 <= self.brain_id < len(self.large_agents):
            brain = self.large_agents[self.brain_id]

        """
        每轮更新前，重新确定层级关系：
        Brain → Large → Small
        """
        # ---------------------
        # 2️⃣ 大节点附属脑节点
        # ---------------------
        # self.large_agents[self.brain_id].son_ids = []
        # for la in self.large_agents:
        #     if la.father_id != self.brain_id:
        #         la.father_id = self.brain_id
        #     if la.id not in self.large_agents[self.brain_id].son_ids:
        #         self.large_agents[self.brain_id].son_ids.append(la.id)

        # ---------------------
        # 3️⃣ 小节点寻找最近的大节点或脑节点
        # ---------------------
        for sa in self.agents:
            # 原父节点
            old_father_id = sa.father_id
            old_father = self.find_agent_by_id(old_father_id)
            # 候选父节点（所有大节点 + 脑节点）
            candidates = self.large_agents.copy()
            # 按距离排序
            candidates = sorted(candidates, key=lambda a: distance(a.pos, sa.pos))
            nearest = candidates[0]
            # 计算距离差异
            if old_father is not None:
                dist_old = distance(sa.pos, old_father.pos)
                dist_new = distance(sa.pos, nearest.pos)
                # 若差异很小（比如<50像素），不切换
                if abs(dist_new - dist_old) < 50:
                    continue
            else:
                print(f"Agent {sa.id} had no father")
                # 候选父节点（所有大节点 + 脑节点）
                candidates = self.large_agents.copy()
                # 按距离排序
                candidates = sorted(candidates, key=lambda a: distance(a.pos, sa.pos))
                nearest = candidates[0]

            # 否则更新父节点
            sa.father_id = nearest.id


        # ---------------------
        # 4️⃣ 清理无效子节点引用
        # ---------------------
        for agent in self.large_agents:
            valid_sons = []
            for sid in agent.son_ids:
                child = self.find_agent_by_id(sid)
                if child is not None and child.alive and child.father_id == agent.id:
                    valid_sons.append(sid)
            agent.son_ids = valid_sons

        # print("[INFO] 层级关系更新完成 ✅")

        # 2) 更新每个代理的感知信息 -> 更新 local_map
        for a in self.agents + self.large_agents:
            try:
                a.update_local_map_from_sensing(self)
                if a.is_large:
                    a.fuse_own_sensing()
            except Exception as e:
                print(f"Error updating local map for agent {a.id}: {e}")

        # 3) periodic communications: small agents send map patches to large agents if within range
        for a in self.agents + self.large_agents:
            # try:
            if getattr(a,'is_large', False):
                a.send_map_patch(comms, [self.large_agents[self.brain_id]], now_time)
                continue
            else:
                # choose nearest large agent (within comm range) else brain
                father = self.find_agent_by_id(a.father_id)
                if father is not None and distance(a.pos, father.pos) <= AGENT_COMM_RANGE:
                    a.send_map_patch(comms, [father], now_time)
            # except Exception:
            #     pass

        # 4) deliver communications queued
        # 多机器人系统之间的层级化的结构与信息互通
        deliveries = comms.deliver(now_time)
        aid_requests = []
        for sender, receiver, msg in deliveries:
            # msg dispatch
            t = msg.get('type', '')
            if t  == 'map_patch':
                # receiver should integrate patch
                if isinstance(receiver, LargeAgent):
                    receiver.integrate_map_patch(msg['patch'])
            elif t in ('aid_request', 'emergency', 'death_report'):
                aid_requests.append((sender, receiver, msg))
            elif t == 'rescue_alert':
                # receiver may prioritize moving to victim
                if isinstance(receiver, AgentBase):
                    receiver.has_goal = True
                    receiver.goal = msg['pos']

        # F) Brain periodic global planning + handling aid requests

        if brain is not None:
            # process incoming aid/emergency messages (reassign or escalate)
            if aid_requests:
                # simple policy: if any emergency -> brain will re-plan and broadcast immediate assists
                # collect urgent positions
                urgents = [m for (_,_,m) in aid_requests if m.get('urgency',0) >= 1.0 or m.get('type','').upper() in ('EMERGENCY','DEATH_REPORT')]
                if urgents:
                    # compute reinforcement: choose closest large agent(s) to each urgent
                    for req in urgents:
                        pos = req.get('pos', None)
                        if pos is None:
                            continue
                        # find nearest alive large agent that isn't the brain itself
                        cand = [la for la in self.large_agents if la.alive and (la is not brain)]
                        if cand:
                            cand.sort(key=lambda la: math.hypot(la.pos[0]-pos[0], la.pos[1]-pos[1]))
                            # dispatch top-1 reinforcement: set goal on that la (and optionally its subordinates)
                            target_la = cand[0]
                            target_la.has_goal = True
                            target_la.goal = pos
                            # also notify via comms
                            comms.send(brain, target_la, {'type':'AID_ASSIGN','from':brain.id, 'pos':pos}, now_time)

            # periodic compute global plan
            if not hasattr(brain, 'brain_planner'):
                brain.brain_planner = BrainGlobalPlanner(plan_interval=5.0)
            try:
                # 限制推理频率
                if now_time - brain.brain_reason_time >= BRAIN_REASON_INTERVAL:
                    if len(self.large_agents) != 0:
                        assigns = brain.brain_reason_and_assign(self.large_agents, now_time)
                    else:
                        assigns = {}
                    brain.brain_reason_time = now_time
                    for aid, wp in assigns.items():
                        Largeagent = next((a for a in self.large_agents if a.id == aid), None)
                        if Largeagent is not None:
                            Largeagent.has_goal = True
                            Largeagent.goal = wp
            except Exception as e:
                print(f"Error in reasoning or task assignment for large agent : {e}")

        # (G) Large agents: reason_and_assign (medium-frequency)
        for la in self.large_agents:
            try:
                # only brain node handles global plan; each large agent will act on messages in its handler
                if now_time - getattr(la, 'last_reason_time', -1e9) >= BRAIN_REASON_INTERVAL:
                    assigns = {}
                    try:
                        sons = [a for a in self.agents if a.father_id == la.id]
                        assigns = la.reason_and_assign(sons, now_time) or {}
                    except Exception:
                        print(f"Error in reason_and_assign for large agent {la.id}")
                    # push assignments to children (via comms or direct set)
                    if isinstance(assigns, dict):
                        for aid, wp in assigns.items():
                            target_agent = next((x for x in self.agents if x.id == aid), None)
                            if target_agent is not None:
                                target_agent.has_goal = True
                                target_agent.goal = wp
                    la.last_reason_time = now_time
            except Exception:
                # tolerate reasoning errors
                pass

        # (H) Small agents: accept & adjust goals, detect emergencies, request aid
        for a in list(self.agents):  # iterate over copy (some may die)
            try:
                if a.has_goal and a.goal is not None:
                    # let agent attempt to locally adjust goal using its local_map
                    try:
                        accepted = a._agent_accept_and_adjust_goal(self)
                    except Exception:
                        pass

                # emergency detection: if agent is in danger zone, or near a dead peer etc.
                sense = a.sense(self)
                desired_vx, desired_vy = a.behavior.decide(a, sense, dt)
                a.step_motion(desired_vx, desired_vy, dt, self)
            except Exception:
                # keep simulation robust
                pass

            # 更新已访问的网格单元
            ci, cj = cell_of_pos(a.pos)
            self.grid_visited_union.add((ci, cj))
            # except Exception as e:
            #     print(f"Error in decision or motion for agent {a.id}: {e}")
        # (I) Large agents also check assigned goals: ensure safety and request reinforcements if not safe
        for la in self.large_agents:
            try:
                if la.has_goal and la.goal is not None:
                    safe = True
                    try:
                        # fallback: check la.known_map grid
                        i,j = cell_of_pos(la.goal)
                        if 0 <= i < GRID_W and 0 <= j < GRID_H:
                            val = la.known_map[j, i]
                            safe = (val != OBSTACLE and val != DANGER)
                    except Exception:
                        print(f"Error checking safety for assigned goal of agent ")
                    if not safe:
                        # if not safe, attempt to find nearby safe alternative
                        try:
                            adjusted = la._agent_accept_and_adjust_goal(self)  # optional method
                        except Exception:
                            print(f"Error adjusting unsafe assigned goal for agent ")
                sense = la.sense(self)
                desired_vx, desired_vy = la.behavior.decide(la, sense, dt)
                la.step_motion(desired_vx, desired_vy, dt, self)
            except Exception:
                pass



    # ======================== 生成固定位置的参数 ========================

    def generate_fixed_map(self, map_id:int):
        """
        根据 map_id (1~10) 生成固定的障碍、危险区、受害者布局。
        """
        # 清空原始数据
        self.obstacles.clear()
        self.danger_zones.clear()
        self.ground_grid[:] = FREE

        if map_id == 1:
            self._add_rect_obstacle(300, 300, 200, 150)
            self._add_danger(100, 100, 60)
            self._add_danger(700, 100, 60)
            self._add_danger(100, 500, 60)
            self._add_danger(700, 500, 60)
            self.victim = Victim(300, 500)
            self.spawn_center = (300, 100)

        elif map_id == 2:
            for x in range(100, 800, 150):
                self._add_rect_obstacle(x, 200, 80, 20)
                self._add_rect_obstacle(x - 50, 400, 80, 20)
            self._add_danger(400, 300, 100)
            self.victim = Victim(600, 50)
            self.spawn_center = (150, 500)

        elif map_id == 3:
            self._add_rect_obstacle(300, 100, 500, 40)
            self._add_rect_obstacle(300, 460, 500, 40)
            self._add_danger(500, 300, 70)
            self.victim = Victim(750, 300)
            self.spawn_center = (100, 300)

        elif map_id == 4:
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                x = 400 + 200 * np.cos(angle)
                y = 300 + 150 * np.sin(angle)
                self._add_rect_obstacle(x-20, y-20, 40, 40)
            self._add_danger(400, 300, 50)
            self.victim = Victim(700, 500)
            self.spawn_center = (150, 150)

        elif map_id == 5:
            for y in range(50, 550, 100):
                self._add_rect_obstacle(400, y, 40, 60)
            self._add_danger(500, 250, 80)
            self.victim = Victim(100, 300)
            self.spawn_center = (700, 100)

        elif map_id == 6:
            self._add_rect_obstacle(200, 150, 100, 200)
            self._add_rect_obstacle(600, 150, 100, 200)
            self._add_danger(400, 300, 90)
            self.victim = Victim(400, 500)
            self.spawn_center = (400, 100)

        elif map_id == 7:
            self._add_rect_obstacle(400, 0, 40, 250)
            self._add_rect_obstacle(400, 350, 40, 250)
            self._add_danger(300, 300, 40)
            self.victim = Victim(750, 300)
            self.spawn_center = (150, 300)

        elif map_id == 8:
            for _ in range(10):
                x = random.randint(100, 700)
                y = random.randint(100, 500)
                self._add_rect_obstacle(x, y, 30, 30)
            self._add_danger(100, 350, 60)
            self.victim = Victim(100, 100)
            self.spawn_center = (100, 500)

        elif map_id == 9:
            self._add_rect_obstacle(300, 0, 40, 300)
            self._add_rect_obstacle(300, 300, 400, 40)
            self._add_danger(650, 200, 75)
            self.victim = Victim(450, 150)
            self.spawn_center = (150, 150)

        elif map_id == 10:
            self._add_rect_obstacle(250, 200, 400, 40)
            self._add_rect_obstacle(250, 400, 400, 40)
            self._add_danger(400, 300, 60)
            self.victim = Victim(750, 550)
            self.spawn_center = (100, 100)

        else:
            raise ValueError("Invalid map_id, must be 1~10")

        # 更新 ground_grid 状态
        self._update_ground_grid()

        print(f"✅ Fixed map {map_id} loaded.")

    # ======================== 辅助函数 ========================
    def _add_rect_obstacle(self, x, y, w, h):
        obs = Obstacle(x, y, w, h)
        self.obstacles.append(obs)

    def _add_danger(self, x, y, r):
        dz = DangerZone(x, y, r)
        self.danger_zones.append(dz)

    def _update_ground_grid(self):
        """根据当前障碍物和危险区更新 ground_grid"""
        self.ground_grid[:] = FREE
        for obs in self.obstacles:
            left = int(obs.rect.left // GRID_CELL)
            right = int(obs.rect.right // GRID_CELL)
            top = int(obs.rect.top // GRID_CELL)
            bottom = int(obs.rect.bottom // GRID_CELL)
            for i in range(left, right):
                for j in range(top, bottom):
                    if 0 <= i < GRID_W and 0 <= j < GRID_H:
                        self.ground_grid[j, i] = OBSTACLE
        for dz in self.danger_zones:
            cx, cy = dz.pos
            for i in range(GRID_W):
                for j in range(GRID_H):
                    gx, gy = pos_of_cell(i, j)
                    if math.hypot(gx - cx, gy - cy) < dz.r:
                        self.ground_grid[j, i] = DANGER


    def find_agent_by_id(self, id_):
        if id_ is None:
            # print(f"Warning: find_agent_by_id called {id_} with None id")
            return None
        for a in self.agents + self.large_agents:
            if a.id == id_:
                return a
        return None