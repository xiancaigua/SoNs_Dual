import heapq
import pygame
import random
import math
import pickle
import numpy as np

from parameters import *
from utils import *
from communicate import Communication
from agent import AgentBase, LargeAgent, BrainAgent
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
    # --- MOD: 危险区改为网格矩形 ---
    def __init__(self, grid_x, grid_y, grid_w, grid_h):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_w = grid_w
        self.grid_h = grid_h

        # 像素级矩形
        self.rect = pygame.Rect(
            grid_x * GRID_CELL,
            grid_y * GRID_CELL,
            grid_w * GRID_CELL,
            grid_h * GRID_CELL
        )

        # 中心点
        self.pos = (
            (self.rect.left + self.rect.right) / 2,
            (self.rect.top + self.rect.bottom) / 2
        )

    def contains(self, p):
        """点是否在危险矩形内"""
        return self.rect.collidepoint(p[0], p[1])

    def draw(self, screen):
        # 半透明矩形
        temp = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        transparent_red = (200, 40, 40, 100)  # 半透明红色
        pygame.draw.rect(temp, transparent_red, (0, 0, self.rect.width, self.rect.height))
        screen.blit(temp, (self.rect.left, self.rect.top))

        # 边框
        pygame.draw.rect(screen, (150, 20, 20), self.rect, 2)

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
        self.spawn_times = NUM_AGENTS
        
        self.obstacles = []
        self.danger_zones = []
        self.victim = None

        self.brain_id = None
        self.agents = []
        self.large_agents = []
        self.wasted_agents = []
        self.wasted_large_agents = []

        # 初始化地面栅格
        self.ground_grid = np.full((GRID_H, GRID_W), FREE, dtype=np.int8)
        self.known_grid = np.full((GRID_H, GRID_W), UNKNOWN, dtype=np.int8)
        self.visited_grid = np.full((GRID_H, GRID_W), UNKNOWN, dtype=np.int8)

        # 环境生成顺序
        self.generate_danger_zones(NUM_DANGER_ZONES)
        self.generate_static_obstacles(NUM_OBSTACLES)

        self.spawn_center = self.random_free_pos(margin=60)

        self.brain = BrainAgent(-1, self.spawn_center[0],self.spawn_center[1])
        self.spawn_agents(NUM_LARGE)
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
            if dz.contains((x,y)):
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

        # === MODIFIED: 全网格化障碍物生成 ===
        for _ in range(n):
            # 障碍物的大小以“网格数量”为单位
            gw = random.randint(2, 12)  # 占据 2~6 个网格宽
            gh = random.randint(2, 12)  # 占据 2~6 个网格高

            # 随机选择左上角的网格索引
            gi = random.randint(0, GRID_W - gw - 1)
            gj = random.randint(0, GRID_H - gh - 1)

            # 计算世界坐标
            x = gi * GRID_CELL
            y = gj * GRID_CELL
            w = gw * GRID_CELL
            h = gh * GRID_CELL

            obs = Obstacle(x, y, w, h)
            self.obstacles.append(obs)

            # 标注所有占据的网格
            for i in range(gi, gi + gw):
                for j in range(gj, gj + gh):
                    self.ground_grid[j, i] = OBSTACLE

    def generate_danger_zones(self, num=NUM_DANGER_ZONES):
        # --- MOD: 危险区域改为网格矩形 ---
        for _ in range(num):

            # 随机矩形尺寸（以格为单位）
            grid_w = random.randint(2, 12)
            grid_h = random.randint(2, 12)

            # 随机位置
            gx = random.randint(0, GRID_W - grid_w - 1)
            gy = random.randint(0, GRID_H - grid_h - 1)

            # 创建危险区
            dz = DangerZone(gx, gy, grid_w, grid_h)
            self.danger_zones.append(dz)

            # ground_grid 填 DANGER
            for j in range(gy, gy + grid_h):
                for i in range(gx, gx + grid_w):
                    self.ground_grid[j, i] = DANGER

    def spawn_agents(self,num=NUM_LARGE):
        """在 base 点生成所有大机器人"""
        init_map = np.full((GRID_H, GRID_W), UNKNOWN, dtype=np.int8)
        init_map[init_map == DANGER] = UNKNOWN
        init_map[init_map == VICTIM] = UNKNOWN
        init_map[init_map == FREE] = UNKNOWN
        for r in range(GRID_H):      # 行 (y)
            for c in range(GRID_W):  # 列 (x)
                # 获取真值地图这一格的状态
                val = self.ground_grid[r, c]
                if val == OBSTACLE:
                    # 如果真值本来就是墙，那肯定是墙
                    init_map[r, c] = OBSTACLE
        self.visited_grid = init_map.copy()
        self.large_agents = []
        bx, by = self.spawn_center   # 基地坐标
        small_global_id = 1000

        for i in range(num):
            x, y = bx, by
            la = LargeAgent(i, x, y)
            # 为每个 large 生成 3 台 small
            for k in range(3):
                sx, sy = bx, by
                sa = AgentBase(
                    id=small_global_id,
                    x=sx,
                    y=sy,
                    sensor_range=SENSOR_SMALL,
                    is_large=False
                )
                sa.father_id = i
                la.son_ids.append(small_global_id)
                self.agents.append(sa)
                small_global_id += 1
            self.large_agents.append(la)
        # 分发给所有大/小机器人
        for la in self.large_agents:
            la.local_map = init_map.copy()
            la.known_map = init_map.copy()

        for sa in self.agents:
            sa.local_map = init_map.copy()
        self.brain.local_map = init_map.copy()
        self.brain.known_map = init_map.copy()
        self.spawn_times += num

    def find_agent_by_id(self, id_):
        if id_ is None:
            # print(f"Warning: find_agent_by_id called {id_} with None id")
            return None
        for a in self.agents + self.large_agents:
            if a.id == id_:
                return a
        return None

    def spawn_reinforcement_agent(self, parent_large_id):
            """
            向基地求援：生成一个新的 Small Agent
            初始位置在基地 (spawn_center)，目标设为跟随父节点
            """
            # 找到父节点对象
            parent_agent = self.find_agent_by_id(parent_large_id)
            
            if not parent_agent:
                return

            # 生成新的 ID (这里简单处理，实际应用可能需要更好的 ID 管理)
            new_id = 1000 + len(self.agents) + len(self.wasted_agents)
            bx, by = self.spawn_center
            
            new_sa = AgentBase(
                id=new_id,
                x=bx, y=by,
                is_large=False
            )
            new_sa.father_id = parent_large_id
            
            # 继承当前的已知地图
            new_sa.local_map = self.known_grid.copy()
            
            new_sa.task_seq = [parent_agent.pos] # 初始目标指向父亲
            
            # 注册关系
            parent_agent.son_ids.append(new_id)
            self.agents.append(new_sa)
            
            print(f"[Reinforcement] New Agent {new_id} dispatched for Large {parent_large_id}")

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

    def check_and_handle_deaths(self):
            """
            处理所有机器人的死亡判定逻辑
            Small: 自身+上下左右5点判定 -> 死亡 -> 上报 Large
            Large: 自身中心判定 -> 死亡
            """
            
            # 1. 检测 Small Agents (5点判定)
            # 偏移量：中心，上，下，左，右 (单位：像素，假设1个像素或者微小偏移)
            # 注意：如果坐标是连续的，这里的 offset 可以是 1 或 2 像素
            offsets = [(0,0)]
            # offsets = [(0,0), (0, 1), (0, -1), (1, 0), (-1, 0)]
            for sa in self.agents:
                if not sa.alive: continue
                sa_cell_pos = cell_of_pos(sa.pos)
                
                hit_danger = False
                fatal_pos = None
                
                for dx, dy in offsets:
                    check_x = sa_cell_pos[0] + dx
                    check_y = sa_cell_pos[1] + dy
                    if 0<=check_x<GRID_W and 0<=check_y<GRID_H:
                        if self.ground_grid[check_y,check_x] == DANGER:
                            hit_danger = True
                            fatal_pos = (check_x, check_y)
                            break
                
                if hit_danger:
                    sa.alive = False
                    print(f"!! Agent {sa.id} DIED at {fatal_pos}. Reporting to Father {sa.father_id}.")
                    
                    # 查找父节点并上报
                    father = next((la for la in self.large_agents if la.id == sa.father_id), None)
                    if father and father.alive:
                        # 父节点记录：(死亡子节点ID, 子节点生前最后一个安全位置)
                        # 假设 sa.last_safe_pos 在每次 move 前更新
                        safe_pos = sa.hist[-2] 
                        father.death_queue.append({
                            'child_id': sa.id,
                            'loc': safe_pos # 这个pos是世界坐标
                        })
                        # 标记致死的危险区域点到父节点地图 (可选)
                        cx, cy = fatal_pos
                        if 0 <= cx < GRID_W and 0 <= cy < GRID_H:
                            father.local_map[cy, cx] = DANGER
                            father.known_map[cy, cx] = DANGER
                            
                            self.visited_grid[cy, cx] = DANGER
                            self.known_grid[cy, cx] = DANGER
                            self.brain.known_map[cy, cx] = DANGER

        # 2. 检测 Large Agents (仅中心判定)
            for la in self.large_agents:
                if not la.alive: continue
                la_cell_pos = cell_of_pos(la.pos)
                if 0<=la_cell_pos[0]<GRID_W and 0<=la_cell_pos[1]<GRID_H:
                    if self.ground_grid[la_cell_pos[1], la_cell_pos[0]] == DANGER:
                        la.alive = False
                        print(f"!!!! Large Agent {la.id} DIED in Danger Zone !!!!")

    def update(self, dt, comms: Communication, now_time):
        """主仿真循环，每一帧调用一次"""

        # 0. 更新时间
        self.time += dt

        # 1. 更新所有机器人本地观测（small/middle/large 统一）
        for a in self.agents + self.large_agents:
            if a.alive:
                a.update_local_map_from_sensing(self)

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
        self.large_agents = [la for la in self.large_agents if la.alive]
        
        # 通常 brain / large 需要得到全局一致地图
        for a in self.agents:
            if a.alive:
                self.large_agents[a.father_id].known_map = np.maximum(self.large_agents[a.father_id].known_map, a.local_map)
                self.known_grid = np.maximum(self.known_grid, self.large_agents[a.father_id].known_map)
        self.brain.known_map = self.known_grid

            # 3. 全局规划（只有 brain 执行）
        if (self.brain is not None and 
            now_time - self.brain.last_reason_time >= BRAIN_REASON_INTERVAL):

            # ======== (A) 区域划分（brain → middle） ========
            middles = [la for la in self.large_agents 
                    if la.alive and not la.is_brain]

            region_assign = self.brain.assign_region_to_middle(middles)

            for mid in middles:
                # region_assign[mid.id] 是 list
                mid.region = region_assign[mid.id][0]

            # ======== (B) middle → small 规划任务 ========
            for mid in middles:
                sons_list = [a for a in self.agents if a.father_id == mid.id]

                assigns = mid.generate_and_dispatch_tasks(
                    child_agents=sons_list,
                    comm=None,
                    region_mask=mid.region,
                    frontier_cells=None,
                    num_points=30,
                    min_dist=4.0,
                    now_time=now_time
                )

                # 给 small 分配
                for son in sons_list:
                    seq = assigns.get(son.id, [])
                    son.task_seq = seq[:]
                    son.has_goal = len(seq) > 0
                    if son.task_seq:
                        son.plan_path_sequence()

            # 更新时间戳
            self.brain.last_reason_time = now_time

        for la in self.large_agents:
            if la.alive:
                sons_list = [a for a in self.agents if a.father_id == la .id]
                la.update_strategy(self,sons_list)


        # 5. 所有 agent 执行 step_motion（跟踪各自的 planned_path）
        for a in self.agents + self.large_agents:
            if a.alive:
                a.step_motion()
                # if not a.is_large
                self.mark_visited(a.pos[0], a.pos[1],a.is_large)
       
        self.check_and_handle_deaths()
        
        # 6.1. 预处理 Large Agents (确定存活/死亡状态)
        # 识别已死亡的 Large Agent ID 集合
        dead_large_ids = {la.id for la in self.large_agents if not la.alive}
        
        # 建立存活 Large Agent 的 ID-对象 映射，方便查找最近的父节点
        alive_large_map = {la.id: la for la in self.large_agents if la.alive} 
        
        # 更新 self.wasted_large_agents 和 self.large_agents 列表
        self.wasted_large_agents.extend([la for la in self.large_agents if not la.alive])
        self.large_agents = list(alive_large_map.values())


        # 6.2. 处理 Small Agents (存亡判断与父节点重分配)
        alive_agents = []
        for a in self.agents:
            if not a.alive:
                # 机器人已死亡，移至浪费列表
                self.wasted_agents.append(a)
            else:
                # 机器人存活，检查其父节点状态
                parent_id = a.father_id
                
                # 如果当前父节点 ID 存在于已死亡的 Large Agent ID 集合中
                if parent_id in dead_large_ids:
                    
                    if alive_large_map:
                        # **核心逻辑：找到最近的存活 Large Agent 作为新父节点**
                        
                        # 使用 distance() 函数计算与所有存活 Large Agent 的距离，并找到最近的一个
                        nearest_la = min(
                            alive_large_map.values(), 
                            key=lambda la: distance(a.pos, la.pos)
                        )
                        
                        # 重新分配父节点 ID
                        # (在实际系统中，你可能还需要更新老父节点的 son_ids 列表，但这需要更复杂的逻辑)
                        a.father_id = nearest_la.id
                        # print(f"Agent {a.id} 的父节点 {parent_id} 死亡，重新分配给 {nearest_la.id}") 

                    else:
                        # 没有存活的 Large Agent 了
                        a.father_id = None
                        break
                        # print(f"Agent {a.id} 失去父节点且没有可用的新父节点。")
                
                alive_agents.append(a)
                
        self.agents = alive_agents
        print(len(self.wasted_agents))
        # 1. 创建当前所有存活 Small Agent 的 ID 集合，用于快速查找
        alive_small_agent_ids = {a.id for a in self.agents}

        # 2. 遍历所有存活 Large Agent，清理其子节点列表
        for la in self.large_agents:
            if not la.alive:
                continue
            initial_count = len(la.son_ids)
            
            # 过滤 la.son_ids，只保留 ID 仍在 alive_small_agent_ids 集合中的子节点
            new_son_ids = [
                child_id for child_id in la.son_ids 
                if child_id in alive_small_agent_ids
            ]
            
            # 如果列表发生了变化，则更新
            if len(new_son_ids) != initial_count:
                la.son_ids = new_son_ids

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

    def coverage_percentage(self):
        total = GRID_W * GRID_H
        # count union of explored cells
        explored = 0
        union_set = set()
        for a in (self.agents + self.large_agents):
            union_set |= a.get_local_explored_cells()
        explored = len(union_set)
        return (explored / total) * 100.0

    def mark_visited(self, x, y, is_large):
        """记录机器人经过的安全栅格"""
        cx, cy = cell_of_pos((x, y))
        if is_large:
            if 0 <= cx < GRID_W and 0 <= cy < GRID_H and \
                self.visited_grid[cy, cx] != OBSTACLE and self.visited_grid[cy, cx] != DANGER:
                self.visited_grid[cy, cx] = FREE
        else:
            dirs = [(0,0)]
            # dirs = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H and \
                    self.visited_grid[ny, nx] != OBSTACLE and self.visited_grid[cy, cx] != DANGER:
                    self.visited_grid[ny, nx] = FREE

    def update_baseline(self, dt, comms:Communication, now_time):
        # Tik
        # 结束判断
        # 死亡节点确认与剔除
        # 动态脑节点选举
        # 分层通信结构：小节点向脑节点周期发送 map_patch，脑节点整合地图
        # 信息整合机制：receiver.integrate_map_patch() 更新全局地图 集体形成共享认知
        # 自组织任务分配 执行 reason_and_assign() 并广播目标 忽视层级结构所有机器人都可以收到，大小结点之间无指挥关系
        self.time += dt
        # 1. 更新所有机器人本地观测（small/middle/large 统一）
        for a in self.agents + self.large_agents:
            if a.alive:
                a.update_local_map_from_sensing(self)

        victim_pos = self.victim.pos
        victim_cell = cell_of_pos(victim_pos)
        for la in self.large_agents:
            if la.known_map[victim_cell[1], victim_cell[0]] != UNKNOWN:
                self.victim.rescued = True

        # 检查哪些节点还活着
        dead_agents = [a for a in self.agents if not a.alive]
        self.wasted_agents.extend(dead_agents)
        self.agents = [a for a in self.agents if a.alive]

        dead_large = [la for la in self.large_agents if not la.alive and not la.is_brain]
        self.wasted_large_agents.extend(dead_large)
        self.large_agents = [la for la in self.large_agents if la.alive]

        # 通常 brain / large 需要得到全局一致地图
        for a in self.agents:
            if a.alive:
                self.large_agents[a.father_id].known_map = np.maximum(self.large_agents[a.father_id].known_map, a.local_map)
                self.known_grid = np.maximum(self.known_grid, self.large_agents[a.father_id].known_map)
        self.brain.known_map = self.known_grid

        for la in self.large_agents:
            if not la.alive:
                continue
            if self.time - la.last_reason_time > BRAIN_REASON_INTERVAL:
                sons_list = [a for a in self.agents if a.father_id == la .id]
                assignments = la.large_reason(sons_list)
                for id, target in assignments.item():
                    

        # 5. 所有 agent 执行 step_motion（跟踪各自的 planned_path）
        for a in self.agents + self.large_agents:
            if a.alive:
                a.step_motion()
                # if not a.is_large
                self.mark_visited(a.pos[0], a.pos[1],a.is_large)
       
        self.check_and_handle_deaths()
        
        # 6.1. 预处理 Large Agents (确定存活/死亡状态)
        # 识别已死亡的 Large Agent ID 集合
        dead_large_ids = {la.id for la in self.large_agents if not la.alive}
        
        # 建立存活 Large Agent 的 ID-对象 映射，方便查找最近的父节点
        alive_large_map = {la.id: la for la in self.large_agents if la.alive} 
        
        # 更新 self.wasted_large_agents 和 self.large_agents 列表
        self.wasted_large_agents.extend([la for la in self.large_agents if not la.alive])
        self.large_agents = list(alive_large_map.values())


        # 6.2. 处理 Small Agents (存亡判断与父节点重分配)
        alive_agents = []
        for a in self.agents:
            if not a.alive:
                # 机器人已死亡，移至浪费列表
                self.wasted_agents.append(a)
            else:
                # 机器人存活，检查其父节点状态
                parent_id = a.father_id
                
                # 如果当前父节点 ID 存在于已死亡的 Large Agent ID 集合中
                if parent_id in dead_large_ids:
                    
                    if alive_large_map:
                        # **核心逻辑：找到最近的存活 Large Agent 作为新父节点**
                        
                        # 使用 distance() 函数计算与所有存活 Large Agent 的距离，并找到最近的一个
                        nearest_la = min(
                            alive_large_map.values(), 
                            key=lambda la: distance(a.pos, la.pos)
                        )
                        
                        # 重新分配父节点 ID
                        # (在实际系统中，你可能还需要更新老父节点的 son_ids 列表，但这需要更复杂的逻辑)
                        a.father_id = nearest_la.id
                        # print(f"Agent {a.id} 的父节点 {parent_id} 死亡，重新分配给 {nearest_la.id}") 

                    else:
                        # 没有存活的 Large Agent 了
                        a.father_id = None
                        break
                        # print(f"Agent {a.id} 失去父节点且没有可用的新父节点。")
                
                alive_agents.append(a)
                
        self.agents = alive_agents
        print(len(self.wasted_agents))
        # 1. 创建当前所有存活 Small Agent 的 ID 集合，用于快速查找
        alive_small_agent_ids = {a.id for a in self.agents}

        # 2. 遍历所有存活 Large Agent，清理其子节点列表
        for la in self.large_agents:
            if not la.alive:
                continue
            initial_count = len(la.son_ids)
            
            # 过滤 la.son_ids，只保留 ID 仍在 alive_small_agent_ids 集合中的子节点
            new_son_ids = [
                child_id for child_id in la.son_ids 
                if child_id in alive_small_agent_ids
            ]
            
            # 如果列表发生了变化，则更新
            if len(new_son_ids) != initial_count:
                la.son_ids = new_son_ids


    def _update(self, dt, comms:Communication, now_time):
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
        overlay = pygame.Surface((GRID_CELL, GRID_CELL), pygame.SRCALPHA)

        for i in range(GRID_W):
            for j in range(GRID_H):

                if self.visited_grid[j, i] == FREE:
                    overlay.fill((120, 180, 255, 90))  # light blue w/ alpha
                    screen.blit(overlay, (i*GRID_CELL, j*GRID_CELL))
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

    def debug_verify_map_fusion(self):
        """
        检查：子节点中的已知格子是否仍然在父节点里是 UNKNOWN
        如果有，说明融合失败。
        """
        for la in self.large_agents:
            father = la
            father_map = father.known_map

            # 遍历该 large 的所有子 small
            for sid in la.son_ids:
                child = next((a for a in self.agents if a.id == sid), None)
                if child is None:
                    continue
                child_map = child.local_map

                # 找出现 "child 已知但 parent 仍未知" 的点
                bad = np.where((child_map != UNKNOWN) & (father_map == UNKNOWN))

                if len(bad[0]) > 0:
                    print(f"\n[DEBUG][MAP FUSION ERROR] Child {sid} → Father {la.id}")
                    print(f"  Count of missed cells: {len(bad[0])}")
                    # 打印前 5 个坐标示例
                    for i in range(min(5, len(bad[0]))):
                        ci = bad[0][i]
                        cj = bad[1][i]
                        print(f"    At cell ({ci},{cj}) child={child_map[ci,cj]}, father=UNKNOWN")
                    print("------------------------------------------------------")
                else:
                    print(f"[DEBUG] Child {sid} merged into Father {la.id} SUCCESSFULLY.")


    def save_state(self, filename="world_state.pkl"):
        """
        使用 pickle 将当前 World 实例的状态保存到文件。
        
        参数:
            filename (str): 保存的文件名 (例如: 'my_world_seed_123.pkl')
        """
        # 1. 创建一个只包含必要状态数据的字典
        # 排除掉可能不需要保存或不兼容 pickle 的部分，通常是临时的计算结果。
        # 注意: 这里的 self.agents, self.large_agents, self.brain 等都将被保存，
        # 它们内部的状态也会递归保存。
        state_to_save = self.__dict__
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(state_to_save, f)
            print(f"✅ 世界状态成功保存到: {filename}")
        except Exception as e:
            print(f"❌ 世界状态保存失败: {e}")

    @classmethod
    def load_state(cls, filename="world_state.pkl"):
        """
        使用 pickle 从文件加载 World 实例的状态。
        注意: 这是一个类方法 (classmethod)，它返回一个新的 World 实例。
        
        参数:
            filename (str): 加载的文件名
            
        返回:
            World: 加载成功的 World 实例，或 None (加载失败)
        """
        if not os.path.exists(filename):
            print(f"❌ 错误: 文件 {filename} 不存在。")
            return None
            
        try:
            with open(filename, 'rb') as f:
                # 1. 加载保存的状态字典
                state = pickle.load(f)
            
            # 2. 创建一个新的 World 实例（不需要运行 __init__）
            # 我们通过 __new__ 跳过 __init__，因为它会重新生成环境
            new_world = cls.__new__(cls)
            
            # 3. 将加载的状态字典赋值给新实例的 __dict__
            new_world.__dict__.update(state)
            
            print(f"✅ 世界状态成功从 {filename} 载入。")
            return new_world
            
        except Exception as e:
            print(f"❌ 世界状态载入失败: {e}")
            # 打印堆栈信息帮助调试，特别是遇到 UnpicklingError 时
            import traceback
            traceback.print_exc() 
            return None