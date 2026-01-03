import pygame
import numpy as np
import torch
import random
import sys
import time
import math
from env import World
from communicate import Communication
from neural.neural_brain_logic import NeuralBrainLogic
from data_collector import extract_comprehensive_state, downsample_map
from parameters import *

# ================= 配置区 =================
TOTAL_EPISODES = 50   # 总共比多少场
DISPLAY_SCALE = 0.6  
WINDOW_W = int(SCREEN_W * DISPLAY_SCALE * 2 + 50) 
WINDOW_H = int(SCREEN_H * DISPLAY_SCALE + 100)    

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (200, 50, 50)
GREEN = (50, 200, 50)
BLUE  = (50, 50, 200)
CYAN  = (0, 200, 200)
GOLD  = (255, 215, 0)
GRAY  = (100, 100, 100)

class RaceRunner:
    def __init__(self, model_path="brain_fitter_v3.pth"):
        self.ai_logic = NeuralBrainLogic(model_path)
        
        # 统计数据容器
        self.stats = {
            "Original": {"wins": 0, "timeouts": 0, "died": 0, "times": [], "coverage": []},
            "AI":       {"wins": 0, "timeouts": 0, "died": 0, "times": [], "coverage": []}
        }
        
        self.visual_mode = True # 默认开启可视化
        if self.visual_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
            pygame.display.set_caption("🔥 最终决战: Expert vs AI (按 'Q' 切换到后台极速跑)")
            self.font = pygame.font.SysFont("Arial", 20)
            self.title_font = pygame.font.SysFont("Arial", 32, bold=True)

    def reset_worlds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.w1 = World(seed=seed) # Expert
        self.c1 = Communication()
        
        random.seed(seed)
        np.random.seed(seed)
        self.w2 = World(seed=seed) # AI
        self.c2 = Communication()
        
        self.sim_time = 0.0
        self.finished = [False, False] 
        self.results = [None, None]
        self.rescue_timer = 0.0 # AI 读条计时

    def check_manual_rescue(self, world):
        if not world.victim: return False
        vic_pos = np.array(world.victim.pos)
        RESCUE_DIST = 20.0 
        for la in world.large_agents:
            if la.alive:
                dist = np.linalg.norm(la.pos - vic_pos)
                if dist < RESCUE_DIST:
                    return True
        return False

    def update_logic(self, dt):
        """ 核心物理循环 (Visual和后台模式通用) """
        BRAIN_INTERVAL = 3.5
        
        # --- LEFT: EXPERT ---
        if not self.finished[0]:
            try:
                if int(self.sim_time/dt) % int(BRAIN_INTERVAL/dt) == 0:
                    actives = [la for la in self.w1.large_agents if la.alive]
                    if actives: self.w1.brain.assign_region_to_middle(actives)
                self.w1.update(dt, self.c1, self.sim_time)
                if self.check_manual_rescue(self.w1) or self.w1.victim.rescued:
                    self.finished[0], self.results[0] = True, "SUCCESS"
                if all(not la.alive for la in self.w1.large_agents) and self.w1.spawn_times >= 3:
                    self.finished[0], self.results[0] = True, "FAILED"
            except Exception: pass 

        # --- RIGHT: AI ENHANCED ---
        if not self.finished[1]:
            try:
                # 1. 视野 & 读条
                in_sight = False
                if self.w2.victim:
                    vic_pos = np.array(self.w2.victim.pos)
                    VISUAL_RANGE = 100.0
                    for la in self.w2.large_agents:
                        if la.alive:
                            dist = np.linalg.norm(la.pos - vic_pos)
                            if dist < VISUAL_RANGE:
                                in_sight = True
                                la.task_seq = [] # 看到即停
                                break
                
                if in_sight: self.rescue_timer += dt
                else: self.rescue_timer = 0.0

                # 2. AI 决策
                if not in_sight:
                    if int(self.sim_time/dt) % int(BRAIN_INTERVAL/dt) == 0:
                        actives = [la for la in self.w2.large_agents if la.alive]
                        if actives:
                            state = extract_comprehensive_state(self.w2)
                            d_map = downsample_map(self.w2.known_grid)
                            raw_goals = self.ai_logic.get_ai_decision(state, d_map)
                            for i, g in enumerate(raw_goals[:len(actives)]):
                                safe_goal = self.ai_logic.validate_and_fix_goal(g, self.w2.grid)
                                if safe_goal:
                                    actives[i].task_seq = [safe_goal]
                                    actives[i].plan_path_sequence()

                # 3. 物理更新
                self.w2.update(dt, self.c2, self.sim_time)
                
                # 4. 判定
                if self.rescue_timer >= 2.0:
                    self.finished[1], self.results[1] = True, "SUCCESS"
                if all(not la.alive for la in self.w2.large_agents) and self.w2.spawn_times >= 3:
                    self.finished[1], self.results[1] = True, "FAILED"
            except Exception: pass

    # ================= 渲染相关 =================
    def draw_world(self, world, status, is_ai=False):
        surf = pygame.Surface((SCREEN_W, SCREEN_H))
        surf.fill((220, 220, 220)) 
        for obs in world.obstacles: pygame.draw.rect(surf, BLACK, obs.rect)
        for dz in world.danger_zones:
            s = pygame.Surface((dz.rect.width, dz.rect.height), pygame.SRCALPHA)
            s.fill((255, 0, 0, 80)) 
            surf.blit(s, dz.rect.topleft)
        for a in world.agents:
            if a.alive:
                color = BLUE if not a.is_large else (0, 0, 150)
                pos = (int(a.pos[0]), int(a.pos[1]))
                pygame.draw.circle(surf, color, pos, 4)
        for la in world.large_agents:
            if la.alive:
                sp = (int(la.pos[0]), int(la.pos[1]))
                pygame.draw.circle(surf, (0, 0, 255), sp, 10)
                if la.task_seq:
                    rt = la.task_seq[0]
                    ep = (int(rt[0]), int(rt[1]))
                    pygame.draw.line(surf, GREEN, sp, ep, 3)
                    pygame.draw.circle(surf, GREEN, ep, 5)
        if world.victim:
            is_suc = (status == "SUCCESS")
            color = (50, 200, 50) if is_suc else GOLD
            vp = (int(world.victim.pos[0]), int(world.victim.pos[1]))
            pygame.draw.circle(surf, color, vp, 8)
            if is_ai and not is_suc and self.rescue_timer > 0:
                pygame.draw.circle(surf, CYAN, vp, 100, 1)
                prog = min(self.rescue_timer/2.0, 1.0)
                rect = pygame.Rect(vp[0]-20, vp[1]-20, 40, 40)
                pygame.draw.arc(surf, RED, rect, 0, 3.14*2*prog, 4)
                
        target_w = int(SCREEN_W * DISPLAY_SCALE)
        target_h = int(SCREEN_H * DISPLAY_SCALE)
        scaled_surf = pygame.transform.smoothscale(surf, (target_w, target_h))
        pygame.draw.rect(scaled_surf, BLACK, (0,0,target_w, target_h), 2)
        return scaled_surf

    def render_frame(self, seed):
        self.screen.fill(WHITE)
        title = self.title_font.render(f"Ep: {seed}/{TOTAL_EPISODES} - {self.sim_time:.1f}s", True, BLACK)
        self.screen.blit(title, (WINDOW_W//2 - title.get_width()//2, 10))
        
        l_surf = self.draw_world(self.w1, self.results[0])
        r_surf = self.draw_world(self.w2, self.results[1], True)
        self.screen.blit(l_surf, (20, 60))
        self.screen.blit(r_surf, (20 + l_surf.get_width() + 10, 60))
        
        # 底部信息
        y_stats = 60 + l_surf.get_height() + 10
        s1 = f"Original: {self.results[0] or 'Running'} | Cov: {self.w1.coverage_percentage():.1f}%"
        self.screen.blit(self.font.render(s1, True, BLACK), (20, y_stats))
        
        ai_st = self.results[1] or ("Scanning..." if self.rescue_timer > 0 else "Running")
        s2 = f"AI Model: {ai_st} | Timer: {self.rescue_timer:.1f}s"
        self.screen.blit(self.font.render(s2, True, BLUE), (20 + l_surf.get_width() + 10, y_stats))
        
        tip = self.font.render("[SPACE] Skip  |  [Q] Exit to Background Mode", True, RED)
        self.screen.blit(tip, (WINDOW_W//2 - 150, WINDOW_H - 30))
        pygame.display.flip()

    # ================= 主流程 =================
    def run_episode(self, episode_idx):
        seed = random.randint(0, 100000)
        self.reset_worlds(seed)
        dt = 0.1
        running = True
        skip = False
        
        print(f"[{episode_idx+1}/{TOTAL_EPISODES}] Running Seed {seed}...", end="\r")

        while running:
            # 1. 交互逻辑 (仅在可视化模式下)
            if self.visual_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE: skip = True # 跳过当前局
                        if event.key == pygame.K_q: # 退出可视化，转后台
                            self.visual_mode = False
                            pygame.quit()
                            print("\n\n⚡ 切换到后台极速模式! 请稍候...")

            # 2. 物理更新
            if self.sim_time < 60.0 and (not self.finished[0] or not self.finished[1]):
                self.update_logic(dt)
                self.sim_time += dt
            else:
                running = False # 时间到或都跑完了

            # 3. 结果修正
            if self.sim_time >= 60.0:
                if not self.finished[0]: self.results[0] = "TIMEOUT"
                if not self.finished[1]: self.results[1] = "TIMEOUT"
                running = False

            if self.finished[0] and self.finished[1]:
                running = False
            
            # 4. 渲染或跳过
            if self.visual_mode and not skip:
                self.render_frame(episode_idx+1)
            
            # 5. 如果 skip 或后台模式，不限制帧率，全速跑
            if skip and self.visual_mode:
                pass # 全速
        
        # === 单局结束，记录数据 ===
        self.record_stats("Original", self.w1, self.results[0])
        self.record_stats("AI", self.w2, self.results[1])

    def record_stats(self, key, world, result):
        if result == "SUCCESS":
            self.stats[key]["wins"] += 1
            self.stats[key]["times"].append(self.sim_time)
        elif result == "TIMEOUT":
            self.stats[key]["timeouts"] += 1
            self.stats[key]["times"].append(60.0)
        else:
            self.stats[key]["died"] += 1
            self.stats[key]["times"].append(60.0)
        
        self.stats[key]["coverage"].append(world.coverage_percentage())

    def print_final_report(self):
        print("\n\n" + "="*60)
        print(f"📊 最终决战报告 (共 {TOTAL_EPISODES} 场随机测试)")
        print("-" * 60)
        
        for key in ["Original", "AI"]:
            s = self.stats[key]
            win_rate = s["wins"] / TOTAL_EPISODES * 100
            avg_time = np.mean(s["times"]) if s["times"] else 0
            avg_cov  = np.mean(s["coverage"]) if s["coverage"] else 0
            
            print(f"[{key} 选手]")
            print(f"  🏆 胜率: {win_rate:.1f}%  (胜:{s['wins']} / 超时:{s['timeouts']} / 牺牲:{s['died']})")
            print(f"  ⏱️ 平均用时: {avg_time:.2f}秒")
            print(f"  🗺️ 平均覆盖率: {avg_cov:.1f}%")
            print("-" * 60)
        
        print("="*60)

    def run_all(self):
        for i in range(TOTAL_EPISODES):
            self.run_episode(i)
        
        self.print_final_report()

if __name__ == "__main__":
    race = RaceRunner()
    race.run_all()