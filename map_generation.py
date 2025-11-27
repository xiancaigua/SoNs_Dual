import pygame
import random
import sys
import time
import numpy as np
import os
from datetime import datetime

# 假设这些是您已有的导入。请确保这些文件和模块在您的运行环境中可用。
from parameters import * 
from utils import * 
from communicate import Communication
from env import World
# 假设 save_simulation_summary, load_font, print_simulation_summary, create_summary_image, save_simulation_screenshot 
# 都已在 utils.py 中定义或已在当前作用域中可用。


# ==================================
# 💡 实验恢复辅助函数 (保持不变)
# ==================================

def get_completed_runs(results_dir):
    """统计指定文件夹中已保存的JSON结果文件数量。"""
    if not os.path.exists(results_dir):
        return 0
    # 统计所有以 .json 结尾的文件
    files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    return len(files)


# -----------------------------
# 主循环与UI (已修改)
# -----------------------------
def main(rounds=1):
    """
    运行一次仿真实验。
    rounds: 当前实验的序号 (从 0 到 1199)
    """
    global SEED
    if SEED is not None:
        # 为每个回合设置一个不同的种子，保证实验的可重复性
        # 这里使用 rounds 作为辅助因子来生成新的种子
        current_seed = SEED + rounds 
        random.seed(current_seed)
        np.random.seed(current_seed)
    
    pygame.init()
    clock = pygame.time.Clock()
    
    # 假设 load_font() 和 SCREEN_W/H, AGENT_RADIUS, LARGE_RADIUS 都已定义

    if VISUALIZE:
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("exploration simulation")
        font = load_font()

    # communication system
    comms = Communication(packet_loss=COMM_PACKET_LOSS, delay=COMM_DELAY)
    
    # 12 个地图的种子
    world_seeds = [110716, 710, 8848, 1107, 233174, 12142325, 258, 8456, 1985, 819, 789654, 666]
    
    # 核心逻辑：rounds // 100 决定了当前运行的是第几个地图 (0到11)
    map_index = rounds // 100
    if map_index >= len(world_seeds):
        print(f"ERROR: rounds {rounds} exceeds total maps.")
        return
        
    world = World(seed=world_seeds[12])
    print(f"--- Running Experiment {rounds + 1} (Map Index: {map_index}, Seed: {world_seeds[map_index]}) ---")

    screenshot_dir = "simulation_screenshots"
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)

    # =============================================================================
    # 📌 新增逻辑：在开始模拟前保存初始地图截图
    # =============================================================================
    if VISUALIZE:
        # 1. 强制绘制初始的世界状态 (T=0.0s)
        world.draw(screen)
        
        # 2. 绘制 HUD/图例，确保截图完整
        try:
            # 绘制 HUD 状态 (T=0.0s)
            alive = sum(1 for a in world.agents if a.alive) + sum(1 for la in world.large_agents if la.alive)
            # coverage = world.coverage_percentage()
            # # sim_time 此时为 0.0
            # hud1 = f"Time: {0.0:.1f}s  Alive agents: {alive}/{len(world.agents)+len(world.large_agents)}  Coverage: {coverage:.2f}%"
            # hud2 = f"Obstacles: {len(world.obstacles)}  Dangers: {len(world.danger_zones)}  Victim: {'rescued' if world.victim.rescued else 'missing'}"
            # screen.blit(font.render(hud1, True, (10, 10, 10)), (8, 6))
            # screen.blit(font.render(hud2, True, (10, 10, 10)), (8, 24))

            # # 绘制 legend
            # pygame.draw.circle(screen, (40, 120, 220), (SCREEN_W - 120, 30), AGENT_RADIUS)
            # screen.blit(font.render("Small Agent", True, (0, 0, 0)), (SCREEN_W - 96, 22))
            # pygame.draw.circle(screen, (200, 160, 60), (SCREEN_W - 120, 60), LARGE_RADIUS)
            # screen.blit(font.render("Large Agent", True, (0, 0, 0)), (SCREEN_W - 96, 52))
            # pygame.draw.circle(screen, (200, 40, 40), (SCREEN_W - 120, 90), 8)
            # screen.blit(font.render("Danger", True, (0, 0, 0)), (SCREEN_W - 96, 82))
        except NameError as e:
            # 如果缺少 AGENT_RADIUS, SCREEN_W 等常量会导致 NameError
            print(f"⚠️ 初始截图：缺少常量或函数定义 ({e})，HUD/图例绘制可能不完整。")
            pass
            
        pygame.display.flip() # 立即更新屏幕
        
        # 3. 保存截图
        map_seed = world_seeds[map_index]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        initial_filename = os.path.join(screenshot_dir, f"initial_map_seed{map_seed}_mapindex{map_index}_{timestamp}.png")
        pygame.image.save(screen, initial_filename)
        print(f"✅ 初始地图截图已保存: {initial_filename}")
        sys.exit(0)
        
    # =============================================================================
    # 📌 新增逻辑结束
    # =============================================================================

    running = True
    paused = False
    sim_time = 0.0
    simulation_result = "unknown"  # 记录仿真结果

    while running:
        dt = clock.tick(FPS) / 1000.0
        sim_time += 0 if paused else dt
        now_time = time.time()

        # 事件处理（仅在可视化模式下启用）
        if VISUALIZE:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_s: 
                        save_simulation_screenshot(screen, world, sim_time, "manual") 
        
        if not paused:
            if BASELINE:
                # world.update_base2(dt, comms, now_time)
                world.update_baseline(dt, comms, now_time)
            else:
                world.update(dt, comms, now_time)

        # 绘制（仅在可视化模式下启用）
        if VISUALIZE:
            world.draw(screen)

            # HUD
            # alive = sum(1 for a in world.agents if a.alive) + sum(1 for la in world.large_agents if la.alive)
            # coverage = world.coverage_percentage()
            # hud1 = f"Time: {sim_time:.1f}s  Alive agents: {alive}/{len(world.agents)+len(world.large_agents)}  Coverage: {coverage:.2f}%"
            # hud2 = f"Obstacles: {len(world.obstacles)}  Dangers: {len(world.danger_zones)}  Victim: {'rescued' if world.victim.rescued else 'missing'}"
            # screen.blit(font.render(hud1, True, (10, 10, 10)), (8, 6))
            # screen.blit(font.render(hud2, True, (10, 10, 10)), (8, 24))

            # # legend
            # pygame.draw.circle(screen, (40, 120, 220), (SCREEN_W - 120, 30), AGENT_RADIUS)
            # screen.blit(font.render("Small Agent", True, (0, 0, 0)), (SCREEN_W - 96, 22))
            # pygame.draw.circle(screen, (200, 160, 60), (SCREEN_W - 120, 60), LARGE_RADIUS)
            # screen.blit(font.render("Large Agent", True, (0, 0, 0)), (SCREEN_W - 96, 52))
            # pygame.draw.circle(screen, (200, 40, 40), (SCREEN_W - 120, 90), 8)
            # screen.blit(font.render("Danger", True, (0, 0, 0)), (SCREEN_W - 96, 82))

            pygame.display.flip()

        # 结束条件判断
        if world.victim.rescued:
            simulation_result = "success"
            print("Mission success: victim rescued")
            # 保存成功截图
            if VISUALIZE:
                final_image = create_summary_image(screen, world, sim_time, "SUCCESS", font) 
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pygame.image.save(final_image, f"simulation_screenshots/success_{timestamp}_{map_index}.png")
            paused = True
            running = False
        elif len(world.large_agents) == 0 or world.spawn_times >= MAX_TOLERATE:
            simulation_result = "failure"
            print("All agents destroyed. Mission failed.")
            # 保存失败截图
            if VISUALIZE:
                final_image = create_summary_image(screen, world, sim_time, "FAILURE", font) 
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pygame.image.save(final_image, f"simulation_screenshots/failure_{timestamp}_{map_index}.png")
            paused = True
            running = False 
        elif sim_time > 60.0:
            simulation_result = "failure"
            print("Max sim time reached.")
            # 保存超时截图
            if VISUALIZE:
                final_image = create_summary_image(screen, world, sim_time, "TIMEOUT", font) 
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pygame.image.save(final_image, f"simulation_screenshots/timeout_{timestamp}_{map_index}.png")
            paused = True
            running = False

    # summary
    print_simulation_summary(world, sim_time, simulation_result)
    
    #--------实验总结---------
    # 保存最终状态的简单截图
    if VISUALIZE and simulation_result != "unknown":
        screenshot_path = save_simulation_screenshot(screen, world, sim_time, f"final_{simulation_result}") 
    else:
        screenshot_path = None
        
    # 保存JSON总结。 (map_index + 1) 是地图ID (1-12)
    json_filename = save_simulation_summary(world, sim_time, simulation_result, screenshot_path, map_id=(map_index + 1)) 

    # 可选：在控制台显示JSON文件路径
    if json_filename:
        print(f"详细仿真结果已保存为JSON文件: {json_filename}")
    if VISUALIZE:
        pygame.quit()


if __name__ == "__main__":
    
    RESULTS_FOLDER = "simulation_results" # 假设这是您保存JSON结果的文件夹
    TOTAL_RUNS = 1200 # 总共的实验次数 (12 个场景 * 100 次)
    
    # 1. 确保 results_dir 存在
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        
    # 2. 统计已完成的实验次数
    completed_runs = get_completed_runs(RESULTS_FOLDER)
    
    print(f"=====================================")
    print(f"        --- 实验恢复模式 ---         ")
    print(f"=====================================")
    print(f"结果文件夹: {RESULTS_FOLDER}")
    print(f"已发现 {completed_runs} 个结果文件。")
    
    if completed_runs >= TOTAL_RUNS:
        print(f"所有 {TOTAL_RUNS} 次实验均已完成。程序退出。")
        sys.exit(0)
    
    start_run = completed_runs
    runs_to_go = TOTAL_RUNS - start_run
    
    # 计算当前应该运行的场景和场景内的次数
    current_map_index = start_run // 100
    current_run_in_scene = start_run % 100
    
    print(f"将从第 {start_run + 1} 次实验开始运行 (剩余 {runs_to_go} 次)。")
    print(f"当前地图场景: Scene {current_map_index + 1}, 该场景内运行次数: {current_run_in_scene + 1} / 100。")
    print(f"=====================================")
    
    # 3. 循环从 start_run 开始到 TOTAL_RUNS 结束
    for i in range(start_run, TOTAL_RUNS):
        main(i) # 传入当前的实验次数 i
    
    print(f"所有 {TOTAL_RUNS} 次实验均已完成。")
    sys.exit(0)