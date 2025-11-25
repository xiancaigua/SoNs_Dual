import pygame
import random
import sys
import time
import numpy as np
import os
from datetime import datetime

# 假设这些是您已有的导入：
from parameters import * 
from utils import * 
from communicate import Communication
from env import World
# 假设 save_simulation_summary, load_font, create_summary_image, save_simulation_screenshot, print_simulation_summary 
# 等辅助函数都在 utils.py 中定义。


# ==================================
# 💡 实验恢复辅助函数
# ==================================

def get_completed_runs(base_dir, ours_dir):
    """
    统计两个文件夹中已保存的JSON结果文件总数。
    并返回 BASE 和 OURS 各自完成的数量。
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        base_count = 0
    else:
        base_count = len([f for f in os.listdir(base_dir) if f.endswith('.json')])

    if not os.path.exists(ours_dir):
        os.makedirs(ours_dir)
        ours_count = 0
    else:
        ours_count = len([f for f in os.listdir(ours_dir) if f.endswith('.json')])
        
    return base_count, ours_count, base_count + ours_count


# -----------------------------
# 主循环与UI
# -----------------------------
def main(rounds, results_folder):
    """
    运行一次仿真实验。
    rounds: 当前实验的序号 (从 0 到 2399)
    results_folder: 当前实验结果应保存的文件夹路径。
    """
    global SEED
    if SEED is not None:
        # 为每个回合设置一个不同的种子，保证实验的可重复性
        current_seed = SEED + rounds 
        random.seed(current_seed)
        np.random.seed(current_seed)
    
    # --- 核心修改 1: 确定当前运行的方法 ---
    # Rounds 0-1199 (1200次) 是 Base 方法
    # Rounds 1200-2399 (1200次) 是 Our 方法
    IS_BASELINE_RUN = rounds < 1200
    method_name = "BASE" if IS_BASELINE_RUN else "OURS"
    
    pygame.init()
    clock = pygame.time.Clock()
    
    if VISUALIZE:
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption(f"exploration simulation - {method_name}")
        font = load_font()

    # communication system
    comms = Communication(packet_loss=COMM_PACKET_LOSS, delay=COMM_DELAY)
    
    # 12 个地图的种子
    world_seeds = [110716, 710, 8848, 1107, 233174, 12142325, 258, 8456, 1985, 819, 789654, 666]
    
    # --- 核心修改 2: 确定当前地图索引 ---
    # 确保 Base 和 Ours 方法在同一轮次 (100次) 使用相同的地图序列
    map_cycle_index = rounds % 1200 # 确定在 1200 次循环中的位置 (0-1199)
    map_index = map_cycle_index // 100 # 确定地图索引 (0-11)
    
    if map_index >= len(world_seeds):
        print(f"ERROR: rounds {rounds} exceeds total maps.")
        return
        
    world = World(seed=world_seeds[map_index])
    print(f"--- Running Experiment {rounds + 1}/{TOTAL_RUNS} | Method: {method_name} | Map Index: {map_index}, Seed: {world_seeds[map_index]} ---")


    running = True
    paused = False
    sim_time = 0.0
    simulation_result = "unknown"  # 记录仿真结果

    screenshot_dir = "simulation_screenshots"
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)

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
                    elif event.key == pygame.K_s:  # 手动保存截图
                        save_simulation_screenshot(screen, world, sim_time, "manual")
        
        # --- 核心修改 3: 执行对应的方法更新 ---
        if not paused:
            if IS_BASELINE_RUN:
                # 使用 Base 方法的更新逻辑
                world.update_baseline(dt, comms, now_time) 
            else:
                # 使用我们自己的方法
                world.update(dt, comms, now_time)

        # 绘制（仅在可视化模式下启用）
        if VISUALIZE:
            world.draw(screen)

            # HUD
            alive = sum(1 for a in world.agents if a.alive) + sum(1 for la in world.large_agents if la.alive)
            coverage = world.coverage_percentage()
            hud1 = f"Time: {sim_time:.1f}s  Alive agents: {alive}/{len(world.agents)+len(world.large_agents)}  Coverage: {coverage:.2f}%"
            hud2 = f"Method: {method_name} | Obstacles: {len(world.obstacles)}  Dangers: {len(world.danger_zones)}  Victim: {'rescued' if world.victim.rescued else 'missing'}"
            screen.blit(font.render(hud1, True, (10, 10, 10)), (8, 6))
            screen.blit(font.render(hud2, True, (10, 10, 10)), (8, 24))

            # legend (保持不变)
            pygame.draw.circle(screen, (40, 120, 220), (SCREEN_W - 120, 30), AGENT_RADIUS)
            screen.blit(font.render("Small Agent", True, (0, 0, 0)), (SCREEN_W - 96, 22))
            pygame.draw.circle(screen, (200, 160, 60), (SCREEN_W - 120, 60), LARGE_RADIUS)
            screen.blit(font.render("Large Agent", True, (0, 0, 0)), (SCREEN_W - 96, 52))
            pygame.draw.circle(screen, (200, 40, 40), (SCREEN_W - 120, 90), 8)
            screen.blit(font.render("Danger", True, (0, 0, 0)), (SCREEN_W - 96, 82))

            pygame.display.flip()

        # 结束条件判断 (截图文件名增加方法标识)
        if world.victim.rescued:
            simulation_result = "success"
            print("Mission success: victim rescued")
            if VISUALIZE:
                final_image = create_summary_image(screen, world, sim_time, "SUCCESS", font)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 截图保存在公共目录，但文件名包含 method_name
                pygame.image.save(final_image, f"simulation_screenshots/success_{timestamp}_{map_index}_{method_name}.png")
            paused = True
            running = False
        elif len(world.large_agents) == 0 or world.spawn_times >= MAX_TOLERATE:
            simulation_result = "failure"
            print("All agents destroyed. Mission failed.")
            if VISUALIZE:
                final_image = create_summary_image(screen, world, sim_time, "FAILURE", font)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pygame.image.save(final_image, f"simulation_screenshots/failure_{timestamp}_{map_index}_{method_name}.png")
            paused = True
            running = False  
        elif sim_time > 60.0:
            simulation_result = "failure"
            print("Max sim time reached.")
            if VISUALIZE:
                final_image = create_summary_image(screen, world, sim_time, "TIMEOUT", font)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pygame.image.save(final_image, f"simulation_screenshots/timeout_{timestamp}_{map_index}_{method_name}.png")
            paused = True
            running = False

    # summary
    print_simulation_summary(world, sim_time, simulation_result)
    
    #--------实验总结---------
    # 保存最终状态的简单截图
    if VISUALIZE and simulation_result != "unknown":
        # 假设 save_simulation_screenshot 也会在公共目录（simulation_screenshots）中保存文件
        screenshot_path = save_simulation_screenshot(screen, world, sim_time, f"final_{simulation_result}_{method_name}") 
    else:
        screenshot_path = None
        
    # --- 核心修改 4: 保存JSON到指定的独立文件夹 ---
    # 假设 save_simulation_summary 接受 results_folder 参数
    json_filename = save_simulation_summary(world, sim_time, simulation_result, screenshot_path, 
                                            map_id=(map_index + 1), results_dir=results_folder) 

    if json_filename:
        print(f"详细仿真结果已保存为JSON文件: {json_filename}")
    if VISUALIZE:
        pygame.quit()


if __name__ == "__main__":
    
    BASE_RESULTS_FOLDER = "simulation_results_BASE"
    OURS_RESULTS_FOLDER = "simulation_results_OURS"
    TOTAL_RUNS = 2400 
    BASE_RUNS = 1200
    
    # 1. 统计已完成的实验次数
    base_completed, ours_completed, completed_runs = get_completed_runs(BASE_RESULTS_FOLDER, OURS_RESULTS_FOLDER)
    
    print(f"=====================================")
    print(f"        --- 实验恢复模式 ---         ")
    print(f"=====================================")
    print(f"BASE 结果文件夹: {BASE_RESULTS_FOLDER} (已完成 {base_completed} 次)")
    print(f"OURS 结果文件夹: {OURS_RESULTS_FOLDER} (已完成 {ours_completed} 次)")
    print(f"总计已完成 {completed_runs} / {TOTAL_RUNS} 次实验。")
    
    if completed_runs >= TOTAL_RUNS:
        print(f"所有 {TOTAL_RUNS} 次实验均已完成。程序退出。")
        sys.exit(0)
    
    # 确定下一轮运行的起始索引
    start_run = 0
    
    # 优先完成 BASE 实验
    if base_completed < BASE_RUNS:
        start_run = base_completed
        runs_to_go = BASE_RUNS - start_run
        
        # 运行 BASE 方法
        print(f"继续运行 BASE 方法 (Rounds 1-{BASE_RUNS})")
        print(f"将从第 {start_run + 1} 次实验开始运行 (剩余 {runs_to_go} 次)。")
        
        for i in range(start_run, BASE_RUNS):
            main(i, BASE_RESULTS_FOLDER) # 传入 BASE 文件夹
            
        # 更新 BASE 完成状态
        base_completed, _, _ = get_completed_runs(BASE_RESULTS_FOLDER, OURS_RESULTS_FOLDER)
        print(f"BASE 方法已完成 {base_completed} 次。")

    
    # 接着运行 OURS 实验 (如果 BASE 已完成)
    if base_completed == BASE_RUNS and ours_completed < BASE_RUNS:
        start_run = BASE_RUNS + ours_completed # OURS 的 rounds 从 1200 开始
        runs_to_go = BASE_RUNS - ours_completed
        
        # 运行 OURS 方法
        print(f"\n开始/继续运行 OURS 方法 (Rounds {BASE_RUNS + 1}-{TOTAL_RUNS})")
        print(f"将从第 {start_run + 1} 次实验开始运行 (剩余 {runs_to_go} 次)。")
        
        for i in range(start_run, TOTAL_RUNS):
            main(i, OURS_RESULTS_FOLDER) # 传入 OURS 文件夹

    
    final_base_completed, final_ours_completed, _ = get_completed_runs(BASE_RESULTS_FOLDER, OURS_RESULTS_FOLDER)
    
    if final_base_completed + final_ours_completed == TOTAL_RUNS:
        print(f"\n=====================================")
        print(f"所有 {TOTAL_RUNS} 次实验均已完成。")
        print(f"=====================================")

    sys.exit(0)