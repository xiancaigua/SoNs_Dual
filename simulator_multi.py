import pygame
import random
import sys
import time
import numpy as np
import os
import multiprocessing
from datetime import datetime

# ==================================
# 💡 全局配置和导入 (假设它们来自 parameters.py, utils.py, etc.)
# ==================================
# 假设这些是您已有的导入：
from parameters import * 
from utils import * 
from communicate import Communication
from env import World

# 实验总次数配置
TOTAL_RUNS = 2400 
BASE_RUNS = 1200
BASE_RESULTS_FOLDER = "simulation_results_BASE"
OURS_RESULTS_FOLDER = "simulation_results_OURS"
# 设置并行进程数（根据您的CPU核心数进行调整，通常设置为核心数或核心数-1）
NUM_PROCESSES = 6 

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

# ==================================
# 💡 多进程 Worker 函数
# ==================================

def run_experiment_worker(run_index):
    """
    供多进程池调用的单个实验运行函数。
    它会根据索引确定使用的方法、结果文件夹，并强制关闭可视化。
    """
    # 确定方法和结果文件夹
    if run_index < BASE_RUNS:
        folder = BASE_RESULTS_FOLDER
    else:
        folder = OURS_RESULTS_FOLDER
        
    try:
        # 调用 main 函数，并强制设置 visualize_override=False
        main(rounds=run_index, results_folder=folder, visualize_override=False)
        return run_index, "SUCCESS"
    except Exception as e:
        print(f"[ERROR] 实验 {run_index} 运行失败: {e}", file=sys.stderr)
        return run_index, f"ERROR: {e}"


# -----------------------------
# 主循环与UI
# -----------------------------
def main(rounds, results_folder, visualize_override=None):
    """
    运行一次仿真实验。
    rounds: 当前实验的序号 (从 0 到 2399)
    results_folder: 当前实验结果应保存的文件夹路径。
    visualize_override: 强制设置可视化模式 (True/False)，用于多进程控制。
    """
    
    # 确定当前可视化状态
    current_visualize = VISUALIZE if visualize_override is None else visualize_override
    
    if SEED is not None:
        # 为每个回合设置一个不同的种子，保证实验的可重复性
        current_seed = SEED + rounds 
        random.seed(current_seed)
        np.random.seed(current_seed)
    
    # --- 核心修改 1: 确定当前运行的方法 ---
    IS_BASELINE_RUN = rounds < BASE_RUNS
    method_name = "BASE" if IS_BASELINE_RUN else "OURS"
    
    clock = None
    screen = None
    font = None
    
    if current_visualize:
        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption(f"exploration simulation - {method_name}")
        font = load_font()

    # communication system
    comms = Communication(packet_loss=COMM_PACKET_LOSS, delay=COMM_DELAY)
    
    # 12 个地图的种子
    world_seeds = [110716, 710, 8848, 1107, 233174, 12142325, 258, 8456, 1985, 819, 789654, 666]
    
    # --- 核心修改 2: 确定当前地图索引 ---
    map_cycle_index = rounds % BASE_RUNS # 确定在 1200 次循环中的位置 (0-1199)
    map_index = map_cycle_index // 100 # 确定地图索引 (0-11)
    
    if map_index >= len(world_seeds):
        print(f"ERROR: rounds {rounds} exceeds total maps.")
        return
        
    world = World(seed=world_seeds[map_index])
    print(f"--- Running Experiment {rounds + 1}/{TOTAL_RUNS} | Method: {method_name} | Map Index: {map_index}, Seed: {world_seeds[map_index]} ---")


    running = True
    paused = False
    
    # 💡 引入 real-time 计时
    last_time = time.time()
    
    sim_time = 0.0
    simulation_result = "unknown"  

    screenshot_dir = "simulation_screenshots"
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)

    while running:
        
        # 💡 使用现实世界时间计算 dt
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        now_time = current_time 

        # 事件处理和帧率控制（仅在可视化模式下启用）
        if current_visualize and clock:
            # 使用 clock.tick 限制绘图循环的帧率，防止 CPU 占用过高。
            # 仿真时间步长 dt 仍然使用 real-time 计算。
            clock.tick(FPS) 
            
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
            
            # 暂停时，不增加仿真时间
            if paused:
                dt = 0.0
        
        # 仿真时间更新：如果是非可视化模式或当前未暂停，则增加 sim_time
        sim_time += dt

        # --- 执行对应的方法更新 ---
        # 即使在非可视化模式下，也始终更新（因为 paused 默认为 False）
        if not paused or not current_visualize: 
            if IS_BASELINE_RUN:
                # 保持与上一文件一致的函数名
                world.update_base2(dt, comms, now_time) 
            else:
                world.update(dt, comms, now_time)

        # 绘制（仅在可视化模式下启用）
        if current_visualize:
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

        # 结束条件判断 (逻辑不变)
        if world.victim.rescued:
            simulation_result = "success"
            print(f"Experiment {rounds+1} | Mission success: victim rescued at {sim_time:.1f}s")
            # 仅在可视化模式下保存截图
            if current_visualize:
                final_image = create_summary_image(screen, world, sim_time, "SUCCESS", font)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pygame.image.save(final_image, f"simulation_screenshots/success_{timestamp}_{map_index}_{method_name}.png")
            running = False
        elif len(world.large_agents) == 0 or world.spawn_times >= MAX_TOLERATE:
            simulation_result = "failure"
            print(f"Experiment {rounds+1} | All agents destroyed. Mission failed at {sim_time:.1f}s.")
            if current_visualize:
                final_image = create_summary_image(screen, world, sim_time, "FAILURE", font)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pygame.image.save(final_image, f"simulation_screenshots/failure_{timestamp}_{map_index}_{method_name}.png")
            running = False  
        elif sim_time > 60.0:
            simulation_result = "failure"
            print(f"Experiment {rounds+1} | Max sim time reached: {sim_time:.1f}s.")
            if current_visualize:
                final_image = create_summary_image(screen, world, sim_time, "TIMEOUT", font)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pygame.image.save(final_image, f"simulation_screenshots/timeout_{timestamp}_{map_index}_{method_name}.png")
            running = False

    # summary
    print_simulation_summary(world, sim_time, simulation_result)
    
    #--------实验总结---------
    # 保存最终状态的简单截图路径 (仅在可视化模式下才会有截图)
    screenshot_path = None
    if current_visualize and simulation_result != "unknown":
        screenshot_path = save_simulation_screenshot(screen, world, sim_time, f"final_{simulation_result}_{method_name}") 
        
    # 保存JSON总结到指定的独立文件夹
    json_filename = save_simulation_summary(world, sim_time, simulation_result, screenshot_path, 
                                            map_id=(map_index + 1), results_dir=results_folder) 

    if json_filename:
        print(f"详细仿真结果已保存为JSON文件: {json_filename}")
    if current_visualize:
        pygame.quit()


if __name__ == "__main__":
    
    # 确保 multiprocessing 启动时，不会再次执行这个 if 块
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # PyInstaller/cx_Freeze environment, skip multiprocessing main guard
        pass
    elif multiprocessing.current_process().name != 'MainProcess':
        # Skip if not the main process (necessary for some systems, but should be fine)
        pass
    else:
        # --- 主进程执行逻辑 ---
        
        # 1. 统计已完成的实验次数
        base_completed, ours_completed, completed_runs = get_completed_runs(BASE_RESULTS_FOLDER, OURS_RESULTS_FOLDER)
        
        print(f"=====================================")
        print(f"        --- 实验恢复模式 ---         ")
        print(f"=====================================")
        print(f"并行进程数: {NUM_PROCESSES}")
        print(f"BASE 结果文件夹: {BASE_RESULTS_FOLDER} (已完成 {base_completed} 次)")
        print(f"OURS 结果文件夹: {OURS_RESULTS_FOLDER} (已完成 {ours_completed} 次)")
        print(f"总计已完成 {completed_runs} / {TOTAL_RUNS} 次实验。")
        
        if completed_runs >= TOTAL_RUNS:
            print(f"所有 {TOTAL_RUNS} 次实验均已完成。程序退出。")
            sys.exit(0)
        
        # 2. 确定需要运行的回合列表
        runs_to_schedule = []
        
        # 优先调度 BASE 实验 (0 - 1199)
        if base_completed < BASE_RUNS:
            runs_to_schedule.extend(range(base_completed, BASE_RUNS))
        
        # 接着调度 OURS 实验 (1200 - 2399)
        if ours_completed < BASE_RUNS:
            start_ours_run = BASE_RUNS + ours_completed
            runs_to_schedule.extend(range(start_ours_run, TOTAL_RUNS))
            
        print(f"共发现 {len(runs_to_schedule)} 次实验需要运行。开始并行调度...")
        print(f"=====================================")
        
        # 3. 使用 multiprocessing Pool 运行实验
        if runs_to_schedule:
            # 使用上下文管理器创建进程池
            with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
                # 使用 imap_unordered 以非阻塞方式获取结果，并打印进度
                for i, (run_index, status) in enumerate(pool.imap_unordered(run_experiment_worker, runs_to_schedule)):
                    current_count = completed_runs + i + 1
                    method = "BASE" if run_index < BASE_RUNS else "OURS"
                    print(f"[PROGRESS] Completed: {current_count}/{TOTAL_RUNS} | Round: {run_index+1} ({method}) | Status: {status}")
                
            # 运行结束后，再次检查最终状态
            final_base_completed, final_ours_completed, final_completed_runs = get_completed_runs(BASE_RESULTS_FOLDER, OURS_RESULTS_FOLDER)
            
            if final_completed_runs == TOTAL_RUNS:
                print(f"\n=====================================")
                print(f"所有 {TOTAL_RUNS} 次实验均已完成。")
                print(f"=====================================")
            else:
                 print(f"\n=====================================")
                 print(f"并行运行结束。总完成数：{final_completed_runs}/{TOTAL_RUNS}。可能存在错误或程序提前退出。")
                 print(f"=====================================")

        sys.exit(0)