import pygame
import random
import sys
import time
import numpy as np
import os
from datetime import datetime

from parameters import *
from utils import *
from communicate import Communication
from env import World




# -----------------------------
# 主循环与UI
# -----------------------------
def main(rounds=1):
    global SEED
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
    pygame.init()
    clock = pygame.time.Clock()
    if VISUALIZE:
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("exploration simulation")
        font = load_font()

    # communication system
    comms = Communication(packet_loss=COMM_PACKET_LOSS, delay=COMM_DELAY)
    map_files = get_map_files("map")
    # save_filename = f"world_seed_233174_init_state.pkl"
    # save_filename = map_files[rounds//100][1]
    # world = World.load_state(save_filename)
    world = World(seed=110716)
    # world.save_state(f"world_seed_{SEED}_init_state.pkl")
    if BASELINE:
        world.set_state()

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
        if not paused:
            if BASELINE:
                world.update_baseline(dt, comms, now_time)
            else:
                world.update(dt, comms, now_time)

        # 绘制（仅在可视化模式下启用）
        if VISUALIZE:
            world.draw(screen)

            # HUD
            alive = sum(1 for a in world.agents if a.alive) + sum(1 for la in world.large_agents if la.alive)
            coverage = world.coverage_percentage()
            hud1 = f"Time: {sim_time:.1f}s  Alive agents: {alive}/{len(world.agents)+len(world.large_agents)}  Coverage: {coverage:.2f}%"
            hud2 = f"Obstacles: {len(world.obstacles)}  Dangers: {len(world.danger_zones)}  Victim: {'rescued' if world.victim.rescued else 'missing'}"
            screen.blit(font.render(hud1, True, (10, 10, 10)), (8, 6))
            screen.blit(font.render(hud2, True, (10, 10, 10)), (8, 24))

            # legend
            pygame.draw.circle(screen, (40, 120, 220), (SCREEN_W - 120, 30), AGENT_RADIUS)
            screen.blit(font.render("Small Agent", True, (0, 0, 0)), (SCREEN_W - 96, 22))
            pygame.draw.circle(screen, (200, 160, 60), (SCREEN_W - 120, 60), LARGE_RADIUS)
            screen.blit(font.render("Large Agent", True, (0, 0, 0)), (SCREEN_W - 96, 52))
            pygame.draw.circle(screen, (200, 40, 40), (SCREEN_W - 120, 90), 8)
            screen.blit(font.render("Danger", True, (0, 0, 0)), (SCREEN_W - 96, 82))

            pygame.display.flip()

        # 结束条件判断
        # print("len(world.large_agents)",len(world.large_agents))
        if world.victim.rescued:
            simulation_result = "success"
            print("Mission success: victim rescued")
            # 保存成功截图
            if VISUALIZE:
                final_image = create_summary_image(screen, world, sim_time, "SUCCESS", font)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pygame.image.save(final_image, f"simulation_screenshots/success_{timestamp}.png")
            paused = True
            running = False
        elif len(world.large_agents) == 0 or world.spawn_times >= 100:
            simulation_result = "failure"
            print("All agents destroyed. Mission failed.")
            # 保存失败截图
            if VISUALIZE:
                final_image = create_summary_image(screen, world, sim_time, "FAILURE", font)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pygame.image.save(final_image, f"simulation_screenshots/failure_{timestamp}.png")
            paused = True
            running = False
            
        elif sim_time > 200.0:
            simulation_result = "timeout"
            print("Max sim time reached.")
            # 保存超时截图
            if VISUALIZE:
                final_image = create_summary_image(screen, world, sim_time, "TIMEOUT", font)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pygame.image.save(final_image, f"simulation_screenshots/timeout_{timestamp}.png")
            paused = True
            running = False

    # summary
    print_simulation_summary(world, sim_time, simulation_result)
    
    #--------实验总结---------
    # 保存最终状态的简单截图
    if VISUALIZE and simulation_result != "unknown":
        screenshot_path = save_simulation_screenshot(screen, world, sim_time, f"final_{simulation_result}")    # Agent trajectories
    else:
        screenshot_path = None
    # 保存JSON总结
    json_filename = save_simulation_summary(world, sim_time, simulation_result, screenshot_path, (1 + rounds//10))

    # 可选：在控制台显示JSON文件路径
    if json_filename:
        print(f"详细仿真结果已保存为JSON文件: {json_filename}")
    if VISUALIZE:
        pygame.quit()


if __name__ == "__main__":
    for i in range(500): 
        main(i)
    sys.exit(0)
