import pygame
import random
import sys
import time
import numpy as np

from parameters import *
from utils import *
from communicate import Communication
from env import World




# -----------------------------
# 主循环与UI
# -----------------------------
def main():
    global SEED
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
    if VISUALIZE:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("exploration simulation")
        clock = pygame.time.Clock()
        font = load_font()

    # communication system
    comms = Communication(packet_loss=COMM_PACKET_LOSS, delay=COMM_DELAY)

    world = World(seed=SEED)

    running = True
    paused = False
    sim_time = 0.0

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
        if not paused:
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

        # end conditions
        if world.victim.rescued and coverage >= 50.0:
            print("Mission success: victim rescued and coverage >= 50%")
            paused = True
            running = False
        if sum(1 for a in world.agents if a.alive) == 0:
            print("All small agents destroyed. Mission failed.")
            paused = True
            running = False
        if sim_time > 300.0:
            print("Max sim time reached.")
            paused = True
            running = False

    # summary
    print("\n=== Simulation summary ===")
    print(f"Sim time: {sim_time:.2f}s")
    alive_small = sum(1 for a in world.agents if a.alive)
    alive_large = sum(1 for la in world.large_agents if la.alive)
    print(f"Alive: small={alive_small}/{len(world.agents)}, large={alive_large}/{len(world.large_agents)}")
    print(f"Coverage: {world.coverage_percentage():.2f}%")
    print(f"Victim rescued: {world.victim.rescued}")
    # Agent trajectories
    for a in world.agents:
        print(f"Agent {a.id} alive={a.alive} traj_len={len(a.hist)} last={a.pos}")
    for la in world.large_agents:
        print(f"LargeAgent {la.id} alive={la.alive} known_cells={np.sum(la.known_map != UNKNOWN)}")

    if VISUALIZE:
        pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
