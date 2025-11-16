# -----------------------------
# 参数配置（可以调整）
# -----------------------------
VISUALIZE = True  # 设置为 False 时禁用可视化
BASELINE = False  # 使用基线智能体

SCREEN_W, SCREEN_H = 1000, 700
WORLD_W, WORLD_H = SCREEN_W, SCREEN_H
FPS = 1200000

NUM_AGENTS = 6
NUM_LARGE = 3
NUM_OBSTACLES = 10
NUM_DANGER_ZONES = 15

GRID_CELL = 10  # 地图格子大小（像素）
GRID_W = WORLD_W // GRID_CELL
GRID_H = WORLD_H // GRID_CELL

AGENT_RADIUS = 6
LARGE_RADIUS = 10

SENSOR_SMALL = 30
SENSOR_LARGE = 100

AGENT_MAX_SPEED = 60.0  # pixels per second
BRAIN_REASON_INTERVAL = 4.0  # seconds

AGENT_COMM_RANGE = 100
COMM_PACKET_LOSS = 0.05  # 报文丢失概率
COMM_DELAY = 0.0  # 可模拟延迟（秒）

DANGER_ZONE_RADIUS = 22

SEED = 110716

# Occupancy codes for maps:
# -1 unknown, 0 free, 1 obstacle, 2 danger, 3 victim
UNKNOWN = -1
FREE = 0
OBSTACLE = 1
DANGER = 2
VICTIM = 3