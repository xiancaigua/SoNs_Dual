import pygame
import math
from parameters import *
# -----------------------------
# 工具函数
# -----------------------------
def load_font():
    # 尝试加载常见中文字体，否则回退系统字体
    try:
        return pygame.font.Font("SimHei.ttf", 16)
    except Exception:
        try:
            return pygame.font.SysFont("SimHei", 16)
        except Exception:
            return pygame.font.SysFont("Arial", 16)

def clamp(x, a, b):
    return max(a, min(b, x))

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def vec_len(v):
    return math.hypot(v[0], v[1])

def normalize(v):
    l = vec_len(v)
    if l == 0:
        return (0,0)
    return (v[0]/l, v[1]/l)

def add(v, w):
    return (v[0]+w[0], v[1]+w[1])

def mul(v, s):
    return (v[0]*s, v[1]*s)

def cell_of_pos(pos):
    return int(pos[0] // GRID_CELL), int(pos[1] // GRID_CELL)

def pos_of_cell(i, j):
    return (i*GRID_CELL + GRID_CELL/2.0, j*GRID_CELL + GRID_CELL/2.0)