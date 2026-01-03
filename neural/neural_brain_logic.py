import torch
import numpy as np
import math
import sys
import os

# 路径防呆处理
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural.simple_train import SimpleBrainFitter 

class NeuralBrainLogic:
    def __init__(self, model_path="brain_fitter_v3.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleBrainFitter(state_dim=12)
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"❌ 模型加载报错: {e}")
        else:
            print(f"⚠️ 警告: 找不到模型文件 {model_path}")

        self.model.to(self.device)
        self.model.eval()

    def get_ai_decision(self, state_vec, downsampled_map):
        """ 获取 AI 的原始预测 """
        s = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        m = torch.tensor(downsampled_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(s, m).cpu().numpy()[0]
        
        # 反归一化
        output[:, 0] *= 1000.0
        output[:, 1] *= 700.0
        return output

    def validate_and_fix_goal(self, goal, grid, search_radius=50):
        """
        🛡️ 增强版安全过滤器：
        同时修正【撞墙】和【自杀】行为。
        """
        x, y = int(goal[0]), int(goal[1])
        h, w = grid.shape 
        
        # 1. 越界强制拉回
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # 2. 获取当前点的地图值
        # 0=Free, 1=Obstacle, 2=Danger, -1=Unknown
        val = grid[y][x]
        
        # 【关键修改】：如果当前点既不是障碍(1)，也不是危险区(2)，才算通过
        if val != 1 and val != 2:
            return (float(x), float(y))
            
        # 3. 如果是坏点，螺旋搜索附近的【安全】空地
        # print(f"⚠️ AI 指令 ({x},{y}) 落入障碍或危险区(Val={val})，正在重定向...")
        
        for r in range(1, search_radius, 2):
            candidates = [
                (x + r, y), (x - r, y), (x, y + r), (x, y - r),
                (x + r, y + r), (x - r, y - r), (x + r, y - r), (x - r, y + r)
            ]
            for cx, cy in candidates:
                if 0 <= cx < w and 0 <= cy < h:
                    c_val = grid[cy][cx]
                    # 只有找到绝对安全的地方(非1且非2)才返回
                    if c_val != 1 and c_val != 2: 
                        return (float(cx), float(cy))
        
        # 4. 实在找不到，返回 None (让机器人启用自带的 explore 逻辑)
        return None