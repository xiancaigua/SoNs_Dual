import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np

# ==========================================
# 🧠 1. 适配后的多模态拟合模型 (SimpleBrainFitter)
# ==========================================
class SimpleBrainFitter(nn.Module):
    def __init__(self, state_dim=12, map_h=35, map_w=50, num_goals=4):
        super().__init__()
        # 地图分支 (CNN) - 处理降采样后的地图 (1, 35, 50)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 输出约为 17x25
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 输出约为 8x12
            nn.Flatten(),
            nn.Linear(32 * 8 * 12, 128), nn.ReLU()
        )
        
        # 状态分支 (MLP) - 处理 12 维增强状态向量
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        
        # 融合输出层
        self.head = nn.Sequential(
            nn.Linear(128 + 64, 256), nn.ReLU(),
            nn.Linear(256, num_goals * 2) # 输出 num_goals 个点的 (x, y)
        )

    def forward(self, state, m):
        m_feat = self.cnn(m)
        s_feat = self.mlp(state)
        combined = torch.cat([m_feat, s_feat], dim=1)
        # 将输出重塑为 (Batch, Goals, 2)
        return self.head(combined).view(-1, 4, 2)

# ==========================================
# 📊 2. 增强的数据加载器 (BrainJSONDataset)
# ==========================================
class BrainJSONDataset(Dataset):
    def __init__(self, json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到数据文件: {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        print(f"成功加载数据，样本总数: {len(self.data)}")

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 状态向量 (12维)
        state = torch.tensor(item["state_vector"], dtype=torch.float32)
        
        # 2. 地图数据 (1, 35, 50)
        m = torch.tensor(item["known_map_downsampled"], dtype=torch.float32).unsqueeze(0)
        
        # 3. 标签归一化 (核心修改)
        # 将世界坐标 (0-1000, 0-700) 缩放到 [0, 1] 之间
        goals = np.array(item["subgoals"])
        if len(goals) > 0:
            goals[:, 0] /= 1000.0  # 假设 SCREEN_W = 1000
            goals[:, 1] /= 700.0   # 假设 SCREEN_H = 700
        
        # 补齐或截断到固定长度 (4个点)
        if len(goals) < 4:
            goals = np.pad(goals, ((0, 4 - len(goals)), (0, 0)), mode='constant')
        else:
            goals = goals[:4]
            
        return state, m, torch.tensor(goals, dtype=torch.float32)

# ==========================================
# 🚀 3. 训练脚本
# ==========================================
def train(data_path="./brain_dataset_v3/brain_expert_latest.json", epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    dataset = BrainJSONDataset(data_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 实例化模型 (state_dim=12)
    model = SimpleBrainFitter(state_dim=12).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss() # 预测归一化坐标，MSE 会变得很小

    print("开始模型拟合训练...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for states, maps, targets in loader:
            states, maps, targets = states.to(device), maps.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(states, maps)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.6f}")

    # 保存模型
    torch.save(model.state_dict(), "brain_fitter_v3.pth")
    print("✅ 训练完成，模型已保存为 brain_fitter_v3.pth")

if __name__ == "__main__":
    # 确保文件夹和文件存在
    train()