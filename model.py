"""
train_diffusion_transformer.py
==============================
用于训练一个 Diffusion Transformer 模型：
输入：state vector（全局抽象状态）
输出：subgoal sequence（若干子目标坐标）

两阶段训练：
1️⃣ 行为克隆 (BC) 预训练
2️⃣ Diffusion fine-tuning

输出模型：models/slow_model.pth
"""

import os
import json
import math
import random
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===============================================================
# 📁 目录配置
# ===============================================================
DATA_DIR = "./dataset"
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ===============================================================
# 🧱 数据集定义
# ===============================================================
class SubgoalDataset(Dataset):
    """
    从 ./dataset/*.json 加载采集数据。
    每条数据包含：
        state_vector: [6]
        subgoals: [(x,y), ...]
    """

    def __init__(self, data_dir=DATA_DIR, max_subgoals=5):
        self.records = []
        self.max_subgoals = max_subgoals

        for fname in os.listdir(data_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(data_dir, fname), "r") as f:
                data = json.load(f)
                state = np.array(data["state_vector"], dtype=np.float32)
                subgoals = np.array(data["subgoals"], dtype=np.float32)
                # padding / trimming to fixed length
                if len(subgoals) < max_subgoals:
                    pad = np.zeros((max_subgoals - len(subgoals), 2))
                    subgoals = np.concatenate([subgoals, pad], axis=0)
                else:
                    subgoals = subgoals[:max_subgoals]
                self.records.append((state, subgoals))

        print(f"✅ Loaded {len(self.records)} samples from {data_dir}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        state, subgoals = self.records[idx]
        return torch.tensor(state, dtype=torch.float32), torch.tensor(subgoals, dtype=torch.float32)


# ===============================================================
# 🧩 模型组件
# ===============================================================

class TransformerEncoder(nn.Module):
    """对状态向量进行条件编码"""

    def __init__(self, input_dim=6, embed_dim=128, num_layers=3, num_heads=4):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, 1, input_dim)
        x = self.fc_in(x)
        return self.encoder(x)  # (B, 1, embed_dim)


class DiffusionDecoder(nn.Module):
    """
    Diffusion-style 子目标序列生成器
    输入：扩散噪声 z_t + 条件 embedding
    输出：去噪子目标序列
    """

    def __init__(self, embed_dim=128, seq_len=5, hidden_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.fc_t = nn.Linear(1, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            batch_first=True,
        )
        self.fc_out = nn.Linear(embed_dim, 2)  # 每个子目标 (x, y)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, z_t, cond_emb, t):
        """
        z_t: (B, seq_len, 2)
        cond_emb: (B, 1, embed_dim)
        t: (B,) 当前扩散步
        """
        B = z_t.size(0)
        t_embed = self.fc_t(t.view(B, 1).unsqueeze(-1))  # (B,1,embed)
        cond_proj = self.proj(cond_emb)
        inp = cond_proj.repeat(1, self.seq_len, 1) + t_embed.repeat(1, self.seq_len, 1)
        out = self.transformer(inp, inp)
        return self.fc_out(out)


# ===============================================================
# 🧠 主模型：SlowModel
# ===============================================================
class SlowModel(nn.Module):
    def __init__(self, state_dim=6, seq_len=5, embed_dim=128):
        super().__init__()
        self.state_encoder = TransformerEncoder(state_dim, embed_dim)
        self.decoder = DiffusionDecoder(embed_dim, seq_len)
        self.seq_len = seq_len

    def forward(self, state, z_t, t):
        cond = self.state_encoder(state.unsqueeze(1))
        return self.decoder(z_t, cond, t)

    def sample(self, state_vec, n_samples=1, n_steps=30):
        """采样扩散生成子目标序列"""
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            state = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            cond = self.state_encoder(state.unsqueeze(1))
            x = torch.randn((n_samples, self.seq_len, 2), device=device)

            for t in reversed(range(1, n_steps + 1)):
                t_tensor = torch.full((n_samples,), t / n_steps, device=device)
                eps_pred = self.decoder(x, cond, t_tensor)
                x = x - 0.1 * eps_pred  # 简化去噪步
            return x.cpu().numpy()


# ===============================================================
# ⚙️ 训练流程
# ===============================================================

def train(model, dataloader, optimizer, device, num_epochs_bc=200, num_epochs_diff=300):
    mse = nn.MSELoss()

    # -------- Stage 1: Behavior Cloning --------
    print("🔧 Stage 1: Behavior Cloning (supervised)")
    for epoch in range(num_epochs_bc):
        model.train()
        total_loss = 0.0
        for state, subgoals in dataloader:
            state, subgoals = state.to(device), subgoals.to(device)
            z_t = torch.randn_like(subgoals)
            t = torch.zeros(state.size(0), device=device)
            pred = model(state, z_t, t)
            loss = mse(pred, subgoals)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[BC] Epoch {epoch+1}/{num_epochs_bc}, Loss={total_loss/len(dataloader):.4f}")

    # -------- Stage 2: Diffusion Fine-tuning --------
    print("🔁 Stage 2: Diffusion Fine-tuning")
    for epoch in range(num_epochs_diff):
        model.train()
        total_loss = 0.0
        for state, subgoals in dataloader:
            state, subgoals = state.to(device), subgoals.to(device)
            B = state.size(0)
            t = torch.rand(B, device=device)
            noise = torch.randn_like(subgoals)
            z_t = subgoals + noise * 0.1 * t.view(-1, 1, 1)
            pred = model(state, z_t, t)
            loss = mse(pred, subgoals)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Diff] Epoch {epoch+1}/{num_epochs_diff}, Loss={total_loss/len(dataloader):.4f}")


# ===============================================================
# 🚀 主程序入口
# ===============================================================
def main():
    dataset = SubgoalDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SlowModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train(model, dataloader, optimizer, device)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "slow_model.pth"))
    print("✅ 模型训练完成并保存到 models/slow_model.pth")

    # 测试采样
    test_state = np.array([0.5, 5, 2, 1.0, 50, 20], dtype=np.float32)
    gen_seq = model.sample(test_state, n_samples=1)
    print("🧩 Generated Subgoals:", gen_seq)


if __name__ == "__main__":
    main()
