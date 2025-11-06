"""
analyze_dataset.py
对 data_collector.py 生成的仿真数据集进行分析和可视化。
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = "./dataset"

def load_dataset(path=DATASET_DIR):
    records = []
    for fname in os.listdir(path):
        if fname.endswith(".json"):
            with open(os.path.join(path, fname), "r") as f:
                records.append(json.load(f))
    return records


def summarize(records):
    rewards = [r["reward"] for r in records]
    coverages = [r["state_vector"][0] for r in records]
    founds = [r["found_victim"] for r in records]

    print(f"总样本数: {len(records)}")
    print(f"平均奖励: {np.mean(rewards):.3f}")
    print(f"平均覆盖率: {np.mean(coverages):.3f}")
    print(f"发现victim比例: {np.mean(founds)*100:.1f}%")

    plt.figure(figsize=(6,4))
    plt.hist(rewards, bins=20, alpha=0.7)
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.scatter(coverages, rewards, c=founds, cmap="coolwarm", alpha=0.7)
    plt.title("Coverage vs Reward (color=FoundVictim)")
    plt.xlabel("Coverage")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    recs = load_dataset()
    summarize(recs)
