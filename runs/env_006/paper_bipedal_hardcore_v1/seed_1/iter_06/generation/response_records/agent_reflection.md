# Response Record

# 设计理由
本轮在正常模式下修改一个组件：将 `health_gate` 从原始的**裸乘积**（`angle_gate * vertical_gate`）变换为**几何平均**（`(angle_gate * vertical_gate)**0.5`）。

**证据**：当前奖励骨架里，两个门控（倾角和垂直速度）通过乘积合成健康门，只要其中一个门趋近 0，奖励就会直接塌缩到 0，使 agent 在该步几乎收不到任何梯度——尤其在 `angle_gate` 较小时（例如倾角接近 0.6 rad），即使垂直速度状态良好，奖励也会被完全抑制。这种“塌缩为 0”的现象导致前进与平衡的耦合过于脆弱，梯度信号断裂，agent 难以在保持稳定和前进间找到平滑的平衡点。实际训练表现为 terminated 率极高（19/20）、总分持续为负且未见回升。  
**变换依据**：#7 中明确指明“乘积 proxy 经常塌缩为 0 → 几何平均”，这是对应的 Level 2 结构变换。几何平均保留了“任一门变差即降低总奖励”的约束性，但在一个门接近 0 时不会立刻归零，而是以开方速率下降，使得梯度仍然存在，有利于学习恢复。  
**系数校准**：主信号 `progress_raw` 的 per‑step 均值约 0.23‑0.27，几何平均后的健康门均值在 0.7‑0.9 之间，总 per‑step reward 会被保持在 ~0.15‑0.25，无额外惩罚负担，符合设计校准中对主信号比例的约束。  
**仅改动一个组件**，完全符合正常模式要求。

# 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取关键观测
    hull_angle = next_obs[0]          # body pitch angle
    horizontal_speed = next_obs[2]   # forward velocity
    vertical_speed = next_obs[3]     # vertical velocity

    # 主进展信号：水平速度（保持每步有梯度）
    progress_raw = horizontal_speed

    # 身体倾角门控：倾角越接近安全上限（0.6 rad），门控越接近0
    angle_threshold = 0.6
    angle_gate = max(0.0, 1.0 - abs(hull_angle) / angle_threshold)

    # 垂直弹跳门控：垂直速度越小越好
    vert_threshold = 2.0
    vertical_gate = max(0.0, 1.0 - abs(vertical_speed) / vert_threshold)

    # 综合健康门控 —— 使用几何平均替代裸乘积，避免塌缩为0
    # 加微小 epsilon 防止数值异常
    health_gate = ((angle_gate * vertical_gate) + 1e-8) ** 0.5

    # 总奖励：在身体状态良好时充分奖励前进，恶化时平缓衰减
    total_reward = progress_raw * health_gate

    components = {
        'progress_raw': progress_raw,
        'angle_gate': angle_gate,
        'vertical_gate': vertical_gate,
        'health_gate': health_gate,
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 乘积 health_gate 在单门控接近 0 时奖励塌缩为 0，导致梯度信号中断；观测使用不全但当前主干问题在门控形式而非缺失信号。
- **behavior**: agent 在倾角接近 0.6 rad 的临界区域频繁摔倒，episode 长度中位数约 300 步且 terminated 率高达 95%。
- **signal**: 健康门塌缩使前进奖励过于脆弱，缺乏柔性的稳定性约束。
- **level**: Level 2
- **hypothesis**: 几何平均会保留约束力同时避免奖励完全归零，使 agent 在接近临界状态时仍能获得前进激励，从而学到更平滑的恢复行为，减少失败。
- **risk**: 极端不稳定步态（如倾角接近 0.6 且垂直速度大）可能仍被少许奖励引导，短期内有可能维持临界状态但未彻底摔倒，延长探索时间；但长期应更稳健。
