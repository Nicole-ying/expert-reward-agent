# Response Record

# 设计理由
将前进奖励从平方形式（`3.0 * v²`）改为线性（`1.0 * max(v,0)`），以降低对高速的过度激励，减少高速带来的不稳定和摔倒。保留 balance gate 和 energy penalty 不变，因为角度惩罚未触发说明 gate 已经防止了大幅倾斜，问题更可能出在激进速度策略。系数 1.0 使 per‑step 奖励约 0.5‑0.8（在 gate 衰减后），低于之前约 0.9 的平均值，以抑制激进加速。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 信号提取 ==========
    horizontal_velocity = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    hip1_torque = action[0]
    knee1_torque = action[1]
    hip2_torque = action[2]
    knee2_torque = action[3]

    # ========== 组件 1: 前进奖励 (线性，仅正向) ==========
    # 使用线性速度，避免平方对高速的过度奖励
    forward_reward = 1.0 * max(horizontal_velocity, 0.0)

    # ========== 组件 2: 平衡约束 (soft_health_gate) ==========
    angle_factor = 1.0 / (1.0 + 2.0 * abs(hull_angle))
    angular_velocity_factor = 1.0 / (1.0 + 0.5 * abs(hull_angular_velocity))
    balance_gate = min(angle_factor, angular_velocity_factor)
    gated_forward_reward = forward_reward * balance_gate

    # ========== 组件 3: 能耗惩罚 (quadratic_penalty) ==========
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * (
        hip1_torque**2 + knee1_torque**2 + hip2_torque**2 + knee2_torque**2
    )

    # ========== 组件 4: 额外平衡惩罚 (hinge) ==========
    angle_safety_threshold = 0.8
    angle_penalty_weight = 1.0
    angle_penalty = -angle_penalty_weight * max(0.0, abs(hull_angle) - angle_safety_threshold)

    # ========== 总奖励 ==========
    total_reward = gated_forward_reward + energy_penalty + angle_penalty

    components = {
        "gated_forward_reward": gated_forward_reward,
        "energy_penalty": energy_penalty,
        "angle_penalty": angle_penalty,
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: 20/20 terminated, len=438.8, score=61.77 (best=264.12), angle_penalty 未触发, gated_forward_reward 平均 0.895/step, 外部得分反向下跌。
- **behavior**: 多数 episode 可能摔倒（得分低），少数高分，整体平均差，内部奖励高但未对应外部高分。
- **signal**: 平方速度奖励过度激励高速，导致策略在快速移动中失去稳定性，增加摔倒率。
- **level**: Level 2（数学结构变换：凸化→线性），由得分骤降且前轮已尝试相同骨架推断需改变激励形态。
- **hypothesis**: 线性速度奖励减少对极端高速的奖励，使策略倾向平稳行进，降低摔倒概率，从而提升外部平均得分。
- **risk**: 速度激励降低可能导致平均步速下降，短期外部得分可能未见明显提升，需进一步观察长度与得分关系。
