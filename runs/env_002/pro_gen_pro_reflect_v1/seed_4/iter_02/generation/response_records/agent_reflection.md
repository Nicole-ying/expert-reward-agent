# Response Record

# 设计理由
本轮将 `angle_penalty`（独立惩罚，active_rate 仅 1.3%）替换为连续稳定性门控 `stability_gate`。原惩罚仅在倾角超过 0.5 rad 时生效，过于滞后且力度不足，无法在倾角逐步增大的过程中提供连续负反馈。改用指数衰减因子 `exp(-α·angle²)` 乘到前进奖励上，使姿态偏差即刻削弱奖励，激励主体始终趋向竖直。α 设为 3.0，在 0.5 rad 时门控约 0.47，惩罚温和但明确，可避免摔倒。移除旧的 `angle_penalty` 以遵循“每轮只改一个组件”原则。保留了 `angvel_penalty` 和 `vert_penalty`，其中 `vert_penalty` 仍然僵尸（active_rate 0%），后续若再无效将调整或移除。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------- 核心观测提取 --------------------
    hull_angle     = float(obs[0])   # 身体倾角 (rad)
    hull_ang_vel   = float(obs[1])   # 身体角速度 (rad/s)
    horiz_vel      = float(obs[2])   # 前向速度 (m/s)
    vert_vel       = float(obs[3])   # 竖直速度 (m/s)

    # -------------------- 主进展奖励 --------------------
    forward_reward = 1.0 * max(0.0, horiz_vel)

    # -------------------- 稳定性门控（替换独立角度惩罚） --------------------
    # 指数衰减：倾角越大，门控值越接近 0，按比例削弱前进奖励
    # α=3.0 使得：0.3 rad → gate≈0.76, 0.5 rad → gate≈0.47, 0.8 rad → gate≈0.15
    alpha = 3.0
    stability_gate = 2.718281828 ** (-alpha * hull_angle ** 2)
    gated_forward_reward = forward_reward * stability_gate

    # -------------------- 角速度二次惩罚（保持不变） --------------------
    angvel_penalty = -0.001 * (hull_ang_vel ** 2)

    # -------------------- 竖直速度抑制（保留，后续可能调整） --------------------
    vert_threshold = 0.5
    vert_penalty = -0.1 * max(0.0, abs(vert_vel) - vert_threshold)

    # -------------------- 总奖励 --------------------
    total_reward = gated_forward_reward + angvel_penalty + vert_penalty

    components = {
        "forward_reward": forward_reward,          # 原始前进奖励（仅观测）
        "stability_gate": stability_gate,          # 门控值（用于诊断）
        "gated_forward_reward": gated_forward_reward,  # 实际计入总奖励的前进部分
        "angvel_penalty": angvel_penalty,
        "vert_penalty": vert_penalty,
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前 reward 严重缺乏有效的摔倒前兆预防，独立角度惩罚触发率低且滞后，竖直速度惩罚完全无效，角速度惩罚过弱。
- **behavior**: agent 能行走但常见摔倒导致提前终止（len 仅 632，多数终端为 failure），前进速度中等但存活时长不足。
- **signal**: 缺少随倾角连续衰减的稳定性信号，独立惩罚在安全‑危险临界区梯度缺失。
- **level**: Level 2（结构变换——用稳定性门控替代独立惩罚）
- **hypothesis**: 将倾角惩罚改为连续门控乘在主奖励上，在每个时间步都给予姿态调整梯度，预期能降低摔倒率，提高平均步长，从而增加总前进距离和得分。
- **risk**: 若 α 过大（门控衰减过强），可能抑制必要的身体摆动，导致步态僵硬、前进速度下降；但 α=3.0 设计温和，风险较低。
