# Response Record

# 设计理由
在上一轮中，agent 虽能存活较长但 70% 的回合仍因高度越界失败，说明姿态稳定性不足是核心问题。历史最佳（iter2）使用了独立的直立偏差惩罚，而当前仅将直立度作为 forward 的门控乘子，未能充分约束姿态。因此，我**新增一个独立的直立惩罚组件** `upright_pen = -2.0 * (quat_x² + quat_y²)`，它等价于 `-(1 - body_up_z)`，在身体倾斜时给予连续梯度惩罚。系数 2.0 经校准：正常行走时 quat_x² + quat_y² 约 0.02，每步惩罚约 -0.04，仅为主信号 forward（典型每步约 0.5）的 8%，不会喧宾夺主；严重倾斜时 forward 因门控同时下降，形成协同约束。保持原有 forward 与 height 惩罚不变。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_x_vel = next_obs[13]

    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, body_up_z)

    w_forward = 1.0
    forward = w_forward * body_x_vel * upright_factor

    w_height = 5.0
    height_pen_low = max(0.0, 0.25 - body_z)
    height_pen_high = max(0.0, body_z - 0.95)
    height_pen = -w_height * (height_pen_low + height_pen_high)

    # 新增：直立偏差惩罚，越倾斜越负，完美直立时为0
    upright_pen = -2.0 * (quat_x**2 + quat_y**2)

    total_reward = forward + height_pen + upright_pen

    components = {
        "forward": forward,
        "height": height_pen,
        "upright_pen": upright_pen
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated=14/20, height_pen active_rate=0.4%，forward active_rate=66%，但外部 score=-338 远小于 target。
- **behavior**: 机器人在大部分时间高度和速度正常，但姿态不够直立且不稳定，最终在近 500 步时跌倒导致失败，且真实任务分极低。
- **signal**: 缺乏对姿态偏斜的独立惩罚，门控乘子不足以迫使策略保持直立，导致步态效率低且易失稳。
- **level**: Level 2（必要职责缺失，增加独立直立惩罚组件）
- **hypothesis**: 加入与直立偏差成正比的无基线惩罚后，策略为了消除该项负奖励会主动保持身体竖直，从而提升步态稳定性和前进效率，减少后期跌倒，提高真实任务得分。
- **risk**: 若系数过大可能抑制探索，使策略过于保守不愿移动；当前系数基于典型步态校准，但极端倾斜时惩罚仍可能造成较大负梯，需关注是否导致训练早期奖励塌缩。
