# Response Record

# 设计理由
将 `height` 组件从独立的线性惩罚（仅在低于 0.25 或高于 0.95 时触发）改为 soft health gate，乘到前进奖励中。原惩罚 active_rate 仅 1%，无法在接近终止边界时提供足够的连续梯度，导致 70% episode 因高度越界提前终止。新 gate 在安全区（0.35–0.9）保持 1.0，在接近边界时线性衰减至 0，提前给予警告信号。系数 w_forward 保持 1.0，阈值根据终止边界的 60–80% 位置设置（下界 safe 0.35、上界 safe 0.9），确保在“仍安全但不理想”区间 gate 值不低于 0.3。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_x_vel = next_obs[13]

    # 直立因子
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, body_up_z)

    # 高度安全 gate
    low_safe = 0.35
    min_z = 0.2
    high_safe = 0.9
    max_z = 1.0

    if body_z >= low_safe:
        gate_low = 1.0
    else:
        gate_low = max(0.0, (body_z - min_z) / (low_safe - min_z))

    if body_z <= high_safe:
        gate_high = 1.0
    else:
        gate_high = max(0.0, (max_z - body_z) / (max_z - high_safe))

    height_gate = min(gate_low, gate_high)

    # 前进奖励（乘以上述两个 gate）
    w_forward = 1.0
    forward = w_forward * body_x_vel * upright_factor * height_gate

    # 直立偏差惩罚（保持上一轮的系数）
    upright_pen = -2.0 * (quat_x**2 + quat_y**2)

    total_reward = forward + upright_pen

    components = {
        "forward": forward,
        "upright_pen": upright_pen,
        "height_gate": height_gate
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated=14/20，height 组件 active_rate 仅 1.0%，高度越界是主要失败原因。
- **behavior**: agent 在存活时能向前移动（forward 活跃 78.7%），但 70% 的 episode 因身体高度超出健康范围而提前终止。
- **signal**: 原有的高度惩罚阈值太靠近边界（0.25/0.95），导致预警步数极少，无法提供有效的避险梯度。
- **level**: Level 2，将独立惩罚转换为 soft_health_gate，乘入主奖励信号。
- **hypothesis**: 通过在 0.35–0.9 外线性衰减前进奖励，agent 将学会远离危险高度，从而降低终止率并提升稳定行走时长。
- **risk**: 若 gate 衰减过强使安全区外几乎无奖励，可能令 agent 过于保守、放弃行走；实际衰减区间已校准为 0.15–0.10，gradient 仍足够。
