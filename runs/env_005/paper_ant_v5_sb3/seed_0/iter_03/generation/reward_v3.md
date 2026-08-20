# 设计理由
修改了 forward 组件的数学形态，从无条件的线性速度奖励转变为受直立姿态门控的前进奖励。利用 `upright_gate = max(0.0, 1.0 - 2.0 * upright_error)` 将姿态误差转换为 0~1 的乘性因子，直接作用于 `body_x_vel`。这样只有在身体接近直立时才能获得完整的前进信号，倾斜过大会削减奖励，从而引导机器人优先维持稳定姿态再提高速度。系数 2.0 使得典型姿态误差（约 0.12）造成约 24% 的奖励衰减，位于每步主信号量级的 0.1~0.3 倍建议区间，同时移除了原有的独立直立惩罚，避免双重约束。高度 hinge 惩罚保持不变，作为安全兜底。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_x_vel = next_obs[13]

    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_error = 1.0 - body_up_z
    upright_gate = max(0.0, 1.0 - 2.0 * upright_error)
    
    w_forward = 1.0
    forward = w_forward * body_x_vel * upright_gate

    w_height = 5.0
    height_pen_low = max(0.0, 0.25 - body_z)
    height_pen_high = max(0.0, body_z - 0.95)
    height_pen = -w_height * (height_pen_low + height_pen_high)

    total_reward = forward + height_pen

    components = {
        "forward": forward,
        "height": height_pen
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated 率 100%，平均 episode 长度 89.65 步，forward 每步均约 0.95 但所有 episode 均提前终止（摔倒或越界）。
- **behavior**: 机器人尝试以较快速度前进，但身体失稳导致在不到 100 步内频繁摔倒，无法积累有效分数。
- **signal**: 原有的独立直立惩罚过弱，未能有效抑制姿态倾斜，使得 agent 在前进时放弃姿态控制。
- **level**: Level 2（将独立惩罚改为乘性门控，修改 forward 组件结构）
- **hypothesis**: 把前进奖励与姿态门控绑定后，agent 必须维持直立才能获得推进信号，这会迫使它在保持平衡的前提下学习前进，从而延长存活并积累更高总分。
- **risk**: 如果门控过强，可能使 agent 过分保守，停留在原地保持直立而不尝试前进；但速度奖励仍可通过小步前移获得，应能逐渐找到平衡。