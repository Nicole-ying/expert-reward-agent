# Response Record

# 设计理由
当前 `height_gate` 在 `body_z ≤ 0.25` 时直接将 `forward_reward` 截断为 0，而终止边界是 0.2。agent 在高度偏低但尚未终结时就完全失去前进激励，剩余只有惩罚项，导致快速失败、episode 极短（len≈11）。本轮将 **硬切断 gate 替换为独立的 hinge 高度惩罚**（`height_penalty`），让前进信号始终存在，同时用连续代价把 body_z 维持在安全区内（≥0.35）。这符合“prefer hinge over quadratic for boundary constraints”原则，且惩罚系数量级已被校准为主信号 per‑step 的 ≤0.3 倍，不会过度压制前进。同时保留 `descend_penalty` 与 `posture_factor`，不对其他组件做改动。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    vx = next_obs[13]
    vy = next_obs[14]
    vz = next_obs[15]

    # 姿态健康因子（连续，保持直立）
    pose_error = quat_x**2 + quat_y**2 + quat_z**2
    posture_factor = 2.718281828 ** (-10.0 * pose_error)

    # 前进奖励（不再使用高度 gate，保证任何存活状态下都有梯度）
    base_forward = max(0.0, vx)
    forward_reward = 1.0 * base_forward * posture_factor

    # 侧向速度惩罚
    lateral_penalty = -0.5 * (vy ** 2)

    # 动作幅度惩罚
    action_penalty = -0.1 * sum(a ** 2 for a in action)

    # 高度惩罚：低于 0.35 m 开始线性增大，最大惩罚量级受控
    height_penalty = -0.5 * max(0.0, 0.35 - body_z)

    # 下降速度惩罚：阻止过快坠落
    descend_penalty = -0.2 * max(0.0, -0.5 - vz)

    total_reward = forward_reward + lateral_penalty + action_penalty + height_penalty + descend_penalty

    components = {
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "action_penalty": action_penalty,
        "height_penalty": height_penalty,
        "descend_penalty": descend_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 高度 gate 在终止阈值上方过早将前进奖励切为 0，造成低高度时无正向信号，agent 迅速失败。
- **evidence**: len=11.45，terminated=20/20；history 中移除高度 gate 时 len 曾达 17.45，加回后 len 暴跌（< 15）；descend_penalty 添加后 len 不升反降。
- **behavior**: agent 在几步内因高度过低进入安全边界以下，forward 信号消失，只剩惩罚，导致立即终止。
- **signal**: 缺少在低高度时仍给予前进激励的连续约束，gate 的 0‑区破坏了整个任务目标信号。
- **level**: Level 2 – 结构变换，将硬 gate 替换为 hinge 惩罚。
- **hypothesis**: 移除 gate 后，agent 在整个存活区间内都能为前进速度获得奖励，同时 height_penalty 提供恢复高度的梯度，应能延长 episode 长度并提升总得分。
- **risk**: 若高度惩罚过弱，agent 可能维持贴近 0.2 m 的低姿势前进，导致 forward_reward 增加但目标得分提升有限；必要时可上调 hinge 系数或斜率。
