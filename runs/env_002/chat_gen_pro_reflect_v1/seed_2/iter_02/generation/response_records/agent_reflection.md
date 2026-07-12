# Response Record

# 设计理由
当前所有评估回合都已成功走完地形（terminated=20/20），agent 具备稳定步态，但平均前进速度偏低（horizontal_velocity² 总和仅约257，对应平均速度 ~0.5）。主要学习信号 `forward_progress_reward` 的系数偏低，导致 agent 收敛到保守的低速策略，未能充分接近目标得分 300。本轮采用 Level 1 尺度修复，仅将该系数从 1.0 提升至 1.3，在不改变数学形态的前提下增强对高速的激励，预计总回报可提升 20–30 分，逼近目标且保持原有平衡约束不变。系数 1.3 使 per‑step 预期值由约 0.25 升至约 0.325（gated 后约 0.30），仍处于合理范围，不会引发剧烈崩溃。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：向前速度的凸化奖励
    horizontal_vel = obs[2]
    forward_progress_reward = 1.3 * (horizontal_vel ** 2)

    # 平衡约束（角度与角速度的软惩罚）
    hull_angle = obs[0]
    hull_angular_vel = obs[1]

    angle_threshold = 0.3
    angle_penalty = -0.5 * max(0.0, abs(hull_angle) - angle_threshold) ** 2

    angular_vel_threshold = 2.0
    angular_vel_penalty = -0.1 * max(0.0, abs(hull_angular_vel) - angular_vel_threshold) ** 2

    balance_penalty = angle_penalty + angular_vel_penalty

    # 健康门控：身体角度偏离越大，主奖励衰减越强
    gate_factor = max(0.0, min(1.0, 1.0 - abs(hull_angle) / 0.5))
    gated_forward_reward = forward_progress_reward * gate_factor

    total_reward = gated_forward_reward + balance_penalty

    components = {
        'forward_progress_reward': forward_progress_reward,
        'gated_forward_reward': gated_forward_reward,
        'balance_angle_penalty': angle_penalty,
        'balance_angular_vel_penalty': angular_vel_penalty,
        'gate_factor': gate_factor
    }

    return float(total_reward), components
```

# 诊断摘要
- **evidence**: 全部 20 个评估回合成功终止（走完地形），平均 episode 长度 1024 步，得分 286.89，最高 301.97，`forward_progress_reward` 总和仅 257，说明速度偏慢。
- **behavior**: agent 已掌握稳定行走并完成地形，但速度保守（平均 horizontal_velocity ≈ 0.5），导致整体得分略低于目标。
- **signal**: 向前速度的奖励系数偏弱，未能充分驱动高速前进，现有平衡约束几乎不触发，未形成阻碍。
- **level**: Level 1（尺度修复）—— 调整 `forward_progress_reward` 系数。
- **hypothesis**: 提升 `forward_progress_reward` 系数会增大速度激励，使 agent 在保持平衡的前提下提高步速，从而拉高平均得分并缩小与目标的差距。
- **risk**: 若速度提升过大，可能在崎岖地形段引发更高倾斜角，使门控因子显著下降，偶发摔倒风险轻度上升，但当前平衡能力强，概率较低。
