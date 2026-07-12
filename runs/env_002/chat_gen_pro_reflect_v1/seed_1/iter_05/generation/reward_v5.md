# 设计理由
当前 angle_penalty 的 active_rate 始终为 0%，说明 hull_angle 未曾超过 0.8 的阈值，其安全功能已被 gate 内部的 angle_factor 覆盖。移除该组件不会改变当前策略的行为，同时能简化奖励结构，消除一个从未参与信号、仅消耗计算的无用项。这属于 Level 2 的“耦合→少量直接组件”清理，不引入新风险。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 信号提取
    horizontal_velocity = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    hip1_torque = action[0]
    knee1_torque = action[1]
    hip2_torque = action[2]
    knee2_torque = action[3]
    # 逐个索引 LIDAR 值以通过验证
    lidar = [obs[14], obs[15], obs[16], obs[17], obs[18], obs[19], obs[20], obs[21], obs[22], obs[23]]

    # 组件1: 前进奖励（线性，仅正向）
    forward_reward = 1.0 * max(horizontal_velocity, 0.0)

    # 组件2: 平衡+地形门控
    angle_factor = 1.0 / (1.0 + 2.0 * abs(hull_angle))
    angular_velocity_factor = 1.0 / (1.0 + 0.5 * abs(hull_angular_velocity))

    # 地形粗糙度因子: 基于LIDAR标准差
    lidar_mean = sum(lidar) / 10.0
    lidar_var = sum((x - lidar_mean) ** 2 for x in lidar) / 10.0
    lidar_std = lidar_var ** 0.5
    terrain_factor = 1.0 / (1.0 + lidar_std)  # std=0 时=1.0, std 增大时缓慢衰减

    # 综合门控: 取最保守值
    balance_gate = min(angle_factor, angular_velocity_factor)
    gate = min(balance_gate, terrain_factor)
    gated_forward_reward = forward_reward * gate

    # 组件3: 能耗惩罚
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * (
        hip1_torque**2 + knee1_torque**2 + hip2_torque**2 + knee2_torque**2
    )

    # 移除未触发的 angle_penalty，其安全作用已被 gate 覆盖
    total_reward = gated_forward_reward + energy_penalty

    components = {
        "gated_forward_reward": gated_forward_reward,
        "energy_penalty": energy_penalty,
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: 20/20 terminated, len≈799, score_range tight (307.5‑310.0), angle_penalty active_rate=0%.
- **behavior**: agent 已学会在崎岖地形上稳定、快速地行走并到达终点，身体倾角始终低于阈值。
- **signal**: gated_forward_reward 与 energy_penalty 已提供完备的学习信号，angle_penalty 从未激活，属于冗余组件。
- **level**: Level 2（移除非必要组件），遵循“每轮只修改一个目标组件”且不涉及系数缩放。
- **hypothesis**: 移除该组件不会改变已收敛策略的行为，但能简化奖励结构，避免未来训练中因无用惩罚产生的潜在干扰。
- **risk**: 极低——当前所有 episode 均未触发该惩罚，且 gate 中的 angle_factor 已对倾角进行抑制；若未来探索中出现更大倾角，缺少额外惩罚可能略微增加摔倒风险，但 gate 的强衰减足以提供约束。