# Response Record

# 设计理由
我将前进奖励从线性 `2.0 * horizontal_velocity` 改为凸形态 `3.0 * horizontal_velocity ** 2`，以让策略更偏好高速度。当前 agent 很可能已走完全程但速度偏慢（所有 episode 都 terminated 成功，无快速摔倒，angle_penalty 永不触发），用平方项可以在速度 >1.0 时给出更强梯度，同时用现有的 balance gate 作为安全约束。系数 3.0 使得在典型速度 ~0.9‑1.0 附近 per‑step 奖励与原值接近，速度上升时奖励增长显著但不极端。

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

    # ========== 组件 1: 前进奖励 (凸化) ==========
    # 将线性奖励改为平方形式，鼓励更高速度
    forward_reward = 3.0 * (horizontal_velocity ** 2)

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

    # ========== 组件 4: 额外平衡惩罚 (hinge，保留但不常触发) ==========
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
- **evidence**: score=264.1 (gap 35.9), terminated=20/20, len=746, angle_penalty active_rate 0% → 所有回合都成功走完全程且姿态极稳，速度可能是拉低最终得分的瓶颈。
- **behavior**: agent 学会了稳定行走并抵达终点，无摔倒，但可能速度偏保守导致得分未达 300。
- **signal**: 线性前进奖励对速度的激励强度不足，需要凸化以在高速区提供更强梯度。
- **level**: Level 2（数学结构变换：线性→凸），因为当前职责已完备，只需改变主奖励形态来推动效率。
- **hypothesis**: 平方奖励会让策略更愿意维持高于 1.0 m/s 的行走速度，从而在保持稳定时压缩行程用时并提升外部 score。
- **risk**: 过高速度可能诱发不稳定步伐，但现有 balance gate 会对身体抖动进行压制，若摔倒率上升需考虑收紧门控阈值。
