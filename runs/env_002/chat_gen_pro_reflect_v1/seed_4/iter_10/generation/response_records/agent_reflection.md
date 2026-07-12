# Response Record

# 设计理由
当前 `gate_factor` 在 0.0～0.25 rad 区间放大（最高 1.6），超过 0.25 rad 即开始衰减。双足行走时身体自然会有轻微前倾，通常落在 0.2～0.3 rad 之间。过于窄的放大区间会迫使 agent 保持极度竖直，从而抑制速度的进一步提升。  
本轮修改将 **放大区间扩宽至 0.3 rad**，同时将 **峰值放大系数从 1.6 降至 1.4**，衰减区间相应调整为 0.3～0.5 rad。这样：
- 正常步态的前倾仍可获得温和放大的速度奖励，鼓励提速；
- 放大幅度降低可避免 agent 过度追求竖直而阻碍加速；
- 衰减终点保持在 0.5 rad 不变，摔倒防护仍有效。

预期在不增加新惩罚的前提下，略微释放速度潜力，推动分数从 ~292.6 向 300 靠拢。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    horizontal_vel = obs[2]
    hull_angle = obs[0]

    # ============================================================
    # Component 1: forward_progress (linear main signal)
    # ============================================================
    forward_reward = 1.0 * max(0.0, horizontal_vel)

    # ============================================================
    # Component 2: soft_health_gate (amplify when mostly upright, attenuate when tilted)
    #   gate ∈ [0.0, 1.4]:
    #     - |angle| ≤ 0.3: 1.4 → 1.0 (gentle amplification)
    #     - |angle| > 0.3: 1.0 → 0.0 at 0.5 rad (attenuation)
    # ============================================================
    abs_angle = abs(hull_angle)
    if abs_angle <= 0.3:
        gate_factor = 1.0 + 0.4 * (1.0 - abs_angle / 0.3)
    else:
        gate_factor = max(0.0, 1.0 - (abs_angle - 0.3) / 0.2)

    gated_forward = forward_reward * gate_factor

    # ============================================================
    # Component 3: energy_penalty (unchanged)
    # ============================================================
    action_power = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.02 * action_power

    # ============================================================
    # Total reward
    # ============================================================
    total_reward = gated_forward + energy_penalty

    components = {
        'forward_reward': forward_reward,
        'gate_factor': gate_factor,
        'gated_forward': gated_forward,
        'energy_penalty': energy_penalty,
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: gate 的放大区间过窄，挤压了正常步态所需的前倾，间接限制了速度。  
- **behavior**: agent 长期保持极度竖直的姿势，速度中等，无法突破更高速度。  
- **signal**: 前进速度已有梯度，但 gate 的过早衰减抑制了适度前倾获得的加速。  
- **level**: Level 2  
- **hypothesis**: 放宽免衰减区间至 0.3 rad 并降低放大峰值至 1.4，能让 agent 敢于前倾加速，提升前进奖励，逼近 300 分。  
- **risk**: 摔倒概率可能轻微上升，但幅度很小，整体应正向。
