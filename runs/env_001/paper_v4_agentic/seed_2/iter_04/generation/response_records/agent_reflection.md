# Response Record

# 设计理由
本轮为第 4 迭代，延续 v3 的 `landing_bonus` 组件，但解决其 **active_rate=0.0%** 的死代码问题。  
根因在于 `landing_bonus` 的联合门控太严格——距离衰减尺度 0.3 使大部分距离奖励近乎为零，且速度/角度硬截止 0.3 将绝大多数着陆瞬间的接触样本排除在外，导致该组件完全未提供梯度，无法影响策略。  
修改方案属于 **Level 1 尺度修复**：保持连续乘积形式，但大幅松弛门控阈值（距离衰减系数 0.3 → 1.0，速度/角度截止 0.3 → 0.5，使因子能在常见的着陆前状态中被激活），并将权重从 80.0 降为 20.0，避免单步奖励过度支配总回报（校准：预计激活时 per‑step ≤ 20 × 0.1~0.3 = 2~6，不超过主信号 proximity_delta 的 2~3 倍，可接受）。其他组件保持不变。  
预期效果：`landing_bonus` 开始提供软着陆信号，引导 agent 在接近目标时降低速度与倾斜角，配合已有的 `velocity_danger` 和 `proximity_delta`，减少高速撞击导致的早期失败，提升分数并延长 episode 长度。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4 — relax landing_bonus thresholds to revive the dead component.
    """
    # ── current state ──
    x_cur = obs[0]
    y_cur = obs[1]
    vx_cur = obs[2]
    vy_cur = obs[3]
    angle_cur = obs[4]
    angvel_cur = obs[5]

    # ── next state ──
    x_next = next_obs[0]
    y_next = next_obs[1]
    left_contact  = next_obs[6]
    right_contact = next_obs[7]

    # ── distance to pad ──
    dist_cur  = (x_cur  ** 2 + y_cur  ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # ── weights / thresholds ──
    w_prox = 50.0
    w_vel  = 0.15
    w_ang  = 5.0
    proximity_threshold = 1.0

    w_land = 20.0                 # reduced from 80.0

    # ── 1. Proximity delta ──
    proximity_delta = w_prox * (dist_cur - dist_next)

    # ── 2. Velocity danger ──
    speed_sq = vx_cur ** 2 + vy_cur ** 2
    velocity_danger = -w_vel * speed_sq / (dist_cur + proximity_threshold)

    # ── 3. Orientation penalty ──
    orientation_penalty = -w_ang * (angle_cur ** 2 + angvel_cur ** 2)

    # ── 4. Soft landing bonus (relaxed) ──
    contact = max(left_contact, right_contact)

    # distance factor: slower decay (sigma 1.0 instead of 0.3)
    dist_factor = 2.718281828 ** (-dist_next / 1.0)

    # velocity and angle factors: wider linear ramp (cutoff 0.5 instead of 0.3)
    yvel_factor = max(0.0, 1.0 - abs(vy_cur) / 0.5)
    xvel_factor = max(0.0, 1.0 - abs(vx_cur) / 0.5)
    angle_factor = max(0.0, 1.0 - abs(angle_cur) / 0.5)

    landing_bonus = w_land * contact * dist_factor * yvel_factor * xvel_factor * angle_factor

    # ── Total reward ──
    total_reward = proximity_delta + velocity_danger + orientation_penalty + landing_bonus

    components = {
        "proximity_delta": proximity_delta,
        "velocity_danger": velocity_danger,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 观测维度全部使用，但 `landing_bonus` 因过严的门控导致 active_rate=0%，信号完全缺失。
- **behavior**: agent 快速向目标移动，但缺乏减速/对齐指导，以较高速度撞击地面，导致早期失败（len≈68）。
- **signal**: 缺软着陆引导，现有速度惩罚亦未能阻止高速撞击。
- **level**: Level 1
- **hypothesis**: 放宽 `landing_bonus` 阈值将使其在着陆瞬间被激活，提供可学习的软着陆梯度，改善终端速度/姿态，延长 episode 并提高分数。
- **risk**: 奖励可能在接触瞬间过大，诱使 agent 做出不稳定的最后时刻微调；若宽度扩展过度，可能模糊软着陆条件。通过降低权重和控制截止值已缓解。
