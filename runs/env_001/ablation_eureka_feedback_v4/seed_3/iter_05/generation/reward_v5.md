# 设计理由
上一轮的常驻状态值奖励（proximity 与 landing_quality）导致 agent 在目标附近徘徊而不触发终止（全部 episode 被截断，未发生成功或失败），典型的 reward hacking。将主信号重构为 **progress × landing_gate**：只奖励向目标的位移改善量，并用连续姿态/接触门控乘到改善量上，从而消除“停在好位置就能持续获奖”的结构。progress 系数设为 10.0，使 per‑step 典型奖励与能量惩罚（-0.01）可比；landing_gate 通过 contact_factor 保持最低 0.3，确保在未接触但姿态合理的过渡区域仍有学习信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current state
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    left, right = obs[6], obs[7]
    # Next state
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nleft, nright = next_obs[6], next_obs[7]

    # 1. Progress: reduction in distance to target
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    dist_next = (nx ** 2 + ny ** 2) ** 0.5
    progress = max(0.0, dist_curr - dist_next)

    # 2. Landing gate: contact factor × posture factor
    # contact_factor encourages at least one contact, more is better; floor 0.3
    contact_factor = 0.3 + 0.7 * (nleft + nright) / 2.0   # range [0.3, 1.0]
    # posture_factor selects for small angle and low speed
    angle_sq = nangle ** 2
    speed_sq = nvx ** 2 + nvy ** 2
    posture_factor = 2.71828 ** (-10.0 * angle_sq) * 2.71828 ** (-2.0 * speed_sq)
    landing_gate = contact_factor * posture_factor

    # 3. Main reward: progress modulated by landing quality
    main_reward = progress * landing_gate * 10.0

    # 4. Energy penalty (unchanged)
    energy_penalty = -0.01 if action != 0 else 0.0

    # 5. Terminal velocity penalty (unchanged from previous round)
    vel_penalty = 0.0
    if ny < 0.05 and abs(nx) < 0.1:
        if nvy < -0.3:
            vel_penalty = -0.5 * max(0.0, -nvy - 0.3)

    total_reward = main_reward + energy_penalty + vel_penalty

    components = {
        "progress_gate_reward": main_reward,
        "energy_penalty": energy_penalty,
        "terminal_velocity_penalty": vel_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: truncated=20/20, len=1000, proximity.mean=646, landing_quality.mean=688, score=-24.9
- **behavior**: agent 在目标附近徘徊，利用状态值奖励持续积累高分却无法完成着陆（reward hacking）
- **signal**: 缺少“进步”信号，常驻状态值奖励导致 exploit；需要结构转为改善量并耦合着陆条件
- **level**: Level 2（状态值→改善量 + proxy对齐任务完成）
- **hypothesis**: 改为 progress × landing_gate 会切断静态奖励，迫使 agent 通过不断靠近目标并同时保持良好姿态来获得奖励，靠近后奖励归零的自然衰减会驱动尽快稳定着陆以终止回合
- **risk**: 如果 landing_gate 在早期探索中衰减过快，可能造成奖励过于稀疏；采用 contact_factor 最低 0.3 和 posture 指数衰减来保证过渡区域有非零信号