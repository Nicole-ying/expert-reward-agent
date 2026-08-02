# 设计理由
本轮对 `progress_gated` 组件做了结构性变换（Level 2）：从“位置距离的绝对减少量 `dist - next_dist`”替换为“速度在朝向原点方向上的投影 `toward_speed`”。原 `delta_dist` 在 agent 未系统性靠近原点时几乎为零，导致该组件虽激活率高但 reward 极小（ep_sum≈0.15），无法为 agent 提供有效的向前指引。新信号直接奖励向目标移动的速度分量，配合 next-state 的姿态门控，能在早期给予稠密梯度，引导 agent 向原点靠近并维持稳定姿态。保留 `proximity_stability` 作为软着陆奖励，保留 `fuel_penalty` 不做更改（其实际贡献极微，不干扰主信号）。系数 `w_progress=8.0` 使正常前进时 per‑step 奖励约 0.8~1.6，与 `proximity_stability` 后期奖励叠加，能有效对抗环境内置的负奖励，同时受 gate 约束不会在姿态恶劣时膨胀。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------- unpack observations -------------------
    x,  y  = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle      = obs[4]
    angvel     = obs[5]
    left_leg   = obs[6]
    right_leg  = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle  = next_obs[4]
    n_angvel = next_obs[5]
    n_left   = next_obs[6]
    n_right  = next_obs[7]

    # ------------------- helper quantities -------------------
    dist      = (x**2  + y**2)  ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    vel_abs       = (vx**2 + vy**2) ** 0.5
    next_vel_abs  = (nvx**2 + nvy**2) ** 0.5

    # ------------------- thresholds & weights -------------------
    w_progress = 8.0
    w_proximity = 10.0
    w_fuel = 0.2

    th_angle  = 0.5
    th_vel    = 1.0
    th_angvel = 2.0
    th_dist   = 0.5

    gate_min = 0.1
    gate_min_stab = 0.2

    # ------------------- 1. velocity-toward-target progress signal -------------------
    # unit vector toward origin from next position
    dir_x = -nx / (next_dist + 1e-6)
    dir_y = -ny / (next_dist + 1e-6)
    toward_speed = nvx * dir_x + nvy * dir_y      # positive if moving toward origin

    # gate based on next state for stability after action
    gate_angle  = max(gate_min, 1.0 - abs(n_angle)  / th_angle)
    gate_vel    = max(gate_min, 1.0 - next_vel_abs   / th_vel)
    gate_angvel = max(gate_min, 1.0 - abs(n_angvel)  / th_angvel)
    gate = gate_angle * gate_vel * gate_angvel

    progress_gated = w_progress * max(0.0, toward_speed) * gate

    # ------------------- 2. proximity + stability reward -------------------
    prox_factor = max(0.0, 1.0 - next_dist / th_dist)

    a_stab  = max(gate_min_stab, 1.0 - abs(n_angle)  / th_angle)
    v_stab  = max(gate_min_stab, 1.0 - next_vel_abs   / th_vel)
    av_stab = max(gate_min_stab, 1.0 - abs(n_angvel)  / th_angvel)
    stab = a_stab * v_stab * av_stab

    contact_flag = 1.0 if (n_left + n_right) >= 1.0 else 0.0
    contact_mult = 1.0 + 0.5 * contact_flag

    proximity_stability_reward = w_proximity * prox_factor * stab * contact_mult

    # ------------------- 3. fuel penalty -------------------
    engine_on = 1.0 if action != 0 else 0.0
    fuel_penalty = -w_fuel * engine_on

    # ------------------- total reward -------------------
    total_reward = progress_gated + proximity_stability_reward + fuel_penalty

    components = {
        'progress_gated':   progress_gated,
        'proximity_stability': proximity_stability_reward,
        'fuel_penalty':      fuel_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测维度均被使用，但前进信号数学形态错误导致有效指引缺失——`delta_dist` 在 agent 未主动靠近原点时恒近乎零，正奖励几乎为 0。
- **behavior**: agent 快速终止（len 68，全部 terminated），原地挣扎或乱飞，未向目标移动，无法获得足以对抗环境惩罚的正奖励。
- **signal**: 缺少有效的“向目标前进”稠密信号；`progress_gated` 虽激活率高但无实质内容，主干奖励几乎为空。
- **level**: Level 2 — 结构变换，将 `progress_gated` 的信号源从距离差改为速度投影。
- **hypothesis**: 新的 `toward_speed` 奖励在每一有向目标移动的 step 上提供正反馈，结合姿态门控将引导 agent 朝向原点并保持稳定，使正奖励显著提升，从而打破快速失败的循环。
- **risk**: 若 agent 学会在姿态不佳时硬冲向原点，可能引发 crash；但 gate 因子（阈值设在终止临界区的 60‑80%）会抑制这种行为，且 `proximity_stability` 在接近时进一步奖励姿态，配合作用应能避免。