# Response Record

## 诊断

**1. Agent 状态：** Episode 全长 1000，无 crash。Score 148（分布 122-182），在 LunarLander 中算"接近但未稳定着陆"。landing_incentive mean=0.298，说明 agent 大部分时间处于 near/slow/upright 状态——但它可能在目标附近**徘徊**而非真正 touchdown。

**2. 根因分析：** 
- `landing_incentive` 的 ratio_to_progress 高达 156。它每步都给显著奖励（nonzero_rate=1.0），因为当前衰减系数（k_dist=3, k_speed=2, k_angle=3）太温和——即使在距离 1.0、速度 0.5 时，乘积仍约 0.25×0.5×0.4×0.5≈0.025，比 progress_delta（0.0019）大一个数量级。
- Agent 被训练成"保持近、慢、正"即可拿分，缺乏动力完成最后几厘米的 touchdown。
- `progress_delta` 太小（0.0019/step）不是因为它弱，而是因为 agent 已经在目标附近晃荡——距离变化自然小。

**3. 上轮动作：** 加了 landing_incentive（连续乘积形态），效果显著（84→148）。说明 landing 引导方向对，但**数学形态太宽**，导致它变成了全局 shaping 而非 touchdown 信号。

**修改方案：** 层次 2——改 landing_incentive 的数学形态。把衰减系数大幅提高，使奖励集中在真正接近着陆的状态（dist<0.3, speed<0.3, angle<0.15），同时提高峰值系数以保持 touchdown 附近的强梯度。这样 landing_incentive 从"全局 hover 奖励"变成"精准着陆引导"，让 progress_delta 在日常导航中重新成为主导信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- potential-based shaping: Φ = -(distance + w_speed*speed + w_angle*|angle| + w_angvel*|angvel|) ----
    # This backbone is unchanged — it provides the main progress gradient.

    # --- positions ---
    dx_obs, dy_obs = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_obs = (dx_obs ** 2 + dy_obs ** 2) ** 0.5
    dist_next = (dx_next ** 2 + dy_next ** 2) ** 0.5

    # --- speeds (L1 norm) ---
    speed_obs = abs(obs[2]) + abs(obs[3])
    speed_next = abs(next_obs[2]) + abs(next_obs[3])

    # --- angle ---
    angle_obs = abs(obs[4])
    angle_next = abs(next_obs[4])

    # --- angular velocity ---
    angvel_obs = abs(obs[5])
    angvel_next = abs(next_obs[5])

    # --- potential values (unchanged weights) ---
    w_speed = 0.05
    w_angle = 0.05
    w_angvel = 0.02

    phi_obs = -(dist_obs + w_speed * speed_obs + w_angle * angle_obs + w_angvel * angvel_obs)
    phi_next = -(dist_next + w_speed * speed_next + w_angle * angle_next + w_angvel * angvel_next)

    potential_reward = phi_next - phi_obs

    # --- landing incentive: sharpened proximity gating ---
    # Old version used k_dist=3, k_speed=2, k_angle=3 — too wide, dominated progress.
    # New version uses much steeper decay so reward concentrates near true touchdown
    # (dist<0.3, speed<0.3, angle<0.15). Peak raised to 1.5 to maintain strong terminal gradient.
    near_factor = 1.0 / (1.0 + 15.0 * dist_next)
    slow_factor = 1.0 / (1.0 + 8.0 * speed_next)
    upright_factor = 1.0 / (1.0 + 8.0 * angle_next)
    landing_incentive = 1.5 * near_factor * slow_factor * upright_factor

    # --- total ---
    total_reward = potential_reward + landing_incentive

    # --- decompose for logging ---
    progress_delta = dist_obs - dist_next
    speed_reduction = w_speed * (speed_obs - speed_next)
    angle_reduction = w_angle * (angle_obs - angle_next)
    angvel_reduction = w_angvel * (angvel_obs - angvel_next)

    components = {
        'progress_delta': progress_delta,
        'speed_reduction': speed_reduction,
        'angle_reduction': angle_reduction,
        'angvel_reduction': angvel_reduction,
        'landing_incentive': landing_incentive,
        'total_reward': total_reward,
    }

    return float(total_reward), components
```
