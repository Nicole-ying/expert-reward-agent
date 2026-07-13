# Response Record

# 设计理由
我将 `w_approach` 从 2.0 提升到 5.0，目的是放大接近进度的梯度，鼓励 agent 在安全速度约束内更快移向着陆垫。当前 progress 的 per‑step 贡献只有速度惩罚的 1/4、landing_quality 的 1/8，导致 agent 偏向缓慢飘移（episode 长度 ~351 步）。将系数提高到 5 使 progress 的 per‑step 贡献约 0.04，与 landing_quality 同一量级但仍低于 safety hinge 的 magnitude，不会淹没安全约束。只做单一系数调整，属于 Level 1 尺度修复。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current and next positions relative to landing pad center
    dx_curr, dy_curr = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_curr = (dx_curr**2 + dy_curr**2) ** 0.5
    dist_next = (dx_next**2 + dy_next**2) ** 0.5

    # Approach progress: positive when getting closer to the pad
    approach_delta = dist_curr - dist_next

    # Next‑step velocity and safety threshold
    vx_next, vy_next = next_obs[2], next_obs[3]
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    safe_speed = 0.2

    # Next‑step orientation and angular velocity
    angle_next = next_obs[4]
    angvel_next = next_obs[5]

    # Contact flags
    left_contact_next = next_obs[6] > 0.5
    right_contact_next = next_obs[7] > 0.5

    # --- Reward weights (best skeleton) ---
    w_approach = 5.0        # increased from 2.0
    w_vel_penalty = 0.5
    w_angle = 0.5
    w_angvel = 0.1
    w_landing = 0.1
    w_grounded = 0.3        # new continuous grounded quality weight

    # Component 1: dense progress towards the pad
    approach_reward = w_approach * approach_delta

    # Component 2: speed constraint (hinge)
    vel_penalty = -w_vel_penalty * max(0.0, speed_next - safe_speed)

    # Component 3: angular stability (quadratic penalties)
    angle_stability = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # Component 4: continuous landing‑quality proxy (unchanged)
    dist_factor = 1.0 / (1.0 + 1.0 * dist_next)
    speed_factor = 1.0 / (1.0 + 1.0 * speed_next)
    landing_reward = w_landing * dist_factor * speed_factor

    # Component 5 (REPLACED): grounded quality — only when leg contact and low speed
    grounded_reward = 0.0
    if left_contact_next or right_contact_next:
        # linear factor: 1 at zero speed, 0 at safe_speed, 0 above
        speed_ratio = max(0.0, 1.0 - speed_next / safe_speed)
        grounded_reward = w_grounded * speed_ratio

    total_reward = approach_reward + vel_penalty + angle_stability + landing_reward + grounded_reward

    components = {
        "approach_progress": approach_reward,
        "velocity_penalty": vel_penalty,
        "angle_stability": angle_stability,
        "landing_quality": landing_reward,
        "grounded_quality": grounded_reward,
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: 所有 episode (20/20) 以 terminated 终止，len=351，score 范围 [229, 291]；progress per‑step 均值 0.008 远小于 landing/grounded 的 0.06/0.04。
- **behavior**: agent 已经稳定着陆，但移速保守，每步位移极小，完成任务步数偏多。
- **signal**: approach_progress 信号太弱，被 landing_quality 和 velocity_penalty 支配，未能充分激励快速靠近。
- **level**: Level 1‑尺度修复（|penalty/progress| ≈ 4 > 0.5 且 velocity_penalty active_rate 估计高，但当前不宜降惩罚，反过来提升 progress）。
- **hypothesis**: 提升 progress 权重到与 landing_quality 同量级，可引导 agent 在安全速度范围内以更短路径接近，减小 episode length，同时不破坏已有的安全着陆策略。
- **risk**: 若权重过大，agent 可能在末端速度超过 hinge 阈值，触发 velocity_penalty 激增从而震荡；系数 5.0 保守，预计速度仍被 0.2 的 hinge 约束。
