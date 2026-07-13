# Response Record

# 设计理由
本轮改动为 **Level 1 尺度修复**：将 `landing_quality`（持续状态奖励）的系数 `w_landing` 从 `0.1` 降至 `0.01`。证据显示该组件平均每 episode 高达 61.55，远高于主进展信号 `approach_progress`（2.79），导致智能体在目标区悬停刷分而非尽快完成着陆，大量 episode 因时间上限而被截断。降低系数后，其 per‑step 预期贡献降至约 0.0077，不超过主进展信号的 2 倍，从而减弱悬停激励，同时仍保留对低距离、低速度的微弱偏好。

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

    # --- Reward weights ---
    w_approach = 2.0
    w_vel_penalty = 0.5           # 保持上轮调整
    w_angle = 0.5
    w_angvel = 0.1
    w_landing = 0.01              # 从 0.1 降至 0.01，削弱悬停收益

    # Component 1: dense progress towards the pad
    approach_reward = w_approach * approach_delta

    # Component 2: speed constraint (hinge – only penalise when exceeding safe threshold)
    vel_penalty = -w_vel_penalty * max(0.0, speed_next - safe_speed)

    # Component 3: angular stability (quadratic penalties)
    angle_stability = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # Component 4: continuous landing‑quality proxy (no contact requirement)
    dist_factor = 1.0 / (1.0 + 1.0 * dist_next)
    speed_factor = 1.0 / (1.0 + 1.0 * speed_next)
    landing_reward = w_landing * dist_factor * speed_factor

    total_reward = approach_reward + vel_penalty + angle_stability + landing_reward

    components = {
        "approach_progress": approach_reward,
        "velocity_penalty": vel_penalty,
        "angle_stability": angle_stability,
        "landing_quality": landing_reward,
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated=9/20, truncated=11/20；`landing_quality` 均值 61.55 远大于 `approach_progress` 2.79。
- **behavior**: 智能体已学会接近目标并保持低速，但因悬停刷取高额 `landing_quality` 奖励，大量 episode 在到达着陆垫后被时间截断。
- **signal**: 持续状态奖励 `landing_quality` 过强，导致完成着陆的动力不足。
- **level**: Level 1 — 尺度修复（降低 `w_landing` 系数）。
- **hypothesis**: 削弱悬停收益后，智能体将更倾向于主动完成最终着陆以终止 episode，从而提高 terminated 比例并减少截断，最终提升得分。
- **risk**: 若降低幅度过大，可能导致智能体完全失去在目标区降速的引导，引发 crash；但新 per‑step 贡献仍为正，可保留降速倾向。
