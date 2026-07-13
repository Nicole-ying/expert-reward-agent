# 设计理由
速度惩罚（velocity_penalty）per‑episode 均值高达 –70.9，占全部负反馈的 90% 以上，而其系数为 5.0，与主进展信号（approach_progress + landing_quality ≈ 2.74 + 40.6 = 43.3）的比例严重失衡（26:1）。由于该惩罚在每一步都处于激活状态且 agent 已经能够成功着陆（terminated=15/20），当前唯一的约束就是过度惩罚可能损害效率。因此我将 w_vel_penalty 从 5.0 降为 0.5，使其 per‑step 量级降至主信号的 0.1‑0.3 倍（典型 speed excess 0.02 时每步约 –0.01），在不移除减速激励的前提下让总奖励更贴近外部任务目标。其他组件保持不变。

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
    w_vel_penalty = 0.5           # 降低为原来的 1/10，使惩罚与主信号成比例
    w_angle = 0.5
    w_angvel = 0.1
    w_landing = 0.1

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
- **evidence**: velocity_penalty 均值 –70.9，主正信号 43.3，比例 26:1；terminated=15/20 说明着陆行为已基本稳定。  
- **behavior**: agent 接近目标并减速着陆，但全程受过度速度惩罚，得分被压低了约 70 分。  
- **signal**: 速度惩罚系数过大，与进展信号严重失衡，不必要的负反馈限制了得分和训练效率。  
- **level**: Level 1 尺度修复 — 降低 w_vel_penalty 至与主信号成比例。  
- **hypothesis**: 降低惩罚系数将直接提升总分，同时保留的减速激励仍能维持安全着陆，有望突破目标。  
- **risk**: 若减速压力过弱，agent 可能在最后阶段速度偏高导致少量 crash，但 landing_quality 中的 speed_factor 会提供额外约束。