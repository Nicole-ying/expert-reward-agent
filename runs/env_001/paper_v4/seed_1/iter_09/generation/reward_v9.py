def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 主学习信号：距离改善 × 姿态门控（保持不变）
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next

    angle_abs = abs(next_obs[4])
    gate      = 1.0 / (1.0 + 5.0 * angle_abs)
    shaped_progress = 5.0 * progress_raw * gate

    # 2. 安全约束：角速度轻量惩罚（保持不变）
    ang_vel_penalty = -0.1 * (next_obs[5] ** 2)

    # 3. 着陆质量：连续稠密信号（sparse→dense，替代稀疏转移事件new_contact门控）
    height = next_obs[1]
    near_ground = 1.0 / (1.0 + 5.0 * abs(height))
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    low_speed = 1.0 / (1.0 + speed)
    upright = 1.0 / (1.0 + 3.0 * angle_abs)
    x_offset = abs(next_obs[0])
    centered = 1.0 / (1.0 + 3.0 * x_offset)

    landing_quality = 0.5 * near_ground * (2.0 * low_speed + 1.0 * upright + 1.0 * centered)

    total_reward = shaped_progress + ang_vel_penalty + landing_quality
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_quality": landing_quality
    }
    return (float(total_reward), components)