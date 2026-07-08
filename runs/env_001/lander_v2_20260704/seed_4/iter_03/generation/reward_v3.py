def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 距离进展：每一步靠近目标的欧氏距离变化
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = 4.0 * (dist_obs - dist_next)

    # 稳定性约束：抑制过度速度和姿态变化
    stability_penalty = (
        -0.01 * (abs(next_obs[2]) + abs(next_obs[3]))
        - 0.01 * abs(next_obs[4])
        - 0.01 * abs(next_obs[5])
    )

    # 稠密着陆代理信号：连续评估各着陆条件的满足度
    D_x = 2.0
    D_y = 2.0
    D_v = 1.0
    D_angle = 0.5

    x_sat = max(0.0, 1.0 - abs(next_obs[0]) / D_x)
    y_sat = max(0.0, 1.0 - abs(next_obs[1]) / D_y)
    vx_sat = max(0.0, 1.0 - abs(next_obs[2]) / D_v)
    vy_sat = max(0.0, 1.0 - abs(next_obs[3]) / D_v)
    angle_sat = max(0.0, 1.0 - abs(next_obs[4]) / D_angle)
    contact_factor = max(0.05, 0.5 * (next_obs[6] + next_obs[7]))

    landing_proxy = x_sat * y_sat * vx_sat * vy_sat * angle_sat * contact_factor
    landing_reward = 0.8 * landing_proxy

    total_reward = progress_reward + stability_penalty + landing_reward

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_proxy': landing_reward
    }

    return float(total_reward), components