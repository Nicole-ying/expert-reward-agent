def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward (保留，核心驱动)
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 2.0 * progress_delta

    # 稳定约束：stability_penalty (进一步削弱，避免主导progress)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.001 * abs(next_obs[4])  # 从0.002降至0.001
    angular_vel_penalty = 0.0005 * abs(next_obs[5])  # 从0.001降至0.0005
    speed_penalty = 0.003 * speed  # 从0.005降至0.003
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # 平滑接近引导：distance_reward (保留，作为小锚点)
    distance_reward = -0.1 * next_dist

    # 平滑接近引导：approach_shaping (保留，远距离时提供额外梯度)
    approach_shaping = 0.0
    if next_dist > 1.0:
        approach_shaping = -0.2 * next_dist

    # 平滑着陆引导：landing_shaping (保留，替代硬阈值landing_bonus)
    near_target = next_dist < 1.0
    low_speed = speed < 1.5
    stable_angle = abs(next_obs[4]) < 0.5
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_shaping = 0.0
    if near_target and low_speed and stable_angle:
        landing_shaping = 0.5 * (1.0 - next_dist / 1.0)
        if both_contact:
            landing_shaping += 0.3

    # 总奖励
    total_reward = progress_reward + stability_penalty + distance_reward + approach_shaping + landing_shaping

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "distance_reward": distance_reward,
        "approach_shaping": approach_shaping,
        "landing_shaping": landing_shaping,
        "total_reward": total_reward
    }

    return float(total_reward), components