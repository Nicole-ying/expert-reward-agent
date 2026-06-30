def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = 6.0 * (current_dist - next_dist)
    distance_anchor = -0.05 * next_dist

    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    speed_penalty = -0.08 * speed
    stability_penalty = -0.08 * abs(next_obs[4]) - 0.04 * abs(next_obs[5])

    near_target_quality = 1.0 / (1.0 + 4.0 * next_dist)
    low_speed_quality = 1.0 / (1.0 + 2.0 * speed)
    angle_quality = 1.0 / (1.0 + 5.0 * abs(next_obs[4]))
    landing_quality = 0.4 * near_target_quality * low_speed_quality * angle_quality

    total_reward = progress_reward + distance_anchor + speed_penalty + stability_penalty + landing_quality
    components = {
        "progress_reward": progress_reward,
        "distance_anchor": distance_anchor,
        "speed_penalty": speed_penalty,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality,
        "total_reward": total_reward,
    }
    return float(total_reward), components