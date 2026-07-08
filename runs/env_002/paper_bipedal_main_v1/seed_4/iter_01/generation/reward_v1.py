def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 前进速度奖励（正值表示向前，直接作为主学习信号）
    forward_speed = next_obs[2]
    forward_reward = forward_speed  # 系数 1.0

    # 稳定性惩罚：惩罚大倾角、高角速度和明显垂直速度（弹跳）
    tilt_angle = next_obs[0]
    angular_vel = next_obs[1]
    vertical_vel = next_obs[3]

    tilt_penalty = -0.5 * (tilt_angle ** 2)
    angular_vel_penalty = -0.1 * (angular_vel ** 2)
    vertical_vel_penalty = -0.5 * (vertical_vel ** 2)
    stability_penalty = tilt_penalty + angular_vel_penalty + vertical_vel_penalty

    total_reward = forward_reward + stability_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components