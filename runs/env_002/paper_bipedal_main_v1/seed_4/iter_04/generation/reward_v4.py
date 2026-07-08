def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 前进速度奖励
    forward_speed = next_obs[2]
    forward_reward = forward_speed

    # 稳定性惩罚
    tilt_angle = next_obs[0]
    angular_vel = next_obs[1]
    vertical_vel = next_obs[3]

    tilt_penalty = -2.5 * (tilt_angle ** 2)
    angular_vel_penalty = -0.5 * (angular_vel ** 2)
    vertical_vel_penalty = -2.5 * (vertical_vel ** 2)
    stability_penalty = tilt_penalty + angular_vel_penalty + vertical_vel_penalty

    # 能量消耗惩罚（动作力矩平方和）—— 系数从-0.05降至-0.03
    energy_penalty = -0.03 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    total_reward = forward_reward + stability_penalty + energy_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty
    }

    return float(total_reward), components