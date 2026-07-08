def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 权重
    w_forward = 1.3
    w_angle = 0.5
    w_vertical_vel = 0.1
    w_angular_vel = 0.05
    w_energy = 0.05

    # 主学习信号：前进速度奖励
    forward_vel = next_obs[2]
    forward_reward = w_forward * forward_vel

    # 稳定/安全约束
    angle = abs(next_obs[0])
    vertical_vel = abs(next_obs[3])
    angular_vel = abs(next_obs[1])
    stability_penalty = -(w_angle * angle + w_vertical_vel * vertical_vel + w_angular_vel * angular_vel)

    # 能量效率：惩罚过大的关节力矩
    torque_sum_sq = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -w_energy * torque_sum_sq

    total_reward = forward_reward + stability_penalty + energy_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty
    }
    return float(total_reward), components