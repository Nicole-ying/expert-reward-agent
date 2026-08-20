def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ====== 主学习信号：前进速度奖励（平方形式，凸化激励） ======
    # 使用 forward_vel * |forward_vel| 保持方向，同时平方增强高速激励
    forward_vel = next_obs[2]
    w_forward = 2.0
    forward_reward = w_forward * forward_vel * abs(forward_vel)

    # ====== 竖直/姿态稳定性惩罚 ======
    hull_angle = obs[0]
    hull_angvel = obs[1]
    vertical_vel = obs[3]
    w_angle = 0.5
    w_angvel = 0.01
    w_vertical = 0.01
    stability_penalty = -(
        w_angle * hull_angle**2 +
        w_angvel * hull_angvel**2 +
        w_vertical * vertical_vel**2
    )

    # ====== 能量效率惩罚 ======
    w_energy = 0.001
    energy_penalty = -w_energy * (
        action[0]**2 + action[1]**2 +
        action[2]**2 + action[3]**2
    )

    total_reward = forward_reward + stability_penalty + energy_penalty
    components = {
        'forward_reward': forward_reward,
        'stability_penalty': stability_penalty,
        'energy_penalty': energy_penalty
    }
    return float(total_reward), components