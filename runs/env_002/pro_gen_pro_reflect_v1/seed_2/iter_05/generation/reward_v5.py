def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ====== 主学习信号：线性前进速度奖励 ======
    forward_vel = next_obs[2]
    w_forward = 1.0
    forward_reward = w_forward * forward_vel

    # ====== 姿态稳定性惩罚（hinge 门控） ======
    hull_angle = obs[0]
    hull_angvel = obs[1]
    vertical_vel = obs[3]

    # 倾角安全阈值：设为 0.4 rad（约 23°），在该范围内不施加惩罚
    safe_angle = 0.4
    angle_violation = max(0.0, abs(hull_angle) - safe_angle)

    w_angle = 1.5
    w_angvel = 0.01
    w_vertical = 0.01

    stability_penalty = -(
        w_angle * angle_violation ** 2 +
        w_angvel * hull_angvel ** 2 +
        w_vertical * vertical_vel ** 2
    )

    # ====== 能量效率惩罚（系数减半以给予更积极的运动自由度） ======
    w_energy = 0.005  # 原 0.01
    energy_penalty = -w_energy * (
        action[0] ** 2 + action[1] ** 2 +
        action[2] ** 2 + action[3] ** 2
    )

    total_reward = forward_reward + stability_penalty + energy_penalty
    components = {
        'forward_reward': forward_reward,
        'stability_penalty': stability_penalty,
        'energy_penalty': energy_penalty
    }
    return float(total_reward), components