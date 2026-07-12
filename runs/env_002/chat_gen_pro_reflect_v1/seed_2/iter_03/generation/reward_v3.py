def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：向前速度的凸化奖励
    horizontal_vel = obs[2]
    forward_progress_reward = 1.3 * (horizontal_vel ** 2)

    # 平衡约束（角度与角速度的软惩罚，保持原有形态但当前未触发）
    hull_angle = obs[0]
    hull_angular_vel = obs[1]

    angle_threshold = 0.3
    angle_penalty = -0.5 * max(0.0, abs(hull_angle) - angle_threshold) ** 2

    angular_vel_threshold = 2.0
    angular_vel_penalty = -0.1 * max(0.0, abs(hull_angular_vel) - angular_vel_threshold) ** 2

    balance_penalty = angle_penalty + angular_vel_penalty

    # 健康门控：身体角度偏离越大，主奖励衰减越强
    gate_factor = max(0.0, min(1.0, 1.0 - abs(hull_angle) / 0.5))
    gated_forward_reward = forward_progress_reward * gate_factor

    # 新增加的能量消耗惩罚（Level 2 干预）
    action_power = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.05 * action_power

    total_reward = gated_forward_reward + balance_penalty + energy_penalty

    components = {
        'forward_progress_reward': forward_progress_reward,
        'gated_forward_reward': gated_forward_reward,
        'balance_angle_penalty': angle_penalty,
        'balance_angular_vel_penalty': angular_vel_penalty,
        'gate_factor': gate_factor,
        'energy_penalty': energy_penalty
    }

    return float(total_reward), components