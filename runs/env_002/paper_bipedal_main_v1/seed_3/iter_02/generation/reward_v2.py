def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 核心进步信号：鼓励沿前进方向快速移动
    forward_velocity = next_obs[2]
    w_fwd = 1.0
    forward_reward = w_fwd * forward_velocity

    # 姿态稳定约束：惩罚躯干倾角偏离直立
    hull_angle = next_obs[0]
    w_angle = 1.0
    angle_penalty = -w_angle * (hull_angle ** 2)

    # 垂直稳定约束：抑制跳跃或剧烈起伏
    vertical_velocity = next_obs[3]
    w_vert = 0.5
    vert_penalty = -w_vert * (vertical_velocity ** 2)

    # 能耗约束：惩罚过大的关节力矩，鼓励高效低能耗步态
    energy_penalty = -0.05 * sum(a ** 2 for a in action)

    total_reward = forward_reward + angle_penalty + vert_penalty + energy_penalty

    components = {
        'forward_reward': forward_reward,
        'angle_penalty': angle_penalty,
        'vert_penalty': vert_penalty,
        'energy_penalty': energy_penalty
    }

    return float(total_reward), components