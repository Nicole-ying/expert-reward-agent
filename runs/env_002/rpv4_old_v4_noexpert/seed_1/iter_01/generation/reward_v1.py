def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度，使用 next_obs 的水平速度
    vx = float(next_obs[2])
    forward_reward = 1.0 - 2.718281828 ** (-max(vx, 0.0) * 1.5)
    forward_reward *= 1.5

    # 稳定性约束：只惩罚过大的躯干倾斜和躯干角速度
    angle_margin = max(0.0, abs(float(next_obs[0])) - 0.15)
    angular_vel_margin = max(0.0, abs(float(next_obs[1])) - 1.0)
    upright_stability_penalty = (
        -2.0 * angle_margin ** 2
        - 0.15 * angular_vel_margin ** 2
    )

    # 上下弹跳约束：轻微惩罚超过死区的垂直速度
    vertical_margin = max(0.0, abs(float(next_obs[3])) - 0.6)
    vertical_oscillation_penalty = -0.3 * vertical_margin ** 2

    total_reward = (
        forward_reward
        + upright_stability_penalty
        + vertical_oscillation_penalty
    )

    components = {
        "forward_velocity_reward": float(forward_reward),
        "upright_stability_penalty": float(upright_stability_penalty),
        "vertical_oscillation_penalty": float(vertical_oscillation_penalty),
    }

    return float(total_reward), components