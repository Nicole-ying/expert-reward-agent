def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前位置与目标垫距离
    x_curr = obs[0]
    y_curr = obs[1]
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5

    x_next = next_obs[0]
    y_next = next_obs[1]
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # 主进展信号：到目标的欧氏距离递减
    progress = dist_curr - dist_next

    # 下一步速度（用于原有速度约束）
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]

    # 速度硬约束（保留但可能保持僵尸）
    x_speed_viol = max(0.0, abs(x_vel_next) - 0.8)
    y_speed_viol = max(0.0, -y_vel_next - 0.8)
    speed_penalty = x_speed_viol + y_speed_viol

    # 姿态与角速度稳定约束
    body_angle_next = next_obs[4]
    angular_vel_next = next_obs[5]
    angle_penalty = body_angle_next ** 2 + angular_vel_next ** 2

    # 替换 contact_bonus：连续软着陆引导
    # 距离因子：越近奖励越大（指数衰减，半衰距离0.5）
    dist_factor = 2.718281828 ** (-dist_next / 0.5)
    # 速度因子：水平与垂直线速度绝对值和越小奖励越大（线性衰减，1.0处归零）
    speed_factor = max(0.0, 1.0 - (abs(x_vel_next) + abs(y_vel_next)) / 1.0)
    landing_reward = dist_factor * speed_factor

    # 加权组合
    total = (
        10.0 * progress
        - 1.0 * speed_penalty
        - 0.5 * angle_penalty
        + 0.01 * landing_reward
    )

    components = {
        "progress": 10.0 * progress,
        "speed_penalty": -1.0 * speed_penalty,
        "angle_penalty": -0.5 * angle_penalty,
        "landing_reward": 0.01 * landing_reward
    }

    return float(total), components