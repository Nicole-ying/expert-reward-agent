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

    # 下一步速度（用于约束）
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]

    # 速度硬约束：过快水平移动或过快下降
    x_speed_viol = max(0.0, abs(x_vel_next) - 0.8)
    # 假定 y_vel 向上为正，下降速度为 -y_vel，限制下降速度 ≤ 0.8
    y_speed_viol = max(0.0, -y_vel_next - 0.8)
    speed_penalty = x_speed_viol + y_speed_viol

    # 姿态与角速度稳定约束
    body_angle_next = next_obs[4]
    angular_vel_next = next_obs[5]
    angle_penalty = body_angle_next ** 2 + angular_vel_next ** 2

    # 软着陆接触奖励：双腿着垫且速度平稳时给予一次性正反馈
    left_next = next_obs[6]
    right_next = next_obs[7]
    contact_bonus = 0.0
    if left_next > 0.5 and right_next > 0.5 and abs(x_vel_next) < 0.3 and abs(y_vel_next) < 0.3:
        contact_bonus = 1.0

    # 加权组合
    total = (
        10.0 * progress
        - 1.0 * speed_penalty
        - 0.5 * angle_penalty
        + 2.0 * contact_bonus
    )

    components = {
        "progress": 10.0 * progress,
        "speed_penalty": -1.0 * speed_penalty,
        "angle_penalty": -0.5 * angle_penalty,
        "contact_bonus": 2.0 * contact_bonus
    }

    return float(total), components