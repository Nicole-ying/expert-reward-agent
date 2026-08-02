def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- 主学习信号：水平前进速度² × 姿态门控 -----
    horizontal_speed = obs[2]
    hull_angle = obs[0]

    # posture_gate: 连续衰减因子，倾角越大前进奖励越小
    posture_gate = 1.0 / (1.0 + 5.0 * abs(hull_angle))
    # 凸化速度奖励：低速时激励弱，高速时激励强，鼓励加速突破
    progress_reward = 0.6 * (horizontal_speed ** 2) * posture_gate

    # ----- 稳定/安全约束 -----
    # 1. 身体角速度惩罚 (quadratic，持续抑制旋转)
    hull_angular_vel = obs[1]
    angular_penalty = -0.06 * (hull_angular_vel ** 2)

    # 2. 垂直速度异常惩罚 (quadratic，弱抑制弹跳)
    vertical_speed = obs[3]
    vertical_penalty = -0.05 * (vertical_speed ** 2)

    # 3. 足部浮空惩罚：双脚同时离地常伴随跳跃或即将摔倒
    leg1_contact = obs[12]
    leg2_contact = obs[13]
    both_airborne = 1.0 if (leg1_contact + leg2_contact == 0.0) else 0.0
    air_penalty = -0.015 * both_airborne

    # ----- 汇总奖励 -----
    total_reward = progress_reward + angular_penalty + vertical_penalty + air_penalty

    components = {
        "progress_reward": progress_reward,
        "posture_gate": posture_gate,
        "angular_penalty": angular_penalty,
        "vertical_penalty": vertical_penalty,
        "air_penalty": air_penalty
    }

    return float(total_reward), components