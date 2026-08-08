def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- 主学习信号：水平前进速度 × 姿态门控 -----
    horizontal_speed = obs[2]
    hull_angle = obs[0]

    # posture_gate: 连续衰减因子，倾角越大前进奖励越小
    # 倾角=0 → gate≈1.0; 倾角=0.5 → gate≈0.29; 倾角=0.8 → gate≈0.20
    posture_gate = 1.0 / (1.0 + 5.0 * abs(hull_angle))
    progress_reward = 0.3 * horizontal_speed * posture_gate

    # ----- 稳定/安全约束 -----
    # 1. 身体角速度惩罚 (quadratic，持续抑制旋转)
    hull_angular_vel = obs[1]
    angular_penalty = -0.06 * (hull_angular_vel ** 2)

    # 2. 垂直速度异常惩罚 (quadratic，弱抑制弹跳，系数降低至原 1/3)
    vertical_speed = obs[3]
    vertical_penalty = -0.05 * (vertical_speed ** 2)

    # ----- 汇总奖励 -----
    total_reward = progress_reward + angular_penalty + vertical_penalty

    components = {
        "progress_reward": progress_reward,
        "posture_gate": posture_gate,
        "angular_penalty": angular_penalty,
        "vertical_penalty": vertical_penalty
    }

    return float(total_reward), components