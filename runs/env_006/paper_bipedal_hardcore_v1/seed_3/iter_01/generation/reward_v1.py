def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- 主学习信号：水平前进速度 -----
    horizontal_speed = obs[2]
    progress_reward = 0.3 * horizontal_speed  # 线性正向驱动

    # ----- 稳定/安全约束 -----
    # 1. 身体倾角惩罚 (hinge，只在倾角过大时生效)
    hull_angle = obs[0]
    angle_threshold = 0.8  # 倾角安全阈值
    posture_penalty = -0.12 * max(0.0, abs(hull_angle) - angle_threshold)

    # 2. 身体角速度惩罚 (quadratic，持续抑制旋转)
    hull_angular_vel = obs[1]
    angular_penalty = -0.06 * (hull_angular_vel ** 2)

    # 3. 垂直速度异常惩罚 (quadratic，抑制弹跳或坠落)
    vertical_speed = obs[3]
    vertical_penalty = -0.15 * (vertical_speed ** 2)

    # ----- 汇总奖励 -----
    total_reward = progress_reward + posture_penalty + angular_penalty + vertical_penalty

    components = {
        "progress_reward": progress_reward,
        "posture_penalty": posture_penalty,
        "angular_penalty": angular_penalty,
        "vertical_penalty": vertical_penalty
    }

    return float(total_reward), components