def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    px, py = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    prev_px, prev_py = obs[0], obs[1]

    curr_dist = (px**2 + py**2)**0.5
    prev_dist = (prev_px**2 + prev_py**2)**0.5

    # 主正向驱动：向中心移动的进步奖励
    progress = 5.0 * (prev_dist - curr_dist)

    # 接近目标的正奖励，替代原来的距离惩罚
    proximity = 0.5 / (1.0 + curr_dist)

    # 速度惩罚（降低系数，允许探索加速）
    vel_penalty = -0.1 * (vx**2 + vy**2)

    # 保持直立
    angle_penalty = -0.5 * (angle**2)

    # 抑制过度旋转
    angvel_penalty = -0.02 * (ang_vel**2)

    # 接触奖励（降低幅度，避免落地后数值占主导）
    contact_reward = 0.5 * left_contact + 0.5 * right_contact + 1.0 * (left_contact * right_contact)

    # 稀疏成功奖励，大幅提高以凸显着陆目标
    success_bonus = 10.0 if (left_contact and right_contact and curr_dist < 0.5) else 0.0

    total = (progress + proximity + vel_penalty + angle_penalty +
             angvel_penalty + contact_reward + success_bonus)

    components = {
        "progress": progress,
        "proximity": proximity,
        "vel_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_reward": contact_reward,
        "success_bonus": success_bonus
    }
    return float(total), components