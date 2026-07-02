def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 读取当前状态
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 从 obs 读取上一步位置，计算距离变化
    px = obs[0]
    py = obs[1]

    prev_dist = (px**2 + py**2) ** 0.5
    curr_dist = (x**2 + y**2) ** 0.5

    # === 1. 主学习信号：进度增量（向目标靠近为正） ===
    progress = prev_dist - curr_dist

    # === 2. 连续着陆质量信号 ===
    # 接近度
    proximity = 1.0 / (1.0 + 5.0 * curr_dist)
    # 速度因子
    speed_val = (vx**2 + vy**2) ** 0.5
    speed_factor = 1.0 / (1.0 + 3.0 * speed_val)
    # 姿态因子
    upright_factor = 1.0 / (1.0 + 3.0 * abs(angle))
    # 接触加成
    contact_bonus = 0.5 + 0.5 * (left_contact + right_contact) / 2.0
    # 连续乘积
    landing_quality = proximity * speed_factor * upright_factor * contact_bonus

    # === 3. 稳定性惩罚：距离门控 ===
    gate = max(0.0, 1.0 - curr_dist / 2.0)
    stability_penalty = -gate * 0.02 * (abs(vx) + abs(vy) + abs(angular_vel))

    # === 阶段权重：早期重探索(progress)，后期重精度(landing_quality) ===
    t = training_progress  # 0.0 → 1.0
    w_progress = 1.0 - 0.8 * t      # 1.0 → 0.2
    w_landing = 0.2 + 0.8 * t       # 0.2 → 1.0

    total_reward = w_progress * progress + w_landing * landing_quality + stability_penalty

    # 注意：components 只放总公式中直接出现的变量，不含 total_reward
    components = {
        "progress": progress,
        "landing_quality": landing_quality,
        "stability_penalty": stability_penalty,
        "w_progress": w_progress,
        "w_landing": w_landing,
    }
    return float(total_reward), components