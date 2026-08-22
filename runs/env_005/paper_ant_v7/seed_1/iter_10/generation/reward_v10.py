def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测拆解
    body_z = obs[0]
    quat_x, quat_y = obs[2], obs[3]
    body_x_velocity = obs[13]

    # 计算躯干对世界 z 轴的投影（直立度，1 为完全直立）
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 健康门控因子 ----------
    height_factor = max(0.0, (body_z - 0.2) / 0.15)      # 0.35→1, 0.2→0
    height_factor = min(1.0, height_factor)

    upright_factor = max(0.0, (body_up_z - 0.7) / 0.2)  # 0.9→1, 0.7→0
    upright_factor = min(1.0, upright_factor)

    # 门控乘积
    health_gate = height_factor * upright_factor

    # ---------- 主奖励：门控前进速度 ----------
    forward_speed_reward = 0.5 * (body_x_velocity ** 2)
    gated_forward_reward = forward_speed_reward * health_gate

    # ---------- 新增：生存奖励（直接鼓励保持高度与直立）----------
    survival_reward = 0.03 * height_factor + 0.04 * upright_factor

    total_reward = gated_forward_reward + survival_reward

    components = {
        "forward_speed_reward": gated_forward_reward,
        "survival_reward": survival_reward,
        "height_factor": height_factor,
        "upright_factor": upright_factor
    }

    return float(total_reward), components