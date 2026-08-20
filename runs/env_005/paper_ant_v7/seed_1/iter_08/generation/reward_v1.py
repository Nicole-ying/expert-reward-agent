def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测拆解
    body_z = obs[0]
    quat_w, quat_x, quat_y, quat_z = obs[1], obs[2], obs[3], obs[4]
    body_x_velocity = obs[13]
    body_y_velocity = obs[14]    # 可能用于横向约束，但v1先不加入
    body_yaw_velocity = obs[18]  # 可能用于偏航约束，但v1先不加入

    next_body_z = next_obs[0]

    # 计算 body_up_z：躯干对世界 z 轴的投影
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 1. 主学习信号：前向速度奖励（凸化形式） ----------
    # 使用平方形式，在 signal 较大时提供更强梯度，推动 agent 从低速平台突破
    forward_speed_reward = 0.5 * (body_x_velocity ** 2)

    # ---------- 2. 健康/安全约束 ----------
    # 2a. 身体高度合页惩罚
    # 安全区域 [0.25, 0.95]，在边界附近开始软惩罚；阈值设在终止边界的 60-80%
    z_lower = 0.25
    z_upper = 0.95
    z_lower_hinge = max(0.0, z_lower - body_z)       # body_z 低于 0.25 时 >0
    z_upper_hinge = max(0.0, body_z - z_upper)       # body_z 高于 0.95 时 >0
    height_penalty = -0.3 * (z_lower_hinge**2 + z_upper_hinge**2)

    # 2b. 直立姿态惩罚（二次形式）
    # body_up_z 的理想范围为 0.85~1.0，中心 1.0
    # 使用 (1 - body_up_z)**2；当 body_up_z 在 0.8 左右时 penalty 约 0.04，在可接受范围
    upright_penalty = -0.05 * ((1.0 - body_up_z) ** 2)

    # ---------- 总奖励 ----------
    total_reward = forward_speed_reward + height_penalty + upright_penalty

    components = {
        "forward_speed_reward": forward_speed_reward,
        "height_penalty": height_penalty,
        "upright_penalty": upright_penalty
    }

    return float(total_reward), components