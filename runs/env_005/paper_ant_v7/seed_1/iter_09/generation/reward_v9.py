def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测拆解
    body_z = obs[0]
    quat_w, quat_x, quat_y, quat_z = obs[1], obs[2], obs[3], obs[4]
    body_x_velocity = obs[13]

    # 计算躯干对世界 z 轴的投影（直立度，1 为完全直立）
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 健康门控因子 ----------
    # 高度因子：body_z 安全区 [0.35, +∞)，危险区 (0.2, 0.35)
    height_factor = max(0.0, (body_z - 0.2) / 0.15)      # 0.35→1, 0.2→0
    height_factor = min(1.0, height_factor)              # 裁剪 >1 的情况

    # 直立因子：body_up_z 理想 ≥0.9，宽容至 0.7 开始惩罚
    upright_factor = max(0.0, (body_up_z - 0.7) / 0.2)  # 0.9→1, 0.7→0
    upright_factor = min(1.0, upright_factor)

    # 综合门控：乘积形式，任一因子恶化到 0 直接切断主奖励
    health_gate = height_factor * upright_factor

    # ---------- 主学习信号：前向速度奖励（凸化形式，门控） ----------
    forward_speed_reward = 0.5 * (body_x_velocity ** 2)
    gated_forward_reward = forward_speed_reward * health_gate

    # ---------- 组件输出（旧独立惩罚归零，保留字段供监控） ----------
    total_reward = gated_forward_reward

    components = {
        "forward_speed_reward": gated_forward_reward,   # 门控后的有效前进奖励
        "height_factor": height_factor,                 # 高度门控因子（监控用）
        "upright_factor": upright_factor,               # 直立门控因子（监控用）
        "height_penalty": 0.0,                          # 已被门控替代
        "upright_penalty": 0.0                          # 已被门控替代
    }

    return float(total_reward), components