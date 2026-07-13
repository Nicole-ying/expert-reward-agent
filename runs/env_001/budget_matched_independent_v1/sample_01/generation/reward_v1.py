def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前观测
    x_old = float(obs[0])
    y_old = float(obs[1])
    # 下一观测
    x_new = float(next_obs[0])
    y_new = float(next_obs[1])
    vx = float(next_obs[2])
    vy = float(next_obs[3])
    angle = float(next_obs[4])
    ang_vel = float(next_obs[5])

    # ---------- 1. 位置进步奖励（主学习信号）----------
    dist_old = (x_old ** 2 + y_old ** 2) ** 0.5
    dist_new = (x_new ** 2 + y_new ** 2) ** 0.5
    progress = dist_old - dist_new   # 接近目标为正

    # ---------- 2. 稳定性软门控（乘在主奖励上）----------
    speed = (vx ** 2 + vy ** 2) ** 0.5
    abs_angle = abs(angle)

    k_speed = 0.5
    k_angle = 1.0
    gate_speed = 1.0 / (1.0 + k_speed * speed)
    gate_angle = 1.0 / (1.0 + k_angle * abs_angle)
    gate = gate_speed * gate_angle

    # ---------- 3. 辅助约束 ----------
    w_angle = 0.1          # 角度平方惩罚权重
    angle_penalty = -w_angle * (angle ** 2)

    w_angvel = 0.01        # 角速度平方惩罚权重
    angular_vel_penalty = -w_angvel * (ang_vel ** 2)

    # ---------- 合成总奖励 ----------
    w_progress = 1.0
    gated_progress = w_progress * progress * gate
    total_reward = gated_progress + angle_penalty + angular_vel_penalty

    components = {
        "gated_progress": float(gated_progress),
        "angle_penalty": float(angle_penalty),
        "angular_vel_penalty": float(angular_vel_penalty)
    }

    return float(total_reward), components