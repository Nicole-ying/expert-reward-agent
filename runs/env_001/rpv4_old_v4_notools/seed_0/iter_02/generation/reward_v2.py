def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: 修复死组件 vel_damping——改为基于高度的速度惩罚，让 agent 在接近地面时减速，
    同时降低 progress 权重以抑制其噪声幅度。
    contact_reward 暂时保留原样，等待 vel_damping 激活后看能否自然触发。
    """
    # 提取信号
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress reward (improvement_delta on Euclidean distance) —— 降权
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # 正值 = 接近
    w_progress = 0.5   # 从 1.0 降低
    progress_reward = w_progress * progress

    # 2. Orientation penalty (保持不变)
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 3. Soft contact reward (保持不变，等待着陆出现)
    contact_flag = left_contact * right_contact
    sigma_angle = 0.1
    sigma_vel = 0.2
    angle_factor = 2.718281828 ** (- (angle ** 2) / (sigma_angle ** 2))
    vel_factor = 2.718281828 ** (- (vx ** 2 + vy ** 2) / (sigma_vel ** 2))
    soft_contact = contact_flag * angle_factor * vel_factor
    w_contact = 2.0
    contact_reward = w_contact * soft_contact

    # 4. 接近地面速度惩罚 (重写 vel_damping) —— 基于高度，不再是全局距离
    altitude = y_curr                   # 假设地面 y=0
    alt_thresh = 1.0                    # 高度 < 1.0 时开始惩罚
    gate_alt = max(0.0, 1.0 - altitude / alt_thresh)   # 0 在高处，1 在地面
    w_vel_damp = 1.0                    # 提升权重以增强信号
    # 主要惩罚垂直速度，也对横向速度适度惩罚
    vel_damping = -w_vel_damp * gate_alt * (abs(vy) + 0.3 * abs(vx))

    total_reward = progress_reward + orientation_penalty + contact_reward + vel_damping

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward,
        "vel_damping": vel_damping
    }
    return float(total_reward), components