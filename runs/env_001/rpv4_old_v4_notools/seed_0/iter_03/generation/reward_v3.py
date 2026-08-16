def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: 重新设计 contact_reward，使其在任意腿触地时即可激活，
    放大权重以提供明确的着陆正激励，引导 agent 完成最终下降。
    progress/ orientation/ vel_damping 保持与 v2 相同。
    """
    # 提取信号
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress reward（保持 v2 权重 0.5）
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr
    w_progress = 0.5
    progress_reward = w_progress * progress

    # 2. Orientation penalty（保持不变）
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 3. Soft contact reward（重写：左右脚之和，放大权重，放宽因子）
    contact_sum = left_contact + right_contact          # 0, 1, 或 2
    sigma_angle = 0.3                                   # 放宽角度容忍度
    sigma_vel = 0.5                                     # 放宽速度容忍度
    angle_factor = 2.718281828 ** (- (angle ** 2) / (sigma_angle ** 2))
    vel_factor = 2.718281828 ** (- (vx ** 2 + vy ** 2) / (sigma_vel ** 2))
    soft_contact = contact_sum * angle_factor * vel_factor
    w_contact = 100.0                                    # 大幅提升权重，一次良好着陆约 200
    contact_reward = w_contact * soft_contact

    # 4. 接近地面速度惩罚（保持 v2，阈值仍为 1.0）
    altitude = y_curr
    alt_thresh = 1.0
    gate_alt = max(0.0, 1.0 - altitude / alt_thresh)
    w_vel_damp = 1.0
    vel_damping = -w_vel_damp * gate_alt * (abs(vy) + 0.3 * abs(vx))

    total_reward = progress_reward + orientation_penalty + contact_reward + vel_damping

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward,
        "vel_damping": vel_damping
    }
    return float(total_reward), components