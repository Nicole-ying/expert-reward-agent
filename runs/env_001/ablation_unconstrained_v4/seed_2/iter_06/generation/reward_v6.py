def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 读取状态 ----------
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 辅助量 ----------
    dist_old = (x_old**2 + y_old**2)**0.5
    dist_new = (x_new**2 + y_new**2)**0.5
    any_contact = max(left_contact, right_contact)
    double_contact = left_contact * right_contact

    # 下降速度因子：要求 agent 必须向下运动才能获取地面相关奖励
    descent_speed = max(0.0, -vy_new)      # 正值表示向下
    descent_factor = min(1.0, descent_speed / 0.2)  # 平滑门控，下降速度 < 0.2 时渐变

    # ---------- 下降进度（密集引导）----------
    w_progress = 2.0
    w_y_progress = 5.0
    progress = w_progress * (dist_old - dist_new)
    y_progress = w_y_progress * (y_old - y_new)

    # ---------- 高度近地奖励（必须伴随下降）----------
    w_height_near = 2.0
    ground_threshold = 0.8
    height_near_raw = max(0.0, ground_threshold - y_new)
    height_near_reward = w_height_near * height_near_raw * descent_factor

    # ---------- 水平对准（下降时生效）----------
    height_threshold = 0.8
    height_gate = max(0.0, 1.0 - y_new / height_threshold)
    alignment_raw = 1.0 / (1.0 + x_new**2)
    w_alignment = 0.5
    alignment = w_alignment * alignment_raw * height_gate * descent_factor

    # ---------- 接触奖励 ----------
    w_any_contact = 2.0
    any_contact_reward = w_any_contact * any_contact

    # ---------- 双腿稳定着陆奖励 ----------
    w_stable = 30.0
    vx_tol, vy_tol, angle_tol, angvel_tol = 0.2, 0.2, 0.1, 0.1
    stability = (max(0.0, 1.0 - abs(vx_new)/vx_tol) *
                 max(0.0, 1.0 - abs(vy_new)/vy_tol) *
                 max(0.0, 1.0 - abs(angle_new)/angle_tol) *
                 max(0.0, 1.0 - abs(angvel_new)/angvel_tol))
    stable_contact_reward = w_stable * double_contact * stability

    # ---------- 水平速度轻惩罚（辅助对准）----------
    w_horiz_penalty = 0.1
    horizontal_penalty = -w_horiz_penalty * (vx_new**2)

    # ---------- 软安全约束（hinge替代二次）----------
    w_vel = 0.05
    w_att = 0.05
    vx_max, vy_max = 0.5, 0.5
    angle_max, angvel_max = 0.2, 0.2

    excess_vx = max(0.0, abs(vx_new) - vx_max)
    excess_vy = max(0.0, abs(vy_new) - vy_max)
    vel_penalty = -w_vel * (excess_vx**2 + excess_vy**2)

    excess_angle = max(0.0, abs(angle_new) - angle_max)
    excess_angvel = max(0.0, abs(angvel_new) - angvel_max)
    att_penalty = -w_att * (excess_angle**2 + excess_angvel**2)

    # ---------- 总奖励 ----------
    total_reward = (progress + y_progress +
                    height_near_reward + alignment +
                    any_contact_reward + stable_contact_reward +
                    horizontal_penalty + vel_penalty + att_penalty)

    components = {
        'progress': progress,
        'y_progress': y_progress,
        'height_near_reward': height_near_reward,
        'alignment': alignment,
        'any_contact_reward': any_contact_reward,
        'stable_contact_reward': stable_contact_reward,
        'horizontal_penalty': horizontal_penalty,
        'vel_penalty': vel_penalty,
        'att_penalty': att_penalty
    }
    return float(total_reward), components