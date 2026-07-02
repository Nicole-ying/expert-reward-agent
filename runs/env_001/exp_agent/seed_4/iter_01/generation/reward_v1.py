def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- HYPERPARAMETERS ----------
    w_progress = 1.0        # 主学习信号：每步接近目标给予奖励
    w_vel     = 0.01        # 速度惩罚权重（水平+垂直）
    w_angle   = 0.01        # 倾斜角度惩罚权重
    w_angvel  = 0.005       # 角速度惩罚权重
    w_landing = 0.3         # 软着陆代理奖励权重

    dist_thresh   = 0.3      # 距离阈值（认为足够接近目标垫）
    vel_thresh    = 0.3      # 速度阈值（认为几乎静止）
    angle_thresh  = 0.2      # 姿态角阈值（认为接近竖直）
    contact_thresh = 0.5     # 接触标志阈值（二值化 1.0/0.0）
    # --------------------------------

    # --- Progress toward target ---
    dist_obs  = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_obs - dist_next
    progress_reward = w_progress * progress_delta

    # --- Stability penalty ---
    vx, vy = next_obs[2], next_obs[3]
    angle  = next_obs[4]
    angvel = next_obs[5]

    abs_v_sum = abs(vx) + abs(vy)
    abs_angle = abs(angle)
    abs_angvel = abs(angvel)

    stability_penalty = -(w_vel * abs_v_sum + w_angle * abs_angle + w_angvel * abs_angvel)

    # --- Soft landing proxy ---
    near_target = (dist_next < dist_thresh)
    low_speed   = (abs_v_sum < vel_thresh)
    upright     = (abs_angle < angle_thresh)
    both_contact = (next_obs[6] > contact_thresh and next_obs[7] > contact_thresh)

    landing_bonus = w_landing if (near_target and low_speed and upright and both_contact) else 0.0

    # --- Total reward ---
    total_reward = progress_reward + stability_penalty + landing_bonus

    components = {
        "total_reward": total_reward,
        "progress_delta_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": landing_bonus
    }

    return float(total_reward), components