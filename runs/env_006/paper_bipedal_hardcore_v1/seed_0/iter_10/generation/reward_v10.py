def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # =========================================================================
    # 1. 提取观测信号
    # =========================================================================
    hull_angle = obs[0]
    hull_ang_vel = obs[1]
    horizontal_speed = obs[2]
    leg_1_contact = obs[12]
    leg_2_contact = obs[13]
    # LIDAR 地形预览（显式索引，避免切片）
    lidar_0 = obs[14]
    lidar_1 = obs[15]
    lidar_2 = obs[16]
    lidar_3 = obs[17]
    lidar_4 = obs[18]
    lidar_5 = obs[19]
    lidar_6 = obs[20]
    lidar_7 = obs[21]
    lidar_8 = obs[22]
    lidar_9 = obs[23]

    next_hull_angle = next_obs[0]
    next_hull_ang_vel = next_obs[1]
    next_horizontal_speed = next_obs[2]
    next_leg_1_contact = next_obs[12]
    next_leg_2_contact = next_obs[13]

    # =========================================================================
    # 2. 前向速度奖励 (主学习信号)
    # =========================================================================
    forward_speed = max(0.0, horizontal_speed)
    forward_reward = 1.0 * forward_speed

    # =========================================================================
    # 3. 姿态稳定门
    # =========================================================================
    tilt_safe_bound = 0.3
    tilt_danger_bound = 0.7
    tilt_margin = tilt_danger_bound - tilt_safe_bound

    abs_tilt = abs(hull_angle)
    if abs_tilt <= tilt_safe_bound:
        tilt_gate = 1.0
    elif abs_tilt >= tilt_danger_bound:
        tilt_gate = 0.0
    else:
        tilt_gate = 1.0 - (abs_tilt - tilt_safe_bound) / tilt_margin

    ang_vel_thresh = 2.0
    ang_vel_margin = 4.0
    abs_ang_vel = abs(hull_ang_vel)
    if abs_ang_vel <= ang_vel_thresh:
        ang_vel_factor = 1.0
    elif abs_ang_vel >= ang_vel_thresh + ang_vel_margin:
        ang_vel_factor = 0.3
    else:
        ang_vel_factor = 1.0 - 0.7 * (abs_ang_vel - ang_vel_thresh) / ang_vel_margin

    stability_gate = tilt_gate * ang_vel_factor

    # =========================================================================
    # 4. 接触切换奖励
    # =========================================================================
    leg1_change = abs(next_leg_1_contact - leg_1_contact)
    leg2_change = abs(next_leg_2_contact - leg_2_contact)
    contact_transition_reward = 0.05 * (leg1_change + leg2_change)

    # =========================================================================
    # 5. 能量效率惩罚
    # =========================================================================
    action_sq_sum = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = 0.005 * action_sq_sum

    # =========================================================================
    # 6. 地形粗糙度惩罚 (新增 — 利用 LIDAR 预判崎岖，减速避险)
    # =========================================================================
    lidar_mean = (lidar_0 + lidar_1 + lidar_2 + lidar_3 + lidar_4 +
                  lidar_5 + lidar_6 + lidar_7 + lidar_8 + lidar_9) / 10.0
    lidar_var = (
        (lidar_0 - lidar_mean) ** 2 +
        (lidar_1 - lidar_mean) ** 2 +
        (lidar_2 - lidar_mean) ** 2 +
        (lidar_3 - lidar_mean) ** 2 +
        (lidar_4 - lidar_mean) ** 2 +
        (lidar_5 - lidar_mean) ** 2 +
        (lidar_6 - lidar_mean) ** 2 +
        (lidar_7 - lidar_mean) ** 2 +
        (lidar_8 - lidar_mean) ** 2 +
        (lidar_9 - lidar_mean) ** 2
    ) / 10.0
    roughness = lidar_var ** 0.5
    roughness_penalty = 0.02 * roughness * abs(horizontal_speed)

    # =========================================================================
    # 7. 组合并返回
    # =========================================================================
    gated_forward = forward_reward * stability_gate
    total_reward = gated_forward + contact_transition_reward - energy_penalty - roughness_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_gate": stability_gate,
        "gated_forward": gated_forward,
        "contact_transition_reward": contact_transition_reward,
        "energy_penalty": -energy_penalty,
        "roughness_penalty": -roughness_penalty
    }

    return float(total_reward), components