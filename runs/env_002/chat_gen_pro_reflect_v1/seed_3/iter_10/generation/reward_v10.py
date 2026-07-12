def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 信号提取
    horizontal_vel = obs[2]
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    # LIDAR 前方距离 (obs[14..23])
    min_lidar = min(obs[14], obs[15], obs[16], obs[17], obs[18],
                    obs[19], obs[20], obs[21], obs[22], obs[23])

    # --- 1. 主奖励：前进速度，线性+凸化 ---
    forward = max(0.0, horizontal_vel)
    forward_reward = 5.0 * forward + 2.0 * (forward ** 2)

    # --- 2. 步态效率：交替接触绑定前进速度 ---
    gait_quality = 1.0 - abs(leg1_contact - leg2_contact)
    gait_efficiency = 0.5 * gait_quality * forward

    # --- 3. 姿态健康门控（平方映射，阈值提前到0.3）---
    angle_abs = abs(hull_angle)
    if angle_abs < 0.3:
        balance_gate = 1.0
    elif angle_abs > 0.5:
        balance_gate = 0.1
    else:
        # 平方映射，从0.3开始急剧压低
        balance_gate = 1.0 - 0.9 * ((angle_abs - 0.3) / 0.2) ** 2

    # 角速度惩罚保持
    angular_vel_penalty = -0.05 * (hull_angular_vel ** 2)
    # 移除 posture_penalty
    balance_modulation = (forward_reward + gait_efficiency) * (balance_gate - 1.0) + angular_vel_penalty

    # --- 4. 双脚离地惩罚 ---
    contact_sum = leg1_contact + leg2_contact
    if contact_sum < 0.5:
        both_off_penalty = -0.1 * (1.0 - contact_sum)
    else:
        both_off_penalty = 0.0

    # --- 5. 动作能耗惩罚 ---
    action_penalty = -0.01 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # --- 6. 新增：LIDAR 近场惩罚（避免冲向障碍物）---
    if min_lidar < 0.3:
        lidar_penalty = -0.5 * (0.3 - min_lidar)
    else:
        lidar_penalty = 0.0

    total_reward = (forward_reward + gait_efficiency + balance_modulation +
                    both_off_penalty + action_penalty + lidar_penalty)

    components = {
        "forward_reward": forward_reward,
        "gait_efficiency": gait_efficiency,
        "balance_modulation": balance_modulation,
        "both_off_penalty": both_off_penalty,
        "action_penalty": action_penalty,
        "lidar_penalty": lidar_penalty,
        "gait_quality": gait_quality,
        "balance_gate": balance_gate,
    }
    return float(total_reward), components