def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号（保持原样，等待 balance 释放后激活）
    # ============================================================
    horizontal_vel = obs[2]
    forward_progress = 2.0 * (max(0.0, horizontal_vel) ** 2)

    # ============================================================
    # 2. gait_bonus: 步态交替质量（保持原样）
    # ============================================================
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    gait_quality = 1.0 - abs(leg1_contact - leg2_contact)
    gait_bonus = 0.3 * gait_quality

    # ============================================================
    # 3. balance_maintenance: 改 hinge 版，允许小幅度前倾探索
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    angle_deviation = abs(hull_angle) - 0.3
    angle_penalty = -1.0 * (max(0.0, angle_deviation) ** 2)
    angular_vel_penalty = -0.05 * (hull_angular_vel ** 2)
    balance_penalty = angle_penalty + angular_vel_penalty

    # ============================================================
    # 4. both_off_ground_penalty: 保留原样（僵尸但无害）
    # ============================================================
    contact_sum = leg1_contact + leg2_contact
    both_off_ground_penalty = -0.2 * (1.0 - contact_sum) if contact_sum < 0.5 else 0.0

    total_reward = forward_progress + gait_bonus + balance_penalty + both_off_ground_penalty

    components = {
        "forward_progress": forward_progress,
        "gait_bonus": gait_bonus,
        "balance_penalty": balance_penalty,
        "both_off_ground_penalty": both_off_ground_penalty,
        "gait_quality": gait_quality
    }

    return float(total_reward), components