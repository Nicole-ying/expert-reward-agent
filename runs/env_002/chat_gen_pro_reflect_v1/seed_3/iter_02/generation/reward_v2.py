def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号（只奖励正向水平速度）
    #    w_up=2.0，凸化平方，拒绝负速度
    # ============================================================
    horizontal_vel = obs[2]
    forward_progress = 2.0 * (max(0.0, horizontal_vel) ** 2)

    # ============================================================
    # 2. gait_quality gate: 步态交替质量作为乘性门控
    #    使用 leg1_contact，leg2_contact 计算交替程度
    #    值域 [0,1]，1 表示两脚接触状态完全相反
    # ============================================================
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    gait_quality = 1.0 - abs(leg1_contact - leg2_contact)

    # 将步态质量作为 0.5~1.0 的乘性因子乘到前进奖励上
    forward_with_gait = forward_progress * (0.5 + 0.5 * gait_quality)

    # ============================================================
    # 3. balance_maintenance: 防摔倒软约束（保持原样）
    #    轻微惩罚倾斜角与角速度
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    balance_penalty = -0.5 * (hull_angle ** 2) - 0.1 * (hull_angular_vel ** 2)

    # ============================================================
    # 4. both_off_ground_penalty: 防止跳跃/双腿腾空（保持原样）
    # ============================================================
    contact_sum = leg1_contact + leg2_contact
    both_off_ground_penalty = -0.2 * (1.0 - contact_sum) if contact_sum < 0.5 else 0.0

    # 组合总奖励（不再包含独立的交替奖励）
    total_reward = forward_with_gait + balance_penalty + both_off_ground_penalty

    components = {
        "forward_with_gait": forward_with_gait,
        "balance_penalty": balance_penalty,
        "both_off_ground_penalty": both_off_ground_penalty,
        "gait_quality": gait_quality,           # 纯诊断，不影响奖励
        "forward_progress_raw": forward_progress # 纯诊断
    }

    return float(total_reward), components