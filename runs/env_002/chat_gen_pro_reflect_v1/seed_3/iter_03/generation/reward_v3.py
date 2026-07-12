def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号（不受门控压制）
    #    w=2.0，凸化平方，拒绝负速度
    # ============================================================
    horizontal_vel = obs[2]
    forward_progress = 2.0 * (max(0.0, horizontal_vel) ** 2)

    # ============================================================
    # 2. gait_bonus: 步态交替质量作为加性奖励（替代乘性门控）
    #    使用 leg1_contact，leg2_contact 计算交替程度
    #    值域 [0, 1]，1 表示两脚接触状态完全相反
    #    bonus 系数 0.3，per-step ≤ 0.3，探索阶段也能获得正向引导
    # ============================================================
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    gait_quality = 1.0 - abs(leg1_contact - leg2_contact)
    gait_bonus = 0.3 * gait_quality

    # ============================================================
    # 3. balance_maintenance: 防摔倒软约束
    #    惩罚倾斜角与角速度
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    balance_penalty = -0.5 * (hull_angle ** 2) - 0.1 * (hull_angular_vel ** 2)

    # ============================================================
    # 4. both_off_ground_penalty: 防止跳跃/双腿腾空
    # ============================================================
    contact_sum = leg1_contact + leg2_contact
    both_off_ground_penalty = -0.2 * (1.0 - contact_sum) if contact_sum < 0.5 else 0.0

    # 组合总奖励：前进主信号 + 步态加成 + 平衡/腾空惩罚
    total_reward = forward_progress + gait_bonus + balance_penalty + both_off_ground_penalty

    components = {
        "forward_progress": forward_progress,
        "gait_bonus": gait_bonus,
        "balance_penalty": balance_penalty,
        "both_off_ground_penalty": both_off_ground_penalty,
        "gait_quality": gait_quality  # 纯诊断
    }

    return float(total_reward), components