def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 主学习信号：距离改善 × 姿态门控（保持不变）
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next

    angle_abs = abs(next_obs[4])
    gate      = 1.0 / (1.0 + 5.0 * angle_abs)
    shaped_progress = 5.0 * progress_raw * gate

    # 2. 安全约束：角速度轻量惩罚（保持不变）
    ang_vel_penalty = -0.1 * (next_obs[5] ** 2)

    # 3. 着陆奖励：软转移事件（替代持续状态奖励）
    # 检测腿部接触的增量：接触增加即给分，完全接触后增量归零自然防刷
    contact_now = next_obs[6] * next_obs[7]
    contact_before = obs[6] * obs[7]
    new_contact = contact_now - contact_before
    if new_contact < 0.0:
        new_contact = 0.0

    # 着陆质量因子：靠近目标、直立、低速
    near_target = 1.0 / (1.0 + 3.0 * abs(next_obs[0]))
    upright = 1.0 / (1.0 + 3.0 * angle_abs)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    low_speed = 1.0 / (1.0 + speed)

    landing_bonus = 15.0 * new_contact * near_target * upright * low_speed

    total_reward = shaped_progress + ang_vel_penalty + landing_bonus
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_bonus": landing_bonus
    }
    return (float(total_reward), components)