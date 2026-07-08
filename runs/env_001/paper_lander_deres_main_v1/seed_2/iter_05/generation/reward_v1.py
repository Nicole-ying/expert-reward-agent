def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 主学习信号: 距离减少奖励 (progress_reward) ---
    # 当前距离
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    # 下一步距离
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = dist_current - dist_next  # 正值 = 靠近目标

    # --- 稳定/安全约束: 姿态惩罚 ---
    # 惩罚角度偏离水平（obs[4] 接近 0 为好）
    orientation_penalty = -0.3 * (obs[4] ** 2)

    # --- 任务完成近似信号 (landing_bonus) ---
    # 当满足多重着陆条件时给予额外奖励
    # 条件: 接近中心, 高度极低, 速度很小, 姿态水平, 双腿同时触地
    landing_bonus = 0.0
    if (abs(next_obs[0]) < 0.5 and
        next_obs[1] < 0.3 and
        abs(next_obs[2]) < 0.5 and
        abs(next_obs[3]) < 0.5 and
        abs(next_obs[4]) < 0.2 and
        next_obs[6] == 1.0 and
        next_obs[7] == 1.0):
        landing_bonus = 5.0

    # --- 总奖励 ---
    total_reward = progress_reward + orientation_penalty + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components