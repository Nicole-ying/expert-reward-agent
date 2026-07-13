def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 主学习信号：接近目标的进度 ---
    # 使用当前和下一步到目标 (0,0) 的欧氏距离差
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = dist_obs - dist_next  # 靠近为正，远离为负

    # --- 稳定和平滑约束 ---
    # 惩罚过大的速度、偏角和角速度，促使平稳着陆
    vel_penalty = 0.05 * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = 0.1 * abs(next_obs[4])
    angvel_penalty = 0.01 * abs(next_obs[5])
    stability_penalty = -(vel_penalty + angle_penalty + angvel_penalty)

    # --- 任务完成近似信号：软着陆奖励 (proxy) ---
    # 当双腿同时接触平台、位置靠近中心、速度足够小、姿态接近水平时给予一次性奖励
    landing_bonus = 0.0
    both_legs_down = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    near_center = (abs(next_obs[0]) < 0.2) and (abs(next_obs[1]) < 0.2)
    slow_enough = (abs(next_obs[2]) < 0.2) and (abs(next_obs[3]) < 0.2)
    upright = abs(next_obs[4]) < 0.1

    if both_legs_down and near_center and slow_enough and upright:
        landing_bonus = 10.0

    # 总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus

    # 组件字典（只包含被加到 total_reward 的项）
    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_bonus': landing_bonus
    }

    return float(total_reward), components