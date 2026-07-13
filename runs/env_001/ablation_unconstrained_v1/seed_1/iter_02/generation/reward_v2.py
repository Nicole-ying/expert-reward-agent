def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 进度信号：向目标 (0,0) 靠近 ---
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = dist_obs - dist_next

    # 当前到目标的距离
    dist = dist_next

    # --- 距离门控：靠近目标时才关心稳定性 ---
    stability_gate = max(0.0, 1.0 - dist / 2.0)

    # 稳定性代价（系数降低约 10 倍）
    vel_sum = abs(next_obs[2]) + abs(next_obs[3])
    angle_abs = abs(next_obs[4])
    angvel_abs = abs(next_obs[5])
    stability_cost = 0.005 * vel_sum + 0.01 * angle_abs + 0.001 * angvel_abs
    stability_penalty = -stability_gate * stability_cost

    # --- 连续着陆奖励：soft product 替代硬性二值 ---
    both_legs = next_obs[6] * next_obs[7]              # 0 或 1，仅双腿同时触地时为 1
    x_ok = max(0.0, 1.0 - abs(next_obs[0]) / 0.3)      # |x| < 0.3 满分
    y_ok = max(0.0, 1.0 - abs(next_obs[1]) / 0.3)      # |y| < 0.3 满分
    vel_ok = max(0.0, 1.0 - vel_sum / 0.5)              # |vx|+|vy| < 0.5 满分
    angle_ok = max(0.0, 1.0 - angle_abs / 0.3)          # |angle| < 0.3 满分

    landing_quality = both_legs * x_ok * y_ok * vel_ok * angle_ok
    landing_reward = 3.0 * landing_quality

    # --- 腿部接触奖励：降低初始探索门槛 ---
    leg_contact_reward = 1.0 * both_legs * max(0.0, 1.0 - dist / 1.5)

    # --- 总奖励 ---
    total_reward = progress + stability_penalty + landing_reward + leg_contact_reward

    components = {
        'progress': progress,
        'stability_penalty': stability_penalty,
        'landing_reward': landing_reward,
        'leg_contact_reward': leg_contact_reward
    }

    return float(total_reward), components