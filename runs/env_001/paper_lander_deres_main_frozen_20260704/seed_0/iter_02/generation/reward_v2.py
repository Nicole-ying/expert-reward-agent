def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # 目标位置（平台中心）定义为 (0,0)，因为 obs[0], obs[1] 是相对偏移
    # ------------------------------------------------------------
    # 1. 主学习信号：朝向目标的进度
    dist_curr = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = 1.0 * (dist_curr - dist_next)  # 正值表示接近目标

    # 2. 稳定性约束：大幅降低系数，使 |penalty/progress| ≈ 0.1
    linear_speed_cost = abs(next_obs[2]) + abs(next_obs[3])
    angle_cost = abs(next_obs[4])
    angular_cost = abs(next_obs[5])

    w_vel = 0.001      # 原 0.01 → 降10倍
    w_angle = 0.001    # 原 0.01 → 降10倍
    w_angvel = 0.0005  # 原 0.005 → 降10倍
    stability_penalty = - w_vel * linear_speed_cost - w_angle * angle_cost - w_angvel * angular_cost

    # 3. 软着陆近似奖励：多条件组合，引导飞行器低速、低角度、双足接触着陆
    near_target = dist_next < 0.5
    low_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5 < 0.5
    stable_angle = abs(next_obs[4]) < 0.1
    both_contacts = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)

    soft_landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contacts) else 0.0

    # 总奖励
    total_reward = progress + stability_penalty + soft_landing_bonus

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }

    return float(total_reward), components