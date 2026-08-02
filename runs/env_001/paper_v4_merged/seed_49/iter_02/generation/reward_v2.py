def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 权重和阈值
    w_progress = 1.0
    w_angle = 0.5
    w_angvel = 0.1
    w_soft_land = 2.0
    w_eff = 0.02

    angle_thresh = 0.3   # rad
    angvel_thresh = 1.0  # rad/s
    max_speed_land = 1.0 # 着陆容许最大合速度
    max_angle_land = 0.5 # 着陆容许最大倾角 rad
    max_safe_vy = 0.5    # 安全下降的垂直速度阈值（m/s）

    # 距离进展（步间距离减少量），加入安全下降门控
    old_dist = (obs[0]**2 + obs[1]**2)**0.5
    new_dist = (next_obs[0]**2 + next_obs[1]**2)**0.5
    delta_dist = old_dist - new_dist   # 正值表示向目标接近

    vy = next_obs[3]                   # 垂直速度
    downward_speed = -vy if vy < 0.0 else 0.0   # 向下速度
    if downward_speed > max_safe_vy:
        overshoot = downward_speed - max_safe_vy
        gate = max(0.0, 1.0 - overshoot / max_safe_vy)
    else:
        gate = 1.0
    progress = w_progress * delta_dist * gate

    # 姿态稳定性（hinge 惩罚）
    angle = next_obs[4]
    angvel = next_obs[5]
    angle_penalty = -w_angle * max(0.0, abs(angle) - angle_thresh)
    angvel_penalty = -w_angvel * max(0.0, abs(angvel) - angvel_thresh)

    # 软着陆奖励（仅在支撑腿接触时有效）
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    any_contact = 1.0 if (left_contact + right_contact) > 0.5 else 0.0

    speed = (next_obs[2]**2 + next_obs[3]**2)**0.5
    speed_factor = 1.0 - min(1.0, speed / max_speed_land)
    angle_factor = 1.0 - min(1.0, abs(angle) / max_angle_land)
    soft_landing_score = speed_factor * angle_factor
    soft_landing = w_soft_land * soft_landing_score * any_contact

    # 发动机使用惩罚（离散动作每次非零动作）
    eff_penalty = -w_eff * (0.0 if action == 0 else 1.0)

    total_reward = progress + angle_penalty + angvel_penalty + soft_landing + eff_penalty

    components = {
        'progress': progress,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty,
        'soft_landing': soft_landing,
        'efficiency': eff_penalty
    }
    return float(total_reward), components