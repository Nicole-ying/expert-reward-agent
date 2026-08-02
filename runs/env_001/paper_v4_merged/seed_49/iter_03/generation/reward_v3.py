def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 权重与阈值
    w_progress = 1.0
    w_angle = 0.5
    w_angvel = 0.1
    w_soft_land = 2.0
    w_eff = 0.02
    w_failure = 5.0              # 失败惩罚权重

    angle_thresh = 0.3
    angvel_thresh = 1.0
    max_speed_land = 1.0
    max_angle_land = 0.5
    max_safe_vy = 0.5

    # 距离进展 + 安全下降门控
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    delta_dist = old_dist - new_dist

    vy = next_obs[3]
    downward_speed = -vy if vy < 0.0 else 0.0
    if downward_speed > max_safe_vy:
        overshoot = downward_speed - max_safe_vy
        gate = max(0.0, 1.0 - overshoot / max_safe_vy)
    else:
        gate = 1.0
    progress = w_progress * delta_dist * gate

    # 姿态稳定性惩罚
    angle = next_obs[4]
    angvel = next_obs[5]
    angle_penalty = -w_angle * max(0.0, abs(angle) - angle_thresh)
    angvel_penalty = -w_angvel * max(0.0, abs(angvel) - angvel_thresh)

    # 软着陆奖励（接触时）
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    any_contact = 1.0 if (left_contact + right_contact) > 0.5 else 0.0
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    speed_factor = 1.0 - min(1.0, speed / max_speed_land)
    angle_factor = 1.0 - min(1.0, abs(angle) / max_angle_land)
    soft_landing_score = speed_factor * angle_factor
    soft_landing = w_soft_land * soft_landing_score * any_contact

    # 发动机使用惩罚
    eff_penalty = -w_eff * (0.0 if action == 0 else 1.0)

    # 新增：终端失败惩罚，在观测到失败状态时施加
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    # 失败条件：水平越界或高度过低（可能坠地/主体触地）
    if abs(x_pos) > 2.0 or y_pos < 0.1:
        failure_penalty = -w_failure
    else:
        failure_penalty = 0.0

    total_reward = (progress + angle_penalty + angvel_penalty +
                    soft_landing + eff_penalty + failure_penalty)

    components = {
        'progress': progress,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty,
        'soft_landing': soft_landing,
        'efficiency': eff_penalty,
        'failure_penalty': failure_penalty
    }
    return float(total_reward), components