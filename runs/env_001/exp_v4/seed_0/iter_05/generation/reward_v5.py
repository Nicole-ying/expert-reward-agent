def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 1. 主学习信号：progress_delta_reward（保持best的80.0系数） ==========
    current_dist_sq = obs[0]**2 + obs[1]**2
    next_dist_sq = next_obs[0]**2 + next_obs[1]**2
    progress_delta = current_dist_sq - next_dist_sq
    progress_scale = 80.0
    progress_delta_reward = progress_delta * progress_scale

    # ========== 2. 稳定/安全约束：stability_penalty（保持best的低权重） ==========
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]

    speed = (x_vel**2 + y_vel**2) ** 0.5
    speed_penalty_weight = 0.02
    speed_penalty = -speed_penalty_weight * speed

    angle_penalty_weight = 0.01
    angle_penalty = -angle_penalty_weight * abs(body_angle)

    angular_vel_penalty_weight = 0.005
    angular_vel_penalty = -angular_vel_penalty_weight * abs(angular_vel)

    stability_penalty = speed_penalty + angle_penalty + angular_vel_penalty

    # ========== 3. 任务完成 proxy：soft_landing_proxy（revert到best的收紧阈值和权重） ==========
    # 收紧条件：0.3/0.3/0.2，权重3.0，避免contact_reward_hacking
    near_target_threshold = 0.3
    low_speed_threshold = 0.3
    stable_angle_threshold = 0.2

    near_target = next_dist_sq < near_target_threshold**2
    low_speed = speed < low_speed_threshold
    stable_angle = abs(body_angle) < stable_angle_threshold
    both_contact = next_obs[6] > 0.5 and next_obs[7] > 0.5

    # 连续塑形：条件越严格满足，奖励越大
    soft_landing_bonus_weight = 3.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_proxy = soft_landing_bonus_weight
    elif near_target and low_speed and stable_angle:
        soft_landing_proxy = soft_landing_bonus_weight * 0.5
    elif near_target and low_speed:
        soft_landing_proxy = soft_landing_bonus_weight * 0.2
    else:
        soft_landing_proxy = 0.0

    # ========== 4. 动作代价（保持小权重） ==========
    action_penalty_weight = 0.005
    if action == 0:
        action_penalty = 0.0
    else:
        action_penalty = -action_penalty_weight

    # ========== 组合总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy + action_penalty

    # ========== 构建 components dict ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "action_penalty": action_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components