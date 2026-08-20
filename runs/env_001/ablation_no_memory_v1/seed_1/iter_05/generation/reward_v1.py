def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Helper: 计算当前和目标距离
    dx_obs, dy_obs = obs[0], obs[1]
    dist_obs = (dx_obs**2 + dy_obs**2)**0.5

    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_next = (dx_next**2 + dy_next**2)**0.5

    # 1. 主学习信号：每一步更接近目标
    progress_delta = dist_obs - dist_next

    # 2. 稳定/安全约束：抑制高速、大角度、高角速度
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    stability_penalty = (-0.01 * (abs(vx) + abs(vy))
                         - 0.05 * abs(angle)
                         - 0.01 * abs(ang_vel))

    # 3. 任务完成代理信号：软着陆的多条件组合
    near_target = abs(next_obs[0]) < 0.2 and abs(next_obs[1]) < 0.2
    low_speed = abs(vx) < 0.5 and abs(vy) < 0.5
    stable_angle = abs(angle) < 0.3
    both_contact = next_obs[6] > 0.5 and next_obs[7] > 0.5

    landing_bonus = 10.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 总奖励
    total_reward = progress_delta + stability_penalty + landing_bonus

    # 组件字典（只包含加到 total_reward 的项）
    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components