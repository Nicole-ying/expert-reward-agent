def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 主学习信号：进度差分奖励（接近目标垫中心）
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = dist_obs - dist_next   # 正值表示接近目标

    # 2. 稳定性约束：惩罚高速、大倾角和高角速度
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angular_speed = abs(next_obs[5])

    w_speed = 0.05
    w_angle = 0.05
    w_angvel = 0.05
    stability_penalty = -(w_speed * speed + w_angle * angle_abs + w_angvel * angular_speed)

    # 3. 任务完成代理：软着陆近似奖励
    near_target = (dist_next < 0.2)
    low_speed = (speed < 0.1)
    stable_angle = (angle_abs < 0.1)
    both_contact = (next_obs[6] == 1.0 and next_obs[7] == 1.0)
    soft_landing = 0.2 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 总奖励
    total_reward = progress_delta + stability_penalty + soft_landing

    components = {
        "progress": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing,
        "total_reward": total_reward
    }

    return float(total_reward), components