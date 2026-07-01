def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward - 奖励每一步更接近目标
    # 目标位置在 (0, 0)，因为 obs[0] 和 obs[1] 是相对于目标平台的坐标
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta  # 权重10，鼓励向目标移动

    # 稳定/安全约束：stability_penalty - 惩罚高速、大姿态角和大角速度
    # 使用 next_obs 的状态，因为奖励基于动作后的结果
    vel_penalty = -0.5 * (next_obs[2] ** 2 + next_obs[3] ** 2)  # 惩罚线速度
    angle_penalty = -0.3 * (next_obs[4] ** 2)  # 惩罚姿态角偏差
    angvel_penalty = -0.2 * (next_obs[5] ** 2)  # 惩罚角速度
    stability_penalty = vel_penalty + angle_penalty + angvel_penalty

    # 任务完成 proxy：soft_landing_proxy - 当接近目标、低速、小姿态角且双接触时给予小奖励
    near_target = next_dist < 0.5
    low_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5 < 0.3
    stable_angle = abs(next_obs[4]) < 0.2
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_bonus = 5.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 效率/动作代价：energy_penalty - 小权重惩罚使用引擎
    # action 0: no_engine, 1: left_orientation, 2: main_engine, 3: right_orientation
    engine_used = 1.0 if action != 0 else 0.0
    energy_penalty = -0.1 * engine_used

    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty

    # 构建组件字典
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components