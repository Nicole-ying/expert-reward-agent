def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward
    # 计算当前位置到目标（0,0）的距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta  # 正奖励表示更接近目标

    # 稳定约束：stability_penalty
    # 惩罚高速、大姿态角和大角速度
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.5 * abs(next_obs[4])  # 姿态角惩罚
    angular_vel_penalty = 0.2 * abs(next_obs[5])  # 角速度惩罚
    speed_penalty = 0.1 * speed  # 速度惩罚
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # 任务完成 proxy：soft_landing_proxy
    # 当接近目标、低速、小姿态角且双接触时给予小奖励
    near_target = next_dist < 0.5
    low_speed = speed < 0.5
    stable_angle = abs(next_obs[4]) < 0.3
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 动作代价：energy_penalty（小权重）
    # 惩罚使用引擎的动作（action 1,2,3）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -0.05 * engine_use

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