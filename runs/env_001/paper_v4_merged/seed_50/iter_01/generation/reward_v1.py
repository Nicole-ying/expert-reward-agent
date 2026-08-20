def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 位置距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 进度奖励：距离减小的量（鼓励靠近目标点）
    progress = current_dist - next_dist
    progress_reward = 2.0 * progress

    # 速度惩罚：抑制冲击速度（二次惩罚）
    velocity_penalty = 0.05 * (next_obs[2] ** 2 + next_obs[3] ** 2)

    # 姿态惩罚：抑制大幅倾斜（二次惩罚）
    angle_penalty = 0.1 * (next_obs[4] ** 2)

    # 软着陆近似奖励：同时满足双腿接触、靠近中心、低速、小角度时给予正向信号
    contact = next_obs[6] * next_obs[7]  # 1.0 仅当双腿都接触
    pos_factor = max(0.0, 1.0 - next_dist / 0.5)
    vel_sum = abs(next_obs[2]) + abs(next_obs[3])
    vel_factor = max(0.0, 1.0 - vel_sum / 0.5)
    angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.2)
    soft_landing = 0.5 * contact * pos_factor * vel_factor * angle_factor

    total_reward = progress_reward - velocity_penalty - angle_penalty + soft_landing
    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "soft_landing_proxy": soft_landing
    }
    return float(total_reward), components