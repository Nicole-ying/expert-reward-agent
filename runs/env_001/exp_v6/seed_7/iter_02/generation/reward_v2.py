def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward —— 每一步靠近目标的正向奖励
    dist_current = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_current - dist_next  # 正值表示靠近

    # 稳定与安全约束：系数降低约10倍，使 penalty 不再主导 progress
    # 原系数 0.05/0.1/0.01 → ratio -4.24，agent 不敢动
    # 新系数 0.005/0.01/0.001 → 预期 ratio ~0.4，低于 0.5 安全线
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    stability_penalty = -0.005 * (abs(vel_x) + abs(vel_y)) - 0.01 * abs(angle) - 0.001 * abs(angular_vel)

    # 软着陆 proxy：当接近目标、低速、姿态小且双腿接触时给予一次性小奖励
    landing_bonus = 0.0
    left_contact = next_obs[6] > 0.5
    right_contact = next_obs[7] > 0.5
    if (dist_next < 0.1 and
        (vel_x**2 + vel_y**2) ** 0.5 < 0.2 and
        abs(angle) < 0.2 and
        left_contact and right_contact):
        landing_bonus = 1.0

    total_reward = progress_delta + stability_penalty + landing_bonus

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components