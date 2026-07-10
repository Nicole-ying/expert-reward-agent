def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 前进速度奖励（只奖励正向移动）
    hor_vel = next_obs[2]
    forward_reward = 2.0 * max(0.0, hor_vel)

    # 2. 平衡/稳定惩罚：身体倾角与角速度的平方惩罚
    angle = next_obs[0]
    ang_vel = next_obs[1]
    stability_penalty = -2.0 * (angle ** 2) - 0.5 * (ang_vel ** 2)

    # 3. 节能惩罚：关节力矩平方和
    energy_penalty = -0.005 * (action[0] ** 2 + action[1] ** 2 +
                               action[2] ** 2 + action[3] ** 2)

    total_reward = forward_reward + stability_penalty + energy_penalty

    components = {
        "forward_velocity": forward_reward,
        "stability": stability_penalty,
        "energy": energy_penalty
    }
    return float(total_reward), components