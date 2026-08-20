def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 势能塑形：引导靠近目标点
    gamma = 0.99
    # 采用负欧几里得距离作为势能函数
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    shaping_reward = gamma * (-dist_next) - (-dist_current)  # = dist_current - gamma * dist_next

    # 2. 稳定性惩罚：基于下一时刻的速度、姿态、角速度
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    w_vel = 0.05        # 速度项系数
    w_ang = 0.02        # 角速度项系数
    w_angle = 0.05      # 姿态角系数

    stability_penalty = -(
        w_vel * (abs(vel_x) + abs(vel_y)) +
        w_ang * abs(ang_vel) +
        w_angle * abs(angle)
    )

    total_reward = shaping_reward + stability_penalty

    components = {
        "potential_shaping_reward": shaping_reward,
        "stability_penalty": stability_penalty,
        "total_reward": total_reward
    }
    return float(total_reward), components