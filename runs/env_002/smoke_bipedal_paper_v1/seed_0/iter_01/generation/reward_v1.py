def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 奖励权重（可调，当前基于典型环境尺度）
    progress_weight = 1.0
    stability_angle_weight = 1.0
    stability_angvel_weight = 0.1

    # 主学习信号：前进速度
    forward_velocity = next_obs[2]          # 水平方向速度 (m/s)
    progress_reward = progress_weight * forward_velocity

    # 稳定约束：身体倾斜与旋转速度
    hull_angle = next_obs[0]                # 身体与竖直方向夹角 (rad)
    hull_ang_vel = next_obs[1]              # 身体角速度 (rad/s)
    stability_penalty = -(stability_angle_weight * abs(hull_angle) +
                          stability_angvel_weight * abs(hull_ang_vel))

    total_reward = progress_reward + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components