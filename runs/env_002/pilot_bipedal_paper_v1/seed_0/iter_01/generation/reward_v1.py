def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度奖励
    forward_velocity = next_obs[2]
    w_forward = 1.0
    forward_reward = w_forward * forward_velocity

    # 稳定性约束：惩罚身体倾角、角速度和剧烈垂直振荡
    hull_angle = next_obs[0]
    hull_angular_vel = next_obs[1]
    vertical_vel = next_obs[3]

    w_angle = 0.5
    w_angvel = 0.1
    w_vvel = 0.2
    stability_cost = (
        -w_angle * abs(hull_angle)
        - w_angvel * abs(hull_angular_vel)
        - w_vvel * abs(vertical_vel)
    )

    total_reward = forward_reward + stability_cost

    components = {
        "forward_reward": forward_reward,
        "stability_cost": stability_cost,
    }

    return float(total_reward), components