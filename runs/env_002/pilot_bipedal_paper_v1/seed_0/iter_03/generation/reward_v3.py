def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度奖励
    forward_velocity = next_obs[2]
    w_forward = 1.0
    forward_reward = w_forward * forward_velocity

    # 稳定性约束：惩罚身体倾角、角速度和剧烈垂直振荡
    hull_angle = next_obs[0]
    hull_angular_vel = next_obs[1]
    vertical_vel = next_obs[3]

    w_angle = 1.0
    w_angvel = 0.2
    w_vvel = 0.4
    stability_cost = (
        -w_angle * abs(hull_angle)
        - w_angvel * abs(hull_angular_vel)
        - w_vvel * abs(vertical_vel)
    )

    # 能量效率：轻量动作代价，抑制高力矩抽搐，鼓励平滑高效步态
    w_action = 0.05
    action_cost = -w_action * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2) / 4.0

    total_reward = forward_reward + stability_cost + action_cost

    components = {
        "forward_reward": forward_reward,
        "stability_cost": stability_cost,
        "action_cost": action_cost,
    }

    return float(total_reward), components