def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 权重（可根据实际训练表现调整）
    w_forward = 1.0
    w_angle = 0.5
    w_vertical_vel = 0.1
    w_angular_vel = 0.05

    # 主学习信号：前进速度奖励
    forward_vel = next_obs[2]  # 水平速度，正值表示向前
    forward_reward = w_forward * forward_vel

    # 稳定/安全约束：抑制身体倾斜、跳动和高速旋转
    angle = abs(next_obs[0])           # 身体相对于竖直的角度
    vertical_vel = abs(next_obs[3])    # 上下跳动速度
    angular_vel = abs(next_obs[1])     # 身体转动角速度
    stability_penalty = - (w_angle * angle + w_vertical_vel * vertical_vel + w_angular_vel * angular_vel)

    total_reward = forward_reward + stability_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty
    }
    return float(total_reward), components