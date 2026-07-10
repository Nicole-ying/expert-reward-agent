def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度（水平速度）
    horizontal_velocity = obs[2]
    # 稳定约束：躯干倾斜角度（保持直立）
    hull_angle = obs[0]

    # 权重设定
    w_forward = 1.0
    w_upright = 0.5

    forward_reward = w_forward * horizontal_velocity
    upright_penalty = -w_upright * (hull_angle ** 2)

    total_reward = forward_reward + upright_penalty

    components = {
        "forward_velocity": forward_reward,
        "upright_penalty": upright_penalty
    }

    return float(total_reward), components