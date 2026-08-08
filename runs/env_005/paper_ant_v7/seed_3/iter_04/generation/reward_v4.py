def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z      = obs[0]
    quat_x      = obs[2]
    quat_y      = obs[3]
    body_x_vel  = obs[13]
    
    # 直立因子（保持）
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)
    upright_gate = max(0.0, body_up_z)
    
    # 高度健康因子（替换独立惩罚）
    low_factor  = min(1.0, max(0.0, (body_z - 0.2) / 0.15))   # 0.2 -> 0.35
    high_factor = min(1.0, max(0.0, (1.0 - body_z) / 0.15))   # 1.0 -> 0.85
    height_factor = low_factor * high_factor
    
    # 前进奖励（推进 + 直立门控 + 高度门控）
    forward_reward = body_x_vel * upright_gate * height_factor
    
    # 动作惩罚（保持）
    action_penalty = -0.01 * sum(action ** 2)
    
    total_reward = forward_reward + action_penalty
    
    components = {
        'forward_gated_height': forward_reward,
        'action_penalty': action_penalty
    }
    return float(total_reward), components