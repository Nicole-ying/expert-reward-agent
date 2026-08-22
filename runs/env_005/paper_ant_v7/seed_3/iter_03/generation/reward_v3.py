def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z      = obs[0]
    quat_x      = obs[2]
    quat_y      = obs[3]
    body_x_vel  = obs[13]
    
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)
    
    # --- 前进奖励（保持原有） ---
    upright_gate = max(0.0, body_up_z)
    forward_reward = body_x_vel * upright_gate
    
    # --- 高度惩罚（保持原有） ---
    height_margin_low  = 0.3
    height_margin_high = 0.9
    below_margin = max(0.0, height_margin_low - body_z)
    above_margin = max(0.0, body_z - height_margin_high)
    height_penalty = below_margin + above_margin
    height_reward = -5.0 * height_penalty
    
    # --- 新增：动作幅度惩罚（抑制高能耗、抖动） ---
    action_penalty = -0.01 * sum(action ** 2)
    
    total_reward = forward_reward + height_reward + action_penalty
    
    components = {
        'forward_gated': forward_reward,
        'height_reward': height_reward,
        'action_penalty': action_penalty
    }
    return float(total_reward), components