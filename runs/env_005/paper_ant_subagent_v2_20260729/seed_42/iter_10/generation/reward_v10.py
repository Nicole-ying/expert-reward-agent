def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Key signals from next_obs
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    
    # uprightness: body_up_z = 1 - 2*(qx^2 + qy^2), ∈[-1,1] (1 = perfectly upright)
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    
    forward_vel = next_obs[13]
    lateral_vel = next_obs[14]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    
    # ---- Soft health gates ----
    # Height gate: safe zone [0.35, 0.9], danger margins [0.2, 1.0]
    lower_safe = 0.35
    lower_danger = 0.2
    upper_safe = 0.9
    upper_danger = 1.0
    gate_lower = min(1.0, max(0.0, (body_z - lower_danger) / (lower_safe - lower_danger)))
    gate_upper = min(1.0, max(0.0, (upper_danger - body_z) / (upper_danger - upper_safe)))
    height_gate = min(gate_lower, gate_upper)
    
    # Uprightness gate: use body_up_z directly, clamp negatives to zero
    uprightness_gate = max(0.0, body_up_z)
    
    # ---- Main progress reward ----
    # Forward velocity scaled by both health gates
    forward_reward = 2.0 * forward_vel * height_gate * uprightness_gate
    
    # ---- Light constraints ----
    lateral_penalty = -0.1 * max(0.0, abs(lateral_vel) - 0.5)
    roll_vel_penalty = -0.005 * roll_vel**2
    pitch_vel_penalty = -0.005 * pitch_vel**2
    
    # Joint velocity smoothing (very light)
    joint_vel_penalty = -0.0002 * sum(next_obs[19:27]**2)   # indices 19..26
    
    # Action magnitude (energy cost, tiny)
    action_penalty = -0.0005 * sum(a**2 for a in action)
    
    total = (forward_reward +
             lateral_penalty +
             roll_vel_penalty +
             pitch_vel_penalty +
             joint_vel_penalty +
             action_penalty)
    
    components = {
        'forward_reward': forward_reward,
        'height_gate': height_gate,
        'uprightness_gate': uprightness_gate,
        'lateral_penalty': lateral_penalty,
        'roll_vel_penalty': roll_vel_penalty,
        'pitch_vel_penalty': pitch_vel_penalty,
        'joint_vel_penalty': joint_vel_penalty,
        'action_penalty': action_penalty
    }
    
    return float(total), components