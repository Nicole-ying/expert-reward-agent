def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    # forward_progress: horizontal_velocity (obs[2])
    horizontal_vel = obs[2]
    
    # balance_maintenance: hull_angle (obs[0]), hull_angular_velocity (obs[1])
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    
    # ============================================================
    # Component 1: forward_progress (main learning signal)
    # Use dense_state_signal: positive linear reward for forward velocity
    # Scale: typical horizontal velocity range ~0-5 m/s, target ~2-3 m/s
    # Weight chosen so that at 2 m/s, this component ≈ 2.0
    # ============================================================
    forward_reward = 1.0 * horizontal_vel
    
    # ============================================================
    # Component 2: balance_maintenance (stability constraint)
    # Use dense_state_signal with quadratic penalty for hull_angle deviation
    # hull_angle is in radians, typical stable range ~[-0.3, 0.3]
    # Penalty is mild: at 0.3 rad (~17 deg), penalty ≈ -0.09
    # Also add small penalty for angular velocity to discourage wobbling
    # ============================================================
    angle_penalty = -2.0 * (hull_angle ** 2)
    angular_vel_penalty = -0.1 * (hull_angular_vel ** 2)
    balance_penalty = angle_penalty + angular_vel_penalty
    
    # ============================================================
    # Component 3: soft_health_gate on forward_reward
    # When hull_angle approaches dangerous territory (>0.5 rad), 
    # reduce forward_reward to discourage "rush then fall" behavior
    # Use linear decay gate: gate = max(0, 1 - |hull_angle| / 0.6)
    # At hull_angle=0.3, gate ≈ 0.5; at hull_angle=0.6, gate ≈ 0.0
    # ============================================================
    gate_factor = max(0.0, 1.0 - abs(hull_angle) / 0.6)
    gated_forward = forward_reward * gate_factor
    
    # ============================================================
    # Total reward
    # ============================================================
    total_reward = gated_forward + balance_penalty
    
    # ============================================================
    # Components dict (for debugging and monitoring)
    # ============================================================
    components = {
        'forward_reward': forward_reward,
        'gated_forward': gated_forward,
        'gate_factor': gate_factor,
        'angle_penalty': angle_penalty,
        'angular_vel_penalty': angular_vel_penalty,
        'balance_penalty': balance_penalty
    }
    
    return float(total_reward), components