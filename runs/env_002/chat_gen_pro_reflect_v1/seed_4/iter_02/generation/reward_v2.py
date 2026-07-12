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
    # CHANGED: hinge penalty for hull_angle instead of quadratic
    # hinge_threshold = 0.3 rad (~17 deg), which is 60% of termination boundary 0.5 rad
    # Penalty kicks in only when |hull_angle| > 0.3, linearly to -0.8 at 0.5 rad
    # This provides stronger gradient near danger zone while not penalizing normal sway
    # Angular velocity penalty kept mild to discourage fast wobbling
    # ============================================================
    angle_threshold = 0.3  # rad, ~17 deg — 60% of 0.5 rad termination boundary
    angle_deviation = max(0.0, abs(hull_angle) - angle_threshold)
    angle_penalty = -4.0 * angle_deviation  # at |hull_angle|=0.5: penalty ≈ -0.8
    
    angular_vel_penalty = -0.1 * (hull_angular_vel ** 2)
    balance_penalty = angle_penalty + angular_vel_penalty
    
    # ============================================================
    # Component 3: soft_health_gate on forward_reward
    # CHANGED: hinge-based gate with relaxed start threshold (0.25 rad)
    # gate = max(0, 1 - (|hull_angle| - 0.25) / (0.5 - 0.25))
    # At hull_angle <= 0.25: gate = 1.0 (full forward reward)
    # At hull_angle = 0.35: gate ≈ 0.6
    # At hull_angle = 0.5: gate = 0.0
    # ============================================================
    gate_lower = 0.25  # no reduction below this angle
    gate_upper = 0.5   # full cutoff at this angle (termination boundary)
    gate_raw = max(0.0, 1.0 - (abs(hull_angle) - gate_lower) / (gate_upper - gate_lower))
    gate_factor = gate_raw  # can't collapse at safe region: at |angle|=0.3, gate≈0.8
    
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
        'balance_penalty': balance_penalty,
        'angle_deviation': angle_deviation
    }
    
    return float(total_reward), components