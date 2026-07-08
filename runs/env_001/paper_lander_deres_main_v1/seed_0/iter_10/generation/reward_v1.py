def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current and next distance to the landing target (target is at (0,0) in relative coordinates)
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    d = (x**2 + y**2) ** 0.5
    nd = (nx**2 + ny**2) ** 0.5
    
    # Main learning signal: dense progress towards the target
    progress_reward = d - nd  # positive when moving closer
    
    # Constraint: penalty for tilting away from upright
    body_angle = abs(next_obs[4])
    attitude_weight = 0.1
    attitude_penalty = -attitude_weight * body_angle
    
    # Total reward
    total_reward = progress_reward + attitude_penalty
    
    # Components dictionary (only the terms directly summed)
    components = {
        'progress_reward': progress_reward,
        'attitude_penalty': attitude_penalty
    }
    
    return float(total_reward), components