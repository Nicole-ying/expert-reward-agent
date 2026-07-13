def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Forward velocity reward - main learning signal
    forward_velocity = next_obs[2]
    # Increase coefficient significantly to dominate alive_bonus
    forward_reward = 5.0 * max(0.0, forward_velocity)
    
    # Alive bonus - reduced to avoid dominating
    hull_angle = next_obs[0]
    is_alive = abs(hull_angle) < 1.0
    alive_bonus = 0.1 if is_alive else 0.0
    
    # Stability penalty - bounded form to avoid over-penalizing
    angle_penalty = -0.2 * (hull_angle ** 2)
    angular_vel_penalty = -0.1 * (next_obs[1] ** 2)
    stability_penalty = angle_penalty + angular_vel_penalty
    
    total_reward = forward_reward + alive_bonus + stability_penalty
    
    components = {
        "progress_reward": forward_reward,
        "alive_bonus": alive_bonus,
        "stability_penalty": stability_penalty
    }
    
    return float(total_reward), components