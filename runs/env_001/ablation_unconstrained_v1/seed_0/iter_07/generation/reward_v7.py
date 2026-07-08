def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current state from obs
    cx_pos = obs[0]
    cy_pos = obs[1]
    
    # Next state from next_obs
    nx_pos = next_obs[0]
    ny_pos = next_obs[1]
    nx_vel = next_obs[2]
    ny_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # 1. Progress-based distance reward (state_to_improvement transform)
    #    Only rewards actual progress toward the target — eliminates proxy farming.
    old_dist = (cx_pos ** 2 + cy_pos ** 2) ** 0.5
    new_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5
    r_progress = 10.0 * max(0.0, old_dist - new_dist)
    
    # 2. Small persistent proximity for terminal gradient maintenance
    #    Keeps a weak pull toward target when progress stalls near zero.
    r_proximity = 0.3 / (1.0 + new_dist)
    
    # 3. Stability penalty — reduced coefficients to match new reward scale
    r_stability = -(
        0.005 * (abs(nx_vel) + abs(ny_vel)) +
        0.05 * abs(body_angle) +
        0.025 * abs(ang_vel)
    )
    
    # 4. Landing proxy with original tight thresholds (restored from best)
    #    Only activates when genuinely close to a soft landing.
    proximity = max(0.0, 1.0 - new_dist / 0.5)
    stillness = max(0.0, 1.0 - (abs(nx_vel) + abs(ny_vel)) / 0.4)
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.3)
    contact   = (left_contact + right_contact) / 2.0
    r_landing = 15.0 * proximity * stillness * upright * (0.1 + 0.9 * contact)
    
    total_reward = r_progress + r_proximity + r_stability + r_landing
    
    components = {
        "progress_reward": r_progress,
        "proximity_reward": r_proximity,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }
    
    return float(total_reward), components