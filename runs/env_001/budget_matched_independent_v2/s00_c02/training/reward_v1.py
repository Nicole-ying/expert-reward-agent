def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract signals from next_obs (state after taking action)
    x_pos = next_obs[0]       # horizontal position relative to target pad
    y_pos = next_obs[1]       # vertical position relative to pad height
    x_vel = next_obs[2]       # horizontal velocity
    y_vel = next_obs[3]       # vertical velocity
    angle = next_obs[4]       # body orientation angle
    ang_vel = next_obs[5]     # angular velocity
    left_contact = next_obs[6]  # left support contact flag (0 or 1)
    right_contact = next_obs[7] # right support contact flag (0 or 1)
    
    # Distance to target (target is at origin)
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    
    # Speed magnitude
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    
    # Whether both legs are on the pad (settled condition)
    both_contact = left_contact * right_contact  # 1 if both contact, 0 otherwise
    
    # 1. Distance reward: encourage approaching the target
    # Exponential decay based on distance, scaled to be in [0, 1]
    distance_reward = 2.718281828 ** (-2.0 * distance)
    
    # 2. Velocity penalty: encourage reducing speed when near target
    # Use sigmoid-like weighting to only penalize speed when close
    near_target = 2.718281828 ** (-3.0 * distance)
    velocity_penalty = -0.5 * speed * near_target
    
    # 3. Orientation reward: encourage upright orientation (angle=0)
    # Penalize deviation from vertical, scaled by angular velocity
    orientation_penalty = -0.3 * (angle ** 2 + 0.1 * ang_vel ** 2)
    
    # 4. Contact reward: reward both legs on the pad
    contact_reward = 2.0 * both_contact
    
    # 5. Fuel efficiency penalty: penalize engine usage
    # Action 2 is main engine, actions 1 and 3 are orientation engines
    engine_used = 1.0 if action in [1, 2, 3] else 0.0
    fuel_penalty = -0.1 * engine_used
    
    # 6. Stability bonus: reward being settled (both contact + low speed + near target)
    stability_bonus = 3.0 * both_contact * (2.718281828 ** (-5.0 * speed)) * (2.718281828 ** (-2.0 * distance))
    
    # Sum all components
    total_reward = distance_reward + velocity_penalty + orientation_penalty + contact_reward + fuel_penalty + stability_bonus
    
    components = {
        "distance_reward": distance_reward,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward,
        "fuel_penalty": fuel_penalty,
        "stability_bonus": stability_bonus
    }
    
    return float(total_reward), components