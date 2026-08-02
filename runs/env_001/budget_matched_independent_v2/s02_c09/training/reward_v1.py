def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract signals from next_obs (post-step state)
    x = next_obs[0]          # horizontal position relative to target
    y = next_obs[1]          # vertical position relative to pad height
    vx = next_obs[2]         # horizontal velocity
    vy = next_obs[3]         # vertical velocity
    angle = next_obs[4]      # body orientation angle
    ang_vel = next_obs[5]    # angular velocity
    left_contact = next_obs[6]   # left support contact flag (0 or 1)
    right_contact = next_obs[7]  # right support contact flag (0 or 1)
    
    # Also extract from obs for delta calculations
    prev_x = obs[0]
    prev_y = obs[1]
    prev_vx = obs[2]
    prev_vy = obs[3]
    
    # ========== Reward Components ==========
    
    # 1. Distance reward: encourage approaching the target (0,0)
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # negative penalty, stronger when far
    
    # 2. Velocity penalty: penalize high speed, especially when close to target
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # Scale penalty by distance: closer = more penalty for speed
    proximity_weight = 1.0 / (distance + 0.5)  # higher when close
    velocity_penalty = -0.05 * speed * proximity_weight
    
    # 3. Orientation reward: encourage upright orientation (angle near 0)
    orientation_penalty = -0.02 * (angle ** 2)  # quadratic penalty for deviation
    
    # 4. Angular velocity penalty: discourage spinning
    angular_penalty = -0.01 * (ang_vel ** 2)
    
    # 5. Contact reward: reward stable contact with both supports
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact
    
    # 6. Progress bonus: reward moving toward target (delta distance reduction)
    prev_distance = (prev_x ** 2 + prev_y ** 2) ** 0.5
    distance_delta = prev_distance - distance  # positive if moving closer
    progress_bonus = 0.2 * max(0, distance_delta)
    
    # 7. Fuel efficiency penalty: penalize engine usage (action 2 = main engine)
    fuel_penalty = -0.05 if action == 2 else 0.0
    
    # 8. Stability bonus: reward being settled (low speed + good orientation + contact)
    is_settled = (speed < 0.1) and (abs(angle) < 0.1) and (abs(ang_vel) < 0.05) and both_contact
    stability_bonus = 1.0 if is_settled else 0.0
    
    # ========== Combine ==========
    total_reward = (distance_reward + velocity_penalty + orientation_penalty + 
                    angular_penalty + contact_reward + progress_bonus + 
                    fuel_penalty + stability_bonus)
    
    components = {
        'distance_reward': distance_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'angular_penalty': angular_penalty,
        'contact_reward': contact_reward,
        'progress_bonus': progress_bonus,
        'fuel_penalty': fuel_penalty,
        'stability_bonus': stability_bonus,
    }
    
    return float(total_reward), components