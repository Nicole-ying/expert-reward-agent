def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract signals from next_obs (post-step state)
    x_pos = next_obs[0]       # horizontal position relative to target pad
    y_pos = next_obs[1]       # vertical position relative to pad height
    x_vel = next_obs[2]       # horizontal velocity
    y_vel = next_obs[3]       # vertical velocity
    angle = next_obs[4]       # body orientation angle
    ang_vel = next_obs[5]     # angular velocity
    left_contact = next_obs[6]  # left support contact flag (0 or 1)
    right_contact = next_obs[7] # right support contact flag (0 or 1)
    
    # Also extract previous state for velocity change calculation
    prev_x_vel = obs[2]
    prev_y_vel = obs[3]
    
    # ========== Reward Components ==========
    
    # 1. Distance reward: encourage approaching the target (0,0)
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -0.1 * distance  # linear penalty on distance
    
    # 2. Velocity penalty: encourage low speed, especially near target
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # Scale penalty by distance: closer to target = higher penalty for speed
    velocity_penalty = -0.05 * speed * (1.0 + 2.0 / (1.0 + distance + 0.1))
    
    # 3. Orientation reward: encourage upright posture (angle near 0)
    angle_penalty = -0.02 * (angle ** 2)
    
    # 4. Angular velocity penalty: discourage spinning
    ang_vel_penalty = -0.01 * (ang_vel ** 2)
    
    # 5. Contact reward: reward stable contact with both supports
    both_contact = left_contact * right_contact  # 1 if both, 0 otherwise
    contact_reward = 0.5 * both_contact
    
    # 6. Fuel efficiency penalty: penalize engine usage
    # Action 2 is main engine, actions 1 and 3 are orientation engines
    engine_used = 1.0 if action in [1, 2, 3] else 0.0
    fuel_penalty = -0.02 * engine_used
    
    # 7. Progress bonus: reward reducing speed over time (smooth landing)
    # Compare current speed to previous speed
    prev_speed = (prev_x_vel ** 2 + prev_y_vel ** 2) ** 0.5
    speed_change = prev_speed - speed  # positive if slowing down
    progress_reward = 0.03 * max(0, speed_change)
    
    # 8. Settlement bonus: reward being stationary and upright near target
    settled = (distance < 0.3) and (speed < 0.1) and (abs(angle) < 0.1) and (abs(ang_vel) < 0.1)
    settlement_bonus = 1.0 if settled else 0.0
    
    # ========== Combine ==========
    total_reward = (distance_reward + velocity_penalty + angle_penalty + 
                    ang_vel_penalty + contact_reward + fuel_penalty + 
                    progress_reward + settlement_bonus)
    
    components = {
        'distance_reward': distance_reward,
        'velocity_penalty': velocity_penalty,
        'angle_penalty': angle_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'contact_reward': contact_reward,
        'fuel_penalty': fuel_penalty,
        'progress_reward': progress_reward,
        'settlement_bonus': settlement_bonus,
    }
    
    return float(total_reward), components