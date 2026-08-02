def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract signals from next_obs (post-action state)
    x = next_obs[0]          # horizontal position relative to target
    y = next_obs[1]          # vertical position relative to pad height
    vx = next_obs[2]         # horizontal velocity
    vy = next_obs[3]         # vertical velocity
    angle = next_obs[4]      # body orientation angle
    ang_vel = next_obs[5]    # angular velocity
    left_contact = next_obs[6]   # left support contact flag (0 or 1)
    right_contact = next_obs[7]  # right support contact flag (0 or 1)
    
    # Also extract previous state for velocity change penalty
    prev_vx = obs[2]
    prev_vy = obs[3]
    
    # --- Reward Components ---
    
    # 1. Distance reward: encourage approaching the target (0,0)
    distance = (x**2 + y**2) ** 0.5
    distance_reward = -0.1 * distance  # negative penalty proportional to distance
    
    # 2. Velocity penalty: encourage low speed, especially near target
    speed = (vx**2 + vy**2) ** 0.5
    # Scale penalty by distance: closer to target -> higher penalty for speed
    velocity_penalty = -0.05 * speed * (1.0 + 2.0 / (1.0 + distance + 0.1))
    
    # 3. Orientation reward: encourage upright orientation (angle=0)
    orientation_penalty = -0.02 * (angle**2 + 0.1 * ang_vel**2)
    
    # 4. Contact reward: reward stable contact with both supports
    contact_reward = 0.5 * (left_contact + right_contact)  # 0, 0.5, or 1.0
    
    # 5. Fuel efficiency penalty: penalize engine usage
    # Action 2 is main engine, actions 1 and 3 are orientation engines
    engine_penalty = 0.0
    if action == 2:  # main engine
        engine_penalty = -0.2
    elif action == 1 or action == 3:  # orientation engines
        engine_penalty = -0.1
    
    # 6. Smoothness penalty: penalize large velocity changes (jerkiness)
    delta_vx = vx - prev_vx
    delta_vy = vy - prev_vy
    acceleration = (delta_vx**2 + delta_vy**2) ** 0.5
    smoothness_penalty = -0.01 * acceleration
    
    # 7. Landing bonus: when both supports contact and near zero velocity
    both_contact = left_contact + right_contact
    landing_bonus = 0.0
    if both_contact >= 1.5 and speed < 0.5 and distance < 0.3:
        landing_bonus = 2.0
    
    # Sum all components
    total_reward = (distance_reward + velocity_penalty + orientation_penalty + 
                    contact_reward + engine_penalty + smoothness_penalty + landing_bonus)
    
    components = {
        'distance_reward': distance_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'contact_reward': contact_reward,
        'engine_penalty': engine_penalty,
        'smoothness_penalty': smoothness_penalty,
        'landing_bonus': landing_bonus
    }
    
    return float(total_reward), components