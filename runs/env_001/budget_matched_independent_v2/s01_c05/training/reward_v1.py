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
    
    # Distance to target (target is at origin)
    distance = (x**2 + y**2) ** 0.5
    
    # Speed magnitude
    speed = (vx**2 + vy**2) ** 0.5
    
    # 1. Proximity reward: encourage being close to target
    # Use a smooth exponential to provide gradient at all distances
    proximity_reward = 2.718281828 ** (-2.0 * distance)
    
    # 2. Velocity penalty: penalize high speed, especially when close to target
    # Scale penalty by distance so it's stronger near target
    velocity_penalty = -0.5 * speed * (1.0 + 2.0 * (2.718281828 ** (-distance)))
    
    # 3. Orientation reward: encourage upright orientation (angle=0 means upright)
    # Penalize deviation from zero angle
    orientation_penalty = -0.3 * (angle ** 2)
    
    # 4. Angular velocity penalty: discourage spinning
    ang_vel_penalty = -0.2 * (ang_vel ** 2)
    
    # 5. Contact bonus: reward stable contact with both supports on the pad
    # Both contacts active indicates successful landing
    contact_bonus = 1.0 * left_contact * right_contact
    
    # 6. Action penalty: discourage unnecessary engine use
    # Action 0 = no engine, actions 1-3 use some thrust
    action_penalty = -0.1 if action != 0 else 0.0
    
    # 7. Landing completion bonus: large reward when settled on pad
    # Settled = close to target, low speed, upright, both contacts
    settled = (distance < 0.3) and (speed < 0.1) and (abs(angle) < 0.1) and (left_contact > 0.5) and (right_contact > 0.5)
    completion_bonus = 5.0 if settled else 0.0
    
    # Sum all components
    total_reward = (proximity_reward + velocity_penalty + orientation_penalty + 
                    ang_vel_penalty + contact_bonus + action_penalty + completion_bonus)
    
    components = {
        "proximity_reward": proximity_reward,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_bonus": contact_bonus,
        "action_penalty": action_penalty,
        "completion_bonus": completion_bonus
    }
    
    return float(total_reward), components