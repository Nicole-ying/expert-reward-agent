def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract signals from next_obs
    x_pos = next_obs[0]       # horizontal coordinate relative to target pad
    y_pos = next_obs[1]       # vertical coordinate relative to pad height
    x_vel = next_obs[2]       # horizontal linear velocity
    y_vel = next_obs[3]       # vertical linear velocity
    body_angle = next_obs[4]  # orientation angle
    ang_vel = next_obs[5]     # angular velocity
    left_contact = next_obs[6]  # left support contact flag (0 or 1)
    right_contact = next_obs[7] # right support contact flag (0 or 1)
    
    # Also extract from obs for delta calculations
    prev_x_pos = obs[0]
    prev_y_pos = obs[1]
    prev_x_vel = obs[2]
    prev_y_vel = obs[3]
    prev_body_angle = obs[4]
    prev_ang_vel = obs[5]
    
    # 1. Distance reward: encourage approaching the target pad
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -0.1 * distance  # negative penalty proportional to distance
    
    # 2. Velocity penalty: encourage reducing speed, especially near target
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # Scale velocity penalty by distance - closer to target = more penalty for speed
    velocity_penalty = -0.05 * speed * (1.0 + 2.0 / (1.0 + distance + 0.1))
    
    # 3. Angular stability: encourage upright orientation and low angular velocity
    # Body angle 0 means upright, penalize deviation
    angle_penalty = -0.02 * (body_angle ** 2)
    angular_vel_penalty = -0.01 * (ang_vel ** 2)
    
    # 4. Contact reward: encourage stable contact with both supports
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact
    
    # 5. Progress reward: reward moving closer to target compared to previous step
    prev_distance = (prev_x_pos ** 2 + prev_y_pos ** 2) ** 0.5
    distance_delta = prev_distance - distance  # positive if moving closer
    progress_reward = 0.2 * max(0.0, distance_delta)
    
    # 6. Fuel efficiency penalty: penalize engine usage
    # Action 0 = no engine, 1 = left orientation, 2 = main, 3 = right orientation
    fuel_penalty = 0.0
    if action == 1 or action == 3:  # orientation engines
        fuel_penalty = -0.02
    elif action == 2:  # main engine
        fuel_penalty = -0.05
    
    # 7. Settling bonus: when very close to target with low velocity and both contacts
    settled = (distance < 0.3 and speed < 0.1 and abs(body_angle) < 0.1 and 
               abs(ang_vel) < 0.05 and both_contact > 0.5)
    settling_bonus = 1.0 if settled else 0.0
    
    # Sum all components
    total_reward = (distance_reward + velocity_penalty + angle_penalty + 
                    angular_vel_penalty + contact_reward + progress_reward + 
                    fuel_penalty + settling_bonus)
    
    components = {
        'distance_reward': distance_reward,
        'velocity_penalty': velocity_penalty,
        'angle_penalty': angle_penalty,
        'angular_vel_penalty': angular_vel_penalty,
        'contact_reward': contact_reward,
        'progress_reward': progress_reward,
        'fuel_penalty': fuel_penalty,
        'settling_bonus': settling_bonus,
    }
    
    return float(total_reward), components