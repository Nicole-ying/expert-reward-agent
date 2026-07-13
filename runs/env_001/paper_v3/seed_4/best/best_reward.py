def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from the next observation
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Proximity to target (assumed at origin)
    dist = (x**2 + y**2) ** 0.5
    proximity = 1.0 / (1.0 + dist)          # bounded [0,1]
    w_proximity = 1.0
    comp_prox = w_proximity * proximity

    # Descent quality: continuous reward for being near ground level
    # with low speed and small attitude deviations
    # Replaces the previous contact-gated soft_landing_bonus that never fired
    height_factor = 1.0 / (1.0 + abs(y))    # peaks at y=0 (platform surface)
    speed_norm = (vx**2 + vy**2) ** 0.5
    factor_vel = 1.0 / (1.0 + speed_norm)
    factor_angle = 1.0 / (1.0 + abs(angle) + abs(angular_vel))
    descent_quality = height_factor * factor_vel * factor_angle
    w_descent = 3.0
    comp_descent = w_descent * descent_quality

    # Quadratic penalties for high velocity and attitude deviations
    w_vel_pen = 0.01
    vel_pen = -w_vel_pen * (vx**2 + vy**2)

    w_att_pen = 0.01
    att_pen = -w_att_pen * (angle**2 + angular_vel**2)

    total = comp_prox + comp_descent + vel_pen + att_pen

    components = {
        'proximity': comp_prox,
        'descent_quality': comp_descent,
        'velocity_penalty': vel_pen,
        'attitude_penalty': att_pen,
    }
    return float(total), components