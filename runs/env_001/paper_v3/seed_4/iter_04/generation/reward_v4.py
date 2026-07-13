def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from observations
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

    # Shared stability factors reused across components
    speed_norm = (vx**2 + vy**2) ** 0.5
    factor_vel = 1.0 / (1.0 + speed_norm)       # bounded [0,1], 1 when still
    factor_angle = 1.0 / (1.0 + abs(angle) + abs(angular_vel))  # bounded [0,1]

    # Descent quality: reward being near ground, slow, and upright
    height_factor = 1.0 / (1.0 + abs(y))        # peaks at y=0
    descent_quality = height_factor * factor_vel * factor_angle
    w_descent = 3.0
    comp_descent = w_descent * descent_quality

    # Sustained contact quality: reward stable, settled contact with both feet
    # Replaces the transition-based contact_bonus that was exploited via bouncing
    both_contact = (left_contact + right_contact) >= 1.5
    if both_contact:
        contact_quality = factor_vel * factor_angle   # bounded [0,1], high when stable
        w_contact = 5.0
        comp_contact = w_contact * contact_quality
    else:
        comp_contact = 0.0

    # Quadratic penalties for high velocity and attitude deviations
    w_vel_pen = 0.01
    vel_pen = -w_vel_pen * (vx**2 + vy**2)

    w_att_pen = 0.01
    att_pen = -w_att_pen * (angle**2 + angular_vel**2)

    total = comp_prox + comp_descent + comp_contact + vel_pen + att_pen

    components = {
        'proximity': comp_prox,
        'descent_quality': comp_descent,
        'contact_quality': comp_contact,
        'velocity_penalty': vel_pen,
        'attitude_penalty': att_pen,
    }
    return float(total), components