def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract signals from current and next observations
    x_old = obs[0]
    y_old = obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Proximity: state-to-improvement transform ---
    # Replace persistent state value with improvement (potential difference).
    # Only rewards actual approach, not occupying a proxy-optimal distance.
    old_dist = (x_old**2 + y_old**2) ** 0.5
    new_dist = (x**2 + y**2) ** 0.5
    old_prox = 1.0 / (1.0 + old_dist)
    new_prox = 1.0 / (1.0 + new_dist)
    proximity_delta = new_prox - old_prox          # positive when getting closer, range ~[-1,1]
    w_proximity = 100.0                             # scales telescoping sum to meaningful range
    comp_prox = w_proximity * proximity_delta

    # --- Descent quality (unchanged from current) ---
    speed_norm = (vx**2 + vy**2) ** 0.5
    factor_vel = 1.0 / (1.0 + speed_norm)
    factor_angle = 1.0 / (1.0 + abs(angle) + abs(angular_vel))
    height_factor = 1.0 / (1.0 + abs(y))
    contact_sum = left_contact + right_contact
    contact_gate = 0.05 + 0.95 * min(1.0, contact_sum)
    descent_quality = contact_gate * height_factor * factor_vel * factor_angle
    w_descent = 3.0
    comp_descent = w_descent * descent_quality

    # --- Contact quality (unchanged from current) ---
    both_contact = (left_contact + right_contact) >= 1.5
    if both_contact:
        contact_quality = factor_vel * factor_angle
        w_contact = 5.0
        comp_contact = w_contact * contact_quality
    else:
        comp_contact = 0.0

    # --- Penalties (unchanged from current) ---
    w_vel_pen = 0.01
    vel_pen = -w_vel_pen * (vx**2 + vy**2)

    w_att_pen = 0.01
    att_pen = -w_att_pen * (angle**2 + angular_vel**2)

    total_reward = comp_prox + comp_descent + comp_contact + vel_pen + att_pen

    reward_components = {
        'proximity': comp_prox,
        'descent_quality': comp_descent,
        'contact_quality': comp_contact,
        'velocity_penalty': vel_pen,
        'attitude_penalty': att_pen,
    }
    return float(total_reward), reward_components