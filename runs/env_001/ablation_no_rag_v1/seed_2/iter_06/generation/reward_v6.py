def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -- Extract state variables --
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, angular_v = obs[4], obs[5]

    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n, angular_v_n = next_obs[4], next_obs[5]
    left_contact_n = next_obs[6]
    right_contact_n = next_obs[7]

    # -- Component 1: Potential-based progress + stability --
    w_dist = 1.0
    w_vel  = 0.3
    dist_current = (x**2 + y**2) ** 0.5
    dist_next    = (x_n**2 + y_n**2) ** 0.5
    vel_current  = abs(vx) + abs(vy)
    vel_next     = abs(vx_n) + abs(vy_n)

    potential_current = w_dist * dist_current + w_vel * vel_current
    potential_next    = w_dist * dist_next    + w_vel * vel_next

    potential_diff = potential_current - potential_next

    # -- Component 2: Attitude penalty --
    k_angle = 0.5
    k_angvel = 0.1
    angle_penalty = -k_angle * (angle_n**2) - k_angvel * (angular_v_n**2)

    # -- Component 3: Sparse completion bonus --
    # Success condition: both support legs in contact, near zero velocity,
    # upright angle, and close to target center.
    success = False
    if left_contact_n == 1.0 and right_contact_n == 1.0:
        if abs(x_n) < 0.2 and abs(y_n) < 0.3:
            if abs(vx_n) < 0.2 and abs(vy_n) < 0.2:
                if abs(angle_n) < 0.2:
                    success = True
    success_bonus = 200.0 if success else 0.0

    # -- Total reward --
    total_reward = potential_diff + angle_penalty + success_bonus

    components = {
        "potential_diff": float(potential_diff),
        "angle_penalty": float(angle_penalty),
        "success_bonus": float(success_bonus)
    }

    return float(total_reward), components