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

    # -- Component 1: Potential-based progress --
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

    # -- Component 3: Dense landing quality reward --
    # Provides per-step reward when both legs touch the ground and the pose is near-ideal.
    # All conditions use a linear decay toward zero outside the thresholds.
    x_ok = max(0.0, 1.0 - abs(x_n) / 0.2)
    y_ok = max(0.0, 1.0 - abs(y_n) / 0.3)
    vx_ok = max(0.0, 1.0 - abs(vx_n) / 0.2)
    vy_ok = max(0.0, 1.0 - abs(vy_n) / 0.2)
    angle_ok = max(0.0, 1.0 - abs(angle_n) / 0.2)
    contact_both = left_contact_n * right_contact_n   # 0.0 or 1.0

    landing_quality = contact_both * x_ok * y_ok * vx_ok * vy_ok * angle_ok
    k_landing = 5.0
    landing_quality_reward = k_landing * landing_quality

    # -- Total reward --
    total_reward = potential_diff + angle_penalty + landing_quality_reward

    components = {
        "potential_diff": float(potential_diff),
        "angle_penalty": float(angle_penalty),
        "landing_quality_reward": float(landing_quality_reward)
    }

    return float(total_reward), components