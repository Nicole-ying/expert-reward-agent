def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lunar-lander-like task.
    
    Args:
        obs, next_obs: shape (8,) arrays with:
            0: x_position (relative to landing pad)
            1: y_position (relative to landing pad height)
            2: x_velocity
            3: y_velocity
            4: body_angle
            5: angular_velocity
            6: left_support_contact (0/1)
            7: right_support_contact (0/1)
        action: discrete 0..3 (unused in this version)
        info: empty dict (do not use)
        training_progress: unused
    """
    # -- Extract state variables --
    # Current state
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, angular_v = obs[4], obs[5]

    # Next state
    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n, angular_v_n = next_obs[4], next_obs[5]

    # -- Component 1: Potential-based progress + stability (main learning signal) --
    # Potential combines distance to target and velocity magnitude.
    # Weight on velocity makes the agent prefer slow, safe approach.
    w_dist = 1.0
    w_vel  = 0.3   # scaling for linear velocity relative to distance
    dist_current = (x**2 + y**2) ** 0.5
    dist_next    = (x_n**2 + y_n**2) ** 0.5
    vel_current  = abs(vx) + abs(vy)
    vel_next     = abs(vx_n) + abs(vy_n)

    potential_current = w_dist * dist_current + w_vel * vel_current
    potential_next    = w_dist * dist_next    + w_vel * vel_next

    # Positive when potential decreases (agent moves closer to target and/or slows down)
    potential_diff = potential_current - potential_next

    # -- Component 2: Attitude penalty (stability constraint) --
    # Penalise large body angle and angular velocity to promote upright landing.
    k_angle = 0.5
    k_angvel = 0.1
    # Use squared terms for smooth gradient
    angle_penalty = -k_angle * (angle_n**2) - k_angvel * (angular_v_n**2)

    # -- Total reward --
    total_reward = potential_diff + angle_penalty

    # -- Components dict (only the two that are summed) --
    components = {
        "potential_diff": float(potential_diff),
        "angle_penalty": float(angle_penalty)
    }

    return float(total_reward), components