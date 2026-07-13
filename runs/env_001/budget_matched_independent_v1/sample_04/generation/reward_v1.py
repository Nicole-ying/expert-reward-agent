def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Implements V1 reward for navigation_goal_reaching (goal_approach_and_soft_contact).
    Uses a combination of proximity, velocity, orientation, and contact signals.
    """
    # Unpack next observation (state after action)
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = next_obs

    # --- Reward components ---
    # Goal proximity: Euclidean distance to target pad center (0,0)
    dist = (x**2 + y**2)**0.5
    goal_proximity = -30.0 * dist

    # Soft landing velocity: quadratic penalty on velocity magnitude
    # Encourages reducing both horizontal and vertical speed
    vel_penalty = -30.0 * (vx**2 + vy**2)

    # Orientation stability: quadratic penalty on body angle deviation from vertical
    angle_dev = angle**2
    orientation_penalty = -30.0 * angle_dev

    # Contact encouragement: bonus for each support leg contacting the pad
    contact_bonus = 10.0 * (left_contact + right_contact)

    # Combine
    total_reward = goal_proximity + vel_penalty + orientation_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "soft_landing_velocity": vel_penalty,
        "orientation_stability": orientation_penalty,
        "contact_encouragement": contact_bonus,
    }

    return float(total_reward), components