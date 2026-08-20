def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # weights
    w_progress = 1.0
    w_angle_penalty = 0.2
    w_landing_event = 25.0       # one-time bonus on first contact transition
    w_landing_stability = 0.05   # tiny sustaining reward while on ground
    w_approach_vel = 0.3
    a_v = 10.0
    b_angle = 10.0

    # distances
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 1. Progress
    progress = w_progress * (dist - next_dist)

    # 2. Orientation stability
    orientation_penalty = -w_angle_penalty * (next_obs[4] ** 2)

    # 3. Landing: split into event (transition) and stability (sustaining)
    contact_prev = max(obs[6], obs[7])
    contact_next = max(next_obs[6], next_obs[7])
    contact_transition = float(contact_next > 0.5 and contact_prev <= 0.5)

    vertical_speed_factor = 1.0 / (1.0 + a_v * (next_obs[3] ** 2))
    angle_factor = 1.0 / (1.0 + b_angle * (next_obs[4] ** 2))
    quality = vertical_speed_factor * angle_factor

    landing_event = w_landing_event * contact_transition * quality
    landing_stability = w_landing_stability * contact_next * quality

    # 4. Approach-phase velocity penalty
    proximity = 1.0 / (1.0 + next_dist)
    velocity_sq = next_obs[2] ** 2 + next_obs[3] ** 2
    approach_velocity_penalty = -w_approach_vel * proximity * velocity_sq

    total_reward = (
        progress
        + orientation_penalty
        + landing_event
        + landing_stability
        + approach_velocity_penalty
    )

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "landing_event": landing_event,
        "landing_stability": landing_stability,
        "approach_velocity_penalty": approach_velocity_penalty
    }

    return float(total_reward), components