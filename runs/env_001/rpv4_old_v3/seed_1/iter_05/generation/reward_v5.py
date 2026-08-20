def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observations
    x_pos = obs[0]
    y_pos = obs[1]
    # Unpack next observations
    x_pos_next = next_obs[0]
    y_pos_next = next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # 1. goal_approach: potential-based shaping (distance to center)
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (x_pos_next ** 2 + y_pos_next ** 2) ** 0.5
    goal_approach = dist_current - dist_next   # positive when approaching

    # 2. soft_landing: full-range velocity, angle, angular velocity penalty
    vel_penalty = abs(x_vel_next) + abs(y_vel_next)
    angle_penalty = abs(body_angle_next)       # rad, should be near 0
    ang_vel_penalty = abs(ang_vel_next)
    soft_landing = -0.2 * vel_penalty - 0.2 * angle_penalty - 0.2 * ang_vel_penalty

    # 3. landing_success: one-time bonus for stable two-leg touchdown
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    STABLE = (abs(x_vel_next) < 1.0 and abs(y_vel_next) < 1.0 and
              abs(body_angle_next) < 0.5 and abs(ang_vel_next) < 1.0)
    if BOTH_LEGS and STABLE:
        landing_success = 100.0
    else:
        landing_success = 0.0

    total_reward = goal_approach + soft_landing + landing_success

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "landing_success": landing_success,
    }

    return float(total_reward), components