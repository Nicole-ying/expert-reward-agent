def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for goal_approach_and_soft_contact (discrete thrusters, 2D lander).
    """
    # ------------------------------------------------------------------
    # unpack observations (current)
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    # unpack next observations
    x_pos_next = next_obs[0]
    y_pos_next = next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    ang_vel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ------------------------------------------------------------------
    # 1. goal_approach (potential‑based shaping using L2 distance to pad)
    #    target is (0,0) on the pad.
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (x_pos_next ** 2 + y_pos_next ** 2) ** 0.5
    goal_approach = dist_current - dist_next   # positive when approaching

    # ------------------------------------------------------------------
    # 2. soft_landing – conditional velocity/angular‑velocity penalty near ground
    #    gate rises as altitude drops below threshold
    LANDING_THRESHOLD = 0.5           # meters above pad
    land_weight = max(0.0, 1.0 - (y_pos_next / LANDING_THRESHOLD))
    # quadratic penalty on linear + angular speed in next state
    soft_landing = -land_weight * (
        x_vel_next ** 2 + y_vel_next ** 2 + 0.1 * ang_vel_next ** 2
    )

    # ------------------------------------------------------------------
    # 3. crash_avoidance – harsh penalty when a crash‑like state is reached
    #    crash = below pad (negative height) AND not both legs on contact
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    CRASH_PENALTY = -10.0
    if y_pos_next < -0.2 and not BOTH_LEGS:
        crash_avoidance = CRASH_PENALTY
    else:
        crash_avoidance = 0.0

    # ------------------------------------------------------------------
    # Combine components
    total_reward = goal_approach + soft_landing + crash_avoidance

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "crash_avoidance": crash_avoidance,
    }

    return float(total_reward), components