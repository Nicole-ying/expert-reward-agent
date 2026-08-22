def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- extract signals ---
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- helper: distance to target pad centre ---
    dist_obs = (x_obs ** 2 + y_obs ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # === 1. progress reward (improvement_delta) ===
    progress = dist_obs - dist_next   # positive when distance shrinks
    w_progress = 50.0
    progress_reward = w_progress * progress

    # === 2. attitude stabilisation (quadratic penalty) ===
    w_angle = 1.0
    w_angvel = 0.5
    attitude_penalty = -w_angle * (abs(angle_next) ** 2) - w_angvel * (angvel_next ** 2)

    # === 3. fuel / engine usage penalty ===
    if action == 0:
        fuel_cost = 0.0
    elif action == 2:   # main engine
        fuel_cost = 0.2
    else:               # left or right orientation engine
        fuel_cost = 0.1
    fuel_reward = -fuel_cost

    # === 4. soft landing component: bonus for dual contact near centre, penalty for high speed ===
    dist_threshold = 0.2
    proximity = max(0.0, 1.0 - dist_next / dist_threshold)   # 1 when dist=0, 0 when dist >= threshold
    dual_contact = left_contact * right_contact               # 1 if both legs touch, else 0
    velocity_mag = (vx_next ** 2 + vy_next ** 2) ** 0.5
    v_safe = 0.5   # maximum allowable landing speed for full bonus; beyond this gives penalty
    speed_factor = 1.0 - velocity_mag / v_safe                # can be negative if speed > v_safe
    w_landing = 40.0
    landing_reward = w_landing * dual_contact * proximity * speed_factor

    # --- total reward ---
    total_reward = progress_reward + attitude_penalty + fuel_reward + landing_reward

    components = {
        "progress": progress_reward,
        "attitude": attitude_penalty,
        "fuel": fuel_reward,
        "landing": landing_reward,
    }

    return float(total_reward), components