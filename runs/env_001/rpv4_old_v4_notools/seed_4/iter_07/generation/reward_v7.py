def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- extract signals ---
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- helpers ---
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    velocity_mag = (vx_next ** 2 + vy_next ** 2) ** 0.5

    # === 1. distance reward — closer is better, but capped ===
    w_dist = 10.0
    distance_reward = w_dist * (1.0 / (1.0 + dist_next))

    # === 2. global velocity penalty — force deceleration everywhere ===
    w_vel = 5.0
    velocity_penalty = -w_vel * velocity_mag

    # === 3. attitude stabilization (as before) ===
    w_angle = 1.0
    w_angvel = 0.5
    attitude_penalty = -w_angle * (angle_next ** 2) - w_angvel * (angvel_next ** 2)

    # === 4. fuel penalty (same as before but slightly scaled) ===
    if action == 0:
        fuel_cost = 0.0
    elif action == 2:   # main engine
        fuel_cost = 0.2
    else:               # side engines
        fuel_cost = 0.1
    fuel_penalty = -fuel_cost

    # === 5. soft landing BONUS — large positive, only on successful touchdown ===
    landing_bonus = 0.0
    if left_contact and right_contact:
        if dist_next < 0.2 and velocity_mag < 0.5:
            landing_bonus = 200.0   # one-shot completion reward

    # --- total ---
    total_reward = (distance_reward +
                    velocity_penalty +
                    attitude_penalty +
                    fuel_penalty +
                    landing_bonus)

    components = {
        "distance": distance_reward,
        "velocity": velocity_penalty,
        "attitude": attitude_penalty,
        "fuel": fuel_penalty,
        "landing": landing_bonus,
    }

    return float(total_reward), components