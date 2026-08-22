def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- extract current and next state ---
    x, y = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- helpers ---
    dist = (x ** 2 + y ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    velocity_mag = (vx_next ** 2 + vy_next ** 2) ** 0.5

    # === 1. distance potential difference ===
    w_dist = 30.0
    potential = 1.0 / (1.0 + dist)
    potential_next = 1.0 / (1.0 + dist_next)
    distance_reward = w_dist * (potential_next - potential)

    # === 2. velocity penalty — gated, only when close to target ===
    w_vel = 3.0
    speed_gate = 1.0 if dist_next < 1.5 else 0.0
    velocity_penalty = -w_vel * velocity_mag * speed_gate

    # === 3. attitude stabilization ===
    w_angle = 1.0
    w_angvel = 0.5
    attitude_penalty = -w_angle * (angle_next ** 2) - w_angvel * (angvel_next ** 2)

    # === 4. fuel penalty ===
    if action == 0:
        fuel_cost = 0.0
    elif action == 2:   # main engine
        fuel_cost = 0.2
    else:               # side engines
        fuel_cost = 0.1
    fuel_penalty = -fuel_cost

    # === 5. soft landing approach + one-shot bonus ===
    approach_reward = 0.0
    if dist_next < 0.8 and velocity_mag < 0.8:
        w_approach = 100.0
        norm_dist = dist_next / 0.8
        norm_vel = velocity_mag / 0.8
        approach_reward = w_approach * (1.0 - norm_dist) * (1.0 - norm_vel)

    landing_one_shot = 0.0
    if left_contact and right_contact and dist_next < 0.2 and velocity_mag < 0.5:
        landing_one_shot = 200.0

    landing_reward = approach_reward + landing_one_shot

    # --- total ---
    total_reward = (distance_reward +
                    velocity_penalty +
                    attitude_penalty +
                    fuel_penalty +
                    landing_reward)

    components = {
        "distance": distance_reward,
        "velocity": velocity_penalty,
        "attitude": attitude_penalty,
        "fuel": fuel_penalty,
        "landing": landing_reward,
    }

    return float(total_reward), components