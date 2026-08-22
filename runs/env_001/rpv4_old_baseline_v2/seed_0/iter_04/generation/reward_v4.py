def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant observations
    x, y = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle = obs[4]
    left_contact, right_contact = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_left, n_right = next_obs[6], next_obs[7]

    # --- Progress: reward reducing distance to the landing pad ---
    old_dist = abs(x) + abs(y)
    new_dist = abs(nx) + abs(ny)
    progress_reward = 0.5 * (old_dist - new_dist)  # positive when distance shrinks

    # --- Landing quality change (potential difference) ---
    # Stability factor: 1 when velocity and angle are near zero
    def stability(vx, vy, angle):
        return 1.0 / (1.0 + abs(vx) + abs(vy) + 5.0 * abs(angle))

    cur_stab = stability(x_vel, y_vel, angle)
    next_stab = stability(nx_vel, ny_vel, n_angle)
    cur_contact = left_contact + right_contact
    next_contact = n_left + n_right

    # Weight for landing potential: 2.0
    landing_change = 2.0 * (next_contact * next_stab - cur_contact * cur_stab)

    # --- Velocity change (penalty for speeding up, reward for slowing down) ---
    vel_change = -0.05 * ((abs(nx_vel) + abs(ny_vel)) - (abs(x_vel) + abs(y_vel)))

    # --- Angle change (penalty for increasing tilt, reward for straightening) ---
    ang_change = -0.1 * (abs(n_angle) - abs(angle))

    # --- Fuel efficiency penalty (only when engine is used) ---
    fuel_penalty = -0.02 * (1.0 if action != 0 else 0.0)

    total_reward = progress_reward + landing_change + vel_change + ang_change + fuel_penalty

    components = {
        "progress_reward": progress_reward,
        "landing_change": landing_change,
        "vel_change": vel_change,
        "ang_change": ang_change,
        "fuel_penalty": fuel_penalty,
    }
    return float(total_reward), components