def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- potential-based shaping: Φ = -(distance + w_speed*speed + w_angle*|angle| + w_angvel*|angvel|) ----
    # reward = Φ(next_obs) - Φ(obs) = sum of improvements across all dimensions.
    # This is unchanged from the successful previous iteration.

    # --- positions ---
    dx_obs, dy_obs = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_obs = (dx_obs ** 2 + dy_obs ** 2) ** 0.5
    dist_next = (dx_next ** 2 + dy_next ** 2) ** 0.5

    # --- speeds (L1 norm) ---
    speed_obs = abs(obs[2]) + abs(obs[3])
    speed_next = abs(next_obs[2]) + abs(next_obs[3])

    # --- angle ---
    angle_obs = abs(obs[4])
    angle_next = abs(next_obs[4])

    # --- angular velocity ---
    angvel_obs = abs(obs[5])
    angvel_next = abs(next_obs[5])

    # --- potential values (unchanged weights) ---
    w_speed = 0.05
    w_angle = 0.05
    w_angvel = 0.02

    phi_obs = -(dist_obs + w_speed * speed_obs + w_angle * angle_obs + w_angvel * angvel_obs)
    phi_next = -(dist_next + w_speed * speed_next + w_angle * angle_next + w_angvel * angvel_next)

    # --- reward from potential improvement ---
    potential_reward = phi_next - phi_obs

    # --- landing incentive: continuous multiplicative signal ---
    # Only gives meaningful reward when near AND slow AND upright simultaneously.
    # Uses bounded 1/(1+kx) form on next_obs so it provides gradient toward good landing states.
    near_factor = 1.0 / (1.0 + 3.0 * dist_next)
    slow_factor = 1.0 / (1.0 + 2.0 * speed_next)
    upright_factor = 1.0 / (1.0 + 3.0 * angle_next)
    landing_incentive = 0.5 * near_factor * slow_factor * upright_factor

    # --- total ---
    total_reward = potential_reward + landing_incentive

    # --- decompose for logging ---
    progress_delta = dist_obs - dist_next
    speed_reduction = w_speed * (speed_obs - speed_next)
    angle_reduction = w_angle * (angle_obs - angle_next)
    angvel_reduction = w_angvel * (angvel_obs - angvel_next)

    components = {
        'progress_delta': progress_delta,
        'speed_reduction': speed_reduction,
        'angle_reduction': angle_reduction,
        'angvel_reduction': angvel_reduction,
        'landing_incentive': landing_incentive,
        'total_reward': total_reward,
    }

    return float(total_reward), components