def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation variables
    x, y = obs[0], obs[1]
    x_v, y_v = obs[2], obs[3]
    angle = obs[4]
    ang_v = obs[5]

    nx, ny = next_obs[0], next_obs[1]
    nx_v, ny_v = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_v = next_obs[5]
    n_left = next_obs[6]
    n_right = next_obs[7]

    # ---------- 1. Progress reward: moving toward the landing pad (0,0) ----------
    dist_curr = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_curr - dist_next
    progress_reward = 1.0 * progress

    # ---------- 2. Horizontal boundary penalty (dead component, kept for backward compat) ----------
    x_limit = 1.2
    x_boundary_penalty = 0.5 * max(0.0, abs(nx) - x_limit)

    # ---------- 3. Landing softness / safety penalty (unchanged) ----------
    v_limit = 0.5
    vx_pen = max(0.0, abs(nx_v) - v_limit)
    vy_pen = max(0.0, abs(ny_v) - v_limit)
    vel_pen = vx_pen + vy_pen

    ang_limit = 1.0
    ang_pen = max(0.0, abs(n_ang_v) - ang_limit)

    tilt_pen = abs(n_angle)

    gate = 1.0 / (1.0 + 5.0 * dist_next)
    landing_safety_penalty = (0.1 * vel_pen + 0.05 * ang_pen + 0.1 * tilt_pen) * gate

    # ---------- 4. Precise landing bonus: product of proximity, velocity, angle, and contact ----------
    # Proximity factor: 1 when distance=0, 0 when distance >= 0.5
    proximity_factor = max(0.0, 1.0 - dist_next / 0.5)
    # Velocity factor: 1 when total speed=0, 0 when >= 0.5
    total_speed = abs(nx_v) + abs(ny_v)
    velocity_factor = max(0.0, 1.0 - total_speed / 0.5)
    # Angle factor: 1 when angle=0, 0 when |angle| >= 0.3
    angle_factor = max(0.0, 1.0 - abs(n_angle) / 0.3)
    # Contact factor: average of both legs
    contact_factor = (n_left + n_right) / 2.0

    precise_landing_bonus = 20.0 * proximity_factor * velocity_factor * angle_factor * contact_factor

    # ---------- Total reward ----------
    total_reward = progress_reward - x_boundary_penalty - landing_safety_penalty + precise_landing_bonus

    components = {
        "progress_reward": float(progress_reward),
        "x_boundary_penalty": float(x_boundary_penalty),
        "landing_safety_penalty": float(landing_safety_penalty),
        "precise_landing_bonus": float(precise_landing_bonus)
    }
    return float(total_reward), components