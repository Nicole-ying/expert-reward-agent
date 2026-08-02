def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observation indices (same as before)
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle, n_ang_vel = next_obs[4], next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # ---------- 1. Main progress: distance to target ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new
    w_progress = 10.0

    # ---------- 2. Stability ----------
    w_angle = 0.5
    w_angvel = 0.1
    stability = -w_angle * (n_angle ** 2) - w_angvel * (n_ang_vel ** 2)

    # ---------- 3. Lateral drift ----------
    w_lat = 0.1
    lateral_drift = -w_lat * (nvx ** 2)

    # ---------- 4. Landing approach (continuous, replaces sparse landing_bonus) ----------
    # Ground proximity: y near 0 (assumed pad altitude) gives high factor
    ground_prox = 1.0 / (1.0 + ny ** 2)
    # Attitude factor: angle within 0.3 rad
    angle_factor = max(0.0, 1.0 - abs(n_angle) / 0.3)
    # Vertical speed factor: gentle vertical speed
    vy_factor = max(0.0, 1.0 - abs(nvy) / 0.3)
    landing_factor = (ground_prox + angle_factor + vy_factor) / 3.0
    w_landing = 0.05
    landing_approach = w_landing * landing_factor

    # ---------- 5. Descending penalty ----------
    safe_vy = -0.3
    w_desc = 0.5
    descending_penalty = w_desc * min(0.0, nvy - safe_vy)

    # Combine
    total_reward = (w_progress * progress
                    + stability
                    + lateral_drift
                    + landing_approach
                    + descending_penalty)

    components = {
        "progress": w_progress * progress,
        "stability_penalty": stability,
        "lateral_drift_penalty": lateral_drift,
        "landing_approach": landing_approach,
        "descending_penalty": descending_penalty
    }

    return float(total_reward), components