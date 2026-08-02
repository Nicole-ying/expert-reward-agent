def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observation indices (assumed 2D lander, similar to LunarLander‑v2)
    # 0: x position, 1: y position, 2: vx, 3: vy, 4: angle, 5: angular velocity
    # 6: left leg contact, 7: right leg contact (0.0 or 1.0)
    # Target landing pad is assumed at (0.0, 0.0)

    # Extract current and next states
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle, n_ang_vel = next_obs[4], next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # ---------- 1. Main progress: distance to target decreasing ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new              # + when approaching target

    w_progress = 10.0

    # ---------- 2. Stability constraint: attitude & angular velocity ----------
    # Quadratic penalty on tilt and rotation
    w_angle = 0.5
    w_angvel = 0.1
    stability = -w_angle * (n_angle ** 2) - w_angvel * (n_ang_vel ** 2)

    # ---------- 3. Lateral drift constraint: horizontal speed ----------
    w_lat = 0.2
    lateral_drift = -w_lat * (nvx ** 2)

    # ---------- 4. Soft landing bonus (joint‑condition proxy) ----------
    landing_bonus = 0.0
    # Conditions: both legs on ground, nearly upright, gentle vertical speed
    if n_left > 0.5 and n_right > 0.5 and abs(n_angle) < 0.1 and abs(nvy) < 0.2:
        landing_bonus = 10.0

    # Combine
    total_reward = (w_progress * progress
                    + stability
                    + lateral_drift
                    + landing_bonus)

    components = {
        "progress": w_progress * progress,
        "stability_penalty": stability,
        "lateral_drift_penalty": lateral_drift,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components