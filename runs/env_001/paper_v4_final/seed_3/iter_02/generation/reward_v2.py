def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observation indices
    # 0: x position, 1: y position, 2: vx, 3: vy, 4: angle, 5: angular velocity
    # 6: left leg contact, 7: right leg contact (0.0 or 1.0)

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
    progress = dist_old - dist_new

    w_progress = 10.0

    # ---------- 2. Attitude gate: suppress progress when angle is dangerous ----------
    # Smooth hinge: gate ≈ 1.0 when |angle| << 0.15, gate → 0.2 when |angle| >> 0.15
    # Use tanh for smooth transition; 0.15 rad ≈ 8.6° is safety threshold
    abs_angle = abs(n_angle)
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (20.0 * (abs_angle - 0.15)) / (1.0 + 2.718281828 ** (20.0 * (abs_angle - 0.15))))

    # ---------- 3. Lateral drift constraint: horizontal speed ----------
    w_lat = 0.2
    lateral_drift = -w_lat * (nvx ** 2)

    # ---------- 4. Angular velocity penalty: small auxiliary smoothing ----------
    w_angvel = 0.1
    angvel_penalty = -w_angvel * (n_ang_vel ** 2)

    # ---------- 5. Landing bonus: soft continuous proxy ----------
    # Both legs touching, nearly upright, gentle speeds
    both_legs = min(n_left, n_right)              # 0.0 to 1.0
    vertical_ok = max(0.0, 1.0 - abs(nvy) / 0.3) # 1.0 when vy≈0, 0 when |vy|>=0.3
    attitude_ok = max(0.0, 1.0 - abs_angle / 0.15) # 1.0 when angle≈0, 0 when |angle|>=0.15

    landing_factor = both_legs * vertical_ok * attitude_ok
    landing_bonus = 3.0 * landing_factor          # up to 3.0, smooth

    # Combine: progress is gated by attitude, then penalties and bonus added
    total_reward = (w_progress * progress * angle_gate
                    + lateral_drift
                    + angvel_penalty
                    + landing_bonus)

    components = {
        "progress_gated": w_progress * progress * angle_gate,
        "lateral_drift_penalty": lateral_drift,
        "angvel_penalty": angvel_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components