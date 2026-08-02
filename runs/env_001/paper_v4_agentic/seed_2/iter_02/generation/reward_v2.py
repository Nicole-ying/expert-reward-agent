def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2 reward — velocity_danger replaces velocity_penalty with smooth inverse-distance weighting.
    Components:
      - proximity_delta:  scaled-up improvement in distance to target (core driving signal)
      - velocity_danger:  speed² / (dist + ε) penalty — continuous, no hard gate
      - orientation_penalty: scaled-up penalty on tilt and angular rate
    """
    # ── current state ──
    x_cur = obs[0]
    y_cur = obs[1]
    vx_cur = obs[2]
    vy_cur = obs[3]
    angle_cur = obs[4]
    angvel_cur = obs[5]

    # ── next state ──
    x_next = next_obs[0]
    y_next = next_obs[1]

    # ── distance to pad (target at 0, 0) ──
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # ── weights / thresholds ──
    w_prox = 50.0           # core progression weight (scaled up 50× from v1)
    w_vel  = 0.15           # velocity danger weight (reduced to keep penalty bounded)
    w_ang  = 5.0            # orientation penalty weight (scaled up 50× from v1)
    proximity_threshold = 1.0  # smoothing constant for inverse-distance (was gate threshold)

    # ── 1. Proximity delta (improvement_delta) ──
    proximity_delta = w_prox * (dist_cur - dist_next)

    # ── 2. Velocity danger (inverse_distance_weighting) ──
    # Continuous penalty: high speed is always warned, severity grows as distance shrinks.
    speed_sq = vx_cur ** 2 + vy_cur ** 2
    velocity_danger = -w_vel * speed_sq / (dist_cur + proximity_threshold)

    # ── 3. Orientation stability (quadratic_penalty) ──
    orientation_penalty = -w_ang * (angle_cur ** 2 + angvel_cur ** 2)

    # ── Total reward ──
    total_reward = proximity_delta + velocity_danger + orientation_penalty

    components = {
        "proximity_delta": proximity_delta,
        "velocity_danger": velocity_danger,
        "orientation_penalty": orientation_penalty,
    }
    return float(total_reward), components