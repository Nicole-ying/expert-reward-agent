def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for the 2D lander soft‑landing task.
    Components:
      - proximity_delta:  improvement in distance to target (core driving signal)
      - velocity_penalty: quadratic speed penalty gated by proximity (soft landing)
      - orientation_penalty: quadratic penalty on tilt and angular rate (safety)
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
    w_prox = 1.0           # core progression weight
    w_vel  = 0.5           # velocity penalty weight
    w_ang  = 0.1           # orientation penalty weight
    proximity_threshold = 0.5   # distance below which we start caring about speed

    # ── 1. Proximity delta (improvement_delta) ──
    # positive when the lander reduces its distance to the target.
    proximity_delta = w_prox * (dist_cur - dist_next)

    # ── 2. Soft landing velocity penalty (quadratic_penalty + conditional_gating) ──
    # gate: 0 when far away, linear ramp to 1 when inside threshold.
    gate = max(0.0, 1.0 - dist_cur / proximity_threshold)
    speed_sq = vx_cur ** 2 + vy_cur ** 2
    velocity_penalty = -w_vel * speed_sq * gate

    # ── 3. Orientation stability (quadratic_penalty) ──
    orientation_penalty = -w_ang * (angle_cur ** 2 + angvel_cur ** 2)

    # ── Total reward ──
    total_reward = proximity_delta + velocity_penalty + orientation_penalty

    components = {
        "proximity_delta": proximity_delta,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
    }
    return float(total_reward), components