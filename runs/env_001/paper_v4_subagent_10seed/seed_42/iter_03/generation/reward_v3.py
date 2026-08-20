def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    REBUILD: Safe-progress skeleton with speed-gated advancement.

    Core idea: progress toward target is only rewarded when approach speed is
    proportional to distance.  This prevents the rush→crash failure mode seen
    in iter 2 while preserving the linear progress gradient that worked in iter 1.

    Components:
      1. safe_progress = progress * speed_gate   (main driver)
      2. orientation_penalty                      (stability – strengthened)
      3. soft_landing                             (success proxy – widened)
    """

    # ── Unpack observations ──────────────────────────────────────────
    px0, py0 = obs[0], obs[1]          # last position
    px1, py1 = next_obs[0], next_obs[1]  # current position
    vx1, vy1 = next_obs[2], next_obs[3]  # current velocity
    angle1  = next_obs[4]                # body angle
    angvel1 = next_obs[5]                # angular velocity
    left_leg  = next_obs[6]              # left contact
    right_leg = next_obs[7]              # right contact

    # ── Derived signals ──────────────────────────────────────────────
    dist_prev  = (px0**2 + py0**2) ** 0.5
    dist_next  = (px1**2 + py1**2) ** 0.5
    speed      = (vx1**2 + vy1**2) ** 0.5

    # ── 1. Safe progress (speed-gated advancement) ───────────────────
    # Linear progress – no convexity (the convexity in iter 2 caused the crash)
    raw_progress = dist_prev - dist_next   # positive when approaching
    progress     = max(0.0, raw_progress)

    # Speed safety gate: expected speed ≈ k_target * dist
    # When the agent moves faster than expected for its distance, the gate
    # decays, cutting the reward for reckless approach.
    k_target      = 1.5                    # expected speed / distance ratio
    gate_strength = 3.0                    # how sharply excess speed is penalised

    expected_speed = k_target * dist_next
    excess_speed   = max(0.0, speed - expected_speed)
    speed_gate     = 1.0 / (1.0 + gate_strength * excess_speed**2)

    safe_progress  = progress * speed_gate

    # ── 2. Orientation / stability penalties (strengthened) ──────────
    # Coefficients raised 10× from iter 1/2 so these constraints actually bite.
    angle_penalty  = -0.1 * (angle1 ** 2)
    angvel_penalty = -0.05 * (angvel1 ** 2)
    orientation_penalty = angle_penalty + angvel_penalty

    # ── 3. Soft landing guidance (widened proximity, added angle factor) ──
    proximity_threshold = 0.3             # widened from 0.2 to raise active_rate
    if dist_next < proximity_threshold:
        contact_factor = (left_leg + right_leg) / 2.0    # ∈ [0, 1]
        speed_factor   = 1.0 / (1.0 + 10.0 * speed)       # decays with speed
        angle_factor   = 1.0 / (1.0 + 5.0 * (angle1**2))  # decays with tilt
        soft_landing   = contact_factor * speed_factor * angle_factor
    else:
        soft_landing = 0.0

    # ── Combine ──────────────────────────────────────────────────────
    total_reward = (
        1.0 * safe_progress
        + 1.0 * orientation_penalty
        + 2.0 * soft_landing          # amplify success proxy relative to per-step progress
    )

    components = {
        "safe_progress":       safe_progress,
        "orientation_penalty": orientation_penalty,
        "soft_landing":        soft_landing,
    }
    return float(total_reward), components