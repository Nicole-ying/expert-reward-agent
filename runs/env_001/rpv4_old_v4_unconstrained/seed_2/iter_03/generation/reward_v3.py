def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── Distance to target (origin at 0, 0) ──
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # ── 1. Progress: strong navigation signal ──
    # potential-based shaping — episode total = dist_start - dist_end
    progress = dist_prev - dist_next          # >0 when approaching target
    progress_reward = 5.0 * progress          # was 3.0, boost to compete with contact

    # ── 2. Adaptive velocity management ──
    vx = next_obs[2]
    vy = next_obs[3]
    speed_sq = vx ** 2 + vy ** 2
    speed = speed_sq ** 0.5

    # 2a. Distance-scaled target speed: fast OK when far, must slow down when close
    #     target_speed → 0.6 when dist is large, → 0 when dist → 0
    target_speed = 0.6 * (1.0 - 2.718281828 ** (-3.0 * dist_next))
    speed_excess = max(0.0, speed - target_speed)
    speed_penalty = -0.08 * speed_excess ** 2   # quadratic penalty on excess only

    # 2b. Safe-descent guidance: allow moderate downward speed, penalise only excess
    safe_descent_rate = 0.4                     # up to 0.4 downward is acceptable
    near_ground = 2.718281828 ** (-3.0 * dist_next)
    descent_excess = max(0.0, -vy - safe_descent_rate)
    descent_penalty = -0.6 * descent_excess * near_ground

    velocity_management = speed_penalty + descent_penalty

    # ── 3. Orientation stabilisation ──
    orientation_penalty = -0.15 * abs(next_obs[4]) - 0.05 * (next_obs[5] ** 2)

    # ── 4. Contact guidance: smooth Gaussian — no more cliff ──
    contact_val = float(next_obs[6] + next_obs[7])   # 0.0 .. 2.0

    # Gaussian proximity: smooth gradient all the way in
    #   dist=0.0 → weight=1.00, dist=0.3 → 0.84, dist=0.5 → 0.61, dist=1.0 → 0.14
    proximity_weight = 2.718281828 ** (-0.5 * (dist_next / 0.5) ** 2)

    # Softer speed gate: 0.5 threshold instead of 0.2
    speed_gate = max(0.0, 1.0 - speed / 0.5)

    # Modest coefficient — smooth function means more steps get partial credit,
    # but lower peak value prevents dominance
    contact_guidance = 0.3 * contact_val * proximity_weight * speed_gate

    # ── Combine ──
    total_reward = (
        progress_reward
        + velocity_management
        + orientation_penalty
        + contact_guidance
    )

    components = {
        "progress": progress_reward,
        "velocity_management": velocity_management,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components