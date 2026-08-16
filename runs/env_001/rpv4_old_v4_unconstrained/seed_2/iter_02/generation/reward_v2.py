def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── Distance to target (origin at 0, 0) ──
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # ── 1. Progress: PRIMARY driver toward target ──
    # potential-based: sum over episode = dist_start - dist_end
    progress = dist_prev - dist_next          # >0 when approaching target
    progress_reward = 3.0 * progress          # was 0.5 → 6x boost

    # ── 2. Velocity management ──
    vx = next_obs[2]
    vy = next_obs[3]
    speed_sq = vx ** 2 + vy ** 2
    speed = speed_sq ** 0.5

    # 2a. General speed penalty — encourages efficiency, prevents oscillation
    speed_penalty = -0.08 * speed_sq

    # 2b. Hard-landing prevention: penalise downward speed near ground
    #     exponential decay with distance → strong only when close to target
    near_ground = 2.718281828 ** (-3.0 * dist_next)
    descent_penalty = -0.8 * max(0.0, -vy) * near_ground

    velocity_management = speed_penalty + descent_penalty

    # ── 3. Orientation stabilisation ──
    orientation_penalty = -0.15 * abs(next_obs[4]) - 0.05 * (next_obs[5] ** 2)

    # ── 4. Contact guidance: tight, conditional, modest ──
    contact_val = float(next_obs[6] + next_obs[7])   # 0.0 .. 2.0

    # Tight proximity: only within 0.3 units of target (was 5.0!)
    proximity_ok = max(0.0, 1.0 - dist_next / 0.3)

    # Tight speed gate: total speed < 0.2 (was vy-only < 0.5)
    speed_ok = max(0.0, 1.0 - speed / 0.2)

    # Modest coefficient 0.3 × tighter gates → much lower effective magnitude
    contact_guidance = 0.3 * contact_val * proximity_ok * speed_ok

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