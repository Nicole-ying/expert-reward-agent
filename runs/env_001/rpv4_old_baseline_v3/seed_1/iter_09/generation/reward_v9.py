def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v9: Restore v5's potential-based progress backbone.
    - Potential = -distance - 0.2 * speed^2: rewards approach AND penalizes speed
    - Drop descent_bonus (redundant, encouraged crashing)
    - Keep one-shot landing_bonus (now reachable since agent slows down)
    - Add crash_penalty for high-speed ground contact as safety guard
    - Drop stability_reward (dead component)
    """
    ox, oy = obs[0], obs[1]
    ovx, ovy = obs[2], obs[3]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_av = next_obs[5]
    o_left, o_right = obs[6], obs[7]
    n_left, n_right = next_obs[6], next_obs[7]

    # ── 1. Potential-based progress (v5's proven backbone) ──
    # Higher potential = closer + slower. Agent is rewarded for improving potential.
    old_dist = (ox**2 + oy**2) ** 0.5
    new_dist = (nx**2 + ny**2) ** 0.5
    old_speed_sq = ovx**2 + ovy**2
    new_speed_sq = nvx**2 + nvy**2

    pot = -new_dist - 0.2 * new_speed_sq
    prev_pot = -old_dist - 0.2 * old_speed_sq
    progress_reward = 15.0 * (pot - prev_pot)

    # ── 2. Attitude penalty (v5's coefficients) ──
    attitude_penalty = 0.5 * (n_angle ** 2) + 0.15 * (n_av ** 2)

    # ── 3. One-shot landing bonus: first transition to stable two-leg contact ──
    o_both = (o_left > 0.5 and o_right > 0.5)
    n_both = (n_left > 0.5 and n_right > 0.5)

    landing_bonus = 0.0
    if n_both and not o_both:
        speed = (nvx**2 + nvy**2) ** 0.5
        # Higher bonus for gentler touchdown
        landing_bonus = 20.0 * (1.0 / (1.0 + 3.0 * speed)) * (1.0 / (1.0 + 3.0 * abs(n_angle)))

    # ── 4. Crash penalty: quadratic penalty for high-speed ground contact ──
    any_contact = (n_left + n_right) > 0.0
    speed = (nvx**2 + nvy**2) ** 0.5
    crash_penalty = 0.0
    if any_contact and speed > 1.0:
        crash_penalty = 2.0 * (speed ** 2)

    total_reward = (
        progress_reward
        - attitude_penalty
        + landing_bonus
        - crash_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "attitude_penalty": -attitude_penalty,
        "landing_bonus": landing_bonus,
        "crash_penalty": -crash_penalty,
    }

    return float(total_reward), components