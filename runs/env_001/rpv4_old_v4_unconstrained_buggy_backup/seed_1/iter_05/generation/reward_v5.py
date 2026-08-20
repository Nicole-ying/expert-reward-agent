def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v5 — Softens landing quality gates and rebalances components.

    Changes from v4:
    - soft_landing_bonus: quality gate temperature 5.0 → 3.0 (softer gradient).
      The multiplicative AND-logic is preserved (contact + low speed + level
      attitude all required), but the penalty for imperfect conditions is less
      severe, reducing the "reward cliff" that caused training oscillation.
    - soft_landing_bonus coefficient: 4.0 → 2.5 to reduce dominance (was 76%
      magnitude share), shifting balance toward process-oriented guidance.
    - proximity_progress: 40.0 → 50.0 to strengthen approach signal now that
      landing bonus is less dominant. This provides more consistent per-step
      guidance throughout the descent.
    - attitude_penalty unchanged.
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 50.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: continuous per-step reward for safe contact ----------
    # Only when both legs are firmly on the pad (binary gate)
    both_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    # Quality gates — temperature reduced 5.0→3.0 for softer gradient
    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 3.0 * v_mag)       # → 1.0 at v=0, → 0.25 at v=1
    angle_quality    = 1.0 / (1.0 + 3.0 * abs(angle))  # → 1.0 at θ=0, → 0.25 at |θ|=1

    # Coefficient 2.5 (was 4.0) — softer max per-step, reduced dominance
    soft_landing_bonus = 2.5 * both_contact * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components