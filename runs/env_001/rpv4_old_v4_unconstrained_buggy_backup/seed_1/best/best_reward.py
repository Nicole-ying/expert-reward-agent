def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v4 — Fixes soft_landing_bonus multi-fire exploit.

    Changes from v3:
    - soft_landing_bonus: transition-based event → continuous per-step quality reward.
      The agent is now rewarded for STAYING in safe contact (settling), not for the
      contact EVENT. This eliminates the bouncing exploit: bouncing means losing
      contact = losing reward. Settling = sustained reward until episode terminates.
    - Coefficient: 30.0 (one-shot) → 4.0 (per-step), bounded by episode termination.
    - proximity_progress: 30.0 → 40.0 to strengthen approach guidance now that the
      landing bonus no longer dominates.
    - attitude_penalty unchanged.
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 40.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: continuous per-step reward for safe contact ----------
    # Only when both legs are firmly on the pad (binary gate)
    both_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    # Quality gates — soft thresholds preserve gradient for learning
    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at θ=0, → 0.17 at |θ|=1

    soft_landing_bonus = 4.0 * both_contact * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components