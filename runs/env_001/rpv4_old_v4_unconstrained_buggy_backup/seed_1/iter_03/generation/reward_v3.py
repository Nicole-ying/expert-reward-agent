def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v3 — Fixes soft_landing_bonus multi-fire exploit with binary contact threshold.

    Changes from v2:
    - Contact detection: continuous product → thresholded binary (0.5 cutoff)
      Prevents partial transition events when contact values fluctuate.
    - Coefficient: 20.0 → 30.0 to maintain meaningful one-shot landing incentive
      now that the bonus truly fires only once per episode.
    - proximity_progress and attitude_penalty unchanged (both healthy).
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: TRUE one-shot reward on first safe contact ----------
    # Binary threshold prevents exploit from continuous contact-value fluctuations
    was_both_contact = 1.0 if (obs[6] > 0.5 and obs[7] > 0.5) else 0.0
    is_both_contact  = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0
    new_contact_event = (1.0 - was_both_contact) * is_both_contact  # 1.0 ONLY on 0→1 transition

    # Quality gates — soft thresholds preserve gradient for learning
    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at θ=0, → 0.17 at |θ|=1

    soft_landing_bonus = 30.0 * new_contact_event * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components