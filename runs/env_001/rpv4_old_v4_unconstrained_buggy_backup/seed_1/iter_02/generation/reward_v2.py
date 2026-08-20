def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v2 — Fixes soft_landing_bonus exploitation by making it transition-based.

    Changes from v1:
    - soft_landing_bonus: state-value → transition-based (only fires on NEW contact)
    - Coefficient raised 5→20 to compensate for one-shot nature
    - proximity_progress and attitude_penalty unchanged (both healthy)
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: ONE-TIME reward on transition into safe contact ----------
    # Detect the EVENT: previously NOT both legs contacting → NOW both legs contacting
    was_both_contact = obs[6] * obs[7]          # 1.0 if both were on pad
    is_both_contact = next_obs[6] * next_obs[7] # 1.0 if both are on pad now
    new_contact_event = (1.0 - was_both_contact) * is_both_contact  # 1.0 ONLY on first-contact step

    # Quality gates — soft thresholds preserve gradient for learning
    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at θ=0, → 0.17 at |θ|=1

    soft_landing_bonus = 20.0 * new_contact_event * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components