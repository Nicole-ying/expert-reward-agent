def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v6 — Adds velocity penalty to bridge approach and soft landing.

    Changes from v5:
    - proximity_progress: 50.0 → 40.0 (revert to v4). v5's boost caused
      the agent to dive at full speed, ignoring landing quality entirely.
    - soft_landing_bonus: coefficient 2.5→4.0, temperature 3.0→5.0 (revert to v4).
      v5's weakened landing bonus collapsed to 1.7% magnitude share.
    - NEW velocity_penalty: quadratic penalty on instantaneous speed.
      This is the key semantic addition — it discourages the high-speed
      diving strategy that proximity_progress naturally incentivizes.
      Quadratic form means near-zero penalty at landing speeds, significant
      penalty at crash-inducing speeds.
    - attitude_penalty unchanged.
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 40.0 * (old_dist - new_dist)

    # ---------- velocity_penalty: discourage high-speed diving ----------
    # Quadratic: gentle at low speeds (landing), significant at high speeds (crashing)
    vx = next_obs[2]
    vy = next_obs[3]
    velocity_penalty = -0.3 * (vx ** 2 + vy ** 2)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: continuous per-step reward for safe contact ----------
    both_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    v_mag = abs(vx) + abs(vy)
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at θ=0, → 0.17 at |θ|=1

    soft_landing_bonus = 4.0 * both_contact * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + velocity_penalty + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "velocity_penalty": velocity_penalty,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components