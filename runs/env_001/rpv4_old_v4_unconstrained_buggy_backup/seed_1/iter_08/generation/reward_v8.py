def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v8 — Rollback to v4 skeleton, reduce proximity_progress 40→30.

    Diagnosis from v5-v7 history:
    - v4 (135.22, 19/20 landed) is the only working framework. Its skeleton:
      continuous per-step soft_landing_bonus + moderate proximity_progress.
    - v5 (proximity↑50, landing↓2.5) → crash diving: proximity dominates gradient.
    - v6 (+velocity_penalty) → permanent hovering: global speed penalty kills descent.
    - v7 (+fuel_penalty on v4) → crash diving: even tiny fuel penalty makes "no engine"
      the dominant early strategy, agent never explores soft landing.

    Single semantic change (L1):
    - Remove fuel_penalty (back to v4 skeleton).
    - Reduce proximity_progress 40.0 → 30.0 (v2/v3 level, proven stable).
      This reduces the crash-diving gradient while keeping approach guidance intact.
    - soft_landing_bonus kept at 4.0 (continuous per-step) — the key signal that
      made v4 achieve 19/20 landings.
    - attitude_penalty unchanged.
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: continuous per-step reward for safe contact ----------
    both_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

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