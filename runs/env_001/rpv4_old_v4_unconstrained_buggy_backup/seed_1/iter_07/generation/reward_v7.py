def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v7 — Adds fuel_penalty to v4 skeleton, encouraging efficient settling.

    Diagnosis from v4-v6 history:
    - v4 (135.22, 19/20 landed) is the only working framework, but soft_landing_bonus
      explodes late in training (cp100%=431.5). Agent bounces and re-lands to farm bonus.
    - v5 (+proximity, -landing) → high-speed crash diving (len=68, all terminated early).
    - v6 (+velocity_penalty) → permanent hovering (len=1000, 0/20 terminated,
      soft_landing_bonus 0% active). Global speed penalty kills landing incentive.
    - The v4 skeleton must be preserved; only v4 achieved reliable landings.

    Single semantic change: NEW fuel_penalty component.
    - Penalizes engine actions (1/2/3) with -0.15 per step.
    - Action 0 (no engine) costs nothing — gravity alone can descend.
    - Once landed and settled, agent can stay on action 0 indefinitely → natural settling incentive.
    - Bouncing requires engine re-fire → bouncing exploit is now taxed per re-fire.
    - Aligns with stated secondary objective: fuel efficiency.

    All other components unchanged from v4:
    - proximity_progress=40.0, soft_landing_bonus=4.0, attitude_penalty same.
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
    both_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at theta=0, → 0.17 at |theta|=1

    soft_landing_bonus = 4.0 * both_contact * velocity_quality * angle_quality

    # ---------- fuel_penalty: discourage unnecessary engine use (NEW) ----------
    # action 0 = no engine (free), actions 1/2/3 = engine fire (cost -0.15)
    fuel_penalty = 0.0 if action == 0 else -0.15

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus + fuel_penalty

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "fuel_penalty": fuel_penalty
    }
    return float(total_reward), components