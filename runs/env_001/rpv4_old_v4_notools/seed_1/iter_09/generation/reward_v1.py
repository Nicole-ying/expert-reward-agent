def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward: progressive distance reduction + attitude stabilisation + height‑dependent landing speed penalty.
    No terminal events, no contact‑based bonuses, no explicit fuel penalty in this iteration.
    """
    # coefficients – tune cautiously in the outer loop
    w_progress = 1.0
    w_attitude = 0.05
    w_velocity = 0.3

    # ---------- goal progress (potential‑based / delta distance) ----------
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    goal_progress = w_progress * (dist_current - dist_next)

    # ---------- upright stabilisation ----------
    # penalty on body angle (next_obs[4]) and angular velocity (next_obs[5])
    attitude_penalty = -w_attitude * (next_obs[4] ** 2 + 0.1 * next_obs[5] ** 2)

    # ---------- soft velocity penalty (height‑gated) ----------
    # activate only when the vehicle is low (y close to 0) to encourage gentle touchdown
    y = next_obs[1]
    gate = max(0.0, 1.0 - y / 0.5)  # linear decay: gate=1 when y=0, gate=0 when y>=0.5
    speed_sq = next_obs[2] ** 2 + next_obs[3] ** 2
    soft_velocity_penalty = -w_velocity * speed_sq * gate

    total_reward = goal_progress + attitude_penalty + soft_velocity_penalty

    components = {
        "goal_progress": goal_progress,
        "attitude_penalty": attitude_penalty,
        "soft_velocity_penalty": soft_velocity_penalty
    }
    return float(total_reward), components