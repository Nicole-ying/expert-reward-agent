def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2 reward: progressive distance reduction + attitude stabilisation +
    distance‑gated velocity penalty + leg‑contact bonus.
    """
    # coefficients
    w_progress = 1.0
    w_attitude = 0.05
    w_velocity = 0.5     # increased to be effective
    w_contact = 0.2      # per leg contact

    # ---------- goal progress (potential‑based delta distance) ----------
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    goal_progress = w_progress * (dist_current - dist_next)

    # ---------- upright stabilisation ----------
    attitude_penalty = -w_attitude * (next_obs[4] ** 2 + 0.1 * next_obs[5] ** 2)

    # ---------- distance‑gated velocity penalty ----------
    # only active near the landing pad to encourage gentle final approach
    vel_penalty = 0.0
    if dist_next < 0.5:
        speed_sq = next_obs[2] ** 2 + next_obs[3] ** 2
        vel_penalty = -w_velocity * speed_sq

    # ---------- leg contact bonus (touchdown feedback) ----------
    # obs[6] and obs[7] are left/right leg contact indicators (0/1)
    leg_contact = 0.0
    if next_obs[6] > 0.5:
        leg_contact += w_contact
    if next_obs[7] > 0.5:
        leg_contact += w_contact

    total_reward = goal_progress + attitude_penalty + vel_penalty + leg_contact

    components = {
        "goal_progress": goal_progress,
        "attitude_penalty": attitude_penalty,
        "velocity_penalty": vel_penalty,
        "leg_contact": leg_contact
    }
    return float(total_reward), components