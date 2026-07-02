def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- position and progress delta (main learning signal) ----
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    d_obs = (px**2 + py**2) ** 0.5
    d_next = (nx**2 + ny**2) ** 0.5
    progress_delta = d_obs - d_next

    # ---- distance penalty (unchanged) ----
    distance_penalty = -0.005 * d_next

    # ---- stability penalty (unchanged) ----
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    stability_penalty = (
        -0.005 * abs(vx)
        - 0.005 * abs(vy)
        - 0.02 * abs(angle)
        - 0.02 * abs(ang_vel)
    )

    # ---- approach quality bonus: potential-based shaping ----
    # KEY CHANGE: switched from state-reward (product form → hover exploit)
    # to potential-based shaping (Φ(next) - Φ(obs)).
    # Φ = proximity * speed_ok * angle_ok — a bounded [0,1] potential that
    # captures "how good is the approach state".
    # Bonus = Φ(next) - Φ(obs) — only rewards IMPROVEMENT, zero when static.
    # This naturally prevents the hover exploit: staying still gives ΔΦ = 0.
    APPROACH_DIST = 2.0
    SPEED_THRESH = 2.0
    ANGLE_THRESH = 0.5

    # Potential at current state (obs)
    prox_now = max(0.0, 1.0 - d_obs / APPROACH_DIST)
    spd_now = max(0.0, 1.0 - (abs(obs[2]) + abs(obs[3])) / SPEED_THRESH)
    ang_now = max(0.0, 1.0 - abs(obs[4]) / ANGLE_THRESH)
    potential_now = prox_now * spd_now * ang_now

    # Potential at next state (next_obs)
    prox_next = max(0.0, 1.0 - d_next / APPROACH_DIST)
    spd_next = max(0.0, 1.0 - (abs(vx) + abs(vy)) / SPEED_THRESH)
    ang_next = max(0.0, 1.0 - abs(angle) / ANGLE_THRESH)
    potential_next = prox_next * spd_next * ang_next

    # γ = 1.0 (standard for potential-based shaping, per Ng 1999)
    approach_bonus = (potential_next - potential_now) * 2.0

    # ---- total ----
    total = progress_delta + distance_penalty + stability_penalty + approach_bonus

    components = {
        "progress_delta_reward": progress_delta,
        "distance_penalty": distance_penalty,
        "stability_penalty": stability_penalty,
        "approach_bonus": approach_bonus,
        "total_reward": total,
    }
    return float(total), components