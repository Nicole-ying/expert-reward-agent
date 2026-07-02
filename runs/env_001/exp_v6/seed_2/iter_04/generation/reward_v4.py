def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- position and progress delta (main learning signal) ----
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    d_obs = (px**2 + py**2) ** 0.5
    d_next = (nx**2 + ny**2) ** 0.5
    progress_delta = d_obs - d_next

    # ---- distance penalty ----
    distance_penalty = -0.005 * d_next

    # ---- stability penalty ----
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    stability_penalty = (
        -0.005 * abs(vx)
        - 0.005 * abs(vy)
        - 0.02 * abs(angle)
        - 0.02 * abs(ang_vel)
    )

    # ---- approach quality bonus (KEY FIX: gated by actual progress) ----
    # Previous version paid agent for "being in a good state" every step,
    # creating a hover-and-collect local optimum.  Now the bonus only fires
    # when the agent is actually approaching (progress_delta > 0), converting
    # it from a "state reward" to a "quality-of-progress amplifier".
    APPROACH_DIST = 2.0
    SPEED_THRESH = 2.0
    ANGLE_THRESH = 0.5

    proximity = max(0.0, 1.0 - d_next / APPROACH_DIST)
    speed_ok = max(0.0, 1.0 - (abs(vx) + abs(vy)) / SPEED_THRESH)
    angle_ok = max(0.0, 1.0 - abs(angle) / ANGLE_THRESH)

    # progress_delta ~0.002 on average when positive; quality factors ~0.5-0.8.
    # scale=2.0 gives bonus ~0.002-0.005 per approaching step, comparable to
    # progress_delta itself — a meaningful amplifier, not a dominating signal.
    approach_bonus = proximity * max(0.0, progress_delta) * speed_ok * angle_ok * 2.0

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