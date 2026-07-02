def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== Helper: distance to goal ==========
    # Goal is assumed to be at (x=0, y=0) relative to the landing pad centre / reference height
    def dist_to_goal(x, y):
        return (x**2 + y**2) ** 0.5

    # ========== 1. Progress delta reward (main learning signal) ==========
    d_current = dist_to_goal(obs[0], obs[1])
    d_next = dist_to_goal(next_obs[0], next_obs[1])
    progress_delta = d_current - d_next   # positive when getting closer

    # ========== 2. Stability penalty (light constraint) ==========
    # Penalise high velocity, large tilt and fast rotation
    w_vel   = 0.1
    w_angle = 0.5
    w_omega = 0.1

    speed_penalty = w_vel * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = w_angle * abs(next_obs[4])
    omega_penalty = w_omega * abs(next_obs[5])

    stability_penalty = -(speed_penalty + angle_penalty + omega_penalty)

    # ========== 3. Soft landing proxy (task‑completion hint) ==========
    # A small bonus when all conditions are met: near target, slow, upright and both supports in contact.
    near_target   = d_next < 0.2
    low_speed     = (abs(next_obs[2]) < 0.2) and (abs(next_obs[3]) < 0.2)
    stable_angle  = abs(next_obs[4]) < 0.1
    both_contact  = (next_obs[6] == 1.0) and (next_obs[7] == 1.0)

    soft_landing_proxy = 0.2 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # ========== Total reward ==========
    # Scale progress_delta to a meaningful magnitude (environment‑dependent, tune later)
    total_reward = 10.0 * progress_delta + stability_penalty + soft_landing_proxy

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components