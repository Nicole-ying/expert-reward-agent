def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: Scale fix — stability penalty coefficients reduced ~10x to stop
    cancelling the progress signal. Progress and soft_landing_proxy unchanged.
    """
    # -- Helper: distance to target (target is at (0,0))
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # -- 1. Progress delta (main driving signal, unchanged)
    delta_dist = dist_current - dist_next
    w_progress = 1.0
    progress_reward = w_progress * delta_dist

    # -- 2. Stability penalty: coefficients reduced ~10x
    w_vel   = 0.001   # was 0.01
    w_angle = 0.001   # was 0.01
    w_angvel= 0.0001  # was 0.001

    stability_penalty = (
        -w_vel   * (abs(next_obs[2]) + abs(next_obs[3]))
        -w_angle * abs(next_obs[4])
        -w_angvel* abs(next_obs[5])
    )

    # -- 3. Soft landing proxy (unchanged)
    dist_threshold   = 0.5
    vel_threshold    = 0.3
    angle_threshold  = 0.2

    near_target = dist_next < dist_threshold
    low_speed   = (abs(next_obs[2]) < vel_threshold) and (abs(next_obs[3]) < vel_threshold)
    upright     = abs(next_obs[4]) < angle_threshold
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)

    w_proxy = 0.5
    soft_landing_proxy = w_proxy if (near_target and low_speed and upright and both_contact) else 0.0

    # -- Total reward
    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components