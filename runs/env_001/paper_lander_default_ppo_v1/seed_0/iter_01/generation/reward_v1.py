def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for 2D lander: drive the agent to reach and settle on the central pad.
    Components:
      - progress_delta_reward: reward reduction in distance to the target (0,0)
      - stability_penalty: small penalty for high speed, tilt and angular velocity
      - soft_landing_proxy: bonus when near target, slow, upright and both feet in contact
    All components use only allowed signals (obs, next_obs, action). No terminal flags used.
    """
    # -- Helper: distance to target (target is at (0,0))
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # -- 1. Progress delta (main driving signal)
    delta_dist = dist_current - dist_next
    w_progress = 1.0
    progress_reward = w_progress * delta_dist

    # -- 2. Stability penalty (light constraint) on next state
    w_vel   = 0.01   # penalty per unit speed sum
    w_angle = 0.01   # penalty per radian of body angle
    w_angvel= 0.001  # penalty per angular velocity unit

    stability_penalty = (
        -w_vel   * (abs(next_obs[2]) + abs(next_obs[3]))
        -w_angle * abs(next_obs[4])
        -w_angvel* abs(next_obs[5])
    )

    # -- 3. Soft landing proxy: conditions for being safely on the pad
    dist_threshold   = 0.5
    vel_threshold    = 0.3
    angle_threshold  = 0.2   # rad (~11.5 deg)

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