def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Parameters
    gamma = 0.995
    w_potential = 10.0  # amplify potential shaping to dominate penalty
    w_v = 0.02  # velocity penalty weight
    w_a = 0.1   # angle penalty weight
    w_w = 0.05  # angular velocity penalty weight
    landing_bonus = 2.0
    # Thresholds for soft landing proxy
    dist_thresh = 0.3
    speed_thresh_x = 0.1
    speed_thresh_y = 0.15
    angle_thresh = 0.15

    # 1. Potential based shaping:
    #    Phi(s) = - distance_to_target
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    phi_obs = -dist_obs
    phi_next = -dist_next
    potential_shaping = gamma * phi_next - phi_obs  # = -gamma*dist_next + dist_obs

    # 2. Stability penalty (encourages low speed, small angle, low angular velocity)
    stability_penalty = - (
        w_v * (abs(next_obs[2]) + abs(next_obs[3])) +
        w_a * abs(next_obs[4]) +
        w_w * abs(next_obs[5])
    )

    # 3. Soft landing proxy (indicates successful touchdown without explicit success flag)
    soft_landing_proxy = 0.0
    if (dist_next < dist_thresh
            and abs(next_obs[2]) < speed_thresh_x
            and abs(next_obs[3]) < speed_thresh_y
            and abs(next_obs[4]) < angle_thresh
            and next_obs[6] == 1.0
            and next_obs[7] == 1.0):
        soft_landing_proxy = landing_bonus

    total_reward = w_potential * potential_shaping + stability_penalty + soft_landing_proxy

    components = {
        "potential_shaping": potential_shaping,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components