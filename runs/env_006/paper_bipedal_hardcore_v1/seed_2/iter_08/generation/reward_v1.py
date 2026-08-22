def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract useful signals ----------
    horizontal_speed = obs[2]          # forward direction
    hull_angle = obs[0]                # tilt (rad)
    hull_angular_velocity = obs[1]     # tilt speed (rad/s)

    # ---------- health gate: close to 1 when upright, decays when tilting ----------
    #   gate = 1 / (1 + 10 * angle^2 + 0.1 * angvel^2)
    #   avoids over-punishing early exploration, but heavily cuts progress reward
    #   when tilt becomes dangerous.
    denom = 1.0 + 10.0 * hull_angle * hull_angle + 0.1 * hull_angular_velocity * hull_angular_velocity
    health_gate = 1.0 / denom

    # ---------- forward progress (only positive direction) ----------
    #   only reward moving forward; ignore backward motion (max to avoid penalizing it)
    fwd_speed = max(0.0, horizontal_speed)
    progress_component = 1.0 * fwd_speed * health_gate   # w_speed = 1.0

    # ---------- moderate action regularisation ----------
    #   small penalty on large joint torques – just enough to avoid extreme signals
    action_sum_sq = action[0]*action[0] + action[1]*action[1] + action[2]*action[2] + action[3]*action[3]
    action_penalty = -0.01 * action_sum_sq

    # ---------- total reward ----------
    total_reward = progress_component + action_penalty

    components = {
        "progress": progress_component,
        "action_penalty": action_penalty
    }
    return float(total_reward), components