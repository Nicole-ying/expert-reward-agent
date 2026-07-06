def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v2 for hopping locomotion: encouraging forward progress while staying upright.
    Components:
        forward_stability_reward  – forward velocity weighted by torso uprightness
        stability_penalty         – penalty for torso tilt and angular velocity
    """
    # --- extract relevant signals from next_obs ---
    torso_angle = next_obs[1]          # rad, 0 = upright
    forward_vel = next_obs[5]          # positive = forward
    torso_ang_vel = next_obs[7]        # rad/s

    # --- forward progress with upright gating ---
    # Exponential decay of the forward reward as the torso tilts.
    temp = 0.3
    upright_factor = 2.718281828 ** (-abs(torso_angle) / temp)
    forward_stability_reward = forward_vel * upright_factor

    # --- light stability penalty ---
    w_angle = 0.1
    w_ang_vel = 0.01
    stability_penalty = -w_angle * (torso_angle ** 2) - w_ang_vel * (torso_ang_vel ** 2)

    # --- total reward ---
    total_reward = forward_stability_reward + stability_penalty

    components = {
        "forward_stability_reward": forward_stability_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components