def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for hopping locomotion: forward progress + vertical activity + stability.
    Components:
        forward_stability_reward – forward velocity weighted by torso uprightness
        vertical_activity         – absolute vertical velocity weighted by uprightness (hopping signal)
        stability_penalty         – penalty for torso tilt and angular velocity
    """
    # --- extract relevant signals from next_obs ---
    torso_angle = next_obs[1]          # rad, 0 = upright
    forward_vel = next_obs[5]          # positive = forward
    vertical_vel = next_obs[6]         # vertical velocity (hopping oscillation)
    torso_ang_vel = next_obs[7]        # rad/s

    # --- shared upright gate ---
    temp = 0.3
    upright_factor = 2.718281828 ** (-abs(torso_angle) / temp)

    # --- forward progress with upright gating ---
    forward_stability_reward = forward_vel * upright_factor

    # --- vertical activity: encourage the up-down dynamics of hopping ---
    w_vert = 0.2
    vertical_activity = w_vert * abs(vertical_vel) * upright_factor

    # --- light stability penalty ---
    w_angle = 0.1
    w_ang_vel = 0.01
    stability_penalty = -w_angle * (torso_angle ** 2) - w_ang_vel * (torso_ang_vel ** 2)

    # --- total reward ---
    total_reward = forward_stability_reward + vertical_activity + stability_penalty

    components = {
        "forward_stability_reward": forward_stability_reward,
        "vertical_activity": vertical_activity,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components