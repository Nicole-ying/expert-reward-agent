def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D bipedal locomotion on rough terrain.
    Drives forward progress while maintaining stable posture.
    No explicit success/failure flags are available in info.
    """
    # ---------- forward progress component ----------
    # Horizontal velocity in the forward direction (next_obs[2]).
    # Positive values correspond to moving forward, negative to backward.
    forward_velocity = next_obs[2]
    w_fwd = 1.0
    forward_reward = w_fwd * forward_velocity

    # ---------- stability penalty component ----------
    # Penalize large hull tilt (deviation from upright) and vertical velocity (jumping/falling).
    # Using absolute values gives smooth, dense gradients.
    hull_angle = next_obs[0]          # tilt angle
    vertical_velocity = next_obs[3]   # vertical speed

    w_angle = 0.5
    w_vertical = 0.1
    stability_penalty = -w_angle * abs(hull_angle) - w_vertical * abs(vertical_velocity)

    # ---------- total reward ----------
    total_reward = forward_reward + stability_penalty

    # components dict: only the terms that are directly summed into total_reward
    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components