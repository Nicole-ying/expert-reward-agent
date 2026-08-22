def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V1 reward for single-leg hopper continuous locomotion.
    Drives forward progress while keeping the torso upright and rotationally stable.
    """
    # Weights
    w_fwd = 1.0          # main progress drive
    w_angle = 0.5        # torso tilt penalty
    w_angvel = 0.1       # torso angular velocity penalty

    # Main learning signal: forward velocity (after transition)
    # Clip to non‑negative so backward exploration is not penalised.
    forward_progress = w_fwd * max(0.0, next_obs[5])

    # Stability constraints based on next observation
    torso_angle_penalty = -w_angle * abs(next_obs[1])
    torso_angular_velocity_penalty = -w_angvel * abs(next_obs[7])

    total_reward = forward_progress + torso_angle_penalty + torso_angular_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "torso_angle_penalty": torso_angle_penalty,
        "torso_angular_velocity_penalty": torso_angular_velocity_penalty
    }

    return float(total_reward), components