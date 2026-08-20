def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V2 reward: adds height-based shaping to encourage dynamic hopping.
    Drives forward progress while keeping the torso upright and at a healthy hopping height.
    """
    # Weights
    w_fwd = 1.0          # main progress drive
    w_angle = 0.5        # torso tilt penalty
    w_angvel = 0.1       # torso angular velocity penalty
    w_height = 0.5       # height tracking reward

    # Main learning signal: forward velocity (after transition)
    forward_progress = w_fwd * max(0.0, next_obs[5])

    # Stability constraints based on next observation
    torso_angle_penalty = -w_angle * abs(next_obs[1])
    torso_angular_velocity_penalty = -w_angvel * abs(next_obs[7])

    # Height tracking: reward proximity to target hopping height
    # Uses a triangular kernel: full reward at target, linearly decreasing to 0 at target +/- tolerance
    target_height = 1.0
    tolerance = 0.5
    height_deviation = abs(next_obs[0] - target_height)
    height_reward = w_height * max(0.0, 1.0 - height_deviation / tolerance)

    total_reward = forward_progress + torso_angle_penalty + torso_angular_velocity_penalty + height_reward

    components = {
        "forward_progress": forward_progress,
        "torso_angle_penalty": torso_angle_penalty,
        "torso_angular_velocity_penalty": torso_angular_velocity_penalty,
        "height_reward": height_reward,
    }

    return float(total_reward), components