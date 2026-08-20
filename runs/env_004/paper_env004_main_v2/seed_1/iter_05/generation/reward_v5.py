def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V5 reward: Replace quadratic forward speed bonus with acceleration reward.
    Uses state_to_improvement: reward speed increases (delta-v) rather than
    rewarding the speed state itself, to help the agent escape the moderate-speed
    plateau and discover faster hopping gaits.
    """
    # Weights
    w_fwd = 1.0          # main linear forward progress
    w_accel = 1.0        # acceleration bonus for speed increases
    w_angle = 0.5        # torso tilt penalty
    w_angvel = 0.1       # torso angular velocity penalty

    # Main learning signal: forward velocity (clipped to non-negative)
    v_fwd = max(0.0, next_obs[5])
    forward_progress = w_fwd * v_fwd

    # Acceleration bonus: reward increases in forward speed
    # Uses obs[5] (pre-step velocity) and next_obs[5] (post-step velocity)
    delta_v = next_obs[5] - obs[5]
    acceleration_bonus = w_accel * max(0.0, delta_v)

    # Stability constraints based on next observation
    torso_angle_penalty = -w_angle * abs(next_obs[1])
    torso_angular_velocity_penalty = -w_angvel * abs(next_obs[7])

    total_reward = forward_progress + acceleration_bonus + torso_angle_penalty + torso_angular_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "acceleration_bonus": acceleration_bonus,
        "torso_angle_penalty": torso_angle_penalty,
        "torso_angular_velocity_penalty": torso_angular_velocity_penalty,
    }

    return float(total_reward), components