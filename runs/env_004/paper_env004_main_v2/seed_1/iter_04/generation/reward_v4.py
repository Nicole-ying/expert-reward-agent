def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V4 reward: V3 base with stronger quadratic forward speed bonus.
    Increases w_fwd2 from 0.1 to 0.25 to provide steeper gradient
    for higher forward velocities and push past the moderate-speed plateau.
    """
    # Weights
    w_fwd = 1.0          # main linear forward progress
    w_fwd2 = 0.25        # quadratic bonus for higher speeds (increased from 0.1)
    w_angle = 0.5        # torso tilt penalty
    w_angvel = 0.1       # torso angular velocity penalty

    # Main learning signal: forward velocity (clipped to non-negative)
    v_fwd = max(0.0, next_obs[5])
    forward_progress = w_fwd * v_fwd

    # Quadratic bonus: disproportionately rewards higher forward speeds
    forward_speed_bonus = w_fwd2 * (v_fwd ** 2)

    # Stability constraints based on next observation
    torso_angle_penalty = -w_angle * abs(next_obs[1])
    torso_angular_velocity_penalty = -w_angvel * abs(next_obs[7])

    total_reward = forward_progress + forward_speed_bonus + torso_angle_penalty + torso_angular_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "forward_speed_bonus": forward_speed_bonus,
        "torso_angle_penalty": torso_angle_penalty,
        "torso_angular_velocity_penalty": torso_angular_velocity_penalty,
    }

    return float(total_reward), components