def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V3 reward: V1 base + quadratic forward speed bonus.
    Restores the successful V1 structure and adds a mild bonus for high forward velocity
    to push the agent beyond a "safe but moderate" speed plateau.
    """
    # Weights
    w_fwd = 1.0          # main linear forward progress
    w_fwd2 = 0.1         # quadratic bonus for higher speeds
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