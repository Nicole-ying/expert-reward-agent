def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V8 reward: V7 base with torso_angular_velocity_penalty restructured as a deadzoned
    penalty (Level 2: unbounded_to_bounded). Moderate angular velocity inherent to hopping
    is now penalty-free; only excess angular velocity beyond 0.5 rad/s is penalized.
    
    Evidence: V7's global angvel penalty suppressed forward speed to avoid oscillations
    needed for fast hopping. Deadzone allows natural rhythmic motion while retaining
    protection against dangerous instability.
    """
    # Weights
    w_fwd = 1.0          # main linear forward progress (unchanged from V7)
    w_fwd2 = 0.1         # quadratic bonus for higher speeds (unchanged from V7)
    w_angle = 0.8        # torso tilt penalty (unchanged from V7)
    
    # Deadzoned angular velocity penalty: only penalize |angvel| > 0.5 rad/s
    w_angvel = 1.0       # increased weight to maintain bite on extreme deviations
    angvel_threshold = 0.5  # natural hopping oscillation allowed up to this threshold

    # Main learning signal: forward velocity (clipped to non-negative)
    v_fwd = max(0.0, next_obs[5])
    forward_progress = w_fwd * v_fwd

    # Quadratic bonus: disproportionately rewards higher forward speeds
    forward_speed_bonus = w_fwd2 * (v_fwd ** 2)

    # Stability constraints
    torso_angle_penalty = -w_angle * abs(next_obs[1])
    
    # Deadzoned: only the excess above threshold is penalized
    excess_angvel = max(0.0, abs(next_obs[7]) - angvel_threshold)
    torso_angular_velocity_penalty = -w_angvel * excess_angvel

    total_reward = forward_progress + forward_speed_bonus + torso_angle_penalty + torso_angular_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "forward_speed_bonus": forward_speed_bonus,
        "torso_angle_penalty": torso_angle_penalty,
        "torso_angular_velocity_penalty": torso_angular_velocity_penalty,
    }

    return float(total_reward), components