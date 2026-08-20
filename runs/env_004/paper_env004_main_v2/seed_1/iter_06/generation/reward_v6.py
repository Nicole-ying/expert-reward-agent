def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V6 reward: V3 best structure + bounded quadratic speed bonus (Level 2 unbounded_to_bounded).
    
    Replaces the unbounded v^2 bonus (which V4 exploited to 0.99/step) with a saturating
    v^2/(1+alpha*v^2) form. This preserves the disproportionate high-speed gradient while
    capping extreme values to prevent destabilization.
    """
    # Weights
    w_fwd = 1.0          # main linear forward progress
    w_fwd2 = 0.15        # bounded quadratic bonus coefficient (slightly > V3's 0.1)
    w_angle = 0.5        # torso tilt penalty
    w_angvel = 0.1       # torso angular velocity penalty

    # Main learning signal: forward velocity (clipped to non-negative)
    v_fwd = max(0.0, next_obs[5])
    forward_progress = w_fwd * v_fwd

    # Bounded quadratic bonus: v^2 / (1 + alpha * v^2)
    # At v=2: bonus=0.50, v=3: bonus=0.93, asymptote=3.0
    # Prevents the unbounded exploitation seen in V4 while retaining gradient
    speed_sq = v_fwd ** 2
    alpha = 0.05
    forward_speed_bonus = w_fwd2 * speed_sq / (1.0 + alpha * speed_sq)

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