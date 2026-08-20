def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V7 reward: V3 best structure + strengthened stability penalties (Level 1 scale fix).
    
    Evidence: V6's bounded speed bonus caused 40% crash rate. V3 was stable at 3260.
    Hypothesis: stability penalties at ~3% of reward are too weak to constrain high-speed
    exploration. Increasing them to ~6-8% should allow safe exploration of faster gaits.
    """
    # Weights
    w_fwd = 1.0          # main linear forward progress (unchanged from V3)
    w_fwd2 = 0.1         # quadratic bonus for higher speeds (unchanged from V3)
    w_angle = 0.8        # torso tilt penalty (V3: 0.5, strengthened for crash prevention)
    w_angvel = 0.2       # torso angular velocity penalty (V3: 0.1, strengthened)

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