def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extracting relevant observations
    hull_angle_abs = abs(next_obs[0])
    hull_ang_vel_abs = abs(next_obs[1])
    horizontal_speed = next_obs[2]
    vertical_speed = next_obs[3]

    # Core forward progress: only reward positive horizontal speed
    forward_speed = max(0.0, horizontal_speed)

    # Soft health gate: reduces forward reward when posture deteriorates
    # Coefficients are chosen so that typical walking produces gate in [0.4, 0.8],
    # while large tilt or fast rotation significantly attenuate the reward.
    k_angle = 5.0
    k_ang_vel = 0.5
    gate = 1.0 / (1.0 + k_angle * hull_angle_abs + k_ang_vel * hull_ang_vel_abs)

    # Gated forward progress (main learning signal)
    w_fwd = 1.0
    progress_gated = w_fwd * forward_speed * gate

    # Vertical bounce penalty: only penalize excessive up/down oscillations
    vert_threshold = 0.5
    if abs(vertical_speed) > vert_threshold:
        excess = abs(vertical_speed) - vert_threshold
        vert_penalty = -0.1 * (excess ** 2)
    else:
        vert_penalty = 0.0

    total_reward = progress_gated + vert_penalty
    components = {
        'progress_gated': progress_gated,
        'vertical_penalty': vert_penalty
    }
    return float(total_reward), components