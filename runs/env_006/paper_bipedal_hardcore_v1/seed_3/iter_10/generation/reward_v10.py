def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extracting relevant observations
    hull_angle_abs = abs(next_obs[0])
    hull_ang_vel_abs = abs(next_obs[1])
    horizontal_speed = next_obs[2]
    leg_1_contact = next_obs[12]
    leg_2_contact = next_obs[13]

    # Core forward progress: only reward positive horizontal speed
    forward_speed = max(0.0, horizontal_speed)

    # Soft health gate: reduces forward reward when posture deteriorates
    k_angle = 5.0
    k_ang_vel = 0.5
    gate = 1.0 / (1.0 + k_angle * hull_angle_abs + k_ang_vel * hull_ang_vel_abs)

    # Gated forward progress (main learning signal)
    w_fwd = 1.0
    progress_gated = w_fwd * forward_speed * gate

    # Air penalty: discourages both feet leaving the ground simultaneously
    # air_factor = 0.0 when both feet on ground, 0.5 when one foot, 1.0 when airborne
    air_factor = 1.0 - 0.5 * (leg_1_contact + leg_2_contact)
    air_penalty = -0.1 * air_factor

    total_reward = progress_gated + air_penalty
    components = {
        'progress_gated': progress_gated,
        'air_penalty': air_penalty
    }
    return float(total_reward), components