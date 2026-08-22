def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals
    body_z = next_obs[0]
    qx = next_obs[2]
    qy = next_obs[3]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]

    # 1. Health gate: height factor (safe range [0.3, 0.9] with soft falloffs)
    # Lower bound: clip to at least 0.1 so gradient never fully disappears
    if body_z < 0.3:
        height_factor = (body_z - 0.2) / 0.1   # reaches 0 at z=0.2
    elif body_z > 0.9:
        height_factor = (1.0 - body_z) / 0.1   # reaches 0 at z=1.0
    else:
        height_factor = 1.0
    height_factor = max(0.1, min(1.0, height_factor))

    # 2. Health gate: upright factor (from body_up_z = 1 - 2*(qx^2 + qy^2))
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    # Map [ -0.2 , 0.5 ] to [0, 1], with floor 0.1
    upright_factor = (body_up_z + 0.2) / 0.7
    upright_factor = max(0.1, min(1.0, upright_factor))

    health_gate = height_factor * upright_factor

    # 3. Main learning signal: squared forward velocity, gated by health
    forward_speed = max(0.0, body_x_vel)
    progress_reward = 2.0 * (forward_speed ** 2) * health_gate

    # 4. Light regularization against lateral drift
    lateral_penalty = -0.5 * (body_y_vel ** 2)

    total_reward = progress_reward + lateral_penalty
    components = {
        'forward_progress': progress_reward,
        'lateral_penalty': lateral_penalty
    }
    return float(total_reward), components