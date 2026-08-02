def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = obs[0]
    y = obs[1]
    vx = obs[2]
    vy = obs[3]
    angle = obs[4]
    angvel = obs[5]
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    next_angle = next_obs[4]
    next_angvel = next_obs[5]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # 1. Progress towards center (0,0)
    dist = (x**2 + y**2)**0.5 + 1e-6
    next_dist = (next_x**2 + next_y**2)**0.5 + 1e-6
    progress_delta = dist - next_dist

    # 2. Orientation stability penalty (hinge)
    angle_threshold = 0.3
    angvel_threshold = 0.5
    angle_violation = max(0.0, abs(next_angle) - angle_threshold)
    angvel_violation = max(0.0, abs(next_angvel) - angvel_threshold)
    orientation_penalty = -0.1 * angle_violation - 0.05 * angvel_violation

    # 3. Speed safety penalty (hinge)
    speed_threshold = 0.5
    vx_violation = max(0.0, abs(next_vx) - speed_threshold)
    vy_violation = max(0.0, abs(next_vy) - speed_threshold)
    speed_penalty = -0.05 * (vx_violation + vy_violation)

    # 4. Contact encouragement (dense reward for feet on ground)
    contact_reward = 0.1 * (next_left + next_right)

    # 5. NEW: Landing progress reward -- drives completion and stabilization
    # Proximity to center: 1.0 when at center, falls to 0 at distance 0.5
    proximity = max(0.0, 1.0 - next_dist / 0.5)
    # Velocity moderation: 1.0 when total speed near zero, falls off
    velocity_moderation = max(0.0, 1.0 - (abs(next_vx) + abs(next_vy)) / 0.5)
    # Orientation stability: 1.0 when angle near zero
    orientation_stability = max(0.0, 1.0 - abs(next_angle) / 0.2)
    # Angular velocity moderation
    angvel_moderation = max(0.0, 1.0 - abs(next_angvel) / 0.3)
    # Contact factor: continuous [0,1] for average contact
    contact_factor = (next_left + next_right) / 2.0

    landing_factor = proximity * velocity_moderation * orientation_stability * angvel_moderation * contact_factor
    landing_progress = 0.15 * landing_factor

    total_reward = progress_delta + orientation_penalty + speed_penalty + contact_reward + landing_progress

    components = {
        'progress_delta': progress_delta,
        'orientation_penalty': orientation_penalty,
        'speed_penalty': speed_penalty,
        'contact_reward': contact_reward,
        'landing_progress': landing_progress
    }

    return float(total_reward), components