def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft, nright = next_obs

    # Distances to target center (0,0)
    dist = (x**2 + y**2)**0.5 + 1e-8
    next_dist = (nx**2 + ny**2)**0.5 + 1e-8

    # 1. Progress towards center
    progress_delta = 5.0 * (dist - next_dist)

    # 2. Completion proxy (geometric mean of conditions)
    proximity = max(0.0, 1.0 - next_dist / 0.8)
    velocity_moderation = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 0.5)
    orientation_stability = max(0.0, 1.0 - abs(nangle) / 0.2)
    angvel_moderation = max(0.0, 1.0 - abs(nangvel) / 0.3)
    # Soft contact factor: always at least 0.1, reaching 1.0 when both feet touch
    contact_factor = 0.1 + 0.9 * (nleft + nright) / 2.0

    proxy_product = proximity * velocity_moderation * orientation_stability * angvel_moderation * contact_factor
    completion_proxy = 1.0 * (proxy_product ** 0.2) if proxy_product > 0 else 0.0

    # 3. Safety penalties (hinge, low thresholds)
    speed_threshold = 0.4
    vx_violation = max(0.0, abs(nvx) - speed_threshold)
    vy_violation = max(0.0, abs(nvy) - speed_threshold)
    speed_penalty = -0.1 * (vx_violation + vy_violation)

    angle_threshold = 0.2
    angle_violation = max(0.0, abs(nangle) - angle_threshold)
    angle_penalty = -0.2 * angle_violation

    angvel_threshold = 0.3
    angvel_violation = max(0.0, abs(nangvel) - angvel_threshold)
    angvel_penalty = -0.1 * angvel_violation

    total_reward = progress_delta + completion_proxy + speed_penalty + angle_penalty + angvel_penalty

    components = {
        'progress_delta': progress_delta,
        'completion_proxy': completion_proxy,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total_reward), components