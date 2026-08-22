def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft, nright = next_obs

    # distance to target center
    dist = (x**2 + y**2)**0.5 + 1e-8
    next_dist = (nx**2 + ny**2)**0.5 + 1e-8

    # baseline progress: moving towards the center
    progress = 5.0 * (dist - next_dist)

    # contact reward (encourage foot contact)
    contact_reward = 0.2 * (nleft + nright)

    # ---------- new completion proxy (product, geometric mean) ----------
    # individual factors (each bounded away from zero to avoid collapse)
    proximity_factor = max(1e-3, 1.0 - next_dist / 0.3)                # closer to center
    velocity_factor  = max(1e-3, 1.0 - (abs(nvx) + abs(nvy)) / 0.3)    # low speed
    angle_factor     = max(1e-3, 1.0 - abs(nangle) / 0.15)             # upright
    angvel_factor    = max(1e-3, 1.0 - abs(nangvel) / 0.2)             # low spin

    # geometric mean avoids single-factor collapse and gives smooth gradient
    completion = 5.0 * (proximity_factor * velocity_factor * angle_factor * angvel_factor) ** 0.25

    # ---------- safety penalties (unchanged from last round) ----------
    speed_penalty  = -0.5 * (max(0.0, abs(nvx) - 0.4) + max(0.0, abs(nvy) - 0.4))
    angle_penalty  = -1.0 * max(0.0, abs(nangle) - 0.15)
    angvel_penalty = -0.3 * max(0.0, abs(nangvel) - 0.3)

    total_reward = (progress + contact_reward + completion +
                    speed_penalty + angle_penalty + angvel_penalty)

    components = {
        'progress': progress,
        'contact_reward': contact_reward,
        'completion': completion,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total_reward), components