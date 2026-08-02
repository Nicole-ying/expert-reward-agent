def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations (all are scalar float)
    x, y, vx, vy, angle, angvel, l_contact, r_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nl_contact, nr_contact = next_obs

    # 1. Main progress signal: distance reduction to target pad (improvement_delta)
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_obs - dist_next  # positive when getting closer

    # 2. Stability constraints (quadratic penalties on next state)
    angle_penalty = -0.1 * (nangle ** 2)
    angvel_penalty = -0.05 * (nangvel ** 2)

    # 3. Soft landing attractor (replaces the dead landing_bonus)
    #    Dense signal that grows as agent approaches, slows, levels, and contacts.
    dist_norm = (nx**2 + ny**2) ** 0.5
    proximity = 2.718281828 ** (-dist_norm / 0.8)         # [0,1], peak at origin

    speed_sum = abs(nvx) + abs(nvy)
    speed_factor = max(0.0, 1.0 - speed_sum / 1.0)       # [0,1], 1 when fully stopped

    angle_factor = max(0.0, 1.0 - abs(nangle) / 0.5)     # [0,1], 1 when level

    contact = float(nl_contact or nr_contact)            # 0 or 1
    contact_boost = 1.0 + 2.0 * contact                  # ×1 without contact, ×3 with

    landing_attractor = proximity * speed_factor * angle_factor * contact_boost

    w_progress = 10.0
    w_attractor = 1.0

    total = (w_progress * progress +
             angle_penalty + angvel_penalty +
             w_attractor * landing_attractor)

    components = {
        "progress": w_progress * progress,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "landing_bonus": w_attractor * landing_attractor
    }

    return float(total), components