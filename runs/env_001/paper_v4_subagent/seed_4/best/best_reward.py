def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, angvel, l_contact, r_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nl_contact, nr_contact = next_obs

    # 1. Main progress signal: distance reduction to target pad
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_obs - dist_next

    # 2. Stability constraints (quadratic penalties on next state)
    #    (kept as in previous version, though they had negligible effect)
    angle_penalty = -0.1 * (nangle ** 2)
    angvel_penalty = -0.05 * (nangvel ** 2)

    # 3. Contact-gated soft landing attractor (tightened conditions)
    #    Only active when support legs are in contact, and rewards are given
    #    only when proximity, speed, and angle are all within narrow bands.
    proximity = max(0.0, 1.0 - dist_next / 0.5)          # dist < 0.5
    speed = abs(nvx) + abs(nvy)
    speed_factor = max(0.0, 1.0 - speed / 0.3)          # speed < 0.3
    angle_factor = max(0.0, 1.0 - abs(nangle) / 0.15)   # |angle| < 0.15

    contact_gate = float(nl_contact or nr_contact)
    landing_attractor = proximity * speed_factor * angle_factor * contact_gate

    w_progress = 10.0
    w_attractor = 2.0    # slightly increased to compensate for tighter gating

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