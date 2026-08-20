def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations (all are scalar float)
    x, y, vx, vy, angle, angvel, l_contact, r_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nl_contact, nr_contact = next_obs

    # 1. Main progress signal: distance reduction to target pad (improvement_delta)
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_obs - dist_next  # positive when getting closer

    # 2. Stability constraints (quadratic penalties on next state)
    angle_penalty = -0.1 * (nangle ** 2)        # penalize tilt
    angvel_penalty = -0.05 * (nangvel ** 2)     # penalize rotation

    # 3. Soft landing proxy (joint_condition_proxy using bounded factors)
    v_thresh = 0.2
    contact = nl_contact * nr_contact                 # 0 or 1
    factor_vx = max(0.0, 1.0 - abs(nvx) / v_thresh)  # [0,1], 1 when speed near 0
    factor_vy = max(0.0, 1.0 - abs(nvy) / v_thresh)  # [0,1]
    landing_bonus = contact * factor_vx * factor_vy   # in [0,1], only when settled

    # Weights
    w_progress = 10.0
    w_landing  = 20.0

    total = (w_progress * progress +
             angle_penalty + angvel_penalty +
             w_landing * landing_bonus)

    components = {
        "progress": w_progress * progress,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "landing_bonus": w_landing * landing_bonus
    }

    return float(total), components