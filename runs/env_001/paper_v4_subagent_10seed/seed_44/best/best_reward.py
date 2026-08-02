def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations: both obs and next_obs are 8-dim vectors
    x,        y        = obs[0], obs[1]
    vx,       vy       = obs[2], obs[3]
    angle,    ang_vel  = obs[4], obs[5]
    # next_obs
    nx,       ny       = next_obs[0], next_obs[1]
    nvx,      nvy      = next_obs[2], next_obs[3]
    nangle,   nang_vel = next_obs[4], next_obs[5]
    lcon,     rcon     = next_obs[6], next_obs[7]  # contact flags at next state

    # ----- potential function (smaller values are better) -----
    def potential(px, py, pvx, pvy, pa):
        dist = (px**2 + py**2) ** 0.5
        vel  = (pvx**2 + pvy**2) ** 0.5
        return -(2.0 * dist + 1.0 * vel + 1.0 * abs(pa))

    # Main progress signal: improvement in potential
    pot_old = potential(x, y, vx, vy, angle)
    pot_new = potential(nx, ny, nvx, nvy, nangle)
    progress = pot_new - pot_old
    # Scale factor can be tuned, keep raw for now. Usually we want reward per step in range ~1.0
    main_progress = progress   # expected range roughly [-?..+?], but typical improvement gives ~0.1-1.0

    # ----- fuel efficiency (action cost) -----
    # action 0 = no engine, 1/2/3 = use engine
    fuel_penalty = -0.02 if action != 0 else 0.0

    # ----- extreme tilt hinge (hard safety) -----
    tilt = abs(nangle)
    tilt_limit = 0.5   # radians, strongly tilted
    if tilt > tilt_limit:
        extreme_tilt_penalty = -0.5 * (tilt - tilt_limit)
    else:
        extreme_tilt_penalty = 0.0

    # ----- soft contact encouragement (only when close to target) -----
    dist_to_target = (nx**2 + ny**2) ** 0.5
    proximity_factor = 1.0 / (1.0 + dist_to_target)   # close → 1, far → 0
    contact_bonus = 0.2 * lcon * rcon * proximity_factor

    # ----- total reward -----
    total_reward = main_progress + fuel_penalty + extreme_tilt_penalty + contact_bonus

    components = {
        "potential_delta": main_progress,
        "fuel_penalty": fuel_penalty,
        "extreme_tilt_penalty": extreme_tilt_penalty,
        "stable_contact_bonus": contact_bonus
    }
    return float(total_reward), components