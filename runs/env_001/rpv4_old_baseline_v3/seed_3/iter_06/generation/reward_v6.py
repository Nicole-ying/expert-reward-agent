def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    xv, yv = obs[2], obs[3]
    angle = obs[4]
    lc, rc = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nxv, nyv = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nlc, nrc = next_obs[6], next_obs[7]

    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # ======= A. Progress: delta-distance toward target =======
    w_progress = 8.0
    progress_reward = w_progress * (dist - next_dist)

    # ======= B. Velocity penalty: proximity-gated speed cost =======
    # Active within 2.0m of target, ramps up as agent gets closer
    vel_proximity = max(0.0, 1.0 - next_dist / 2.0)
    next_speed = (nxv**2 + nyv**2) ** 0.5
    w_vel = 2.0
    velocity_penalty = -w_vel * vel_proximity * next_speed

    # ======= C. Stable contact: per-step reward for sustained platform contact =======
    # Only rewards double-contact when near the target (prevents fake landings far away)
    on_platform = (nlc > 0.5 and nrc > 0.5 and next_dist < 0.5)
    w_contact = 5.0
    stable_contact_reward = w_contact if on_platform else 0.0

    # ======= D. Touchdown transition: small one-time signal for achieving contact =======
    prev_on_platform = (lc > 0.5 and rc > 0.5 and dist < 0.5)
    next_on_platform = (nlc > 0.5 and nrc > 0.5 and next_dist < 0.5)
    touchdown_bonus = 0.0
    if not prev_on_platform and next_on_platform:
        soft_landing = abs(nxv) < 0.5 and abs(nyv) < 0.4 and abs(nangle) < 0.3
        touchdown_bonus = 8.0 if soft_landing else 3.0

    # ======= E. Orientation penalty: discourage tilt =======
    w_orient = 0.3
    orientation_penalty = -w_orient * abs(nangle)

    # ======= F. Fuel efficiency: small cost for engine use =======
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = (
        progress_reward
        + velocity_penalty
        + stable_contact_reward
        + touchdown_bonus
        + orientation_penalty
        + fuel_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "stable_contact_reward": stable_contact_reward,
        "touchdown_bonus": touchdown_bonus,
        "orientation_penalty": orientation_penalty,
        "fuel_penalty": fuel_penalty,
    }
    return float(total_reward), components