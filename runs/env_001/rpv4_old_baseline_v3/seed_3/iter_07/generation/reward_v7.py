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

    # === A. Proximity: dense positive shaping toward target ===
    # Always active, grows smoothly as agent approaches.
    proximity_reward = 2.5 / (1.0 + next_dist)

    # === B. Progress: bonus for reducing distance this step ===
    progress_reward = 4.0 * (dist - next_dist)

    # === C. Touchdown: one-time transition bonus for first platform contact ===
    prev_on_platform = (lc > 0.5 and rc > 0.5 and dist < 0.5)
    on_platform = (nlc > 0.5 and nrc > 0.5 and next_dist < 0.5)
    touchdown_bonus = 0.0
    if not prev_on_platform and on_platform:
        touchdown_bonus = 20.0

    # === D. Sustained contact: per-step reward while on platform ===
    contact_reward = 4.0 if on_platform else 0.0

    # === E. Speed penalty: only active while ON platform (discourage sliding) ===
    next_speed = (nxv**2 + nyv**2) ** 0.5
    speed_penalty = -1.5 * next_speed if on_platform else 0.0

    # === F. Orientation penalty: discourage tilt ===
    orientation_penalty = -0.15 * abs(nangle)

    # === G. Fuel cost: mild regularization for engine use ===
    fuel_cost = -0.02 if action != 0 else 0.0

    total = (proximity_reward + progress_reward + touchdown_bonus +
             contact_reward + speed_penalty + orientation_penalty + fuel_cost)

    components = {
        "proximity_reward": proximity_reward,
        "progress_reward": progress_reward,
        "touchdown_bonus": touchdown_bonus,
        "contact_reward": contact_reward,
        "speed_penalty": speed_penalty,
        "orientation_penalty": orientation_penalty,
        "fuel_cost": fuel_cost,
    }
    return float(total), components