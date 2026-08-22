def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nxv, nyv = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nangv = next_obs[5]
    nlc, nrc = next_obs[6], next_obs[7]
    olc, orc = obs[6], obs[7]

    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    next_speed = (nxv**2 + nyv**2) ** 0.5

    # === A. Progress: potential-based proximity (zero for hovering) ===
    # Potential Φ(s) = 1/(1+dist). Reward = γ * (Φ(next) - Φ(current))
    # Approaching: positive. Hovering: zero. Retreating: negative.
    potential = 1.0 / (1.0 + dist)
    next_potential = 1.0 / (1.0 + next_dist)
    progress_reward = 10.0 * (next_potential - potential)

    # === B. Contact: per-step reward for leg contact, no distance condition ===
    prev_both = (olc > 0.5 and orc > 0.5)
    curr_both = (nlc > 0.5 and nrc > 0.5)
    curr_one = ((nlc > 0.5) or (nrc > 0.5)) and not curr_both

    # Both legs on platform: massive per-step reward (the main driver)
    landing_reward = 50.0 if curr_both else (5.0 if curr_one else 0.0)

    # === C. Touchdown: one-time transition bonus for first full contact ===
    touchdown_bonus = 500.0 if (not prev_both and curr_both) else 0.0

    # === D. Speed penalty: platform-only (free movement during approach) ===
    on_platform = curr_both
    speed_penalty = -1.5 * next_speed if on_platform else 0.0

    # === E. Orientation: stay upright, always active but mild ===
    orientation_penalty = -0.5 * abs(nangle) - 0.15 * abs(nangv)

    # === F. Fuel cost: mild regularization ===
    fuel_cost = -0.03 if action != 0 else 0.0

    total = (progress_reward + landing_reward + touchdown_bonus +
             speed_penalty + orientation_penalty + fuel_cost)

    components = {
        "progress_reward": progress_reward,
        "landing_reward": landing_reward,
        "touchdown_bonus": touchdown_bonus,
        "speed_penalty": speed_penalty,
        "orientation_penalty": orientation_penalty,
        "fuel_cost": fuel_cost,
    }
    return float(total), components