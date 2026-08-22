def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, omega, lc, rc = obs
    nx, ny, nvx, nvy, nangle, nomega, nlc, nrc = next_obs

    # Distance to pad centre (current step, previous step)
    new_dist = (nx * nx + ny * ny) ** 0.5
    old_dist = (x * x + y * y) ** 0.5

    # 1. Dense proximity – strongly rewards being on/near the pad
    proximity_reward = 10.0 / (0.5 + new_dist)

    # 2. Contact – large reward for each leg touching the ground
    contact_reward = 5.0 * (nlc + nrc)

    # 3. Settle – continuous high reward once soft‑landing conditions hold
    settled = (
        nlc > 0.5 and nrc > 0.5 and
        abs(nvx) < 0.2 and abs(nvy) < 0.2 and
        abs(nangle) < 0.3
    )
    settle_reward = 10.0 if settled else 0.0

    # 4. Progress – small incentive to move toward the pad every step
    progress_reward = 0.1 * (old_dist - new_dist)

    # 5. Fuel penalty – gentle pressure not to fire engines unnecessarily
    fuel_penalty = -0.05 if action != 0 else 0.0

    total = proximity_reward + contact_reward + settle_reward + progress_reward + fuel_penalty

    components = {
        "proximity_reward": proximity_reward,
        "contact_reward": contact_reward,
        "settle_reward": settle_reward,
        "progress_reward": progress_reward,
        "fuel_penalty": fuel_penalty,
    }
    return float(total), components