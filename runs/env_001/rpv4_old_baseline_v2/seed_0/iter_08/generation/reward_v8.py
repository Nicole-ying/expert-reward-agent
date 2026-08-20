def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, omega, lc, rc = obs
    nx, ny, nvx, nvy, nangle, nomega, nlc, nrc = next_obs

    # 1. Potential-based shaping: encourage approaching the pad,
    #    slowing down, and staying upright simultaneously.
    dist_prev = (x * x + y * y) ** 0.5
    dist_curr = (nx * nx + ny * ny) ** 0.5
    speed_prev = abs(vx) + abs(vy)
    speed_curr = abs(nvx) + abs(nvy)

    potential_prev = -(1.0 * dist_prev + 0.5 * speed_prev + 0.5 * abs(angle))
    potential_curr = -(1.0 * dist_curr + 0.5 * speed_curr + 0.5 * abs(nangle))
    shaping_reward = potential_curr - potential_prev

    # 2. Small contact bonus – every leg contact gives a tiny signal.
    contact_bonus = 0.1 * (nlc + nrc)

    # 3. Tiny fuel penalty for any engine use.
    fuel_penalty = -0.01 if action != 0 else 0.0

    # 4. Sparse success bonus – big reward for a stable two‑legged landing.
    success = (
        nlc > 0.5 and nrc > 0.5 and
        ny < 0.3 and
        abs(nangle) < 0.3 and
        abs(nvx) < 0.2 and abs(nvy) < 0.2
    )
    success_bonus = 200.0 if success else 0.0

    total = shaping_reward + contact_bonus + fuel_penalty + success_bonus

    components = {
        "shaping_reward": shaping_reward,
        "contact_bonus": contact_bonus,
        "fuel_penalty": fuel_penalty,
        "success_bonus": success_bonus,
    }
    return float(total), components