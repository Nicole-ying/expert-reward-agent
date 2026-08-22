def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, omega, lc, rc = obs
    nx, ny, nvx, nvy, nangle, nomega, nlc, nrc = next_obs

    # 1. Shaped distance-to-target reward (stronger)
    d_prev = (x * x + y * y) ** 0.5
    d_curr = (nx * nx + ny * ny) ** 0.5
    dist_reward = 10.0 * (d_prev - d_curr)

    # 2. Soft-landing velocity penalty – only active near the ground
    ground_factor = 1.0 if ny < 1.5 else 0.0
    vel_penalty = -0.1 * (abs(nvx) + abs(nvy)) * ground_factor

    # 3. Gentle angle penalty (always active)
    angle_penalty = -0.08 * abs(nangle)

    # 4. Light step penalty to encourage speed
    step_penalty = -0.02

    # 5. Tiny fuel penalty for any non-idle action
    fuel_penalty = -0.01 if action != 0 else 0.0

    # 6. Dense touch bonus – reward every leg contact with the pad
    touch_bonus = 0.4 * (nlc + nrc)

    # 7. Sparse success bonus for a stable two‑legged landing
    success = (
        nlc > 0.5 and nrc > 0.5 and
        ny < 0.3 and
        abs(nangle) < 0.3 and
        abs(nvx) < 0.2 and abs(nvy) < 0.2
    )
    success_bonus = 100.0 if success else 0.0

    total = (
        dist_reward
        + vel_penalty
        + angle_penalty
        + step_penalty
        + fuel_penalty
        + touch_bonus
        + success_bonus
    )

    components = {
        "dist_reward": dist_reward,
        "vel_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "step_penalty": step_penalty,
        "fuel_penalty": fuel_penalty,
        "touch_bonus": touch_bonus,
        "success_bonus": success_bonus,
    }
    return float(total), components