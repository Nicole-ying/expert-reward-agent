def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, omega, lc, rc = obs
    nx, ny, nvx, nvy, nangle, nomega, nlc, nrc = next_obs

    # Distance to the pad centre
    dist = (nx * nx + ny * ny) ** 0.5
    # Total linear speed
    speed = abs(nvx) + abs(nvy)
    # Absolute tilt angle
    tilt = abs(nangle)

    # 1. Proximity reward: always >= 0, highest when on the pad
    proximity_reward = 1.0 / (1.0 + dist)

    # 2. Speed reward: encourages gentle touchdown
    speed_reward = 1.0 / (1.0 + speed)

    # 3. Orientation reward: favours staying upright
    angle_reward = 1.0 / (1.0 + 5.0 * tilt)

    # 4. Contact bonus: tiny reward for each leg touching the ground
    contact_reward = 0.2 * (nlc + nrc)

    # 5. Fuel penalty: minuscule cost to promote efficiency
    fuel_penalty = -0.01 if action != 0 else 0.0

    # 6. Sparse success bonus – unchanged loose condition from previous function
    success = (
        nlc > 0.5 and nrc > 0.5 and
        ny < 0.3 and
        tilt < 0.3 and
        abs(nvx) < 0.2 and abs(nvy) < 0.2
    )
    success_bonus = 200.0 if success else 0.0

    total = proximity_reward + speed_reward + angle_reward + contact_reward + fuel_penalty + success_bonus

    components = {
        "proximity_reward": proximity_reward,
        "speed_reward": speed_reward,
        "angle_reward": angle_reward,
        "contact_reward": contact_reward,
        "fuel_penalty": fuel_penalty,
        "success_bonus": success_bonus,
    }
    return float(total), components