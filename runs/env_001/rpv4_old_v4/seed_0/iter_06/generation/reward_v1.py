def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 1. Approach target (main progress) ──
    distance = (x * x + y * y) ** 0.5
    approach_reward = -1.0 * distance

    # ── 2. Soft landing gate ──
    gate = 1.0 / (1.0 + 3.0 * distance)   # close to 1 when near pad

    vel_penalty = -0.3 * gate * (vx * vx + vy * vy + 0.1 * angular_vel * angular_vel)
    orientation_penalty = -0.2 * gate * (angle * angle)

    # ── 3. Fuel efficiency (light penalty) ──
    if action == 2:               # main engine
        fuel_penalty = -0.02
    elif action == 1 or action == 3:  # orientation engines
        fuel_penalty = -0.005
    else:                         # no engine
        fuel_penalty = 0.0

    total_reward = approach_reward + vel_penalty + orientation_penalty + fuel_penalty

    components = {
        'approach_reward': approach_reward,
        'vel_penalty': vel_penalty,
        'orientation_penalty': orientation_penalty,
        'fuel_penalty': fuel_penalty,
    }
    return float(total_reward), components