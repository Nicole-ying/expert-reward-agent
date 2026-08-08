def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distance to target pad center (0,0)
    dist = (x * x + y * y) ** 0.5 + 1e-8

    # 1. Radial velocity reward: always active, positive when approaching target
    dir_x = -x / dist
    dir_y = -y / dist
    radial_vel = vx * dir_x + vy * dir_y
    radial_reward = 3.0 * radial_vel   # can be positive or negative

    # 2. Velocity shaping: penalise horizontal drift and reward appropriate vertical speed
    # Desired downward speed depends on altitude; avoid hard coding sign, assume vy<0 is downward
    desired_vy = -0.3 if y > 0.3 else -0.1
    vy_error = (vy - desired_vy) ** 2
    vx_error = vx ** 2
    vel_penalty = -0.8 * vx_error - 0.4 * vy_error

    # 3. Attitude penalty: stay upright
    angle_penalty = -1.5 * abs(angle)

    # 4. Engine usage penalty (encourage no_engine)
    if action == 0:
        engine_penalty = 0.0
    elif action in (1, 3):   # orientation engines
        engine_penalty = -0.2
    elif action == 2:        # main engine
        engine_penalty = -0.4
    else:
        engine_penalty = 0.0

    # 5. Proximity bonus when low and near the pad: replaced old near_reward
    # Gives positive reward for being close to the pad while at low altitude
    if y < 0.5 and dist < 1.5:
        # bonus stronger when closer to (0,0) and near ground
        proximity_bonus = 3.0 * (1.5 - dist) * max(0.0, 0.5 - y)
    else:
        proximity_bonus = 0.0

    # 6. Contact and landing bonuses
    any_contact = (left_contact > 0.5 or right_contact > 0.5)
    full_contact = (left_contact > 0.5 and right_contact > 0.5)
    near_target = (abs(x) < 0.4 and y < 0.3)

    if full_contact and near_target:
        # Check for soft landing conditions
        if abs(vx) < 0.6 and abs(vy) < 0.6 and abs(angle) < 0.5:
            landing_bonus = 200.0   # perfect landing
        else:
            landing_bonus = 40.0    # landed but a bit rough, still reward
    elif full_contact and not near_target:
        # Landed on wrong place – penalise
        landing_bonus = -50.0
    elif any_contact and not full_contact:
        # Only one leg touching: encourage exploration
        landing_bonus = 5.0 if near_target else -10.0
    else:
        landing_bonus = 0.0

    # 7. Time penalty (per step)
    time_penalty = -0.05

    total_reward = (radial_reward + vel_penalty + angle_penalty +
                    engine_penalty + proximity_bonus + landing_bonus + time_penalty)

    components = {
        'radial_reward': radial_reward,
        'vel_penalty': vel_penalty,
        'angle_penalty': angle_penalty,
        'engine_penalty': engine_penalty,
        'proximity_bonus': proximity_bonus,
        'landing_bonus': landing_bonus,
        'time_penalty': time_penalty,
    }

    return float(total_reward), components