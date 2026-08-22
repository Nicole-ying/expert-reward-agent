def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Parameters
    w_radial = 8.0             # reward for speed toward the target
    w_slow = 1.5               # reward for low speed (globally)
    w_angle = 1.0              # reward for upright attitude
    w_near = 3.0               # shaping reward when close to the pad
    w_orient = -0.05           # penalty for orientation engine
    w_main = -0.1              # penalty for main engine
    w_time = -0.01             # tiny time penalty
    w_landing = 200.0          # successful landing reward
    w_crash = -50.0            # penalty for non‑pad contact

    # Positions and distances
    x = next_obs[0]
    y = next_obs[1]
    dist = (x * x + y * y) ** 0.5 + 1e-8

    # Velocities
    vx = next_obs[2]
    vy = next_obs[3]

    # Attitude
    angle = next_obs[4]

    # Contacts
    left = next_obs[6]
    right = next_obs[7]

    # 1. Radial velocity reward: encourage moving toward the target
    dir_x = -x / dist
    dir_y = -y / dist
    radial_vel = vx * dir_x + vy * dir_y   # positive -> closing distance
    if radial_vel > 0:
        radial_reward = w_radial * radial_vel
    else:
        radial_reward = 0.0

    # 2. Low speed reward (global, decays when speed > 2.0)
    speed = (vx * vx + vy * vy) ** 0.5
    slow_reward = w_slow * max(0.0, 1.0 - speed / 2.0)

    # 3. Upright reward (angle in radians, reward falls linearly for |angle| > 0.5)
    angle_reward = w_angle * max(0.0, 1.0 - abs(angle) / 0.5)

    # 4. Proximity shaping: extra reward when inside a 2.0‑radius neighbourhood
    if dist < 2.0:
        near_reward = w_near * (2.0 - dist)
    else:
        near_reward = 0.0

    # 5. Engine usage penalty
    if action == 0:
        act_pen = 0.0
    elif action in (1, 3):
        act_pen = w_orient
    elif action == 2:
        act_pen = w_main
    else:
        act_pen = 0.0

    # 6. Constant time penalty
    time_pen = w_time

    # 7. Successful landing: both legs contact the pad near the target with low speed and upright
    full_contact = (left > 0.5 and right > 0.5)
    near_target = (abs(x) < 0.3 and abs(y) < 0.3)
    low_speed = (abs(vx) < 0.5 and abs(vy) < 0.5)
    upright = abs(angle) < 0.5
    if full_contact and near_target and low_speed and upright:
        landing_reward = w_landing
    else:
        landing_reward = 0.0

    # 8. Non‑pad contact penalty
    any_contact = (left > 0.5 or right > 0.5)
    if any_contact and not near_target:
        crash_pen = w_crash
    else:
        crash_pen = 0.0

    total_reward = (radial_reward + slow_reward + angle_reward +
                    near_reward + act_pen + time_pen +
                    landing_reward + crash_pen)

    components = {
        'radial_reward': radial_reward,
        'slow_reward': slow_reward,
        'angle_reward': angle_reward,
        'near_reward': near_reward,
        'act_pen': act_pen,
        'time_pen': time_pen,
        'landing_reward': landing_reward,
        'crash_pen': crash_pen,
    }

    return float(total_reward), components