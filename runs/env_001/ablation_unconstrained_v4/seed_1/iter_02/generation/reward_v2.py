def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- extract current and next state ----
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # ---- distances ----
    prev_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # proximity gate: ~1.0 near pad, decays to ~0 far away
    near_pad = 2.718281828 ** (-next_dist**2 / 0.5)

    # Component A: distance progress (strengthened main signal)
    progress = prev_dist - next_dist
    c_progress = 3.0 * progress

    # Component B: continuous proximity shaping (dense gradient everywhere)
    c_proximity = 1.0 / (1.0 + next_dist)

    # Component C: decomposed contact reward (additive, not product)
    contact_count = n_left_contact + n_right_contact  # [0.0, 1.0, 2.0]
    c_contact = 3.0 * contact_count * near_pad

    # Component D: speed penalty gated by proximity
    speed_sq = nvx**2 + nvy**2
    c_speed = -0.5 * speed_sq * near_pad

    # Component E: posture shaping (mild global + focused near-pad)
    c_angle_global = -0.03 * (n_angle**2)
    c_angle_near   = -0.5 * (n_angle**2) * near_pad
    c_angvel_global = -0.003 * (n_angvel**2)
    c_angvel_near   = -0.05 * (n_angvel**2) * near_pad

    total_reward = (
        c_progress
        + c_proximity
        + c_contact
        + c_speed
        + c_angle_global
        + c_angle_near
        + c_angvel_global
        + c_angvel_near
    )

    components = {
        'distance_progress': c_progress,
        'proximity_shaping': c_proximity,
        'contact_reward': c_contact,
        'speed_penalty': c_speed,
        'angle_global': c_angle_global,
        'angle_near': c_angle_near,
        'angvel_global': c_angvel_global,
        'angvel_near': c_angvel_near,
    }

    return (float(total_reward), components)