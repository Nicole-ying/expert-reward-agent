def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_prev, y_prev = obs[0], obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 1. Progress reward (delta improvement on distance) ──
    #     Positive when approaching the target pad, negative when retreating.
    #     w=2.0 gives strong approach gradient.
    dist_prev = (x_prev * x_prev + y_prev * y_prev) ** 0.5
    dist_curr = (x * x + y * y) ** 0.5
    progress = dist_prev - dist_curr
    progress_reward = 2.0 * progress

    # ── 2. Soft landing gate (velocity + orientation penalties near pad) ──
    #     Exponential gate: decays faster than the old rational 1/(1+3d),
    #     localizing penalties to the near-pad region where they matter.
    #     At d=3.0, gate≈0.01 (effectively zero) vs old gate≈0.10,
    #     fixing orientation_penalty's 100% active-rate noise issue.
    distance = dist_curr
    gate = 2.718281828 ** (-1.5 * distance)

    vel_penalty = -0.3 * gate * (vx * vx + vy * vy + 0.1 * angular_vel * angular_vel)
    orientation_penalty = -0.2 * gate * (angle * angle)

    # ── 3. Fuel efficiency (light penalty on engine use) ──
    if action == 2:                      # main engine
        fuel_penalty = -0.02
    elif action == 1 or action == 3:     # orientation engines
        fuel_penalty = -0.005
    else:                                # no engine
        fuel_penalty = 0.0

    total_reward = progress_reward + vel_penalty + orientation_penalty + fuel_penalty

    components = {
        'progress_reward': progress_reward,
        'vel_penalty': vel_penalty,
        'orientation_penalty': orientation_penalty,
        'fuel_penalty': fuel_penalty,
    }
    return float(total_reward), components