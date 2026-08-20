def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # unpack observations
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nl_contact, nr_contact = next_obs

    # compute distances to target (0,0)
    dist = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5

    # 1. main progress signal: improvement_delta on distance
    delta_dist = dist - dist_next  # positive when approaching target
    delta_dist_clipped = max(-0.5, min(delta_dist, 0.5))  # bound extreme jumps
    progress_reward = 1.0 * delta_dist_clipped

    # 2. speed penalty when near the target to encourage gentle approach
    close_threshold = 0.3
    if dist_next < close_threshold:
        speed_penalty = -0.5 * (abs(nvx) + abs(nvy))
    else:
        speed_penalty = 0.0

    # 3. attitude stability: hinge penalty on body angle
    safe_angle = 0.2  # radians
    angle_excess = max(0.0, abs(nangle) - safe_angle)
    angle_penalty = -0.1 * angle_excess

    # 4. fuel efficiency: small penalty for main engine usage
    fuel_cost = 0.0
    if action == 2:  # main engine
        fuel_cost = -0.02

    # 5. soft landing bonus: proxy success condition using observable signals
    success_dist_thresh = 0.1
    success_speed_thresh = 0.2
    success_angle_thresh = 0.1
    soft_landing_bonus = 0.0
    if (dist_next < success_dist_thresh and
        abs(nvx) < success_speed_thresh and
        abs(nvy) < success_speed_thresh and
        abs(nangle) < success_angle_thresh and
        nl_contact == 1 and nr_contact == 1):
        soft_landing_bonus = 10.0

    total_reward = progress_reward + speed_penalty + angle_penalty + fuel_cost + soft_landing_bonus
    components = {
        "progress_reward": progress_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "fuel_cost": fuel_cost,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components