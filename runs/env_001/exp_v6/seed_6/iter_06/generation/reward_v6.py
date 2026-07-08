def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract states ----------
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 1. main learning signal: progress toward (0,0) ----------
    # DIAGNOSIS: scale=1 gave mean=0.003, too weak vs landing proxy (73x).
    # Skeleton recommends scale 5~20. Using 5 brings mean to ~0.015,
    # ratio to ~14x, and pushes stability_penalty below 10% of progress.
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress_scale = 5.0
    progress_delta_reward = progress_scale * (dist_obs - dist_next)

    # ---------- 2. stability / smoothness penalty ----------
    vel_penalty = abs(vx) + abs(vy)
    angle_penalty = abs(angle)
    ang_vel_penalty = abs(ang_vel)

    w_vel = 0.001
    w_angle = 0.005
    w_angvel = 0.001
    stability_penalty = -(w_vel * vel_penalty + w_angle * angle_penalty + w_angvel * ang_vel_penalty)

    # ---------- 3. soft landing proxy (CONTINUOUS, max(0,1-x/D) form) ----------
    # distance factor: 1.0 at dist=0, 0.0 at dist>=0.5
    dist_factor = max(0.0, 1.0 - dist_next / 0.5)

    # speed factor: 1.0 at zero speed, 0.0 at total_speed>=0.5
    total_speed = abs(vx) + abs(vy)
    speed_factor = max(0.0, 1.0 - total_speed / 0.5)

    # angle factor: 1.0 at angle=0, 0.0 at |angle|>=0.2
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.2)

    # contact: soft factor [0.33, 1.0], rewards any contact
    contact_factor = 0.33 + 0.335 * (left_contact + right_contact)

    soft_landing_proxy = 0.4 * dist_factor * speed_factor * angle_factor * contact_factor

    # ---------- total ----------
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward,
    }

    return float(total_reward), components