def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract states ----------
    # obs
    x, y = obs[0], obs[1]
    # next_obs
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 1. main learning signal: progress toward (0,0) ----------
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress_delta_reward = dist_obs - dist_next

    # ---------- 2. stability / smoothness penalty ----------
    # lightweight penalties to encourage gentle, stable landing
    vel_penalty = abs(vx) + abs(vy)          # prefer zero horizontal/vertical speed
    angle_penalty = abs(angle)               # prefer upright orientation
    ang_vel_penalty = abs(ang_vel)           # prefer no rotation

    w_vel = 0.01
    w_angle = 0.05
    w_angvel = 0.01
    stability_penalty = - (w_vel * vel_penalty + w_angle * angle_penalty + w_angvel * ang_vel_penalty)

    # ---------- 3. soft landing proxy (small bonus for likely safe touchdown) ----------
    # Criteria: near target, low speed, upright, both legs in contact
    near_target = dist_next < 0.2
    low_speed = abs(vx) < 0.1 and abs(vy) < 0.1
    upright = abs(angle) < 0.05
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)

    soft_landing_proxy = 0.1 if (near_target and low_speed and upright and both_contact) else 0.0

    # ---------- total ----------
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy

    # ---------- components dict ----------
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components