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
    vel_penalty = abs(vx) + abs(vy)
    angle_penalty = abs(angle)
    ang_vel_penalty = abs(ang_vel)

    w_vel = 0.001
    w_angle = 0.005
    w_angvel = 0.001
    stability_penalty = - (w_vel * vel_penalty + w_angle * angle_penalty + w_angvel * ang_vel_penalty)

    # ---------- 3. soft landing proxy (CONTINUOUS) ----------
    # DIAGNOSIS: binary trigger at 0.1 with 62% nonzero_rate dominates progress 23x.
    # Agent exploits loose binary by hovering, not landing precisely.
    # FIX: replace binary if/else with continuous product of bounded factors.
    # Each factor uses 1/(1+kx) — smooth, bounded [0,1], gradient everywhere.
    # Coefficient 0.15 at max; typical hovering ~0.01-0.02, ratio ~3-7x progress.

    # distance: 1/(1+10*dist) — 0.5 at dist=0.1, 0.09 at dist=1.0
    dist_factor = 1.0 / (1.0 + 10.0 * dist_next)

    # speed: 1/(1+5*(|vx|+|vy|)) — 0.5 at total_speed=0.2, 0.17 at 1.0
    speed_factor = 1.0 / (1.0 + 5.0 * (abs(vx) + abs(vy)))

    # angle: 1/(1+20*|angle|) — 0.5 at |angle|=0.05, 0.33 at 0.1
    angle_factor = 1.0 / (1.0 + 20.0 * abs(angle))

    # contact: soft factor — 0.5 with no contact, 1.0 with both legs
    contact_factor = 0.5 + 0.25 * (left_contact + right_contact)

    soft_landing_proxy = 0.15 * dist_factor * speed_factor * angle_factor * contact_factor

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