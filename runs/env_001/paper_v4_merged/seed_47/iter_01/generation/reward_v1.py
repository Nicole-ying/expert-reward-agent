def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, n_angle, n_ang_vel, n_left, n_right = next_obs

    # ---------- 1. Main progress: improvement in Euclidean distance to landing pad ----------
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    delta_dist = dist - next_dist                # positive when getting closer
    progress_reward = 2.0 * delta_dist

    # ---------- 2. Attitude safety constraint: penalise large body angle and angular velocity ----------
    angle_err = abs(n_angle)
    ang_vel_abs = abs(n_ang_vel)
    attitude_penalty = -0.5 * (angle_err**2 + (0.5 * ang_vel_abs)**2)

    # ---------- 3. Soft success proxy: combination of near‑target, upright, stationary and two‑leg contact ----------
    prox = max(0.0, 1.0 - next_dist / 5.0)           # close to pad centre and ground
    upright = max(0.0, 1.0 - angle_err / 0.5)        # nearly vertical
    speed = (nvx**2 + nvy**2) ** 0.5
    stationary = max(0.0, 1.0 - speed / 1.0)         # low linear velocity
    contact = (n_left + n_right) / 2.0                # 1.0 when both legs touch
    success_proxy = prox * upright * stationary * contact
    success_reward = 5.0 * success_proxy

    # ---------- Aggregate ----------
    total_reward = progress_reward + attitude_penalty + success_reward

    components = {
        "progress_reward": progress_reward,
        "attitude_penalty": attitude_penalty,
        "success_reward": success_reward
    }
    return float(total_reward), components