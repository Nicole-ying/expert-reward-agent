def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, n_angle, n_ang_vel, n_left, n_right = next_obs

    # ---------- 1. Main progress: improvement in Euclidean distance to landing pad ----------
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    delta_dist = dist - next_dist                # positive when getting closer
    progress_reward = 2.0 * delta_dist

    # ---------- 2. Attitude safety constraint ----------
    angle_err = abs(n_angle)
    ang_vel_abs = abs(n_ang_vel)
    attitude_penalty = -0.5 * (angle_err**2 + (0.5 * ang_vel_abs)**2)

    # ---------- 3. Landing approach reward (continuous multi-factor, replaces dead success_reward) ----------
    prox = max(0.0, 1.0 - next_dist / 5.0)
    upright = max(0.0, 1.0 - angle_err / 0.5)
    speed = (nvx**2 + nvy**2) ** 0.5
    stationary = max(0.0, 1.0 - speed / 1.0)
    contact = (n_left + n_right) / 2.0
    landing_reward = 1.0 * (prox + upright + stationary + contact) / 4.0

    # ---------- Aggregate ----------
    total_reward = progress_reward + attitude_penalty + landing_reward

    components = {
        "progress_reward": progress_reward,
        "attitude_penalty": attitude_penalty,
        "landing_reward": landing_reward
    }
    return float(total_reward), components