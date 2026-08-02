def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, n_angle, n_ang_vel, n_left, n_right = next_obs

    # ---------- 1. Distance progress (keep) ----------
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    delta_dist = dist - next_dist
    progress_reward = 2.0 * delta_dist

    # ---------- 2. Attitude safety penalty (keep) ----------
    angle_err = abs(n_angle)
    ang_vel_abs = abs(n_ang_vel)
    attitude_penalty = -0.5 * (angle_err**2 + (0.5 * ang_vel_abs)**2)

    # ---------- 3. Landing potential difference (replaces state-based landing_reward) ----------
    # Potential function: lower is better (closer, more upright, slower)
    speed = (vx**2 + vy**2) ** 0.5
    next_speed = (nvx**2 + nvy**2) ** 0.5
    angle_err_prev = abs(angle)
    angle_err_next = abs(n_angle)

    pot_prev = - (5.0 * dist + 10.0 * angle_err_prev + 5.0 * speed)
    pot_next = - (5.0 * next_dist + 10.0 * angle_err_next + 5.0 * next_speed)
    landing_potential_diff = pot_next - pot_prev

    # Contact gain: reward for newly establishing leg contacts
    contact_gain = (n_left + n_right) - (left_contact + right_contact)   # 0, 1, or 2
    contact_bonus = 10.0 * contact_gain

    # Success bonus: approximately landed
    success = (
        n_left == 1.0 and n_right == 1.0 and
        abs(nvx) < 0.2 and abs(nvy) < 0.2 and
        abs(n_angle) < 0.2 and
        abs(nx) < 0.5 and abs(ny) < 0.5
    )
    success_bonus = 100.0 if success else 0.0

    landing_reward = landing_potential_diff + contact_bonus + success_bonus

    # ---------- Aggregate ----------
    total_reward = progress_reward + attitude_penalty + landing_reward

    components = {
        "progress_reward": progress_reward,
        "attitude_penalty": attitude_penalty,
        "landing_potential_diff": landing_potential_diff,
        "contact_bonus": contact_bonus,
        "success_bonus": success_bonus
    }
    return float(total_reward), components