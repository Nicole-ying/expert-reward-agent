def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- observations ----------
    x_pos, y_pos = obs[0], obs[1]
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_left_contact, n_right_contact = next_obs[6], next_obs[7]

    # ---------- distances ----------
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    n_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5

    # ---------- 1. progress reward (approach incentive) ----------
    progress_reward = 5.0 * (dist - n_dist)  # positive when moving toward target

    # ---------- 2. touchdown bonus (task completion) ----------
    # Large one-time reward when both legs touch the platform.
    # Assuming the episode terminates immediately after this condition,
    # so no repeated accumulation of this bonus.
    both_contacts = (n_left_contact > 0.5) and (n_right_contact > 0.5)
    touchdown_bonus = 200.0 if both_contacts else 0.0

    # ---------- 3. crash / boundary constraints ----------
    x_violation = max(0.0, abs(nx_pos) - 1.5)
    v_violation = max(0.0, -ny_vel - 3.0)   # excessive downward speed
    crash_penalty = -1.0 * x_violation - 1.0 * v_violation

    total_reward = progress_reward + touchdown_bonus + crash_penalty

    components = {
        "progress_reward": progress_reward,
        "touchdown_bonus": touchdown_bonus,
        "crash_penalty": crash_penalty,
    }
    return float(total_reward), components