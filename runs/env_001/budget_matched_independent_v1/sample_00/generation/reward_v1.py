def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for 2D lander approach‑and‑soft‑contact task.
    Components:
      - goal_proximity_reward   (main learning signal)
      - safe_landing_reward     (task‑completion soft proxy)
      - energy_penalty          (mandatory secondary goal)
      - orientation_penalty     (safety / stability constraint)
    """
    # Extract signals from next_obs (post‑transition state)
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 1. Goal proximity (dense state signal) ----------
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    w_dist = 1.0
    goal_proximity = -w_dist * distance

    # ---------- 2. Safe landing proxy (joint condition on contact + stability) ----------
    both_contact = float(left_contact > 0.5 and right_contact > 0.5)
    # stability: high reward when velocities and angles are near zero
    vel_sq = x_vel ** 2 + y_vel ** 2
    angle_sq = body_angle ** 2
    omega_sq = angular_vel ** 2
    stability = (1.0 / (1.0 + 10.0 * vel_sq)) * \
                (1.0 / (1.0 + 10.0 * angle_sq)) * \
                (1.0 / (1.0 + 10.0 * omega_sq))
    w_land = 10.0
    safe_landing = w_land * both_contact * stability

    # ---------- 3. Energy penalty ----------
    # penalise any engine activation; action 0 = no engine, 1/2/3 = engine on
    w_energy = 0.02
    if action == 0:
        energy_penalty = 0.0
    else:
        energy_penalty = -w_energy

    # ---------- 4. Orientation penalty ----------
    # light penalty to keep the lander upright
    w_angle = 0.1
    orientation_penalty = -w_angle * (body_angle ** 2)

    # ---------- Assemble total reward ----------
    total_reward = goal_proximity + safe_landing + energy_penalty + orientation_penalty

    components = {
        'goal_proximity': goal_proximity,
        'safe_landing': safe_landing,
        'energy_penalty': energy_penalty,
        'orientation_penalty': orientation_penalty
    }

    return float(total_reward), components