def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observation
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]

    # Unpack next observation
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_left = next_obs[6]
    n_right = next_obs[7]

    # ------------------------------------------------------------
    # Component 1: potential-based shaping (dense progress signal)
    # ------------------------------------------------------------
    # potential = -(distance penalty + speed penalty + attitude penalty)
    dist = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5
    attitude = abs(angle)
    potential = - (2.0 * dist) - (0.5 * speed) - (1.0 * attitude)

    n_dist = (nx ** 2 + ny ** 2) ** 0.5
    n_speed = (nvx ** 2 + nvy ** 2) ** 0.5
    n_attitude = abs(n_angle)
    n_potential = - (2.0 * n_dist) - (0.5 * n_speed) - (1.0 * n_attitude)

    shaping_reward = n_potential - potential   # positive when state improves

    # -------------------------------------------
    # Component 2: terminal success event reward
    # -------------------------------------------
    success_landing = (n_left > 0.5 and n_right > 0.5 and n_speed < 0.5)
    success_bonus = 200.0 if success_landing else 0.0

    # -------------------------------------------
    # Component 3: landing safety gate (soft)
    # -------------------------------------------
    y_thresh = 0.3
    safe_down_speed = 0.2
    if ny < y_thresh:
        danger = max(0.0, -nvy - safe_down_speed)          # >0 when descending too fast
        gate = 1.0 / (1.0 + 5.0 * danger + 3.0 * n_attitude)
    else:
        gate = 1.0

    # -------------------------------------------
    # Component 4: action efficiency
    # -------------------------------------------
    action_cost = -0.02 if action != 0 else 0.0   # discourage unnecessary engine use

    # -------------------------------------------
    # Combine
    # -------------------------------------------
    total_reward = shaping_reward * gate + success_bonus + action_cost

    components = {
        "shaping": shaping_reward,
        "success_bonus": success_bonus,
        "action_cost": action_cost,
        "gate_factor": gate
    }
    return float(total_reward), components