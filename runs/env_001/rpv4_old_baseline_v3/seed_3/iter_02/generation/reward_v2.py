def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: Progressive landing reward with annealed thresholds + distance shaping + slam prevention
    
    Core insight: contact_bonus was dead because thresholds (y<0.1, x<0.2) were impossibly tight.
    This version uses generous initial thresholds that tighten via training_progress,
    plus intermediate rewards (any contact, proximity) to provide discoverable gradient.
    """
    # --- unpack ---
    x_c, y_c = obs[0], obs[1]
    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n = next_obs[4]
    angvel_n = next_obs[5]
    lc_n = next_obs[6]
    rc_n = next_obs[7]

    # --- annealed landing thresholds ---
    # Early training: loose thresholds so agent can discover landing reward.
    # Late training: tighten for precision.
    tp = training_progress  # 0.0 → 1.0
    x_thresh = 0.6 - 0.35 * tp   # 0.60 → 0.25
    y_thresh = 0.4 - 0.25 * tp   # 0.40 → 0.15
    vel_thresh = 0.6 - 0.25 * tp # 0.60 → 0.35
    ang_thresh = 0.5 - 0.25 * tp # 0.50 → 0.25

    # --- 1. progress_reward: potential-based delta-distance ---
    dist_c = (x_c ** 2 + y_c ** 2) ** 0.5
    dist_n = (x_n ** 2 + y_n ** 2) ** 0.5
    progress = 3.0 * (dist_c - dist_n)

    # --- 2. proximity_bonus: continuous reward for being near pad ---
    # Provides gradient even before any contact
    proximity = max(0.0, 1.0 - dist_n / 4.0)  # radius 4.0
    proximity_bonus = 0.3 * proximity

    # --- 3. contact_reward: encourage any leg contact ---
    # Intermediate stepping stone toward full landing
    contact_reward = 1.5 * (lc_n + rc_n)

    # --- 4. landing_reward: BIG bonus for full landing state ---
    # Product of soft conditions — only fires when ALL are satisfied
    x_ok = max(0.0, 1.0 - abs(x_n) / x_thresh)
    y_ok = max(0.0, 1.0 - abs(y_n) / y_thresh)
    speed = (vx_n ** 2 + vy_n ** 2) ** 0.5
    vel_ok = max(0.0, 1.0 - speed / vel_thresh)
    ang_ok = max(0.0, 1.0 - abs(angle_n) / ang_thresh)
    both_legs = lc_n * rc_n

    landing_quality = x_ok * y_ok * vel_ok * ang_ok
    landing_reward = 8.0 * landing_quality * both_legs

    # --- 5. orientation_cost: mild regularizer ---
    orient_cost = -0.15 * abs(angle_n) - 0.03 * (angvel_n ** 2)

    # --- 6. slam_cost: penalize high downward speed near ground ---
    near_ground = max(0.0, 1.0 - abs(y_n) / 2.0)
    downward_speed = max(0.0, -vy_n)
    slam_cost = -4.0 * near_ground * downward_speed

    # --- 7. fuel_cost: tiny efficiency incentive ---
    if action == 2:
        fuel_cost = -0.08
    elif action in (1, 3):
        fuel_cost = -0.04
    else:
        fuel_cost = 0.0

    # --- assemble ---
    total_reward = (
        progress
        + proximity_bonus
        + contact_reward
        + landing_reward
        + orient_cost
        + slam_cost
        + fuel_cost
    )

    components = {
        'progress': progress,
        'proximity_bonus': proximity_bonus,
        'contact_reward': contact_reward,
        'landing_reward': landing_reward,
        'orient_cost': orient_cost,
        'slam_cost': slam_cost,
        'fuel_cost': fuel_cost,
    }

    return total_reward, components