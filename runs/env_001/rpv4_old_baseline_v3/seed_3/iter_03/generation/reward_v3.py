def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: Remove proximity_bonus (local optimum trap), fix slam_cost (threshold-based),
    add descent_reward, boost progress/contact/landing, add step_cost.
    
    Core insight: v2 failed because proximity_bonus let agent hover forever near pad
    collecting ~0.25/step, while slam_cost punished ALL downward motion — so agent
    never descended to make leg contact. This version removes the trap completely.
    """
    # --- unpack ---
    x_c, y_c = obs[0], obs[1]
    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n = next_obs[4]
    angvel_n = next_obs[5]
    lc_n = next_obs[6]
    rc_n = next_obs[7]

    # --- 1. progress: potential-based, strong gradient toward (0,0) ---
    # Self-extinguishes at target — no local optimum
    dist_c = (x_c ** 2 + y_c ** 2) ** 0.5
    dist_n = (x_n ** 2 + y_n ** 2) ** 0.5
    progress = 10.0 * (dist_c - dist_n)

    # --- 2. x_centering: extra pressure to align horizontally ---
    x_improvement = abs(x_c) - abs(x_n)
    x_bonus = 2.0 * x_improvement

    # --- 3. descent_reward: reward controlled descent when x is near center ---
    # This replaces the old proximity_bonus — it pushes agent DOWN onto the pad
    x_near = max(0.0, 1.0 - abs(x_n) / 2.0)
    descending = max(0.0, y_c - y_n)
    speed_ok = max(0.0, 1.0 - (vx_n ** 2 + vy_n ** 2) ** 0.5 / 1.5)
    descent_reward = 3.0 * x_near * descending * speed_ok

    # --- 4. contact_reward: substantial per-leg reward ---
    # Only fires when legs actually touch — discoverable once agent descends
    contact_reward = 8.0 * (lc_n + rc_n)

    # --- 5. landing_bonus: big bonus for both legs on pad near origin ---
    near_origin = max(0.0, 1.0 - (x_n ** 2 + y_n ** 2) ** 0.5 / 0.8)
    landing_bonus = 20.0 * lc_n * rc_n * near_origin

    # --- 6. slam_cost: ONLY penalize EXCESSIVE downward speed near ground ---
    # Safe descent (< 0.5 units/step downward) is NOT penalized
    # This is the critical fix — v2 penalized ALL downward speed
    near_ground = max(0.0, 1.0 - abs(y_n) / 2.0)
    excess_downward = max(0.0, -vy_n - 0.5)
    slam_cost = -10.0 * near_ground * excess_downward

    # --- 7. orient_cost: mild regularizer (kept from v2) ---
    orient_cost = -0.15 * abs(angle_n) - 0.03 * (angvel_n ** 2)

    # --- 8. step_cost: tiny per-step penalty to discourage infinite hovering ---
    step_cost = -0.03

    # --- 9. fuel_cost: reduced from v2 to avoid discouraging necessary engine use ---
    if action == 2:
        fuel_cost = -0.04
    elif action in (1, 3):
        fuel_cost = -0.02
    else:
        fuel_cost = 0.0

    # --- assemble ---
    total_reward = (
        progress
        + x_bonus
        + descent_reward
        + contact_reward
        + landing_bonus
        + slam_cost
        + orient_cost
        + step_cost
        + fuel_cost
    )

    components = {
        'progress': progress,
        'x_bonus': x_bonus,
        'descent_reward': descent_reward,
        'contact_reward': contact_reward,
        'landing_bonus': landing_bonus,
        'slam_cost': slam_cost,
        'orient_cost': orient_cost,
        'step_cost': step_cost,
        'fuel_cost': fuel_cost,
    }

    return total_reward, components