def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for lunar lander style 2D flyer landing.
    Main goal: reach target platform (x≈0, y≈0) with low velocity, upright.
    
    Components:
        A: progress (distance reduction towards target)       -- main learning signal (STRENGTHENED)
        B: velocity damping (next_obs squared velocity)       -- stability constraint
        C: orientation stabilization (next_obs angle/angvel)  -- stability constraint
        D: fuel penalty (per action cost)                     -- efficiency / mild action cost
        E: landing_quality (per-step reward while landed)     -- continuous soft-landing signal (L2: was sparse bonus)
    
    iter_11 change (L2): Replaced sparse success_landing_bonus (transition-triggered, 
    w=100 one-time) with continuous landing_quality reward (per-step while both feet 
    on ground, scaled by proximity/speed/tilt quality). This prevents exploitation 
    via bouncing (iter_10: 61 triggers/episode at w=100 → ep_sum=6100, 99.3% share)
    and restores gradient to dense shaping signals.
    """
    # Extract states
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    
    # Current and next distance to target (Euclidean)
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    
    # Component A: progress – positive when moving closer to (0,0)
    progress = dist - next_dist  # positive if approaching
    w_progress = 20.0  # iter_10: verified boostable, kept at 20
    component_A = w_progress * progress

    # Component B: velocity damping – penalise large speeds in next step
    vx_n, vy_n = next_obs[2], next_obs[3]
    vel_penalty = vx_n**2 + vy_n**2
    w_vel = 0.05
    component_B = -w_vel * vel_penalty

    # Component C: orientation stabilization – penalise tilt and rotation
    angle_n, angvel_n = next_obs[4], next_obs[5]
    ang_penalty = angle_n**2 + angvel_n**2
    w_ang = 0.05
    component_C = -w_ang * ang_penalty

    # Component D: fuel penalty – light cost for any engine use
    fuel_cost = 0.0 if action == 0 else 1.0
    w_fuel = 0.01
    component_D = -w_fuel * fuel_cost

    # Component E: landing quality – per-step reward while both feet are on ground
    # Replaces the sparse transition-based bonus that was exploited via bouncing.
    # Reward is continuous: every step with both feet down earns quality-scaled reward.
    # Quality drops to 0 for bad landings (high speed, tilt, or far from center).
    # Bouncing interrupts reward → no incentive to bounce.
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]
    curr_both_contact = (n_left_contact > 0.5) and (n_right_contact > 0.5)

    # Landing quality metrics (0=perfect, threshold=zero-quality)
    landing_dist = next_dist  # already computed above
    landing_speed = (vx_n**2 + vy_n**2) ** 0.5
    landing_tilt = abs(angle_n)

    # Quality scores: 1.0 at perfect, linear decay to 0 at threshold
    prox_score = max(0.0, 1.0 - landing_dist / 0.4)      # 0 at dist≥0.4
    speed_score = max(0.0, 1.0 - landing_speed / 1.0)    # 0 at speed≥1.0
    tilt_score = max(0.0, 1.0 - landing_tilt / 0.5)      # 0 at tilt≥0.5

    landing_quality = prox_score * speed_score * tilt_score  # 0.0 to 1.0

    w_landing = 5.0  # per-step reward when landed; ~100 total over 20 stable steps
    component_E = w_landing * landing_quality if curr_both_contact else 0.0

    total_reward = component_A + component_B + component_C + component_D + component_E

    components = {
        "progress": component_A,
        "vel_damping": component_B,
        "angle_stab": component_C,
        "fuel_penalty": component_D,
        "landing_quality": component_E
    }

    return float(total_reward), components