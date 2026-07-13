def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current and next positions relative to landing pad center
    dx_curr, dy_curr = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_curr = (dx_curr**2 + dy_curr**2) ** 0.5
    dist_next = (dx_next**2 + dy_next**2) ** 0.5

    # Approach progress: positive when getting closer to the pad
    approach_delta = dist_curr - dist_next

    # Next‑step velocity and safety threshold
    vx_next, vy_next = next_obs[2], next_obs[3]
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    safe_speed = 0.2

    # Next‑step orientation and angular velocity
    angle_next = next_obs[4]
    angvel_next = next_obs[5]

    # Contact flags
    left_contact_curr = obs[6] > 0.5
    right_contact_curr = obs[7] > 0.5
    left_contact_next = next_obs[6] > 0.5
    right_contact_next = next_obs[7] > 0.5

    # --- Reward weights (best skeleton, no coefficient changes) ---
    w_approach = 2.0
    w_vel_penalty = 0.5
    w_angle = 0.5
    w_angvel = 0.1
    w_landing = 0.1          # restored to best value
    w_touchdown = 10.0       # one‑time landing success bonus

    # Component 1: dense progress towards the pad
    approach_reward = w_approach * approach_delta

    # Component 2: speed constraint (hinge – only penalise when exceeding safe threshold)
    vel_penalty = -w_vel_penalty * max(0.0, speed_next - safe_speed)

    # Component 3: angular stability (quadratic penalties)
    angle_stability = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # Component 4: continuous landing‑quality proxy
    dist_factor = 1.0 / (1.0 + 1.0 * dist_next)
    speed_factor = 1.0 / (1.0 + 1.0 * speed_next)
    landing_reward = w_landing * dist_factor * speed_factor

    # Component 5 (NEW): one‑time touchdown bonus
    # fires on the step where at least one leg first makes contact
    touchdown_bonus = 0.0
    if (not left_contact_curr and not right_contact_curr) and (left_contact_next or right_contact_next):
        touchdown_bonus = w_touchdown

    total_reward = approach_reward + vel_penalty + angle_stability + landing_reward + touchdown_bonus

    components = {
        "approach_progress": approach_reward,
        "velocity_penalty": vel_penalty,
        "angle_stability": angle_stability,
        "landing_quality": landing_reward,
        "touchdown_success": touchdown_bonus,
    }
    return float(total_reward), components