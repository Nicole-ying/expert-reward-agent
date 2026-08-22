def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- unpack observations ---
    x_pos = obs[0]           # horizontal distance to target
    y_pos = obs[1]           # vertical distance to target
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    # --- hyperparameters ---
    w_dist = 0.5            # approach reward weight
    w_vel = 0.02            # velocity penalty weight
    w_angle = 0.1           # body angle penalty weight
    w_angvel = 0.05         # angular velocity penalty weight
    w_boundary = 0.5        # horizontal out-of-bounds penalty
    w_landing = 1.0         # landing quality bonus weight

    safe_x_bound = 1.0       # x beyond this is considered unsafe
    vel_thresh = 0.5         # velocity below which landing is considered stable
    angle_thresh = 0.2       # body angle below which landing is considered upright

    # 1. Approach reward (main learning signal)
    dist_sq = x_pos**2 + y_pos**2
    approach = -w_dist * dist_sq

    # 2. Stability penalties (soft constraints)
    vel_penalty = -w_vel * (x_vel**2 + y_vel**2)
    angle_penalty = -w_angle * body_angle**2 - w_angvel * ang_vel**2

    # 3. Horizontal boundary penalty (prevent drifting out of viewport)
    if abs(x_pos) > safe_x_bound:
        boundary_penalty = -w_boundary * (abs(x_pos) - safe_x_bound)**2
    else:
        boundary_penalty = 0.0

    # 4. Landing quality bonus (joint condition proxy)
    both_legs = left_contact * right_contact  # = 1.0 only if both legs touch
    # bounded stability factors
    vel_stable = max(0.0, 1.0 - abs(x_vel) / vel_thresh)
    yvel_stable = max(0.0, 1.0 - abs(y_vel) / vel_thresh)
    angle_stable = max(0.0, 1.0 - abs(body_angle) / angle_thresh)
    landing_quality = both_legs * vel_stable * yvel_stable * angle_stable
    landing_bonus = w_landing * landing_quality

    total_reward = approach + vel_penalty + angle_penalty + boundary_penalty + landing_bonus

    components = {
        "approach_reward": approach,
        "velocity_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "boundary_penalty": boundary_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components