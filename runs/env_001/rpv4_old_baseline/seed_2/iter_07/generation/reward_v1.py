def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- signal extraction (current and next) -----
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next, y_vel_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ----- distance to landing pad -----
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5

    # ----- component A: proximity progress (dense improvement delta) -----
    # Encourage moving closer to the pad; scale to make it a principal driver.
    w_prox = 0.15
    proximity_progress = w_prox * (dist_curr - dist_next)   # positive when distance shrinks

    # ----- component B: soft landing when legs contact the pad -----
    contact_any = 1.0 if (left_contact_next > 0.5 or right_contact_next > 0.5) else 0.0

    # thresholds for a safe landing (tune via experiment)
    vy_thresh = 0.2   # vertical speed very low
    vx_thresh = 0.2   # horizontal speed very low
    angle_thresh = 0.1 # radians, small tilt

    # bounded factors: each goes to 1 when the condition is perfectly met, 0 when threshold exceeded
    vy_factor = max(0.0, 1.0 - abs(y_vel_next) / vy_thresh)
    vx_factor = max(0.0, 1.0 - abs(x_vel_next) / vx_thresh)
    angle_factor = max(0.0, 1.0 - abs(angle_next) / angle_thresh)

    landing_quality = vy_factor * vx_factor * angle_factor  # joint condition proxy

    # The reward is given only when contact is active, thus encourages gentle touch-down.
    w_land = 0.8
    soft_landing_reward = contact_any * w_land * landing_quality

    # ----- component C: orientation stability (light penalty) -----
    w_ang = 0.05
    orientation_penalty = -w_ang * (angle_next**2 + ang_vel_next**2)

    # ----- assemble reward -----
    total_reward = proximity_progress + soft_landing_reward + orientation_penalty

    components = {
        'proximity_progress': proximity_progress,
        'soft_landing_reward': soft_landing_reward,
        'orientation_penalty': orientation_penalty
    }

    return float(total_reward), components