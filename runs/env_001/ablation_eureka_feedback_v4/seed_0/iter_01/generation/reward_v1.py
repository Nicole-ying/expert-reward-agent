def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next_obs
    x = next_obs[0]            # horizontal position relative to landing pad
    y = next_obs[1]            # vertical position relative to pad
    vx = next_obs[2]           # horizontal velocity
    vy = next_obs[3]           # vertical velocity
    angle = next_obs[4]        # body orientation angle
    ang_vel = next_obs[5]      # angular velocity
    left_contact = next_obs[6] # left leg contact flag (0/1)
    right_contact = next_obs[7]# right leg contact flag (0/1)

    # --- Component 1: position proximity (main learning signal) ---
    distance = (x**2 + y**2) ** 0.5
    proximity = -1.0 * distance   # negative Euclidean distance, encouraging approach

    # --- Component 2: movement stability penalty (soft landing assistance) ---
    # Quadratic penalties on linear velocity, angular velocity, and body angle.
    vel_penalty = -0.05 * (vx**2 + vy**2)
    ang_penalty = -0.02 * (ang_vel**2)
    angle_penalty = -0.5 * (angle**2)
    stability = vel_penalty + ang_penalty + angle_penalty

    # --- Component 3: leg contact reward (encouraging safe settlement) ---
    contact = 0.5 * (left_contact + right_contact)

    # Total reward
    total = proximity + stability + contact

    components = {
        "proximity": proximity,
        "stability": stability,
        "contact": contact
    }
    return float(total), components