def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === Unpack next_obs ===
    x       = next_obs[0]
    y       = next_obs[1]   # height above landing pad, assumed non‑negative
    vx      = next_obs[2]
    vy      = next_obs[3]
    angle   = next_obs[4]
    angvel  = next_obs[5]
    left_c  = next_obs[6]   # 0 or 1
    right_c = next_obs[7]   # 0 or 1

    # === Useful derived quantities ===
    height = max(y, 0.0)              # protect against negative penetration
    dist_to_target = (x**2 + height**2) ** 0.5
    speed_sq = vx**2 + vy**2
    inv_height_scale = 1.0 / (0.1 + height)   # increases as height decreases

    # === Hyper‑parameters (v1) ===
    w_prox    = 0.2
    w_vel     = 0.05
    w_orient  = 0.02
    w_contact = 5.0

    temp_speed  = 0.2      # velocity penalty / safe‑contact temperature
    temp_angle  = 0.04     # angle temperature for safe contact
    temp_x      = 0.2      # horizontal distance temperature for safe contact

    # === 1. Target approach proximity (main learning signal) ===
    # Negative euclidean distance – every step gives gradient towards target
    proximity_reward = -w_prox * dist_to_target

    # === 2. Velocity norm penalty (safety constraint) ===
    # Quadratic penalty on speed, scaled by 1/(0.1+height) -> low altitude penalties are stronger
    velocity_penalty = -w_vel * speed_sq * inv_height_scale

    # === 3. Orientation stability (safety constraint) ===
    # Quadratic penalty on angle + angular velocity, also height‑scaled
    orientation_penalty = -w_orient * (angle**2 + angvel**2) * inv_height_scale

    # === 4. Safe contact encouragement (soft proxy for landing) ===
    # Only active when both support legs touch the pad.
    # Further modulated by near‑zero velocity, near‑upright angle, and near‑center horizontal position.
    safe_contact_bonus = 0.0
    if left_c == 1 and right_c == 1:
        # Use exponential factors because numpy is forbidden and we want smooth [0,1] scaling.
        e = 2.718281828
        speed_factor   = e ** (-speed_sq / temp_speed)
        angle_factor   = e ** (-angle**2 / temp_angle)
        x_prox_factor  = e ** (-x**2 / temp_x)

        safe_contact_bonus = w_contact * speed_factor * angle_factor * x_prox_factor

    # === Assemble total reward ===
    total_reward = proximity_reward + velocity_penalty + orientation_penalty + safe_contact_bonus

    components = {
        "proximity_reward":      proximity_reward,
        "velocity_penalty":      velocity_penalty,
        "orientation_penalty":   orientation_penalty,
        "safe_contact_bonus":    safe_contact_bonus
    }

    return float(total_reward), components