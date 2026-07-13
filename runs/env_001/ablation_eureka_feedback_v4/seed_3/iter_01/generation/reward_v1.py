def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next observation
    x = next_obs[0]          # x position relative to target pad center
    y = next_obs[1]          # y position relative to target pad height
    vx = next_obs[2]         # horizontal velocity
    vy = next_obs[3]         # vertical velocity (negative = downward)
    angle = next_obs[4]      # body angle
    angvel = next_obs[5]     # angular velocity
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # A. Goal proximity shaping (mandatory)
    dist = (x**2 + y**2) ** 0.5
    proximity = 2.71828 ** (-dist)   # bounded positive: [0,1], peak at (0,0)

    # B. Soft landing condition (mandatory, task‑completion proxy)
    dist_factor = 2.71828 ** (-10.0 * dist)
    speed_sq = vx**2 + vy**2
    vel_factor = 2.71828 ** (-10.0 * speed_sq)
    angle_factor = 2.71828 ** (-10.0 * angle**2)
    angvel_factor = 2.71828 ** (-10.0 * angvel**2)
    contact_factor = left_contact + right_contact   # 0 when no contact, >0 otherwise
    soft_landing = dist_factor * vel_factor * angle_factor * angvel_factor * contact_factor

    # C. Energy efficiency (mandatory)
    energy_penalty = -0.01 if action != 0 else 0.0

    # D. Terminal velocity penalty (conditional, only near the pad)
    vel_penalty = 0.0
    if y < 0.05 and abs(x) < 0.1:
        # penalize excessively large downward speed
        if vy < -0.3:
            vel_penalty = -0.5 * max(0.0, -vy - 0.3)

    total_reward = 1.0 * proximity + 2.0 * soft_landing + energy_penalty + vel_penalty

    components = {
        "proximity": proximity,
        "soft_landing": soft_landing,
        "energy_penalty": energy_penalty,
        "terminal_velocity_penalty": vel_penalty
    }
    return float(total_reward), components