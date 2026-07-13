def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next observation
    x = next_obs[0]          # x position relative to target pad center
    y = next_obs[1]          # y position relative to target pad height
    vx = next_obs[2]         # horizontal velocity
    vy = next_obs[3]         # vertical velocity (negative = downward)
    angle = next_obs[4]      # body angle
    angvel = next_obs[5]     # angular velocity (unused)
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # A. Goal proximity shaping (unchanged)
    dist = (x**2 + y**2) ** 0.5
    proximity = 2.71828 ** (-dist)   # [0,1], peak at (0,0)

    # B. Landing success reward (replaces soft_landing)
    # Only active when legs are in contact with the pad, gated by safe posture
    speed_sq = vx**2 + vy**2
    angle_sq = angle**2
    safety_gate = 2.71828 ** (-10.0 * angle_sq) * 2.71828 ** (-5.0 * speed_sq)
    landing_success = 1.0 * (left_contact + right_contact) * safety_gate

    # C. Energy efficiency (unchanged)
    energy_penalty = -0.01 if action != 0 else 0.0

    # D. Terminal velocity penalty (unchanged)
    vel_penalty = 0.0
    if y < 0.05 and abs(x) < 0.1:
        if vy < -0.3:
            vel_penalty = -0.5 * max(0.0, -vy - 0.3)

    total_reward = 1.0 * proximity + 1.0 * landing_success + energy_penalty + vel_penalty

    components = {
        "proximity": proximity,
        "landing_success": landing_success,
        "energy_penalty": energy_penalty,
        "terminal_velocity_penalty": vel_penalty
    }
    return float(total_reward), components