def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- Progress reward: reward moving closer to the target ---
    old_dist = abs(obs[0]) + abs(obs[1])
    new_dist = abs(next_obs[0]) + abs(next_obs[1])
    progress = old_dist - new_dist  # positive when distance shrinks
    progress_reward = 0.5 * progress

    # --- Landing / contact reward (greatly reduced scale) ---
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    contact_factor = (left_contact + right_contact) / 2.0

    vel_abs = abs(next_obs[2]) + abs(next_obs[3])
    angle_abs = abs(next_obs[4])
    # stability is high when velocity and angle are small
    stability = 1.0 / (1.0 + vel_abs + 5.0 * angle_abs)
    landing_reward = 1.0 * contact_factor * stability   # old was 5.0

    # --- Light penalties for orientation and speed (conditioned softly) ---
    # Keep small throughout so they don't hinder approach but suppress extremes.
    ang_penalty = -0.05 * angle_abs       # old was -0.5 * |angle|
    vel_penalty = -0.01 * vel_abs         # old was -0.1 * (|x_vel|+|y_vel|)

    # --- Fuel efficiency ---
    fuel_penalty = -0.02 * (1.0 if action != 0 else 0.0)

    total_reward = progress_reward + landing_reward + ang_penalty + vel_penalty + fuel_penalty

    components = {
        "progress_reward": progress_reward,
        "landing_reward": landing_reward,
        "ang_penalty": ang_penalty,
        "vel_penalty": vel_penalty,
        "fuel_penalty": fuel_penalty,
    }
    return float(total_reward), components