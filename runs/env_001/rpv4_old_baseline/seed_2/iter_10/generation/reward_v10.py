def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    old_x, old_y = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    old_dist = (old_x ** 2 + old_y ** 2) ** 0.5
    new_dist = (x ** 2 + y ** 2) ** 0.5

    # Strong approach shaping: reward moving closer to target
    approach_reward = (old_dist - new_dist) * 30.0

    # Speed / angle penalty only when near the target, to encourage soft landing
    proximity_threshold = 1.5
    if new_dist < proximity_threshold:
        speed_penalty = -0.5 * (vx ** 2 + vy ** 2) - 0.3 * angvel ** 2 - 0.3 * angle ** 2
    else:
        speed_penalty = 0.0

    # Landing bonus: large fixed bonus minus quality deficits
    if left_contact > 0.5 and right_contact > 0.5:
        landing_quality = 50.0 - 20.0 * angle ** 2 - 10.0 * (vx ** 2 + vy ** 2) - 10.0 * angvel ** 2
        landing_reward = max(0.0, landing_quality)
    else:
        landing_reward = 0.0

    # Negligible fuel penalty, not dominating
    fuel_penalty = -0.001 if action in [1, 2, 3] else 0.0

    total = approach_reward + speed_penalty + landing_reward + fuel_penalty

    components = {
        "approach_reward": approach_reward,
        "speed_penalty": speed_penalty,
        "landing_reward": landing_reward,
        "fuel_penalty": fuel_penalty,
    }
    return float(total), components