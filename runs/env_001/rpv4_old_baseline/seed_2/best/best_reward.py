def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next state
    old_x, old_y = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- Approach reward: distance-based shaping toward target (0,0) ----
    old_dist = (old_x ** 2 + old_y ** 2) ** 0.5
    new_dist = (x ** 2 + y ** 2) ** 0.5
    approach_reward = (old_dist - new_dist) * 5.0   # positive when moving closer

    # ---- Smoothness penalty: discourage high speed and rotation ----
    speed_penalty = -0.2 * (vx ** 2 + vy ** 2) - 0.1 * (angle ** 2 + angvel ** 2)

    # ---- Landing reward: both legs on platform and vehicle stable ----
    if left_contact > 0.5 and right_contact > 0.5:
        # Quality of touchdown: near upright, negligible velocity
        landing_quality = 10.0 - 15.0 * angle ** 2 - 3.0 * vx ** 2 - 3.0 * vy ** 2 - 3.0 * angvel ** 2
        landing_reward = max(0.0, landing_quality)
    else:
        landing_reward = 0.0

    # ---- Fuel penalty: discourage unnecessary engine use ----
    fuel_penalty = -0.05 if action in [1, 2, 3] else 0.0

    # ---- Small per-step penalty to prevent lingering ----
    time_penalty = -0.01

    total = approach_reward + speed_penalty + landing_reward + fuel_penalty + time_penalty

    components = {
        "approach_reward": approach_reward,
        "speed_penalty": speed_penalty,
        "landing_reward": landing_reward,
        "fuel_penalty": fuel_penalty,
        "time_penalty": time_penalty
    }
    return float(total), components