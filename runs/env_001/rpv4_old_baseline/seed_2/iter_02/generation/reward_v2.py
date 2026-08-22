def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant state variables from post-action observation
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    body_angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Small distance penalty to encourage approaching the target
    dist = (x * x + y * y) ** 0.5
    proximity_reward = -0.05 * dist

    # 2. Soft speed reward: prefers low total speed (including angular)
    speed = (vx * vx + vy * vy + angvel * angvel) ** 0.5
    speed_reward = 1.0 / (1.0 + 5.0 * speed)

    # 3. Upright reward: prefers near-vertical body angle
    angle_reward = 1.0 / (1.0 + 10.0 * abs(body_angle))

    # 4. Strong sparse landing bonus for safe touchdown
    contact_ok = float(left_contact > 0.5 and right_contact > 0.5)
    speed_ok = float(abs(vx) < 0.2 and abs(vy) < 0.2 and abs(angvel) < 0.1)
    angle_ok = float(abs(body_angle) < 0.1)
    landing_bonus = 200.0 * contact_ok * speed_ok * angle_ok

    total = proximity_reward + speed_reward + angle_reward + landing_bonus
    components = {
        "proximity_reward": proximity_reward,
        "speed_reward": speed_reward,
        "angle_reward": angle_reward,
        "landing_bonus": landing_bonus
    }
    return total, components