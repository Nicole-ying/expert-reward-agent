def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs (the state after action)
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    av = next_obs[5]
    lc = next_obs[6]
    rc = next_obs[7]

    # Distance to target (landing platform)
    dist = (x**2 + y**2) ** 0.5

    # 1. Goal distance reduction: negative quadratic penalty encourages moving towards zero
    proximity_reward = -0.1 * (x**2 + y**2)

    # 2. Soft landing velocity: velocity penalty gated by proximity
    #    Gate = 1/(1 + k*dist) – strong penalty only when close to platform
    vel_gate = 1.0 / (1.0 + 5.0 * dist)
    velocity_penalty = -0.1 * (vx**2 + vy**2) * vel_gate

    # 3. Upright orientation: penalize tilt and angular velocity
    orientation_penalty = -0.5 * (angle**2) - 0.1 * (av**2)

    # 4. Dual leg contact: sparse bonus when both legs touch the platform
    contact_bonus = 10.0 * lc * rc

    total_reward = proximity_reward + velocity_penalty + orientation_penalty + contact_bonus

    components = {
        'proximity_reward': proximity_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'contact_bonus': contact_bonus
    }

    return float(total_reward), components