def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    LunarLander-style discrete control environment.
    Obs: [x, y, vx, vy, angle, angvel, left_contact, right_contact]
    Goal: reach target pad (x≈0,y≈0) with both legs down, low speed, upright attitude.
    """
    # Unpack observations
    x = obs[0]
    y = obs[1]
    vx = obs[2]
    vy = obs[3]
    angle = obs[4]
    angvel = obs[5]
    left = obs[6]
    right = obs[7]

    # ---------- helper quantities ----------
    distance = (x**2 + y**2) ** 0.5
    speed = (vx**2 + vy**2) ** 0.5

    # ---------- 1. Goal proximity (main driving signal) ----------
    w_prox = 1.0
    proximity_penalty = -w_prox * distance

    # ---------- 2. Velocity control (gated, near target) ----------
    w_vel = 0.5
    d_threshold = 0.5
    gate = max(0.0, 1.0 - distance / d_threshold)   # active when close
    vel_penalty = -w_vel * (vx**2 + vy**2) * gate

    # ---------- 3. Attitude stabilization ----------
    w_angle = 0.5
    w_angvel = 0.1
    attitude_penalty = -w_angle * (angle**2) - w_angvel * (angvel**2)

    # ---------- 4. Stable contact proxy ----------
    # Smooth reward for both legs touching, maximised when close & slow
    w_contact = 5.0
    kd = 5.0            # distance sensitivity
    kv = 10.0           # speed sensitivity
    proximity_factor = 1.0 / (1.0 + kd * distance)
    speed_factor = 1.0 / (1.0 + kv * speed)
    contact_bonus = w_contact * left * right * proximity_factor * speed_factor

    # ---------- Assemble total reward ----------
    total_reward = proximity_penalty + vel_penalty + attitude_penalty + contact_bonus

    components = {
        "proximity_penalty": proximity_penalty,
        "vel_penalty": vel_penalty,
        "attitude_penalty": attitude_penalty,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components