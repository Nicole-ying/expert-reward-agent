def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract next state signals
    x = next_obs[0]
    y = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distance to target platform center (goal is (0,0))
    distance = (x**2 + y**2) ** 0.5

    # ---------- main learning signal: goal proximity ----------
    w_prox = 2.0
    # Bounded positive reward: maximum 2.0 at distance 0, decays smoothly
    prox_reward = w_prox / (1.0 + distance)

    # ---------- near‑factor for soft‑landing constraints ----------
    near = 1.0 / (1.0 + distance)   # 1 when close, ~0 when far

    # ---------- velocity penalty ----------
    w_vel = 0.1
    vel_penalty = -w_vel * near * (x_vel**2 + y_vel**2)

    # ---------- body angle penalty ----------
    w_angle = 0.2
    angle_penalty = -w_angle * near * (angle**2)

    # ---------- angular velocity damping ----------
    w_angvel = 0.1
    angvel_penalty = -w_angvel * near * (ang_vel**2)

    # ---------- contact bonus (both feet on ground) ----------
    w_contact = 1.0
    both_contacts = left_contact * right_contact   # 0 or 1
    contact_reward = w_contact * near * both_contacts

    # ---------- total reward ----------
    total_reward = prox_reward + vel_penalty + angle_penalty + angvel_penalty + contact_reward

    components = {
        "proximity_reward": prox_reward,
        "velocity_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "angular_velocity_penalty": angvel_penalty,
        "contact_bonus": contact_reward
    }

    return float(total_reward), components