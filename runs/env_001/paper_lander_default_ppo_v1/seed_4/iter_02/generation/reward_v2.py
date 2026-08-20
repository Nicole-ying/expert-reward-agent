def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft_contact, nright_contact = next_obs

    # ----------------------------------------------------------------
    # Core mandatory roles
    # ----------------------------------------------------------------
    # 1. Goal proximity: potential-based delta shaping
    #    Positive when moving toward target, negative when moving away
    distance = (x**2 + y**2)**0.5
    next_distance = (nx**2 + ny**2)**0.5
    proximity_reward = 2.0 * (distance - next_distance)

    # 2. Orientation penalty: keep body upright and stable
    orientation_penalty = -0.1 * (angle**2) - 0.1 * (angvel**2)

    # ----------------------------------------------------------------
    # Soft landing & settling (mandatory role)
    # ----------------------------------------------------------------
    # Gating factor: strong influence only when close to the target
    proximity_gate = 1.0 / (1.0 + 5.0 * distance)

    # Speed penalty: punish high velocities when near the pad
    speed_sq = vx**2 + vy**2
    speed_penalty_gated = -0.5 * speed_sq * proximity_gate

    # ----------------------------------------------------------------
    # Conditional role: safe contact bonus
    # ----------------------------------------------------------------
    # Reward stable two-leg contact when close to the pad
    contact_bonus = 0.5 * left_contact * right_contact * proximity_gate

    # ----------------------------------------------------------------
    # Combine components
    # ----------------------------------------------------------------
    total_reward = (
        proximity_reward +
        orientation_penalty +
        speed_penalty_gated +
        contact_bonus
    )

    components = {
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "speed_penalty_gated": speed_penalty_gated,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components