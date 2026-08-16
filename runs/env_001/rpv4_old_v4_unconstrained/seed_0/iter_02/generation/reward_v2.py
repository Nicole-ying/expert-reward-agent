def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: Complete rewrite.
    - Negative distance to pad (smooth, always active goal proximity)
    - Orientation penalty (quadratic on angle + angvel)
    - Velocity damping via exponential proximity gate (always active)
    - Progressive contact reward (any foot → bonus, both → more)
    - Landing success bonus (large per-step reward when settled)
    - Action efficiency (reward idle, penalize engine use)
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ──── 1. Goal proximity: negative distance to origin ────
    dist = (x ** 2 + y ** 2) ** 0.5
    w_proximity = 0.3
    proximity_reward = -w_proximity * dist

    # ──── 2. Orientation penalty ────
    w_angle = 0.08
    w_angvel = 0.03
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ──── 3. Velocity damping (exponential gate, always active) ────
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # gate ≈ 1 when dist ≈ 0, gate → 0 when dist ≫ σ
    sigma_dist = 1.5
    proximity_gate = 2.718281828 ** (- (dist ** 2) / (sigma_dist ** 2))
    w_damp = 0.4
    vel_damping = -w_damp * proximity_gate * speed

    # ──── 4. Progressive contact reward ────
    # 0, 1, or 2 feet on ground — any contact is progress
    contact_sum = left_contact + right_contact
    w_contact = 0.5
    contact_reward = w_contact * contact_sum

    # ──── 5. Landing success bonus ────
    both_on_ground = (left_contact > 0.5 and right_contact > 0.5)
    near_pad = (dist < 0.3)
    upright = (abs(angle) < 0.15)
    slow = (speed < 0.3)
    is_good_landing = both_on_ground and near_pad and upright and slow
    w_landing = 10.0
    landing_bonus = w_landing * (1.0 if is_good_landing else 0.0)

    # ──── 6. Action efficiency ────
    if action == 0:          # no engine — reward gliding
        action_reward = 0.03
    elif action == 2:        # main engine — highest cost
        action_reward = -0.08
    else:                    # orientation engines (1 or 3) — low cost
        action_reward = -0.03

    # ──── Aggregate ────
    total_reward = (
        proximity_reward
        + orientation_penalty
        + vel_damping
        + contact_reward
        + landing_bonus
        + action_reward
    )

    components = {
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "action_reward": action_reward,
    }
    return float(total_reward), components