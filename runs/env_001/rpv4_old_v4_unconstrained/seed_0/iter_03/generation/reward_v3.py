def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: Radical rewrite — eliminate suicide incentive.
    
    Core principles:
    - Proximity is ALWAYS POSITIVE (bounded, 1/(1+dist)) — no suicide gradient
    - Progress reward (delta-distance) for efficient navigation
    - Velocity damping only penalizes EXCESS speed near pad (safe_speed threshold)
    - Relaxed landing conditions for discoverability
    - Contact as progress signal, not sparse rarity
    - Modest action/orientation shaping
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    x_prev = obs[0]
    y_prev = obs[1]

    # ── 1. Proximity bonus: always positive, bounded ──
    # Maximum +2.0 at dist=0, decays smoothly, never negative
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    proximity_bonus = 2.0 / (1.0 + dist_curr)

    # ── 2. Progress reward: delta-distance shaping ──
    # Positive when approaching pad, negative when retreating
    # Bounded total: roughly 2 * (initial_dist - final_dist)
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    progress_reward = 2.0 * (dist_prev - dist_curr)

    # ── 3. Orientation penalty (moderate) ──
    orientation_penalty = -0.15 * (angle ** 2) - 0.06 * (angvel ** 2)

    # ── 4. Velocity damping: only EXCESS speed near pad ──
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # Gaussian gate: ~1.0 at pad, ~0.37 at dist=1.0, ~0.02 at dist=2.0
    proximity_gate = 2.718281828 ** (- (dist_curr ** 2) / 1.0)
    safe_speed = 0.35          # speed below this is free near pad
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -1.0 * proximity_gate * excess_speed

    # ── 5. Contact reward: any foot on ground is progress ──
    contact_sum = left_contact + right_contact
    contact_reward = 1.0 * contact_sum

    # ── 6. Landing success bonus (relaxed thresholds) ──
    both_on_ground = (left_contact > 0.5 and right_contact > 0.5)
    near_pad = (dist_curr < 0.8)
    upright = (abs(angle) < 0.4)
    slow = (speed < 0.8)
    is_good_landing = both_on_ground and near_pad and upright and slow
    landing_bonus = 10.0 if is_good_landing else 0.0

    # ── 7. Action efficiency ──
    if action == 0:            # glide — small reward for not wasting fuel
        action_reward = 0.02
    elif action == 2:          # main engine — highest cost
        action_reward = -0.06
    else:                      # orientation engines (1, 3) — moderate cost
        action_reward = -0.03

    # ── Aggregate ──
    total_reward = (
        proximity_bonus
        + progress_reward
        + orientation_penalty
        + vel_damping
        + contact_reward
        + landing_bonus
        + action_reward
    )

    components = {
        "proximity_bonus": proximity_bonus,
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "action_reward": action_reward,
    }
    return float(total_reward), components