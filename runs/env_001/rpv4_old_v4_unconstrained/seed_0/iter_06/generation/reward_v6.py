def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v6: Restore v4 penalty scales + landing transition detection + modest near-landing bridge.

    Core fixes from v5 regression:
    - ALL penalties restored to v4 levels (vel_damping: -0.5, orientation: -0.1/-0.04, progress: 1.5)
    - Landing bonus uses transition detection: first good-landing step gets +200,
      subsequent sustain gets +5/step — prevents the cp90% 5387 exploit seen in v4 training
    - Near-landing bridge kept at 2/step (not v5's 10) for gradient without dominance
    - Contact reward slightly boosted (0.75) for stronger ground-discovery signal
    - Action costs at v4 levels: -0.04 main, -0.02 side, 0 glide
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Previous observation for transition detection and progress
    px = obs[0]
    py = obs[1]
    pvx = obs[2]
    pvy = obs[3]
    p_angle = obs[4]
    p_left = obs[6]
    p_right = obs[7]

    # ── Derived quantities ──
    dist = (x ** 2 + y ** 2) ** 0.5
    prev_dist = (px ** 2 + py ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5
    prev_speed = (pvx ** 2 + pvy ** 2) ** 0.5

    # ── 1. Progress: delta-distance (v4 scale: 1.5) ──
    progress_reward = 1.5 * (prev_dist - dist)

    # ── 2. Orientation penalty (v4 scale: -0.1 angle², -0.04 angvel²) ──
    orientation_penalty = -0.1 * (angle ** 2) - 0.04 * (angvel ** 2)

    # ── 3. Velocity damping near pad (v4 scale: narrow gate sigma=0.5, coeff=-0.5) ──
    # Only activates when very close to pad — does not interfere with descent
    proximity_gate = 2.718281828 ** (- (dist ** 2) / 0.5)
    safe_speed = 0.3
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -0.5 * proximity_gate * excess_speed

    # ── 4. Contact reward (slightly boosted: 0.75/foot for stronger ground discovery) ──
    contact_sum = left_contact + right_contact
    contact_reward = 0.75 * contact_sum

    # ── 5. Landing bonus with transition detection ──
    both_on_ground = (left_contact > 0.5 and right_contact > 0.5)
    near_pad = (dist < 0.6)
    upright = (abs(angle) < 0.35)
    slow = (speed < 0.5)
    is_good_landing = both_on_ground and near_pad and upright and slow

    # Detect previous landing state from obs (transition detection)
    prev_both = (p_left > 0.5 and p_right > 0.5)
    prev_near = (prev_dist < 0.6)
    prev_upright = (abs(p_angle) < 0.35)
    prev_slow = (prev_speed < 0.5)
    was_good_landing = prev_both and prev_near and prev_upright and prev_slow

    # Near-landing: on pad but not perfectly settled — bridges the gap
    is_near_landing = both_on_ground and near_pad and not is_good_landing

    if is_good_landing and not was_good_landing:
        # First step of successful landing: large one-time bonus
        landing_bonus = 200.0
    elif is_good_landing:
        # Sustaining good landing: modest per-step to encourage staying settled
        landing_bonus = 5.0
    elif is_near_landing:
        # On pad but not yet perfect: small bridge incentive
        landing_bonus = 2.0
    else:
        landing_bonus = 0.0

    # ── 6. Action efficiency (v4 scale) ──
    if action == 0:
        action_reward = 0.0
    elif action == 2:          # main engine — highest cost
        action_reward = -0.04
    else:                      # orientation engines (1, 3) — moderate cost
        action_reward = -0.02

    # ── Aggregate ──
    total_reward = (
        progress_reward
        + orientation_penalty
        + vel_damping
        + contact_reward
        + landing_bonus
        + action_reward
    )

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "action_reward": action_reward,
    }
    return float(total_reward), components