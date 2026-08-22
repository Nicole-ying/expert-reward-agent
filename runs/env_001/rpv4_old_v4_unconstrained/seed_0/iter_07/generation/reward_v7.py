def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v7: Continuous landing quality — eliminate binary cliffs that cause policy collapse.

    Root cause of v4/v6 crashes: landing_bonus was a sparse/semi-sparse spike (50/step
    or 200-transition) that dominated the reward landscape. Agent overfit to triggering
    it, then the brittle policy collapsed under continued exploration.

    Fix: Replace binary landing check with a continuous 0–1 quality score.
    - Product of 4 sub-scores: dist, angle, speed, contact — all must be good for high reward
    - Max 12.0/step (vs 50 in v4, 200+5 in v6) — balanced against other components
    - Only activates when at least one foot touches ground (no hovering credit)
    - Contact reward (0.5/foot) kept for ground-discovery gradient
    - Progress boosted to 2.0 to compensate for reduced landing signal
    - All other components at v4-proven scales
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    px = obs[0]
    py = obs[1]

    # ── Derived quantities ──
    dist = (x ** 2 + y ** 2) ** 0.5
    prev_dist = (px ** 2 + py ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # ── 1. Progress: delta-distance (slightly boosted for navigation gradient) ──
    progress_reward = 2.0 * (prev_dist - dist)

    # ── 2. Orientation penalty (v4 scale: proven stable) ──
    orientation_penalty = -0.1 * (angle ** 2) - 0.04 * (angvel ** 2)

    # ── 3. Velocity damping near pad (v4 scale: sigma=0.5, coeff=-0.5) ──
    proximity_gate = 2.718281828 ** (- (dist ** 2) / 0.5)
    safe_speed = 0.3
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -0.5 * proximity_gate * excess_speed

    # ── 4. Contact reward: modest ground-discovery incentive ──
    contact_reward = 0.5 * (left_contact + right_contact)

    # ── 5. Continuous landing quality (replaces binary landing_bonus) ──
    any_contact = (left_contact > 0.5 or right_contact > 0.5)

    if any_contact:
        # Each sub-score: 0 (worst) to 1 (perfect), linear decay outside target zone
        dist_quality = max(0.0, 1.0 - dist / 0.5)          # 1 at pad center, 0 at dist>=0.5
        angle_quality = max(0.0, 1.0 - abs(angle) / 0.4)   # 1 when upright, 0 at |angle|>=0.4
        speed_quality = max(0.0, 1.0 - speed / 0.5)        # 1 when still, 0 at speed>=0.5
        contact_quality = (left_contact + right_contact) / 2.0  # 0, 0.5, or 1.0

        # Product: ALL factors matter — partial credit for partial progress
        landing_quality = dist_quality * angle_quality * speed_quality * contact_quality
        landing_bonus = 12.0 * landing_quality
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