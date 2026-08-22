def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v8: Sum-based touchdown quality + one-time bonus. Eliminates per-step landing accumulation.

    Root cause of v4/v6/v7 failures:
    - Per-step landing rewards (binary 50/step or product 12/step) accumulate while agent
      sits on ground, inevitably dominating the reward landscape.
    - Product formulation (v7) creates multiplicative cliffs: one bad factor → reward=0.
    - This causes the classic exploit-collapse pattern at cp80% in training.

    Fixes:
    1. PRODUCT → SUM: landing_quality = dist + angle + speed + contact_quality.
       Each factor contributes independently → smooth gradient, no cliffs.
    2. ONE-TIME touchdown bonus: reward is given only at the moment of first ground
       contact (air→ground transition), proportional to landing quality.
       Max ~80, no per-step sustain → cannot dominate through accumulation.
    3. contact_reward reduced to 0.2/foot — retains ground-discovery gradient without
       excessive accumulation.
    4. All other components at v4/v7-proven scales.
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
    p_left = obs[6]
    p_right = obs[7]

    # ── Derived quantities ──
    dist = (x ** 2 + y ** 2) ** 0.5
    prev_dist = (px ** 2 + py ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # ── 1. Progress: delta-distance (v7 scale, proven navigation signal) ──
    progress_reward = 2.0 * (prev_dist - dist)

    # ── 2. Orientation penalty (v7 scale: stable, always active) ──
    orientation_penalty = -0.1 * (angle ** 2) - 0.04 * (angvel ** 2)

    # ── 3. Velocity damping near pad (v4/v7 scale: narrow gate, modest coefficient) ──
    proximity_gate = 2.718281828 ** (- (dist ** 2) / 0.5)
    safe_speed = 0.3
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -0.5 * proximity_gate * excess_speed

    # ── 4. Contact reward: modest ground-discovery incentive (reduced from 0.5) ──
    contact_reward = 0.2 * (left_contact + right_contact)

    # ── 5. Landing bonus: ONE-TIME touchdown quality (SUM-based, no per-step sustain) ──
    # Detect air→ground transition
    was_airborne = (p_left < 0.5 and p_right < 0.5)
    touching_now = (left_contact > 0.5 or right_contact > 0.5)
    just_touched_down = was_airborne and touching_now

    if just_touched_down:
        # Each quality factor: 0 (worst) to 1 (perfect), linear decay
        dist_quality = max(0.0, 1.0 - dist / 0.5)           # 1 at center, 0 at dist>=0.5
        angle_quality = max(0.0, 1.0 - abs(angle) / 0.4)    # 1 when upright, 0 at |angle|>=0.4
        speed_quality = max(0.0, 1.0 - speed / 0.5)         # 1 when still, 0 at speed>=0.5
        contact_quality = (left_contact + right_contact) / 2.0  # 0, 0.5, or 1.0

        # SUM: each factor contributes independently — smooth gradient, no cliffs
        td_quality = dist_quality + angle_quality + speed_quality + contact_quality
        # Max td_quality = 4.0 → max bonus = 80.0 (one-time, no accumulation)
        landing_bonus = 20.0 * td_quality
    else:
        landing_bonus = 0.0

    # ── 6. Action efficiency (v4/v7 scale) ──
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