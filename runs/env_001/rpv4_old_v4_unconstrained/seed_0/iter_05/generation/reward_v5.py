def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v5: Smoothed landing gradient + strengthened shaping.

    Core changes from v4:
    - near_landing_bonus (10/step) bridges the gap between contact and full landing
    - Wider vel_damping gate (sigma=1.5) for earlier deceleration
    - Stronger orientation penalty so it actually matters
    - Stronger progress_reward for clearer navigation gradient
    - Landing bonus (50/step) remains the dominant terminal incentive
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

    # ── 1. Progress: delta-distance (strengthened navigation signal) ──
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    progress_reward = 2.5 * (dist_prev - dist_curr)

    # ── 2. Orientation penalty (strengthened — now actually matters) ──
    orientation_penalty = -0.25 * (angle ** 2) - 0.1 * (angvel ** 2)

    # ── 3. Velocity damping near pad (widened gate + stronger coefficient) ──
    # sigma=1.5 means gate ≈ 0.64 at dist=1.0, ≈ 0.90 at dist=0.5
    # Agent feels deceleration pressure from much farther out
    speed = (vx ** 2 + vy ** 2) ** 0.5
    proximity_gate = 2.718281828 ** (- (dist_curr ** 2) / 1.5)
    safe_speed = 0.3
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -4.0 * proximity_gate * excess_speed

    # ── 4. Contact reward (strengthened ground-exploration incentive) ──
    contact_sum = left_contact + right_contact
    contact_reward = 1.0 * contact_sum

    # ── 5. Landing bonuses: two-tier for smooth gradient ──
    both_on_ground = (left_contact > 0.5 and right_contact > 0.5)
    near_pad = (dist_curr < 0.6)
    upright = (abs(angle) < 0.35)
    slow = (speed < 0.5)
    is_good_landing = both_on_ground and near_pad and upright and slow

    # near_landing: on the pad but not yet perfectly settled
    # This creates a stepping stone — agent gets rewarded for just being on the pad
    is_near_landing = both_on_ground and near_pad and not is_good_landing

    if is_good_landing:
        landing_bonus = 50.0
    elif is_near_landing:
        landing_bonus = 10.0
    else:
        landing_bonus = 0.0

    # ── 6. Action efficiency (slightly relaxed) ──
    if action == 0:
        action_reward = 0.0
    elif action == 2:          # main engine — highest cost
        action_reward = -0.03
    else:                      # orientation engines (1, 3) — moderate cost
        action_reward = -0.01

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