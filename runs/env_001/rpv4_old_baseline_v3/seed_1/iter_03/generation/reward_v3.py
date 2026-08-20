def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v3: back to v1's proven skeleton + descent pressure + stronger landing.

    Core changes from v2:
    - Remove: flat speed_penalty, step_cost, fuel_cost, hard_landing, position_quality
    - Restore: strong proximity_progress (15x, between v1's 30x and v2's 5x)
    - Add: altitude_reward for continuous descent incentive
    - Add: partial_contact for easier landing discovery
    - Strengthen: soft_landing 3x vs v1 (15.0 vs 5.0)
    - Add: proactive ground_speed_penalty (fires before crash, not after)
    - Revert: attitude_penalty to v1's mild coefficients
    """
    # ── state extraction ──
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    old_x, old_y = obs[0], obs[1]

    # ── 1. proximity_progress: weighted delta-distance ──
    # Vertical weighted 2x — descending is the critical action.
    # 15x coefficient is between v1 (30x) and v2 (5x).
    old_wdist = (old_x ** 2 + 2.0 * old_y ** 2) ** 0.5
    new_wdist = (x ** 2 + 2.0 * y ** 2) ** 0.5
    proximity_progress = 15.0 * (old_wdist - new_wdist)

    # ── 2. altitude_reward: dense incentive to descend ──
    # Max 0.1 at y=0, decays smoothly with height.
    # Gives the agent a reason to commit to going down instead of hovering.
    altitude_reward = 0.1 / (1.0 + 3.0 * y)

    # ── 3. soft_landing: contact-based rewards ──
    speed = (vx ** 2 + vy ** 2) ** 0.5
    both_contact = left_contact * right_contact
    any_contact = max(left_contact, right_contact)

    # Quality gates: landing is only rewarded when gentle and upright
    speed_factor = 1.0 / (1.0 + 5.0 * speed)
    angle_factor = 1.0 / (1.0 + 3.0 * abs(angle))

    # Full bonus for perfect two-leg landing (3x stronger than v1)
    soft_landing = 15.0 * both_contact * speed_factor * angle_factor
    # Partial credit for any leg contact — easier to discover
    partial_contact = 3.0 * any_contact * speed_factor * angle_factor

    # ── 4. attitude_penalty: mild, same as v1 ──
    attitude_penalty = -0.5 * (angle ** 2) - 0.2 * (ang_vel ** 2)

    # ── 5. ground_speed_penalty: proactive crash prevention ──
    # Only activates when near ground AND moving fast — warns before impact.
    # Scales linearly with proximity to ground: stronger as y→0.
    ground_speed_penalty = 0.0
    if y < 0.5 and speed > 0.5:
        ground_speed_penalty = -3.0 * speed * (0.5 - y) / 0.5

    # ── total ──
    total_reward = (
        proximity_progress
        + altitude_reward
        + soft_landing
        + partial_contact
        + attitude_penalty
        + ground_speed_penalty
    )

    components = {
        "proximity_progress": proximity_progress,
        "altitude_reward": altitude_reward,
        "soft_landing": soft_landing,
        "partial_contact": partial_contact,
        "attitude_penalty": attitude_penalty,
        "ground_speed_penalty": ground_speed_penalty,
    }
    return float(total_reward), components