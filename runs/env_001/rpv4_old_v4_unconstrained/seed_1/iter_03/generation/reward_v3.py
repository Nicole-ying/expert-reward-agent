def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v3 — fixed net-negative-per-step bug from v2.

    Core changes from v2:
    - speed_penalty is now conditional on near_ground proximity (was global).
      Fast descent at high altitude is efficient, not penalised.
    - Added descent_incentive: explicit per-step reward for moving downward,
      supplementing the potential-based approach_reward.
    - Reduced fuel_penalty from -0.25 to -0.05 so the agent dares to use engines.
    - Removed step_penalty entirely — it incentivised early termination.
    - Boosted landing_reward coefficient to create a strong terminal attractor.
    - All coefficients rebalanced so the approach phase is net-positive.
    """

    # ── current state ──────────────────────────────────────────────
    x  = next_obs[0]          # horizontal position relative to pad centre
    y  = next_obs[1]          # vertical position (height above pad)
    vx = next_obs[2]          # horizontal velocity
    vy = next_obs[3]          # vertical velocity
    angle      = next_obs[4]  # body tilt angle
    angular_vel = next_obs[5] # angular velocity
    left_contact  = next_obs[6]  # 1.0 if left leg on ground
    right_contact = next_obs[7]  # 1.0 if right leg on ground

    ox = obs[0]               # previous x
    oy = obs[1]               # previous y

    # ── 1. approach_reward ─────────────────────────────────────────
    # Potential-based shaping: reward closing Euclidean distance to pad.
    # Modest coefficient (was 30.0 in v1, 5.0 in v2) — enough to guide,
    # not so large it drowns out landing signals.
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    approach_reward = 3.0 * (old_dist - new_dist)

    # ── 2. descent_incentive ───────────────────────────────────────
    # Explicit reward for each unit of downward progress.
    # Supplements approach_reward: even if horizontal drift dilutes the
    # distance signal, descending is always rewarded.
    descent_incentive = 0.6 * max(0.0, oy - y)

    # ── 3. altitude_bonus ──────────────────────────────────────────
    # Small bonus for staying at low altitude.  Creates a gentle
    # "basin" near the pad without being large enough to cause hovering.
    altitude_bonus = 0.02 / (1.0 + 3.0 * (y ** 2) ** 0.5)

    # ── 4. speed_penalty ───────────────────────────────────────────
    # Penalise high speed ONLY when close to the ground, where it risks
    # a crash.  At high altitude, speed is unpenalised — fast efficient
    # descent is actually desirable.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    near_ground = 1.0 / (1.0 + 8.0 * (y ** 2) ** 0.5)   # ~1 at y=0, ~0 when high
    speed_penalty = -0.08 * speed * near_ground

    # ── 5. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating throughout the trajectory.
    # Moderate coefficients — strong enough to matter, not so strong
    # that the agent fears using orientation engines.
    attitude_penalty = -0.2 * (angle ** 2) - 0.05 * (angular_vel ** 2)

    # ── 6. landing_reward ──────────────────────────────────────────
    # Reward leg-ground contact (0, 1, or 2 legs), scaled by quality:
    # low speed, upright attitude, and proximity to pad centre.
    # Continuous n_contact (not binary product) gives partial credit
    # for single-leg contact, providing a learning gradient.
    n_contact = left_contact + right_contact          # 0.0, 1.0, or 2.0
    landing_quality = 1.0 / (
        (1.0 + 4.0 * speed)
        * (1.0 + 3.0 * (angle ** 2) ** 0.5)
        * (1.0 + 2.0 * (x ** 2) ** 0.5)
    )
    landing_reward = 0.8 * n_contact * landing_quality

    # ── 7. fuel_penalty ────────────────────────────────────────────
    # Small penalty for engine use (actions 1,2,3).
    # Kept intentionally light so the agent is willing to use engines
    # for braking and attitude control.  -0.05 vs v2's -0.25.
    fuel_penalty = -0.05 if action != 0 else 0.0

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        approach_reward
        + descent_incentive
        + altitude_bonus
        + speed_penalty
        + attitude_penalty
        + landing_reward
        + fuel_penalty
    )

    components = {
        "approach_reward":   approach_reward,
        "descent_incentive": descent_incentive,
        "altitude_bonus":    altitude_bonus,
        "speed_penalty":     speed_penalty,
        "attitude_penalty":  attitude_penalty,
        "landing_reward":    landing_reward,
        "fuel_penalty":      fuel_penalty,
    }

    return float(total_reward), components