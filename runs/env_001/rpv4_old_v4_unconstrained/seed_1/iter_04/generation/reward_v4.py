def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v4 — return to potential-based shaping backbone (iter_1 proven path),
    with exploit-proof landing reward and dead-component cleanup.

    Core changes from v3:
    - approach_reward: 3.0 → 12.0 — strong gradient that iter_1 proved works.
    - REMOVED altitude_bonus — dead component, flat 0.44 across all checkpoints.
    - Speed handling: replaced pure penalty with slow_bonus (reward low speed
      near ground) + mild speed_penalty (only near ground). Net-positive framing.
    - landing_reward: now NEW-CONTACT-BASED. Uses max(0, curr-prev) so that
      sustained contact gives zero additional reward — prevents the bounce
      exploit that caused iter_1's soft_landing_bonus to spike to 256.5.
    - ADDED centering_penalty: light horizontal-deviation cost to prevent
      the agent drifting out of the viewport.
    - Fuel penalty: 0.05 → 0.02 — minimal disincentive, just enough to
      prefer action=0 when hovering.
    - No step_penalty — iter_2 showed this accelerates crashing.
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
    # Coefficient 12.0 is a middle ground: iter_1 used 30.0 (worked but
    # dominated 83% of magnitude), iter_3 used 3.0 (too weak, agent
    # closed only ~1.1 units of distance over the whole episode).
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    approach_reward = 12.0 * (old_dist - new_dist)

    # ── 2. descent_incentive ───────────────────────────────────────
    # Explicit reward for downward progress.  Supplements approach_reward
    # so that even if horizontal drift dilutes the distance signal,
    # descending is always rewarded.
    descent_incentive = 0.5 * max(0.0, oy - y)

    # ── 3. centering_penalty ───────────────────────────────────────
    # Light quadratic penalty for horizontal deviation from pad centre.
    # Prevents the agent from drifting sideways out of the viewport
    # (a failure mode seen in iter_2/3 where episodes terminate early).
    centering_penalty = -0.08 * (x ** 2)

    # ── 4. speed guidance (dual: slow bonus + excess penalty) ──────
    # Positive framing: reward LOW speed near the ground (where it
    # matters for safe landing).  Mild penalty for excess speed near
    # ground as a backstop.  Both decay exponentially with altitude
    # so high-altitude fast descent is unpenalised.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    ground_factor = 2.718281828 ** (-2.5 * y)   # 1.0 at y=0, ~0.08 at y=1, ~0.007 at y=2
    slow_bonus = 0.15 * ground_factor / (1.0 + speed)
    speed_penalty = -0.06 * speed * ground_factor

    # ── 5. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating.  Moderate coefficients —
    # strong enough to matter, not so strong the agent fears using
    # orientation engines entirely.
    attitude_penalty = -0.3 * (angle ** 2) - 0.08 * (angular_vel ** 2)

    # ── 6. landing_reward (new-contact-based) ──────────────────────
    # KEY FIX: reward only NEW contact events, not sustained contact.
    # This prevents the bounce exploit where the agent repeatedly
    # taps the pad to farm contact bonuses (iter_1 soft_landing_bonus
    # reached 256.5 at cp80% due to this).
    #
    # new_contact ∈ {0, 1, 2}: how many legs *just now* made contact.
    # Once both legs are down, subsequent steps give ZERO landing reward.
    prev_contact = obs[6] + obs[7]              # 0.0, 1.0, or 2.0
    curr_contact = left_contact + right_contact  # 0.0, 1.0, or 2.0
    new_contact = max(0.0, curr_contact - prev_contact)

    # Quality multiplier: higher bonus for soft, upright, centred contact
    contact_quality = 1.0 / (
        (1.0 + 5.0 * speed)
        * (1.0 + 3.0 * (angle ** 2) ** 0.5)
        * (1.0 + 2.0 * (x ** 2) ** 0.5)
    )
    landing_reward = 8.0 * new_contact * contact_quality

    # ── 7. fuel_penalty ────────────────────────────────────────────
    # Minimal penalty — just enough to break ties in favour of action=0
    # when other signals are neutral.  -0.02 is 6x smaller than iter_2's
    # -0.25, so engines can be used freely for braking and control.
    fuel_penalty = -0.02 if action != 0 else 0.0

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        approach_reward
        + descent_incentive
        + centering_penalty
        + slow_bonus
        + speed_penalty
        + attitude_penalty
        + landing_reward
        + fuel_penalty
    )

    components = {
        "approach_reward":   approach_reward,
        "descent_incentive": descent_incentive,
        "centering_penalty": centering_penalty,
        "slow_bonus":        slow_bonus,
        "speed_penalty":     speed_penalty,
        "attitude_penalty":  attitude_penalty,
        "landing_reward":    landing_reward,
        "fuel_penalty":      fuel_penalty,
    }

    return float(total_reward), components