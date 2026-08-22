def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v6 — dense continuous landing-quality reward replaces sparse settled_reward.

    CORE INSIGHT from iter_4 & iter_5:
    - iter_4: landing_reward (transition-based new_contact * quality) was exploitable
      via bouncing — agent farmed 927.0 ep_sum. But the STRUCTURE was right:
      approach_reward (12.0) + slow_bonus (continuous near-ground speed reward)
      achieved score=230.8 at cp30% before exploits took over.
    - iter_5: settled_reward (binary: speed<0.4 & angle<0.25 & x<0.8 & both_legs)
      was a DEAD COMPONENT — 0.4% active_rate, 0.75 ep_sum. Binary thresholds
      created a reward cliff the agent could never discover.

    STRATEGY (v6):
    - Keep approach_reward at 12.0 (proven backbone, iter_4 cp30%).
    - REPLACE binary settled_reward with DENSE, CONTINUOUS soft_landing reward:
      per-step quality = ground_factor * speed_factor * angle_factor * x_factor * contact_factor
      Each factor is a smooth function → gradient everywhere, no dead zones.
      Per-step (not transition-based) → bouncing REDUCES reward, can't be farmed.
    - Keep descent_incentive (explicit vertical progress signal).
    - Keep attitude_penalty, centering_penalty, fuel_penalty as mild regularizers.
    - REMOVE height_penalty (redundant with descent_incentive).
    - REMOVE contact_discovery (tiny seed bonus, made obsolete by dense soft_landing).
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
    # Potential-based Euclidean distance shaping. 12.0 coefficient is the
    # sweet spot from iter_4 (score=230 at cp30% before landing_reward
    # exploitation took over).
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    approach_reward = 12.0 * (old_dist - new_dist)

    # ── 2. soft_landing (DENSE, CONTINUOUS) ─────────────────────────
    # Per-step reward proportional to how "settled-like" the state is.
    #
    # KEY PROPERTIES:
    # - DENSE: active every step (not 0.4% like settled_reward).
    # - CONTINUOUS: smooth factors, no binary thresholds → gradient everywhere.
    # - PER-STEP: not transition-based → bouncing REDUCES reward because
    #   breaking contact lowers contact_factor immediately.
    #
    # Each factor ∈ [0, 1] or [0.3, 1.0] for contact.  Multiplicative
    # combination means ALL conditions must be partially met for high reward,
    # providing a clear gradient toward the target state.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    abs_angle = abs(angle)
    abs_x = abs(x)

    # Ground proximity: 1.0 at y=0, decays exponentially with height.
    # At y=2: ~0.14, at y=5: ~0.007 → landing reward is negligible at altitude.
    ground_factor = 2.718281828 ** (-y)

    # Speed quality: 1.0 when stationary, ~0.2 at speed=1.0.
    speed_factor = 1.0 / (1.0 + 4.0 * speed)

    # Angle quality: 1.0 when level, ~0.25 at angle=1.0 rad (~57°).
    angle_factor = 1.0 / (1.0 + 3.0 * abs_angle)

    # Horizontal centering: 1.0 when x=0, ~0.33 at x=1.0.
    x_factor = 1.0 / (1.0 + 2.0 * abs_x)

    # Contact factor: scales from 0.3 (no contact) to 1.0 (both legs).
    # The 0.3 floor ensures gradient exists even before first contact.
    contact_factor = 0.3 + 0.7 * (left_contact + right_contact) / 2.0

    # Combined quality: product of all factors.
    # Perfect landing (y=0, speed=0, angle=0, x=0, both legs):
    #   3.0 * 1.0 * 1.0 * 1.0 * 1.0 * 1.0 = 3.0 per step.
    # Hovering near ground without contact (y≈0, speed≈0, no legs):
    #   3.0 * 1.0 * 1.0 * 1.0 * 1.0 * 0.3 = 0.9 per step — modest, won't
    #   incentivize hovering because episode terminates on settle.
    soft_landing = 3.0 * ground_factor * speed_factor * angle_factor * x_factor * contact_factor

    # ── 3. descent_incentive ────────────────────────────────────────
    # Explicit reward for downward progress. Supplements approach_reward
    # so descending is always locally rewarded, even when horizontal
    # movement dilutes the Euclidean distance signal.
    # Coefficient 0.3 (slightly reduced from iter_4's 0.5 to avoid
    # competing with soft_landing for magnitude share).
    descent_incentive = 0.3 * max(0.0, oy - y)

    # ── 4. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating. Proven coefficients from
    # iter_4/iter_5: ep_sum ~-1.3 to -1.7, non-intrusive but effective.
    attitude_penalty = -0.3 * (angle ** 2) - 0.08 * (angular_vel ** 2)

    # ── 5. centering_penalty ───────────────────────────────────────
    # Mild quadratic penalty for horizontal deviation. Prevents the
    # agent from drifting sideways out of the viewport (failure mode
    # seen in iter_2/3). Coefficient 0.08 from iter_4, non-intrusive.
    centering_penalty = -0.08 * (x ** 2)

    # ── 6. fuel_penalty ────────────────────────────────────────────
    # Minimal disincentive for engine use. -0.02 is just enough to
    # prefer action=0 when hovering, but cheap enough that braking
    # and attitude control are essentially free.
    fuel_penalty = -0.02 if action != 0 else 0.0

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        approach_reward
        + soft_landing
        + descent_incentive
        + attitude_penalty
        + centering_penalty
        + fuel_penalty
    )

    components = {
        "approach_reward":   approach_reward,
        "soft_landing":      soft_landing,
        "descent_incentive": descent_incentive,
        "attitude_penalty":  attitude_penalty,
        "centering_penalty": centering_penalty,
        "fuel_penalty":      fuel_penalty,
    }

    return float(total_reward), components