def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v9 — restore iter_4 landing strength + hardened anti-bounce + amplified brakes.

    DIAGNOSIS (iter_8 failure, score=-93.26):
    - slow_bonus=0.15 too weak: ~0.014/step near ground vs approach ~0.19/step.
      Agent never learned to brake, crashed at step ~70, slow_bonus never accumulated.
    - contact_gain=3.0 + loss=-4.0: expected value of attempting contact is NEGATIVE
      for any quality < 0.75. Agent learned to AVOID contact. Active rate flatlined at 2.3%.
    - All checkpoint trends FLAT: agent repeated the same crash trajectory every episode.

    STRATEGY (v9):
    1. AMPLIFY slow_bonus: 0.15 → 0.8 base, widen ground_factor from e^(-2.5y) → e^(-1.5y).
       This gives the agent a longer braking window (activates at y≈2 instead of y≈1)
       and makes the brake signal competitive with approach_reward near the ground.
    2. RESTORE strong landing_reward: 10.0 * new_contact * quality (iter_4 used 8.0).
       Large enough to be discovered even at moderate quality.
    3. HARDEN anti-bounce: loss penalty -8.0/leg ensures ANY bounce is net-negative
       given realistic quality values (quality < 0.8 for any imperfect landing).
       Proper landing: gain once → sustain → episode terminates → net positive.
    4. RELAX quality gate: speed_factor=3.0 (was 4.0), angle_factor=1.0 (was 1.5),
       x_factor=1.0 (was 1.0). Wider tolerance so first successful contacts give
       meaningful gradient.
    5. STRONGER speed_penalty: -0.3 (was -0.06), proportional to the amplified
       slow_bonus as a backstop against excess speed near ground.
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
    # Potential-based Euclidean distance shaping. 12.0 is the proven
    # backbone across all iterations. Stable, never exploited.
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    approach_reward = 12.0 * (old_dist - new_dist)

    # ── 2. slow_bonus (AMPLIFIED) ──────────────────────────────────
    # Positive reward for low speed near the ground.  THE critical brake
    # signal.  Amplified 5.3× from iter_8 (0.15 → 0.80), and ground_factor
    # widened from e^(-2.5y) → e^(-1.5y) so the gradient activates earlier
    # (meaningful at y≈2 instead of y≈1), giving agent time to decelerate.
    #
    # Per-step values (illustrative):
    #   y=2.0, speed=3: 0.80*0.050/4.0 = 0.010  (just activating)
    #   y=1.0, speed=2: 0.80*0.223/3.0 = 0.059  (gradient building)
    #   y=0.5, speed=1: 0.80*0.472/2.0 = 0.189  (strong incentive to slow)
    #   y=0.1, speed=0.5: 0.80*0.861/1.5 = 0.459 (dominant signal)
    # Cannot explode: max at y=0,speed=0 is 0.80, and episode terminates on settle.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    ground_factor = 2.718281828 ** (-1.5 * y)
    slow_bonus = 0.80 * ground_factor / (1.0 + speed)

    # ── 3. speed_penalty (AMPLIFIED backstop) ──────────────────────
    # Negative penalty for excess speed near ground.  Amplified 5× from
    # iter_8 (-0.06 → -0.30), proportional to slow_bonus amplification.
    # At y=1.0, speed=2: penalty ≈ -0.30*2*0.223 = -0.134.
    # Net brake signal at y=1.0, speed=2: 0.059 - 0.134 = -0.075 (penalty).
    # Net brake signal at y=0.5, speed=1: 0.189 - 0.142 = +0.047 (reward).
    # Gradient: fast near ground → penalised; slow near ground → rewarded.
    speed_penalty = -0.30 * speed * ground_factor

    # ── 4. descent_incentive ────────────────────────────────────────
    # Explicit reward for downward progress.  Back to iter_4 level (0.5,
    # was 0.3 in iter_7/8).  Supplements approach_reward so descending is
    # always locally rewarded even when horizontal drift dilutes the
    # Euclidean distance signal.
    descent_incentive = 0.5 * max(0.0, oy - y)

    # ── 5. landing_reward + contact_loss_penalty ──────────────────
    # RESTORED strong landing reward (10.0, iter_4 used 8.0) with
    # HARDENED anti-bounce loss penalty (-8.0 per leg).
    #
    # Bounce economics (per leg, quality-gated gain, flat loss):
    #   Quality=0.4 (speed≈0.5, angle≈0.2, x≈0.3): gain=4.0, loss=8.0 → net -4.0
    #   Quality=0.7 (speed≈0.15, angle≈0.05, x≈0.1): gain=7.0, loss=8.0 → net -1.0
    #   ANY realistic quality < 0.8 → bouncing is NET NEGATIVE.
    #
    # Proper landing economics:
    #   0→1 leg: gain 10.0*q1, 1→2 legs: gain 10.0*q2, then episode terminates.
    #   No loss fires.  Net strongly positive (10-20 for decent quality).
    #
    # Quality gate RELAXED vs iter_8: speed_factor=3.0 (was 4.0),
    # angle_factor=1.0 (was 1.5).  Wider tolerance → first contacts give
    # meaningful gradient → agent can discover the landing signal.
    prev_contact_sum = obs[6] + obs[7]               # 0.0, 1.0, or 2.0
    curr_contact_sum = left_contact + right_contact   # 0.0, 1.0, or 2.0
    new_contact = max(0.0, curr_contact_sum - prev_contact_sum)
    contact_loss = max(0.0, prev_contact_sum - curr_contact_sum)

    contact_quality = 1.0 / (
        (1.0 + 3.0 * speed)
        * (1.0 + 1.0 * abs(angle))
        * (1.0 + 1.0 * abs(x))
    )

    landing_reward = 10.0 * new_contact * contact_quality
    contact_loss_penalty = -8.0 * contact_loss  # per leg, flat (no quality gate on loss)

    # ── 6. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating.  Proven coefficients across
    # iter_4/6/7/8: ep_sum ~-1.0 to -2.0, non-intrusive but effective.
    attitude_penalty = -0.3 * (angle ** 2) - 0.08 * (angular_vel ** 2)

    # ── 7. centering_penalty ───────────────────────────────────────
    # Mild quadratic penalty for horizontal deviation. Prevents
    # drifting out of viewport (failure mode in iter_2/3).
    centering_penalty = -0.08 * (x ** 2)

    # ── 8. fuel_penalty ────────────────────────────────────────────
    # Minimal disincentive for engine use. -0.02 is cheap enough that
    # braking and attitude control are essentially free, but the agent
    # prefers action=0 when other signals are neutral.
    fuel_penalty = -0.02 if action != 0 else 0.0

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        approach_reward
        + slow_bonus
        + speed_penalty
        + descent_incentive
        + landing_reward
        + contact_loss_penalty
        + attitude_penalty
        + centering_penalty
        + fuel_penalty
    )

    components = {
        "approach_reward":      approach_reward,
        "slow_bonus":           slow_bonus,
        "speed_penalty":        speed_penalty,
        "descent_incentive":    descent_incentive,
        "landing_reward":       landing_reward,
        "contact_loss_penalty": contact_loss_penalty,
        "attitude_penalty":     attitude_penalty,
        "centering_penalty":    centering_penalty,
        "fuel_penalty":         fuel_penalty,
    }

    return float(total_reward), components