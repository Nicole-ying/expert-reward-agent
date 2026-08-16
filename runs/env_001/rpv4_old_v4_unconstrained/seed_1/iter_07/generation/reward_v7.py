def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v7 — anti-bounce contact mechanism replaces exploitable soft_landing.

    CORE INSIGHT from all 6 iterations:
    - Any per-step POSITIVE reward near the target state gets exploited:
      * iter_4: transition-based landing_reward → bouncing exploit (927 ep_sum)
      * iter_6: dense per-step soft_landing → hovering exploit (306 ep_sum)
      * iter_1: per-step soft_landing_bonus → spike to 256.5
    - Sparse/binary rewards die (iter_5 settled_reward: 0.4% active).
    - approach_reward (potential-based distance shaping) is the only
      consistently stable positive signal across all iterations.

    STRATEGY (v7):
    - Keep approach_reward at 12.0 (proven backbone).
    - REPLACE soft_landing with contact_gain_reward + contact_loss_penalty:
      * Reward NEW contact (gain), quality-gated by speed/angle/x.
      * Penalize LOSING contact (loss) MORE than the gain reward.
      * Net effect: bouncing (gain→loss→gain) is net NEGATIVE.
        Proper landing (gain→sustain→settle) is net POSITIVE.
    - Use speed_penalty (≤0, near ground only) instead of slow_bonus:
      can't be exploited — best case is speed=0 → penalty=0.
    - Remove all per-step positive rewards that don't require contact.
    - Keep mild regularizers (attitude, centering, fuel).
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
    # Potential-based Euclidean distance shaping. Coefficient 12.0 is the
    # proven sweet spot across iter_1/4/5/6. Stable, never exploited.
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    approach_reward = 12.0 * (old_dist - new_dist)

    # ── 2. descent_incentive ────────────────────────────────────────
    # Explicit reward for downward progress. Supplements approach_reward
    # so descending is always locally rewarded even when horizontal
    # movement dilutes the Euclidean distance signal.
    descent_incentive = 0.3 * max(0.0, oy - y)

    # ── 3. contact handling: gain reward + loss penalty ────────────
    # KEY INNOVATION: penalize losing contact MORE than rewarding gain.
    # This makes bouncing net-negative while proper landing is net-positive.
    #
    # Bouncing cycle:  gain ~1.0 → lose -2.0 → net negative per bounce.
    # Proper landing:  gain ~1.5 → sustain (no loss) → episode terminates.
    #
    # Contact quality gates ensure only gentle touchdowns get high reward.
    prev_contact_sum = obs[6] + obs[7]          # 0.0, 1.0, or 2.0
    curr_contact_sum = left_contact + right_contact  # 0.0, 1.0, or 2.0
    contact_gain = max(0.0, curr_contact_sum - prev_contact_sum)
    contact_loss = max(0.0, prev_contact_sum - curr_contact_sum)

    speed = (vx ** 2 + vy ** 2) ** 0.5
    # Quality: high when speed low, angle small, centered.
    # Speed factor dominates: at speed=2.0 quality drops to ~1/11 ≈ 0.09.
    contact_quality = 1.0 / (
        (1.0 + 5.0 * speed)
        * (1.0 + 2.0 * abs(angle))
        * (1.0 + abs(x))
    )

    contact_gain_reward = 3.0 * contact_gain * contact_quality
    contact_loss_penalty = -2.0 * contact_loss  # per leg lost, independent of quality

    # ── 4. speed_penalty (near ground only) ─────────────────────────
    # Penalize high speed near the ground to encourage braking.
    # Always ≤0: at speed=0 penalty=0 — no positive reward to exploit.
    # Decays exponentially with height: at y=3 ground_factor < 0.001.
    ground_factor = 2.718281828 ** (-2.5 * y)
    speed_penalty = -0.3 * speed * ground_factor

    # ── 5. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating. Proven coefficients from
    # iter_4/5/6: ep_sum ~-1.0 to -1.7, non-intrusive but effective.
    attitude_penalty = -0.3 * (angle ** 2) - 0.08 * (angular_vel ** 2)

    # ── 6. centering_penalty ───────────────────────────────────────
    # Mild quadratic penalty for horizontal deviation. Prevents
    # drifting out of viewport (failure mode in iter_2/3).
    centering_penalty = -0.08 * (x ** 2)

    # ── 7. fuel_penalty ────────────────────────────────────────────
    # Minimal disincentive for engine use. -0.02 is just enough to
    # prefer action=0 when hovering, but cheap enough that braking
    # and attitude control are essentially free.
    fuel_penalty = -0.02 if action != 0 else 0.0

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        approach_reward
        + descent_incentive
        + contact_gain_reward
        + contact_loss_penalty
        + speed_penalty
        + attitude_penalty
        + centering_penalty
        + fuel_penalty
    )

    components = {
        "approach_reward":     approach_reward,
        "descent_incentive":   descent_incentive,
        "contact_gain_reward": contact_gain_reward,
        "contact_loss_penalty": contact_loss_penalty,
        "speed_penalty":       speed_penalty,
        "attitude_penalty":    attitude_penalty,
        "centering_penalty":   centering_penalty,
        "fuel_penalty":        fuel_penalty,
    }

    return float(total_reward), components