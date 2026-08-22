def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v5 — settled-state reward replaces exploitable transition bonuses.

    CORE INSIGHT from iter_1 & iter_4:
    - Any contact-transition bonus (new_contact-based) gets exploited via bouncing.
    - Any unconditional near-ground low-speed bonus (slow_bonus) encourages hovering.
    - iter_4 cp30% hit score=230.8 with approach shaping alone, then exploits took over.

    STRATEGY:
    - Keep the approach_reward backbone (12.0 coefficient, proven at cp30%).
    - REPLACE landing_reward + slow_bonus + speed_penalty with a single
      **settled_reward**: per-step reward ONLY when the agent is actually in the
      target state (dual contact + low speed + level + centered).
    - This is inherently bounce-proof: bouncing breaks the settled condition,
      so the agent wants to STAY settled, not bounce.
    - REPLACE descent_incentive + centering_penalty (dead-weight components,
      each <2% magnitude share) with a mild height_penalty to discourage hovering.
    - Tiny contact_discovery seed reward to help the agent discover pad contact.
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
    # proven sweet spot: iter_1 (30.0, dominated 83%), iter_3 (3.0, too weak),
    # iter_4 (12.0, worked — score=230 at cp30% before exploits kicked in).
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    approach_reward = 12.0 * (old_dist - new_dist)

    # ── 2. settled_reward ──────────────────────────────────────────
    # Per-step reward for being in the actual target state.
    # This REPLACES: landing_reward (exploitable transition bonus),
    # slow_bonus (exploitable hovering reward), and speed_penalty.
    #
    # KEY: sustained settled state IS the task goal. Bouncing breaks the
    # condition (loses contact or gains speed), so the agent wants to stay.
    # This is inherently aligned with the external evaluator.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    both_legs_down = (left_contact > 0.5 and right_contact > 0.5)
    is_settled = (
        both_legs_down
        and speed < 0.4           # nearly stationary
        and abs(angle) < 0.25     # roughly level (≈14°)
        and abs(x) < 0.8          # near pad centre
    )
    settled_reward = 2.5 if is_settled else 0.0

    # ── 3. contact_discovery ───────────────────────────────────────
    # TINY seed bonus for any new leg contact. Helps the agent discover
    # that touching the pad is good, without being worth farming:
    # max 0.6 per dual-touchdown vs 2.5 per settled step.
    # Once settled, the agent never breaks contact, so this becomes silent.
    prev_contact = obs[6] + obs[7]
    curr_contact = left_contact + right_contact
    new_contact = max(0.0, curr_contact - prev_contact)
    contact_discovery = 0.3 * new_contact

    # ── 4. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating. Same coefficients as iter_4
    # (proven non-intrusive, ep_sum ~-1.3).
    attitude_penalty = -0.3 * (angle ** 2) - 0.08 * (angular_vel ** 2)

    # ── 5. fuel_penalty ────────────────────────────────────────────
    # Mild disincentive for engine use. -0.03 is slightly stronger than
    # iter_4's -0.02 to better discourage unnecessary firing (which enables
    # bouncing). Still low enough that braking and attitude control are cheap.
    fuel_penalty = -0.03 if action != 0 else 0.0

    # ── 6. height_penalty ──────────────────────────────────────────
    # Mild linear penalty for altitude. Replaces the dead centering_penalty
    # and descent_incentive. Provides constant downward pressure so the agent
    # doesn't loiter at high altitude. At y=5 this is -0.05/step, negligible
    # against approach_reward's ~0.5-1.0 per step during active descent.
    height_penalty = -0.01 * y

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        approach_reward
        + settled_reward
        + contact_discovery
        + attitude_penalty
        + fuel_penalty
        + height_penalty
    )

    components = {
        "approach_reward":   approach_reward,
        "settled_reward":    settled_reward,
        "contact_discovery": contact_discovery,
        "attitude_penalty":  attitude_penalty,
        "fuel_penalty":      fuel_penalty,
        "height_penalty":    height_penalty,
    }

    return float(total_reward), components