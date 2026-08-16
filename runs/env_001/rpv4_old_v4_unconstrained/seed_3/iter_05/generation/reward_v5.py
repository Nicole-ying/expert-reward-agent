def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v5: restore one-shot contact_bonus as primary landing driver; gut landing_approach;
        widen settled_reward gates; squared-distance proximity.

    Core thesis (from iter_1→4 evidence):
    - contact_bonus is the ONLY signal that has ever produced landing behavior.
      iter_2: per-step contact_bonus → 20/20 landing. iter_3: one-shot → 9/20 + hover.
      iter_4: removed → score collapse to 10.
    - landing_approach at high weight causes hover parasitism (iter_3: 304 ep_sum).
    - Fix: HUGE one-shot contact_bonus + TINY landing_approach + discoverable settled_reward.

    Changes from v4:
    - RESTORED one-shot transition contact_bonus, w=50 (was 8 in v3, removed in v4).
    - landing_approach: 0.20 → 0.05 (anti-hover).
    - settled_reward: wider gates (spatial 0.04→0.10, speed 0.03→0.08), w 3.5→2.0.
    - goal_proximity: squared distance for stronger far-field gradient.
    - contact_speed_penalty: kept, w reduced 2.0→1.2.
    """

    # ── unpack ──────────────────────────────────────────────────────
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_c_cur, right_c_cur = obs[6], obs[7]
    left_c_nxt, right_c_nxt = next_obs[6], next_obs[7]

    # ── derived ─────────────────────────────────────────────────────
    dist_cur_sq = x_cur ** 2 + y_cur ** 2
    dist_nxt_sq = x_nxt ** 2 + y_nxt ** 2
    dist_cur = dist_cur_sq ** 0.5
    dist_nxt = dist_nxt_sq ** 0.5
    speed_sq = vx_nxt ** 2 + vy_nxt ** 2
    both_legs = left_c_nxt * right_c_nxt        # 0.0 or 1.0

    # ══════════════════════════════════════════════════════════════════
    # 1. goal_proximity — squared-distance penalty + approach bonus
    #    Squared gives stronger gradient far away, gentler near target.
    # ══════════════════════════════════════════════════════════════════
    w_dist = 0.10
    proximity_penalty = -w_dist * dist_nxt_sq       # -2.5 at dist=5, -0.1 at dist=1

    w_approach = 1.0
    delta_dist = dist_cur - dist_nxt                 # positive = getting closer
    approach_bonus = w_approach * max(0.0, delta_dist)

    goal_proximity = proximity_penalty + approach_bonus

    # ══════════════════════════════════════════════════════════════════
    # 2. velocity_penalty — gated speed penalty + anti-slam near ground
    # ══════════════════════════════════════════════════════════════════
    w_v = 0.30
    k_gate = 5.0
    gate = 1.0 / (1.0 + k_gate * dist_nxt)          # ~1 near pad, ~0 far away

    # anti-slam: penalise downward velocity when very close to ground
    w_slam = 1.5
    slam_gate = 1.0 / (1.0 + 25.0 * dist_nxt)        # tight: only |dist| < ~0.15 matters
    downward_v = max(0.0, -vy_nxt)                   # positive = moving downward

    velocity_penalty = (
        -w_v * gate * speed_sq
        - w_slam * slam_gate * (downward_v ** 2)
    )

    # ══════════════════════════════════════════════════════════════════
    # 3. orientation_penalty — stay upright, suppress spin
    # ══════════════════════════════════════════════════════════════════
    w_ang = 0.05
    w_angvel = 0.01
    orientation_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # ══════════════════════════════════════════════════════════════════
    # 4. landing_approach — TINY continuous shaping, anti-hover
    #    Weight gutted from 0.20 (v4) / 0.60 (v3) → 0.05.
    #    Only provides gradient; NOT profitable as a steady state.
    # ══════════════════════════════════════════════════════════════════
    near_y   = 2.718281828 ** (-abs(y_nxt) / 0.25)
    near_x   = 2.718281828 ** (-abs(x_nxt) / 0.35)
    slow     = 2.718281828 ** (-speed_sq / 0.60)
    upright  = 2.718281828 ** (-abs(angle_nxt) / 0.30)
    landing_approach = 0.05 * near_y * near_x * slow * upright

    # ══════════════════════════════════════════════════════════════════
    # 5. contact_bonus — ONE-SHOT transition bonus (RESTORED)
    #    The essential landing driver. Fires ONCE on first touchdown.
    #    Tight spatial + speed gates prevent bounce-exploit:
    #    a re-trigger after bouncing has worse position & speed → small bonus.
    # ══════════════════════════════════════════════════════════════════
    w_contact = 50.0
    had_contact = left_c_cur * right_c_cur             # were we already touching?
    has_contact = both_legs                            # are we touching now?
    new_contact = has_contact * (1.0 - had_contact)    # 1 ONLY on transition step

    # tight gates: only a clean landing at the pad gets full bonus
    spatial_gate = 2.718281828 ** (-dist_nxt_sq / 0.06)    # σ²=0.06 → tight
    speed_gate   = 2.718281828 ** (-speed_sq / 0.08)       # σ²=0.08 → soft landing
    contact_bonus = w_contact * new_contact * spatial_gate * speed_gate

    # ══════════════════════════════════════════════════════════════════
    # 6. settled_reward — per-step reward for STAYING landed
    #    Wider gates than v4 so agent can discover it.
    #    Only active when both legs are in sustained contact.
    # ══════════════════════════════════════════════════════════════════
    w_settled = 2.0
    s_spatial  = 2.718281828 ** (-dist_nxt_sq / 0.10)   # was 0.04 in v4
    s_slow     = 2.718281828 ** (-speed_sq / 0.08)      # was 0.03 in v4
    s_upright  = 2.718281828 ** (-abs(angle_nxt) / 0.15) # was 0.10 in v4
    settled_reward = w_settled * both_legs * s_spatial * s_slow * s_upright

    # ══════════════════════════════════════════════════════════════════
    # 7. contact_speed_penalty — anti-bounce: punish speed while in contact
    # ══════════════════════════════════════════════════════════════════
    w_contact_spd = 1.2
    contact_speed_penalty = -w_contact_spd * both_legs * speed_sq

    # ══════════════════════════════════════════════════════════════════
    # 8. fuel_penalty — small regularisation
    # ══════════════════════════════════════════════════════════════════
    w_fuel = 0.005
    fuel_penalty = -w_fuel * (0.0 if action == 0 else 1.0)

    # ══════════════════════════════════════════════════════════════════
    # assemble
    # ══════════════════════════════════════════════════════════════════
    total_reward = (
        goal_proximity
        + velocity_penalty
        + orientation_penalty
        + landing_approach
        + contact_bonus
        + settled_reward
        + contact_speed_penalty
        + fuel_penalty
    )

    components = {
        'goal_proximity': goal_proximity,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'landing_approach': landing_approach,
        'contact_bonus': contact_bonus,
        'settled_reward': settled_reward,
        'contact_speed_penalty': contact_speed_penalty,
        'fuel_penalty': fuel_penalty,
    }

    return total_reward, components