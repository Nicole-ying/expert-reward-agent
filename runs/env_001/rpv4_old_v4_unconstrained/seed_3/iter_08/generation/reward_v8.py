def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v8: return to iter_5 contact_bonus architecture (proven, score=175.7)
        with modest anti-bounce hardening.

    Diagnosis (iter_5→7 regression):
    - iter_5 contact_bonus: w=50, σ²_spatial=0.06, σ²_speed=0.08 → ep_sum=1177,
      score=175.7. Bounce phase at cp80-90% self-resolved by cp100%.
    - iter_7 over-corrected: tightened gates (σ²→0.04 each) + approach_scale →
      contact_bonus collapsed to 24.4 ep_sum, agent never learned to land.
    - The bounce exploit in iter_5 was self-limiting: agent naturally converged
      to settled_reward dominance by cp100%. No heavy anti-bounce needed.

    Changes from iter_7 (→ iter_5):
    1. contact_bonus: RESTORE iter_5 gates (σ² 0.06/0.08), REMOVE approach_scale.
       w=50 (same). This is the single most critical fix.
    2. settled_reward: w 6→3.5. iter_7's w=6 was compensating for dead contact_bonus
       but created extreme volatility. Moderate boost over iter_5's w=2.
    3. contact_speed_penalty: w 2.0→1.8 (between iter_5 1.2 and iter_7 2.0).
    4. goal_proximity: iter_5-style squared distance for strong far-field gradient.
       No approach_bonus (avoids oscillation cycles).
    5. velocity_penalty: iter_5-style gated + anti-slam, slightly wider slam gate.
    6. landing_approach: w 0.03→0.04, whisper only.
    7. orientation_penalty & fuel_penalty: same as iter_5.
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
    dist_nxt_sq = x_nxt ** 2 + y_nxt ** 2
    dist_nxt = dist_nxt_sq ** 0.5
    speed_sq = vx_nxt ** 2 + vy_nxt ** 2
    had_contact = left_c_cur * right_c_cur          # 0.0 or 1.0
    has_contact = left_c_nxt * right_c_nxt           # 0.0 or 1.0

    # ══════════════════════════════════════════════════════════════════
    # 1. goal_proximity — squared-distance penalty
    #    Strong far-field gradient (at dist=5: -2.5), gentle near pad.
    #    No approach_bonus: avoids oscillation (approach→overshoot→re-approach).
    # ══════════════════════════════════════════════════════════════════
    w_dist = 0.10
    goal_proximity = -w_dist * dist_nxt_sq            # -2.5 at dist=5, -0.1 at dist=1

    # ══════════════════════════════════════════════════════════════════
    # 2. velocity_penalty — gated speed cost + anti-slam near ground
    #    Gate allows free motion far from pad.
    #    Anti-slam penalises downward velocity when very close to ground.
    # ══════════════════════════════════════════════════════════════════
    k_gate = 5.0
    proximity_gate = 1.0 / (1.0 + k_gate * dist_nxt)   # ~1 near pad, ~0.17 at dist=1

    w_v = 0.30
    gated_speed_penalty = -w_v * proximity_gate * speed_sq

    w_slam = 2.0
    slam_gate = 2.718281828 ** (-dist_nxt_sq / 0.04)    # tight: only dist < ~0.25 matters
    downward_v = max(0.0, -vy_nxt)
    anti_slam = -w_slam * slam_gate * (downward_v ** 2)

    velocity_penalty = gated_speed_penalty + anti_slam

    # ══════════════════════════════════════════════════════════════════
    # 3. orientation_penalty — stay upright, suppress spin
    # ══════════════════════════════════════════════════════════════════
    w_ang = 0.05
    w_angvel = 0.01
    orientation_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # ══════════════════════════════════════════════════════════════════
    # 4. landing_approach — whisper-level shaping
    #    Tiny weight provides soft gradient toward slow+near+upright
    #    without creating a profitable hover steady-state.
    # ══════════════════════════════════════════════════════════════════
    near_y   = 2.718281828 ** (-abs(y_nxt) / 0.25)
    near_x   = 2.718281828 ** (-abs(x_nxt) / 0.35)
    slow     = 2.718281828 ** (-speed_sq / 0.60)
    upright  = 2.718281828 ** (-abs(angle_nxt) / 0.30)
    landing_approach = 0.04 * near_y * near_x * slow * upright

    # ══════════════════════════════════════════════════════════════════
    # 5. contact_bonus — ONE-SHOT transition bonus (iter_5 RESTORED)
    #
    #    This is THE essential landing driver. Fires ONCE on first touchdown.
    #    Gates at iter_5 levels (σ² 0.06/0.08) — tight enough to require
    #    a reasonable landing but loose enough to be discoverable.
    #
    #    NO approach_scale: the iter_5 evidence shows bounce exploit is
    #    self-limiting. Agent converged from bounce (cp80-90%) to settled
    #    (cp100%) naturally. Over-engineering kills the signal.
    #
    #    Anti-bounce now comes from: boosted settled_reward (w=3.5 vs 2)
    #    + stronger contact_speed_penalty (w=1.8 vs 1.2).
    # ══════════════════════════════════════════════════════════════════
    w_contact = 50.0
    new_contact = has_contact * (1.0 - had_contact)       # 1 ONLY on transition step

    spatial_gate = 2.718281828 ** (-dist_nxt_sq / 0.06)   # σ²=0.06: iter_5 level
    speed_gate   = 2.718281828 ** (-speed_sq / 0.08)      # σ²=0.08: iter_5 level

    contact_bonus = w_contact * new_contact * spatial_gate * speed_gate

    # ══════════════════════════════════════════════════════════════════
    # 6. settled_reward — per-step reward for STAYING landed
    #
    #    w=3.5 (iter_5 was 2.0, iter_7 was 6.0). Moderate boost makes
    #    sustained settling clearly dominate bouncing without creating
    #    the extreme volatility seen in iter_7.
    #
    #    Gates: intermediate between iter_5 and iter_7. Wide enough to
    #    reward imperfect landings, tight enough to incentivise improvement.
    # ══════════════════════════════════════════════════════════════════
    w_settled = 3.5
    s_spatial  = 2.718281828 ** (-dist_nxt_sq / 0.12)    # σ²=0.12: moderate
    s_slow     = 2.718281828 ** (-speed_sq / 0.10)       # σ²=0.10: moderate
    s_upright  = 2.718281828 ** (-abs(angle_nxt) / 0.20) # σ²=0.20: tolerant
    settled_reward = w_settled * has_contact * s_spatial * s_slow * s_upright

    # ══════════════════════════════════════════════════════════════════
    # 7. contact_speed_penalty — anti-bounce friction
    #    w=1.8 (iter_5 was 1.2, iter_7 was 2.0). Any speed while in contact
    #    incurs penalty, making bounce cycles less profitable.
    # ══════════════════════════════════════════════════════════════════
    w_contact_spd = 1.8
    contact_speed_penalty = -w_contact_spd * has_contact * speed_sq

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