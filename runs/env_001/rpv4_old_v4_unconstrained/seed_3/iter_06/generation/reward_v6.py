def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v6: anti-bounce overhaul.

    Diagnosis (from iter_5 checkpoint):
    - contact_bonus spiked to 2454-2711 at cp80-90% (active 10.5%), then collapsed to 26.
      Agent discovered bouncing exploit: lose contact → regain → re-trigger one-shot bonus.
    - settled_reward at w=2.0 couldn't compete with bounce profits (~25-40 per bounce cycle).
    - goal_proximity was dead (~-10 flat across all checkpoints).
    - Score peaked at 249 (cp50%) then crashed to 114 (cp80%) → exploitative policy collapsed.

    Changes from v5:
    1. contact_bonus: w 50→15 — reduce bounce profit; still clear landing signal.
    2. settled_reward: w 2.0→4.0 — make staying put clearly better than bouncing.
    3. NEW contact_break_penalty: -6.0 when losing contact — direct anti-bounce cost.
    4. goal_proximity: linear distance (not squared), remove approach_bonus (oscillation risk).
    5. NEW pre_landing_bonus: reward good posture near pad WITHOUT contact — bridges approach→landing.
    6. landing_approach: 0.05→0.03 — further anti-hover.
    7. velocity_penalty: gated, stronger anti-slam near ground.
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
    speed = speed_sq ** 0.5
    had_contact = left_c_cur * right_c_cur          # 0.0 or 1.0
    has_contact = left_c_nxt * right_c_nxt           # 0.0 or 1.0
    both_legs = has_contact

    # ══════════════════════════════════════════════════════════════════
    # 1. goal_proximity — linear distance penalty (simple, effective)
    #    Removed approach_bonus: it rewarded oscillation (approach→overshoot→re-approach).
    #    Linear gives consistent gradient: stronger far, gentle near.
    # ══════════════════════════════════════════════════════════════════
    w_dist = 0.25
    goal_proximity = -w_dist * dist_nxt               # -1.25 at dist=5, -0.25 at dist=1

    # ══════════════════════════════════════════════════════════════════
    # 2. velocity_penalty — gated speed cost + strong anti-slam near ground
    #    Gate: speed penalty only matters near the pad (allows free motion far away).
    #    Anti-slam: harsh penalty for downward speed when very close to ground.
    # ══════════════════════════════════════════════════════════════════
    k_gate = 4.0
    proximity_gate = 1.0 / (1.0 + k_gate * dist_nxt)   # ~1 at pad, ~0.2 at dist=1

    w_v = 0.40
    gated_speed_penalty = -w_v * proximity_gate * speed_sq

    # anti-slam: penalise downward velocity near ground (prevents crash-landing)
    w_slam = 3.0
    slam_gate = 2.718281828 ** (-dist_nxt_sq / 0.03)    # tight: only dist < ~0.2 matters
    downward_v = max(0.0, -vy_nxt)                      # positive = moving down
    anti_slam = -w_slam * slam_gate * (downward_v ** 2)

    velocity_penalty = gated_speed_penalty + anti_slam

    # ══════════════════════════════════════════════════════════════════
    # 3. orientation_penalty — stay upright, suppress spin
    # ══════════════════════════════════════════════════════════════════
    w_ang = 0.06
    w_angvel = 0.015
    orientation_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # ══════════════════════════════════════════════════════════════════
    # 4. landing_approach — micro shaping, anti-hover
    #    Further reduced from 0.05 → 0.03. Only a whisper of gradient.
    # ══════════════════════════════════════════════════════════════════
    near_y   = 2.718281828 ** (-abs(y_nxt) / 0.25)
    near_x   = 2.718281828 ** (-abs(x_nxt) / 0.35)
    slow     = 2.718281828 ** (-speed_sq / 0.60)
    upright  = 2.718281828 ** (-abs(angle_nxt) / 0.30)
    landing_approach = 0.03 * near_y * near_x * slow * upright

    # ══════════════════════════════════════════════════════════════════
    # 5. pre_landing_bonus — reward good approach posture near pad
    #    Bridges the gap between "接近目标" and "着陆接触".
    #    ONLY active when NOT in contact — once landed, settled_reward takes over.
    #    This gives the agent a reason to slow down and align BEFORE touching down.
    # ══════════════════════════════════════════════════════════════════
    w_preland = 0.40
    near_pad = 2.718281828 ** (-dist_nxt_sq / 0.15)       # σ²=0.15: active within ~0.5 dist
    pre_slow = 2.718281828 ** (-speed_sq / 0.25)           # reward speed < ~0.5
    pre_upright = 2.718281828 ** (-abs(angle_nxt) / 0.25)  # reward angle < ~0.5
    not_contacted = 1.0 - has_contact                       # 0 when in contact
    pre_landing_bonus = w_preland * near_pad * pre_slow * pre_upright * not_contacted

    # ══════════════════════════════════════════════════════════════════
    # 6. contact_bonus — ONE-SHOT transition bonus (REDUCED)
    #    w reduced from 50→15. Still a clear signal to land, but no longer
    #    profitable to bounce. Bounce cycle: +15*gates - 6(break_penalty) < 0.
    #    Tight spatial + speed gates ensure only clean landings get full bonus.
    # ══════════════════════════════════════════════════════════════════
    w_contact = 15.0
    new_contact = has_contact * (1.0 - had_contact)       # 1 ONLY on transition step

    spatial_gate = 2.718281828 ** (-dist_nxt_sq / 0.05)   # σ²=0.05: tight
    speed_gate   = 2.718281828 ** (-speed_sq / 0.06)      # σ²=0.06: soft landing only
    contact_bonus = w_contact * new_contact * spatial_gate * speed_gate

    # ══════════════════════════════════════════════════════════════════
    # 7. contact_break_penalty — NEW: punish losing contact
    #    Direct anti-bounce signal. When agent was in contact and loses it,
    #    apply a meaningful penalty. This makes bouncing unprofitable:
    #    bounce_profit ≈ 15*0.5 - 6 = +1.5 at best (with bad gates, net negative).
    #    Staying settled: ~4.0/step — clearly superior.
    # ══════════════════════════════════════════════════════════════════
    w_break = 6.0
    lost_contact = had_contact * (1.0 - has_contact)      # 1 when contact→no contact
    contact_break_penalty = -w_break * lost_contact

    # ══════════════════════════════════════════════════════════════════
    # 8. settled_reward — per-step reward for STAYING landed (INCREASED)
    #    w doubled from 2.0→4.0. Now the dominant positive signal:
    #    ~4.0 * 0.9³ ≈ 2.9/step when well-settled. Over 150 settled steps ≈ 435.
    #    This dwarfs any bounce profit and anchors the policy in stable landing.
    # ══════════════════════════════════════════════════════════════════
    w_settled = 4.0
    s_spatial  = 2.718281828 ** (-dist_nxt_sq / 0.10)
    s_slow     = 2.718281828 ** (-speed_sq / 0.08)
    s_upright  = 2.718281828 ** (-abs(angle_nxt) / 0.15)
    settled_reward = w_settled * has_contact * s_spatial * s_slow * s_upright

    # ══════════════════════════════════════════════════════════════════
    # 9. contact_speed_penalty — anti-bounce: punish speed while in contact
    #    Retained as a secondary anti-bounce measure. Works alongside
    #    contact_break_penalty: speed while contacted → penalty; losing contact → penalty.
    # ══════════════════════════════════════════════════════════════════
    w_contact_spd = 1.5
    contact_speed_penalty = -w_contact_spd * has_contact * speed_sq

    # ══════════════════════════════════════════════════════════════════
    # 10. fuel_penalty — small regularisation
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
        + pre_landing_bonus
        + contact_bonus
        + contact_break_penalty
        + settled_reward
        + contact_speed_penalty
        + fuel_penalty
    )

    components = {
        'goal_proximity': goal_proximity,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'landing_approach': landing_approach,
        'pre_landing_bonus': pre_landing_bonus,
        'contact_bonus': contact_bonus,
        'contact_break_penalty': contact_break_penalty,
        'settled_reward': settled_reward,
        'contact_speed_penalty': contact_speed_penalty,
        'fuel_penalty': fuel_penalty,
    }

    return total_reward, components