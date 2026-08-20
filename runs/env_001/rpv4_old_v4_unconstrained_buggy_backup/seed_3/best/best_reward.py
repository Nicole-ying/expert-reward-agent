def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v9: transition-based contact reward — fix per-step over-stay degradation

    Diagnosis from v8 (score=154, cp80% peaked at 240 then degraded to 160):
    - Per-step contact_bonus (w=65/step) created a persistent over-stay incentive.
      Agent learned to land then stayed collecting ~57 steps of bonus (3695 total),
      drowning out efficiency signals (goal_proximity ~2.3, velocity ~-6.6).
    - cp80%→cp100%: contact rebounded +17%, approach efficiency dropped -17%,
      score fell from 240→160. Classic per-step reward exploitation.
    - Weight tuning alone (v6=80, v7=40→collapsed, v8=65) cannot fix this:
      contact must be strong enough to drive landing, but per-step always over-stays.

    Changes from v8:
    1. contact_bonus: restructured from per-step accumulation to transition-based:
       - landing_bonus (500, one-time): rewards the FIRST step achieving safe contact.
         Equivalent to ~8 old steps — concentrated signal for learning to land.
       - sustain_bonus (3/step, per-step): tiny gradient to stay on pad until
         natural termination. 3 vs old 65 = 22x weaker over-stay incentive.
    2. All other components unchanged from v8 proven backbone.

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations (current + next) ---
    x_cur, y_cur = obs[0], obs[1]
    left_cont_cur, right_cont_cur = obs[6], obs[7]

    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity: potential-based progress reward ---
    # Rewards reduction in Euclidean distance to target (0,0).
    # Positive when approaching, negative when retreating.
    # Weight 2.0 (v6/v8 proven): consistent approach gradient.
    w_pot = 2.0
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = w_pot * (dist_cur - dist_nxt)

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Dampens high-speed approaches near the ground. Gate keeps far-field
    # maneuvering unpenalized. Weight 0.3 (v6/v8 proven).
    w_v = 0.3
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep lander upright.
    # Weights unchanged from v6/v8 — stable across all iterations.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: transition-based landing reward ---
    # KEY CHANGE (v9): switch from per-step accumulation to event-based.
    #
    # Problem: per-step bonus created over-stay exploitation.
    #   Old: +65 every step on pad → 3695 total, 99.7% magnitude share.
    #
    # Solution: one-time landing bonus + tiny sustain.
    #   - landing_bonus: rewards the TRANSITION into safe contact (first touch).
    #     Concentrated 500-point signal — strong enough to drive landing learning.
    #   - sustain_bonus: 3/step for maintaining contact. Provides gentle gradient
    #     to stay on pad until natural termination, but 22x weaker than old 65/step
    #     so over-stay is no longer the dominant strategy.
    #
    # Safe contact definition: both legs on pad, near target center.
    w_land = 500.0      # one-time landing bonus
    w_sustain = 3.0     # per-step sustain (was 65 in v8 → 22x reduction)

    near_target_cur = (abs(x_cur) < 0.5) and (abs(y_cur) < 0.3)
    both_legs_cur = (left_cont_cur > 0.5) and (right_cont_cur > 0.5)
    was_safe = near_target_cur and both_legs_cur

    near_target_nxt = (abs(x_nxt) < 0.5) and (abs(y_nxt) < 0.3)
    both_legs_nxt = (left_cont_nxt > 0.5) and (right_cont_nxt > 0.5)
    is_safe = near_target_nxt and both_legs_nxt

    # Transition: first step entering safe contact
    landing_bonus = w_land if (is_safe and not was_safe) else 0.0
    # Sustain: every step maintaining safe contact
    sustain_bonus = w_sustain if is_safe else 0.0

    contact_bonus = landing_bonus + sustain_bonus

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
    }

    return total_reward, components