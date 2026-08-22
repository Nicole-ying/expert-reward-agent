def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v7: rebalanced coefficients — moderate contact + strong approach gradient

    Diagnosis from v6 (score=154, best so far but 23% below target):
    - contact_bonus dominated at 99.7% magnitude share, drowning all other signals.
    - Checkpoint trend revealed score peaked at cp80% (240, above target) then
      degraded to cp100% (160). At cp80%, contact_bonus was lower (3888) and
      goal_proximity was higher (2.703) — the agent had efficient approach.
    - As training progressed, agent learned to maximize contact_bonus by spending
      more time on pad, while approach efficiency decayed (goal_proximity fell).
    - Root cause: w_pot=2.0 too weak to compete with w_contact=80 during late training.

    Changes from v6:
    1. w_contact: 80 → 40 — halve per-step contact bonus to reduce dominance.
       Still well above iter_5's failed w=25, but low enough to let other signals
       shape behavior during late training.
    2. w_pot: 2.0 → 6.0 — 3x stronger potential-based progress reward.
       At cp80% behavior levels, goal_proximity would be ~8.1 instead of ~2.7,
       making approach efficiency a meaningful signal throughout training.
    3. w_v: 0.3 → 0.5 — stronger velocity damping near ground to prevent
       aggressive dives that a stronger w_pot might otherwise encourage.
    4. w_ang: 0.05 → 0.08 — slightly stronger orientation penalty for
       better landing posture quality.

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity: potential-based progress reward ---
    # Rewards reduction in Euclidean distance to target (0,0).
    # Positive when approaching, negative when retreating.
    # Weight 6.0 (up from 2.0): provides strong approach gradient that
    # remains competitive with contact_bonus throughout training,
    # preventing the late-training degradation seen in v6.
    w_pot = 6.0
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = w_pot * (dist_cur - dist_nxt)

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Dampens high-speed approaches near the ground. Gate keeps far-field
    # maneuvering unpenalized. Weight 0.5 (up from 0.3) provides stronger
    # braking incentive near the pad to ensure soft landings.
    w_v = 0.5
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep lander upright.
    # Slightly increased angle weight (0.05→0.08) for better landing posture.
    w_ang = 0.08
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: per-step reward for safe pad contact ---
    # Per-step design (iter_2/v6 proven): rewards every step with both legs
    # on pad near target. Weight 40 (down from 80): still the largest single
    # signal but no longer at 99.7% dominance. This lets goal_proximity and
    # velocity_penalty meaningfully shape approach behavior.
    w_contact = 40.0
    near_target = (abs(x_nxt) < 0.5) and (abs(y_nxt) < 0.3)
    both_legs = (left_cont_nxt > 0.5) and (right_cont_nxt > 0.5)
    contact_bonus = w_contact if (near_target and both_legs) else 0.0

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
    }

    return total_reward, components