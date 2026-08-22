def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v8: v6 backbone with modest contact reduction (80→65) to curb late-training
    over-stay degradation while staying safely above the activation threshold.

    Diagnosis from v7 (score=-121, total regression from v6's 154):
    - contact 80→40 collapsed landing behavior (active 13.7%→0.1%).
      History shows contact must be ≥80 to reliably activate (iter_2=100✓,
      iter_6=80✓, iter_7=40✗, iter_5=25✗).
    - velocity 0.3→0.5 + potential 2→6 created conflicting near-ground signals
      that made approach net-positive but landing net-negative vs staying aloft.

    Diagnosis from v6 (best=154, cp80%=240, cp100%=160):
    - Architecture is sound: potential-based progress + per-step contact.
    - cp80%→cp100% degradation: agent over-stayed on pad (+8 contact steps),
      approach efficiency dropped (goal_proximity 2.703→2.242).
    - Contact at 99.7% magnitude_share drowns efficiency signals in late training.

    Changes from v7 (effectively reverting toward v6):
    1. w_contact: 40 → 65 — modest reduction from v6's 80 (19% cut, not 50%).
       Still well above the failure threshold seen at 40 and 25.
       Reduces per-step incentive to over-stay without killing landing motivation.
    2. w_pot: 6.0 → 2.0 — revert to v6's proven approach gradient.
       v7's 6.0 created conflicting incentives near ground.
    3. w_v: 0.5 → 0.3 — revert to v6's velocity damping.
       v7's 0.5 was too punitive near ground, discouraging final approach.
    4. w_ang: 0.08 → 0.05 — revert to v6's orientation weight.

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
    # Weight 2.0 (v6 proven): provides consistent approach gradient
    # without conflicting with contact incentive near ground.
    w_pot = 2.0
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = w_pot * (dist_cur - dist_nxt)

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Dampens high-speed approaches near the ground. Gate keeps far-field
    # maneuvering unpenalized. Weight 0.3 (v6 proven): enough to deter
    # high-speed crashes without discouraging final approach.
    w_v = 0.3
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep lander upright.
    # Weights unchanged from v6 — stable across all iterations.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: per-step reward for safe pad contact ---
    # Per-step design (v6/v2 proven): rewards every step with both legs
    # on pad near target. Weight 65: modest reduction from v6's 80 to
    # reduce late-training over-stay incentive while staying safely above
    # the ~40-50 threshold where landing behavior collapses.
    w_contact = 65.0
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