def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v5: per-step contact bonus (v2 skeleton) with reduced weight + stronger velocity damping

    Key diagnosis from v4 failure:
    - v4's one-time transition bonus (+200) created a flat reward landscape:
      contact_bonus stayed at ~142 across ALL checkpoints with zero growth.
      Agent learned "dive fast, grab bonus, die" (len=68.55, all early terminate).
    - v2's per-step bonus grew from 125→2820 during training — reward scaled
      naturally with landing quality, creating a curriculum from touch to settle.

    Changes from v4:
    1. contact_bonus: revert from one-time transition (+200) + per-step settling (+2)
       back to pure per-step bonus (v2 approach). Weight reduced 100→25 to address
       the 99.6% magnitude dominance that plagued v2.
    2. velocity_penalty: strengthened 0.15→0.4 (v4 had 0.15) to provide meaningful
       gradient against high-speed approaches, now that contact_bonus is smaller.

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity: direct distance penalty (v2/v4 approach, proven stable) ---
    # Consistent gradient toward (0,0) at every step.
    w_dist = 0.05
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = -w_dist * dist_nxt

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Strengthened (0.15→0.4) to deter high-speed crashes now that contact_bonus
    # is moderate. Gate ensures far-field maneuvering isn't overly penalized.
    w_v = 0.4
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Keep lander upright, suppress wobble. Unchanged from v2/v4.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: per-step reward for safe pad contact ---
    # Reverted from one-time transition (v4) back to per-step (v2 skeleton).
    # Weight 25 (vs v2's 100) keeps contact as the anchor signal while giving
    # velocity/orientation penalties room to shape behavior.
    # Per-step design means longer settled stays → higher reward, creating
    # natural curriculum from first touch to stable landing.
    w_contact = 25.0
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