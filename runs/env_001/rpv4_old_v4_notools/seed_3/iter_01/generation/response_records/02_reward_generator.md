# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward: goal proximity + velocity damping + orientation penalty + safe contact bonus
    
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

    # --- 1. goal_proximity (improvement_delta) ---
    # encourage moving toward (0,0) by reducing squared distance
    # larger weight on vertical axis to avoid hard landings
    w_x, w_y = 1.0, 2.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2   # positive when approaching horizontally
    delta_y_sq = y_cur ** 2 - y_nxt ** 2   # positive when approaching vertically
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (quadratic penalty, gated by distance) ---
    # penalise high speed mainly when close to the pad
    w_v = 0.05
    k_gate = 1.0
    dist_nxt = abs(x_nxt) + abs(y_nxt)   # simple Manhattan gate; could use Euclidean
    gate = 1.0 / (1.0 + k_gate * dist_nxt)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. safe_contact_bonus (joint condition proxy) ---
    # reward when both legs touch and position is near (0,0)
    w_contact = 10.0
    y_th = 0.1   # within 0.1 vertical distance
    x_th = 0.2   # within 0.2 horizontal offset
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt   # 0.0 or 1.0
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
    }

    return total_reward, components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- **agent**: 2D flyer with main engine and two orientation thrusters, two bottom contacts.

## selected reward roles
We follow the mandatory/conditional roles from `environment_card.md`:

| role | status | why selected |
|------|--------|--------------|
| `goal_proximity` | **included** (mandatory) | core driving signal: reduce distance to pad (x→0, y→0) |
| `velocity_damping` | **included** (mandatory) | prevent high-speed crashes on landing; gated by distance to not punish far-field cruising |
| `orientation_penalty` | **included** (mandatory) | encourage upright posture, suppress wobble |
| `safe_contact_bonus` | **included** (conditional) | reward correct two-leg landing near the centre; activated only when y & x are close and both contacts engaged |
| `fuel_efficiency_penalty` | **excluded for v1** | following v1 efficiency‑omission rule; to be added later if fuel waste is observed |

## role-to-signal mapping and chosen formula operators
- `goal_proximity` → signals: `x_position, y_position` from `obs` and `next_obs`  
  **operator**: `improvement_delta` on squared distance (`x_cur^2 - x_nxt^2`, `y_cur^2 - y_nxt^2`), with higher weight on vertical to discourage hard landings.
- `velocity_damping` → signals: `x_velocity, y_velocity` from `next_obs`  
  **operator**: `quadratic_penalty` on speed magnitude, modulated by a distance‑based gate (`1/(1 + k * dist)`). This makes the penalty weak far away and strong near the pad, avoiding suppression of necessary transit speed.
- `orientation_penalty` → signals: `body_angle, angular_velocity` from `next_obs`  
  **operator**: `dense_state_signal` penalty (`abs(angle)`, `angular_velocity^2`).
- `safe_contact_bonus` → signals: `y_position, x_position, left_support_contact, right_support_contact` from `next_obs`  
  **operator**: `joint_condition_proxy` using continuous bounded factors (`y_close`, `x_close`) multiplied by the binary product of leg contacts. This gives a smooth reward ramp when approaching the correct landing configuration, peaking when all conditions are met.

## excluded roles and justification
- `terminal_success_reward` – **not used**: no explicit success flag in `info` (`explicit_success_flag_available=false`).
- `terminal_failure_penalty` – **not used**: no explicit failure flag; crash/out‑of‑bound detection would require unreliable inference from termination, which is not available in step‑level `info`.
- `fuel_efficiency_penalty` – **deferred to later iteration**: v1 focuses on task acquisition; efficiency costs can be added once the agent reliably reaches the pad.
- `early_settle_penalty` / `final_bonus_only` – explicitly listed as **avoid roles** due to missing or ambiguous signals.

## design choices against failure modes
| failure mode | protective mechanism |
|--------------|----------------------|
| **high‑speed crash into ground** | `velocity_damping` activated strongly near the pad, combined with `goal_proximity` guiding descent instead of dive. |
| **horizontal fly‑out** | `goal_proximity` gives positive reward only for approaching x=0; moving away is penalised. |
| **"fake landing" off‑centre** | `safe_contact_bonus` requires both x and y to be near zero before rewarding contact. |
| **excessive wobble / topple** | `orientation_penalty` suppresses large angles and fast angular motion. |
| **free‑fall without engine use** | `goal_proximity` still rewards descent to y→0, but the agent must learn to fire main engine to decelerate; the velocity penalty encourages low speed near ground, motivating controlled landing. |

## what is left for future iterations
- Incorporating `fuel_efficiency_penalty` (small cost for non‑idle actions) once landing is robust.
- Possibly introducing a `soft_health_gate` that factors orientation or velocity risk into the main proximity reward if crashes persist.
- Tuning thresholds and weights using observed episode statistics (tying `y_th, x_th` to actual physics).

## expected post‑training observations
- **episode success rate** (both legs touching near centre with low speed).
- **crash rate** (high velocity at y≈0), **out‑of‑bound rate**.
- **average velocity penalty** near termination → indicates whether braking is learned.
- **orientation penalty** over time – too high may indicate excessive stiffness; if still high, reduce `w_ang` slightly.
- **contact bonus frequency** – if never triggered, relax `y_th`/`x_th` or increase `w_contact`.

This v1 design gives the agent dense, shaped feedback for approach and landing while avoiding hidden dependencies on unavailable terminal flags.
