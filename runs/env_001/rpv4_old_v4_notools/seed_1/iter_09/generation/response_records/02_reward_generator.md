# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward: progressive distance reduction + attitude stabilisation + height‑dependent landing speed penalty.
    No terminal events, no contact‑based bonuses, no explicit fuel penalty in this iteration.
    """
    # coefficients – tune cautiously in the outer loop
    w_progress = 1.0
    w_attitude = 0.05
    w_velocity = 0.3

    # ---------- goal progress (potential‑based / delta distance) ----------
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    goal_progress = w_progress * (dist_current - dist_next)

    # ---------- upright stabilisation ----------
    # penalty on body angle (next_obs[4]) and angular velocity (next_obs[5])
    attitude_penalty = -w_attitude * (next_obs[4] ** 2 + 0.1 * next_obs[5] ** 2)

    # ---------- soft velocity penalty (height‑gated) ----------
    # activate only when the vehicle is low (y close to 0) to encourage gentle touchdown
    y = next_obs[1]
    gate = max(0.0, 1.0 - y / 0.5)  # linear decay: gate=1 when y=0, gate=0 when y>=0.5
    speed_sq = next_obs[2] ** 2 + next_obs[3] ** 2
    soft_velocity_penalty = -w_velocity * speed_sq * gate

    total_reward = goal_progress + attitude_penalty + soft_velocity_penalty

    components = {
        "goal_progress": goal_progress,
        "attitude_penalty": attitude_penalty,
        "soft_velocity_penalty": soft_velocity_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`

## selected reward roles
1. **goal_proximity** (mandatory) – core learning signal, implemented as improvement delta:  
   `w * (distance_current - distance_next)`.  
   Provides dense, per‑step gradient directly aligned with the primary objective of reaching the target platform.
2. **upright_stabilization** (mandatory) – quadratic penalty on body angle and angular velocity, active throughout the episode. Keeps the lander’s attitude close to horizontal, preventing unrecoverable flips.
3. **soft_landing** (mandatory) – conditional velocity penalty gated by height (`y` close to 0).  
   Suppresses large horizontal / vertical speeds only when the vehicle is near the ground, avoiding premature suppression of descent during the approach phase.

## role_to_signal_mapping
| role | signals used | formula operator |
|---|---|---|
| `goal_proximity` | `obs[0,1]`, `next_obs[0,1]` | `improvement_delta` (distance reduction) |
| `upright_stabilization` | `next_obs[4,5]` | `quadratic_penalty` |
| `soft_landing` | `next_obs[1,2,3]` | `conditional_scale` (height‑gated quadratic penalty) |

## excluded roles and reasons
- **throttle_efficiency** – excluded from v1. The primary mission is to teach smooth, accurate landing; explicit fuel cost can be added in a later iteration once the basic task is solved.
- **terminal_settle_boost** – excluded due to high risk of misattribution. No explicit success flag exists, and a contact‑based bonus could be gamed (e.g. agent learning to stay on the pad with marginal velocity to repeatedly collect rewards without terminating correctly).
- **terminal_success_reward / terminal_failure_penalty** – not used because the environment provides **no explicit success/failure flags** and `info` is empty.

## why no terminal events
The environment card states `explicit_success_flag_available: false` and `explicit_failure_flag_available: false`. All attempts to design a reliable terminal bonus or crash penalty would rely on noisy, unverifiable approximations; v1 avoids them entirely.

## responsibilities deferred to later iterations
- Fuel efficiency (action cost)
- Time‑to‑land pressure
- Dynamic curriculum or soft‑health gates
- Contact‑quality bonuses (only after a more stable baseline is achieved without them)

## post‑training failure modes to watch
- **Hovering or slow descent**: episode runs long, `y` stays large. → May need a small altitude‑reduction incentive (future).
- **High‑speed impact**: `y_velocity` large at termination. → Increase `w_velocity` or refine the height gate (e.g. exponential decay).
- **Attitude oscillations**: `body_angle` or `angular_velocity` remains large. → Increase `w_attitude` or introduce an angular‑velocity damping term.
- **Horizontal drift outside viewport**: `x_position` diverges. → Strengthen the lateral component within `goal_progress` by scaling x‑distance separately (future refinement).
