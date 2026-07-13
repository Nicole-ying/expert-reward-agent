# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    x = obs[0]            # horizontal position relative to target pad
    y = obs[1]            # vertical position relative to target pad
    vx = obs[2]           # horizontal velocity
    vy = obs[3]           # vertical velocity
    angle = obs[4]        # body angle
    ang_vel = obs[5]      # angular velocity
    left_contact = obs[6] # left support leg contact (0 or 1)
    right_contact = obs[7]# right support leg contact (0 or 1)

    # Hyperparameters (tunable in later iterations)
    w_goal = 1.0
    alpha_proximity = 5.0          # controls the activation radius for proximity-based terms
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_contact = 2.0

    # Distance to target center (squared)
    dist_sq = x**2 + y**2

    # Soft proximity weight: ~1 when close to target, ~0 when far
    proximity = 1.0 / (1.0 + alpha_proximity * dist_sq)

    # 1. Main progress: drive toward target center (dense quadratic penalty on distance)
    goal_proximity = -w_goal * dist_sq

    # 2. Soft landing velocity penalty: active only near the target
    velocity_penalty = -w_vel * (vx**2 + vy**2) * proximity

    # 3. Orientation stability penalty: penalize tilt and spin everywhere (light weight)
    orientation_penalty = -w_angle * (angle**2) - w_angvel * (ang_vel**2)

    # 4. Contact reward: encourage both legs touching, gated by proximity
    both_legs_contact = left_contact * right_contact  # 1 only if both are 1
    contact_reward = w_contact * both_legs_contact * proximity

    # Total reward
    total_reward = goal_proximity + velocity_penalty + orientation_penalty + contact_reward

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**:  
  `navigation_goal_reaching` / `goal_approach_and_soft_contact` — a 2D lander must reach and settle on the central pad with both support legs in contact and a stable upright orientation.

- **selected reward roles** (mandatory from environment_card):  
  1. **goal_proximity_reward** — core driving signal: quadratic penalty on distance to the target center (`-w_goal * dist²`).  
  2. **soft_landing_velocity_penalty** — suppresses high horizontal/vertical velocity only near the pad, using a continuous **proximity gate** (`proximity = 1/(1+α·dist²)`) multiplied by squared velocity penalty.  
  3. **orientation_stability_reward** — light quadratic penalty on body angle and angular velocity, applied step‑by‑step to discourage flips and wobbling.  
  4. **contact_reward** — rewards simultaneous leg contact (`left_contact * right_contact`) weighted by the same proximity gate, so that contact far from the target (e.g., on boundaries) is not rewarded.

- **role‑to‑signal mapping**: each role directly uses the signals declared in `role_to_signal_mapping` (positions, velocities, angle, angvel, binary contact flags).

- **formula operators selected**:  
  - `goal_proximity`: `dense_state_signal` (quadratic penalty on state).  
  - `velocity_penalty`: `quadratic_penalty` combined with `masked_by_proximity` (product with soft proximity weight).  
  - `orientation_penalty`: `quadratic_penalty` (on angle and angular velocity).  
  - `contact_reward`: `boolean_and` (product of binary contacts) multiplied by proximity soft gate.

- **excluded roles and reasons**:  
  - `settlement_bonus`: excluded for v1 because it requires tight multi‑condition thresholds and would be sparse; learning can progress without a hard “success” proxy.  
  - `fuel_efficiency_penalty`: deferred to later iterations — early training must focus on reaching and landing, and an action‑cost penalty could freeze exploration.  
  - `explicit_crash_penalty` and `time_optimality_penalty`: excluded as per environment_card (no usable crash/time signals, forbidden info fields).

- **no terminal success/failure reward**: `explicit_success_flag_available=false` and `explicit_failure_flag_available=false`, so no `terminal_success_reward` or `terminal_failure_penalty` can be constructed reliably. The `contact_reward` with proximity gating acts as a **soft proxy** for successful settlement.

- **deferred roles**: fuel efficiency and a hard settlement bonus will be considered in later reward iterations, once the agent demonstrates consistent stable landings.

- **expected failure modes after early training**:  
  - hard crashes on the pad (high velocity) → may require stronger velocity penalty or a slightly earlier activation radius.  
  - hovering without descending (fear of penalty) → could be due to over‑penalising fuel or speed; our design avoids fuel penalty, and speed penalty is only active near the pad, so this is less likely.  
  - landing on one leg or tilting → the orientation and contact rewards should mitigate this, but weights may need adjustment.  
  - drifting out of viewport → the strong distance‑based goal penalty encourages staying near centre.