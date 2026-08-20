# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D lander reaching a central platform.
    Components:
    - proximity_penalty: quadratic distance to target center (0,0)
    - velocity_penalty: speed penalty gated by proximity to target
    - orientation_penalty: penalty for tilt and angular velocity
    - contact_bonus: small reward for simultaneous two‐leg contact, scaled by proximity
    """
    # Current position (relative to target)
    x_pos, y_pos = obs[0], obs[1]
    # Next state
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Proximity to target – encourage getting near (0,0)
    dist_sq = nx_pos ** 2 + ny_pos ** 2
    prox_weight = 2.0
    proximity_reward = -prox_weight * dist_sq

    # 2. Velocity penalty – active only when close to target
    # Proximity factor: near 1 when close, near 0 when far
    dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5
    proximity_factor = 1.0 / (1.0 + 10.0 * dist)   # smooth gating
    vel_weight = 1.0
    velocity_penalty = -vel_weight * proximity_factor * (nx_vel ** 2 + ny_vel ** 2)

    # 3. Orientation stability – keep body upright and avoid spinning
    orient_weight = 0.5
    orientation_penalty = -orient_weight * (n_angle ** 2 + 0.2 * n_angvel ** 2)

    # 4. Dual leg contact – small bonus for proper landing stance, gated by proximity
    contact_weight = 0.5
    contact_product = left_contact * right_contact   # 1.0 only if both legs touch
    contact_bonus = contact_weight * contact_product * proximity_factor

    total_reward = proximity_reward + velocity_penalty + orientation_penalty + contact_bonus

    components = {
        'proximity': float(proximity_reward),
        'velocity_penalty': float(velocity_penalty),
        'orientation_penalty': float(orientation_penalty),
        'contact_bonus': float(contact_bonus)
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**  
  `navigation_goal_reaching` with `goal_approach_and_soft_contact` dynamics, discrete control.

- **selected reward roles**  
  Mandatory roles from `reward_role_decomposition`:
  1. `target_proximity` – guide the agent toward the landing platform.
  2. `soft_landing_velocity_control` – reduce speed when near the target.
  3. `orientation_stability` – keep the body horizontal and rotational motion low.
  4. `dual_leg_contact` – encourage simultaneous two‑leg contact for a stable landing.

- **role_to_signal_mapping**  
  - `target_proximity`: uses `x_position` and `y_position` → quadratic penalty on squared distance to `(0,0)`.
  - `soft_landing_velocity_control`: uses `x_velocity`, `y_velocity` and the distance to the target (via next state) → speed penalty gated by a proximity factor (`1/(1+10*dist)`) that activates near the pad.
  - `orientation_stability`: uses `body_angle` and `angular_velocity` → quadratic penalties.
  - `dual_leg_contact`: uses `left_support_contact`, `right_support_contact` → multiplicative bonus, further scaled by the same proximity factor.

- **formula operators chosen from Expert Schema**  
  - `quadratic_penalty` for position error and orientation.
  - `gated_penalty` (soft gate via proximity factor) for velocity control – a form of `product_with_proximity`.
  - `conditional_reward` (multiplication) for dual‑leg contact, gated by proximity.

- **excluded roles and justification**  
  - `explicit_success_bonus` / `explicit_failure_penalty`: no success/failure flag exists; not allowed.
  - `fuel_consumption_penalty`: conditional role deferred to later iterations because the agent must first learn the task; early efficiency pressure can prevent exploration.
  - `time_pressure`: no step count signal available, and discounting naturally favors faster arrival without explicit penalty.

- **why no terminal_success_reward / terminal_failure_penalty**  
  `explicit_success_flag_available=false` and `explicit_failure_flag_available=false`; the environment provides no such labels. Using state‑based proxies would be fragile and is avoided in v1.

- **deferred to future iterations**  
  - Fuel/action cost optimization.  
  - More sophisticated contact gating (e.g., only reward when settled).  
  - Dynamic curriculum weighting or adaptive thresholds.  
  - Fine‑tuned orientation‑velocity coordination for extra smooth landings.

- **expected failure modes after initial training**  
  - **Hard landing / crash**: agent may still approach too fast; need to monitor high contact velocities and increase gated speed penalty weight.
  - **Hovering / refusal to descend**: excessive orientation penalty or velocity penalty with poor proximity gate could cause the agent to stall above the pad.
  - **Premature sleep**: landing far from the center if proximity reward is too weak; consider increasing `prox_weight`.
  - **Orientation instability**: if the agent oscillates, tune the angular velocity coefficient.
  - **Single‑leg landing followed by drift**: if `contact_bonus` dominates late behaviour, it might encourage slamming one leg down; the proximity gate mitigates this by only rewarding dual contact near the pad.
