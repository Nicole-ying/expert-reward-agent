# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Goal: Guide a 2D lander to softly touch down on the target pad with both feet,
    staying upright and near zero velocity.
    """
    # Unpack next_obs (post-action state)
    x = next_obs[0]          # horizontal offset from pad
    y = next_obs[1]          # vertical offset (positive = above pad)
    vx = next_obs[2]         # horizontal velocity
    vy = next_obs[3]         # vertical velocity
    angle = next_obs[4]      # body angle
    ang_vel = next_obs[5]    # angular velocity
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Component 1: Approach and soft landing ---
    # Distance to target (the pad is at (0,0))
    dist = (x**2 + y**2) ** 0.5

    # Positive reward that increases as distance decreases (bounded)
    approach_reward = 1.0 / (1.0 + dist)

    # Velocity penalty gated by proximity: heavy only when close
    gate_vel = 1.0 / (1.0 + dist)
    vel_penalty = (vx**2 + vy**2) * gate_vel

    w_approach = 1.0
    w_vel      = 0.1
    comp_approach_landing = w_approach * approach_reward - w_vel * vel_penalty

    # --- Component 2: Upright stabilization ---
    angle_penalty = angle**2
    angvel_penalty = ang_vel**2
    w_angle  = 0.1
    w_angvel = 0.1
    comp_stabilization = - w_angle * angle_penalty - w_angvel * angvel_penalty

    # --- Component 3: Successful contact reward ---
    # Give a clear signal when both landing legs touch simultaneously
    w_contact = 3.0
    comp_contact = w_contact * (left_contact * right_contact)

    total_reward = comp_approach_landing + comp_stabilization + comp_contact
    components = {
        "approach_and_soft_landing": comp_approach_landing,
        "upright_stabilization": comp_stabilization,
        "contact_reward": comp_contact
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **Task family & dynamics subtype**  
  `navigation_goal_reaching` with `goal_approach_and_soft_contact` – a 2D lander must reach the target pad, decelerate, stay upright, and touch down with both legs.

- **Selected reward roles (mandatory)**  
  1. **Approach_target + soft_landing_velocity** – merged into a single `approach_and_soft_landing` component.  
     - *Signals*: `obs[0]` (x), `obs[1]` (y), `obs[2]` (vx), `obs[3]` (vy).  
     - *Formula*:  
       - Bounded distance reward: `1/(1+dist)` – always positive, maximal at touchdown.  
       - Velocity penalty gated by distance: `- w_vel * (vx²+vy²) * 1/(1+dist)` – forces deceleration only near the pad.  
  2. **Upright_stabilization** – keeps the lander level.  
     - *Signals*: `obs[4]` (angle), `obs[5]` (angular velocity).  
     - *Formula*: quadratic penalty on both `angle²` and `ang_vel²`.  
  3. **Successful_contact_reward** – the only hard evidence of a good landing.  
     - *Signals*: `obs[6]` (left contact), `obs[7]` (right contact).  
     - *Formula*: logical product `left_contact * right_contact` multiplied by a weight, providing a sparse but strong positive signal when both legs touch.

- **Role‑to‑signal‑mapping adherence**  
  Exactly follows the environment card’s mapping: distance/velocity from `obs[0..3]`, orientation from `obs[4..5]`, contact from `obs[6..7]`. No missing signals.

- **Excluded roles and reasons**  
  - `fuel_efficiency`, `time_penalty`: conditional roles intended for later iterations, not needed for basic task learning.  
  - `orientation_engine_usage_penalty_alone`: no way to distinguish useful from wasteful attitude engine firing; would harm learning.  
  - `exact_contact_sequence_bonus`: no sequential contact data available.  
  - `success_flag_based_reward`: environment card confirms no explicit success/failure flags, `info` is empty.

- **Why no terminal success/failure reward**  
  Explicit flags are unavailable (`explicit_success_flag_available = false`, `info = {}`). The design instead relies on continuous distance‑reward, soft velocity gating, and the bilateral contact proxy.

- **Future iterations**  
  - Add a small time‑penalty (per‑step small negative reward) or fuel‑efficiency term once the agent reliably reaches the pad.  
  - Introduce a more sophisticated `joint_condition_proxy` that multiplies proximity, low velocity, and stable attitude, making the terminal reward even more aligned with soft‑landing criteria.

- **Failure modes to monitor after initial training**  
  1. *Hovering above the pad without descending* – check if the approach reward alone is sufficient; may need a vertical‑progress bonus.  
  2. *High‑speed crash* – examine if the velocity gate weight is too small.  
  3. *Tilted landing, only one foot in contact* – increase `w_angle` or add a height‑dependent angle penalty.  
  4. *Persistent side‑to‑side oscillation* – adjust angular velocity penalty.  
  5. *Fuel hoarding, no main‑engine use* – the design deliberately avoids fuel penalties, so this should be rare; if observed, add a mild action‑penalty later.
