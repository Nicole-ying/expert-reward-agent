# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation (using next_obs, which represents the state after action)
    x       = next_obs[0]   # horizontal offset from target pad center
    y       = next_obs[1]   # vertical offset from target pad (height)
    vx      = next_obs[2]   # horizontal velocity
    vy      = next_obs[3]   # vertical velocity
    angle   = next_obs[4]   # body tilt angle (rad)
    angvel  = next_obs[5]   # angular velocity
    left_c  = next_obs[6]   # left leg contact flag (0 or 1)
    right_c = next_obs[7]   # right leg contact flag (0 or 1)

    # ------------------------------------------------------------------
    # Component A – goal proximity + soft landing  (main learning signal)
    # ------------------------------------------------------------------
    # Encourage small positional errors, low velocities, upright attitude.
    w_prox  = 2.0
    w_vel   = 1.0
    w_ang   = 2.0

    comp_prox  = w_prox * ( -abs(x) - abs(y) )
    comp_vel   = w_vel  * ( -vx**2 - vy**2 )
    comp_angle = w_ang  * ( -angle**2 - angvel**2 )

    # ------------------------------------------------------------------
    # Component B – contact quality (stability constraint)
    # ------------------------------------------------------------------
    # Reward simultaneous leg contact, but only when the body is already
    # near the pad and moving slowly.  This discourages single‑leg touch.
    w_contact = 10.0

    pos_thresh = 0.3            # meters
    vel_thresh = 0.3            # m/s

    gate_x   = max(0.0, 1.0 - abs(x) / (pos_thresh + 1e-8))
    gate_y   = max(0.0, 1.0 - abs(y) / (pos_thresh + 1e-8))
    gate_vx  = max(0.0, 1.0 - abs(vx) / (vel_thresh + 1e-8))
    gate_vy  = max(0.0, 1.0 - abs(vy) / (vel_thresh + 1e-8))

    # Both legs must touch (binary product)
    leg_contact = left_c * right_c
    comp_contact = w_contact * leg_contact * gate_x * gate_y * gate_vx * gate_vy

    # ------------------------------------------------------------------
    # Component C – success proxy (task‑completion soft signal)
    # ------------------------------------------------------------------
    # Geometric mean of near‑perfect conditions.  Only active when all
    # signals are close to ideal, providing a final shaping pull.
    w_proxy = 20.0

    eps_x     = 0.1
    eps_y     = 0.1
    eps_vx    = 0.15
    eps_vy    = 0.15
    eps_angle = 0.15   # radians

    score_x     = max(0.0, 1.0 - abs(x) / eps_x)
    score_y     = max(0.0, 1.0 - abs(y) / eps_y)
    score_vx    = max(0.0, 1.0 - abs(vx) / eps_vx)
    score_vy    = max(0.0, 1.0 - abs(vy) / eps_vy)
    score_angle = max(0.0, 1.0 - abs(angle) / eps_angle)
    score_contact = (left_c + right_c) * 0.5   # [0,1]

    # Use geometric mean to avoid product collapse.
    # 1e-12 avoids zero division.
    prod = (score_x * score_y * score_vx * score_vy * score_angle * score_contact)
    if prod > 1e-12:
        comp_proxy = w_proxy * (prod ** (1.0 / 6.0))
    else:
        comp_proxy = 0.0

    # ------------------------------------------------------------------
    # Component D – thrust penalty (efficiency cost, secondary)
    # ------------------------------------------------------------------
    w_thrust = 0.05
    comp_thrust = -w_thrust * float(action != 0)   # any engine use

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = (comp_prox + comp_vel + comp_angle +
                    comp_contact + comp_proxy + comp_thrust)

    components = {
        "goal_proximity": comp_prox,
        "velocity_penalty": comp_vel,
        "angle_penalty": comp_angle,
        "contact_quality": comp_contact,
        "success_proxy": comp_proxy,
        "thrust_penalty": comp_thrust,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **task_family:** `navigation_goal_reaching`  
- **dynamics_subtype:** `goal_approach_and_soft_contact` (discrete control, two‑leg lander, main + orientation thrusters)

## selected reward roles (from environment_card)
| role | selected | mapping to component(s) |
|---|---|---|
| `goal_proximity` | ✓ | `goal_proximity` (direct distance penalty) |
| `soft_landing` | ✓ | merged into `velocity_penalty` + `angle_penalty` |
| `contact_quality` | ✓ | `contact_quality` (conditional bonus) |
| `success_shaping` | ✓ | `success_proxy` (geometric mean of near‑success conditions) |
| `thrust_penalty` | ✓ | `thrust_penalty` (small fixed action penalty) |
| `angular_damping` | ✓ (covered) | absorbed in `angle_penalty` via angular velocity term |

## role‑to‑signal mapping used
- **proximity:** `next_obs[0]` (x), `next_obs[1]` (y) → \( - |x| - |y| \)  
- **velocity penalty:** `next_obs[2]` (vx), `next_obs[3]` (vy) → \( -vx^2 - vy^2 \)  
- **angle penalty:** `next_obs[4]` (angle), `next_obs[5]` (angvel) → \( -angle^2 - angvel^2 \)  
- **contact quality:** `next_obs[6]`, `next_obs[7]` → `left_contact * right_contact` gated by position & velocity thresholds  
- **success proxy:** all above signals + contact score → geometric‑mean composite  
- **thrust penalty:** `action != 0`

## formula operators chosen for each role
- **goal_proximity:** `dense_state_signal` (negative penalty on absolute position errors)  
- **soft_landing:** `quadratic_penalty` on velocities and angle (continuous, smaller‑is‑better)  
- **contact_quality:** `conditional_bonus` – binary leg‑contact multiplied by linear decay gates for position and velocity (resembles `soft_health_gate` logic)  
- **success_shaping:** `joint_condition_proxy` using geometric mean of six bounded scores – a soft, smooth success signal  
- **thrust_penalty:** `constant_penalty` applied to any non‑idle action (discrete equivalent of action efficiency)

## excluded roles and reasoning
- `explicit_success_reward` – **excluded** because `explicit_success_flag_available = false` (info dict is empty)  
- `explicit_failure_penalty` – **excluded** for the same reason; no termination‑reason signal in `info`  
- `progress_time_penalty` – **excluded** because no step count or max‑steps are provided  
- `fuel_exact_penalty` – **excluded** because fuel quantity is not an observed signal (only action index is available); a light action penalty is used instead  

## why terminal success / failure are not used
The environment does **not** expose a success flag (`info["success"]` does not exist) and does not report failure reasons. Writing a hard terminal reward would require guessing the termination cause from observations, which is fragile and prone to false positives/negatives in this version. Instead, `contact_quality` and `success_proxy` provide dense, learnable signals that shape the agent toward successful soft landing.

## what is left for future iterations
- **Fuel‑efficiency optimisation:** The current `thrust_penalty` is minimal; a more sophisticated fuel‑based cost (e.g., impulse‑weighted penalty) can be added once fuel observations become available or after a working landing policy emerges.  
- **Dynamic curriculum:** If velocity/angle constraints hinder early exploration, a schedule that gradually increases their weights or tightens thresholds could be introduced.  
- **Gate scheduling:** Soft gates on contact and success could be made tighter over training progress.  
- **Boundary‑exit prevention:** If drifting out of the horizontal viewport becomes a common failure, an additional hinge penalty on extreme `|x|` can be added.

## failure modes to watch after initial training
- **Hovering / refusal to descend:** `goal_proximity` too weak compared to `velocity_penalty` → agent stays high; increase `w_prox` or decrease `w_vel`.  
- **High‑speed impact:** `velocity_penalty` too light → contact occurs with large `vy`; raise `w_vel` or tighten gates.  
- **Excessive attitude oscillations:** angle/angvel penalty insufficient → large `w_ang` may be needed.  
- **Single‑leg touchdown & tip‑over:** `contact_quality` not strong enough; increase `w_contact` or relax gates slightly to make the reward more attainable.  
- **False success outside pad:** Agent settles with low velocity/angle but far from centre; success proxy may reinforce it. Tighten `eps_x`, `eps_y` or increase `goal_proximity` to prevent resting off‑target.