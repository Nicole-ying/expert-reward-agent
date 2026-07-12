# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals
    horizontal_velocity = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]

    # ----- 1. Forward progress reward -----
    # Encourage positive horizontal velocity only (discourage backward motion implicitly via other penalties)
    forward_speed_reward = 2.0 * max(0.0, horizontal_velocity)

    # ----- 2. Upright stability penalty -----
    # Penalize deviation from upright orientation and rapid angular motion
    angle_penalty = 1.0 * (hull_angle ** 2)
    angvel_penalty = 0.1 * (hull_angular_velocity ** 2)
    upright_penalty = -(angle_penalty + angvel_penalty)

    # ----- 3. Energy efficiency cost -----
    # Light penalty on joint torque magnitudes to discourage wasteful actions
    torque_sq_sum = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    action_cost = -0.01 * torque_sq_sum

    total_reward = forward_speed_reward + upright_penalty + action_cost
    components = {
        "forward_speed_reward": forward_speed_reward,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**: locomotion_continuous_control / planar_bipedal_gait  
- **selected reward roles**:  
  1. `forward_progress_reward` (mandatory)  
  2. `upright_stability_penalty` (mandatory, treated as a stability/safety constraint)  
  3. `energy_efficiency_penalty` (conditional, kept intentionally small in v1)  
- **role_to_signal_mapping**:  
  - `forward_progress_reward` uses `horizontal_velocity` (obs[2]), clipped to non‑negative values.  
  - `upright_stability_penalty` uses `hull_angle` (obs[0]) and `hull_angular_velocity` (obs[1]), both with quadratic penalties.  
  - `energy_efficiency_penalty` uses the 4‑dimensional `action` vector, summing squared torques.  
- **formula operators selected**:  
  - `forward_progress_reward`: `dense_state_signal` (linear positive with zero‑clipping).  
  - `upright_stability_penalty`: `quadratic_penalty` (posture error squared).  
  - `energy_efficiency_penalty`: `quadratic_penalty` (action magnitude squared).  
- **excluded roles and reasons**:  
  - `terminal_success_bonus` – no explicit success flag available (info is empty, `explicit_success_flag_available=false`).  
  - `terminal_failure_penalty` – no explicit failure flag and no `termination_reason` in info.  
  - `contact_alternation_incentive` – excluded from v1 because mechanical alternating rewards can conflict with natural dynamics and cause instability; it is reserved for later iterations if abnormal gait patterns appear.  
  - `terrain_aware_preplanning_reward` – LIDAR usage is indirect and risks destabilising early training.  
- **why no terminal_success_reward / terminal_failure_penalty**: the environment card states `explicit_success_flag_available: false` and `explicit_failure_flag_available: false`, and `info` is an empty dictionary. No signal exists to reliably detect success or failure inside the reward function.  
- **roles left for future iterations**:  
  - A `soft_health_gate` that scales main progress reward based on a continuous posture health metric (if falling remains the dominant failure mode and the agent “rushes then dies”).  
  - `contact_alternation_incentive` or other gait‑shaping components (if stepping patterns become abnormal).  
  - Explicit `vertical_velocity` penalty (obs[3]) if bouncing/hopping appears.  
  - Curriculum‑weighting of the energy term (increase over training progress) once forward locomotion is stable.  
- **failure modes to observe after initial training**:  
  1. Agent jitters in place without net forward displacement → check if `forward_speed_reward` weight is too low or if the posture penalty is too restrictive.  
  2. Frequent early falls (short episodes) → evaluate whether the posture penalty (angle + angular velocity) is sufficient or needs a hinge‑based gate.  
  3. Excessive energy consumption with little speed → the action cost may be too small; could be increased gradually, but only after stable locomotion is achieved.  
  4. Stiff limbs or one‑leg dragging → later introduce a contact‑alternation incentive with a very low weight.
