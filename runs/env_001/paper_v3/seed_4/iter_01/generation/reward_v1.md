# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from the next observation
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Proximity to target (assumed at origin)
    dist = (x**2 + y**2) ** 0.5
    proximity = 1.0 / (1.0 + dist)          # bounded_signal [0,1]
    w_proximity = 1.0
    comp_prox = w_proximity * proximity

    # Soft landing bonus: encourages two‑leg contact with low speed and small angle
    contact_avg = (left_contact + right_contact) * 0.5
    speed_norm = (vx**2 + vy**2) ** 0.5
    factor_vel = 1.0 / (1.0 + speed_norm)
    factor_angle = 1.0 / (1.0 + abs(angle) + abs(angular_vel))
    landing_bonus = contact_avg * factor_vel * factor_angle
    w_bonus = 1.0
    comp_bonus = w_bonus * landing_bonus

    # Quadratic penalties for high velocity and attitude deviations
    w_vel_pen = 0.01
    vel_pen = -w_vel_pen * (vx**2 + vy**2)

    w_att_pen = 0.01
    att_pen = -w_att_pen * (angle**2 + angular_vel**2)

    total = comp_prox + comp_bonus + vel_pen + att_pen

    components = {
        'proximity': comp_prox,
        'soft_landing_bonus': comp_bonus,
        'velocity_penalty': vel_pen,
        'attitude_penalty': att_pen,
    }
    return float(total), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**: `navigation_goal_reaching` / `goal_approach_and_soft_contact`
- **selected reward roles**:  
  - `goal_reaching_and_safe_contact` (mandatory)  
  - `soft_landing_quality` (mandatory)
- **role‑to‑signal mapping**:  
  - `goal_reaching_and_safe_contact` uses positions (`x`, `y`), contact flags, velocities, and angles to shape approach and successful touch‑down.  
  - `soft_landing_quality` uses velocities, body angle, angular velocity, and contacts to penalise hard landings and encourage simultaneous two‑leg contact.
- **formula operators used**:  
  - Proximity: `bounded_signal` ( \(1/(1+|\text{dist}|)\) ) gives dense, saturating progress toward the target.  
  - Soft landing bonus: `joint_condition_proxy` multiplies contact average, velocity factor, and angle factor to reward good landing conditions.  
  - Velocity/attitude penalties: `quadratic_penalty` with very small weights to softly suppress excessive speed and orientation errors.
- **excluded roles**:  
  - `terminal_success_reward` / `terminal_failure_penalty`: environment has no explicit success or failure flags (`info` is empty).  
  - `action/efficiency cost`: deferred to later iterations (v1 focuses on task acquisition).  
  - `curriculum_weighting`: not needed; v1 uses fixed coefficients.
- **why no terminal rewards**: the environment provides no `terminated` signal to the reward function, so any attempt to detect episode end would be a proxy and is omitted in v1.
- **deferred to future iterations**: fuel/engine efficiency, more sophisticated contact‑gate logic, and possible dynamic weight tuning.
- **expected post‑training failure modes**:  
  - Agent may approach rapidly but fail to decelerate sufficiently, leading to hard bounces or crashing.  
  - Oscillatory behaviour around the target if the penalty weights are not correctly tuned.  
  - Learning to “hover” just above the pad without full contact if the bonus is insufficient to overcome the penalty of touching down.