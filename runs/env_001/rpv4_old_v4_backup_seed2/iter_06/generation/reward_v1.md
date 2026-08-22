# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lunar lander-like navigation task.
    Components:
      A. distance_progress   – reduction in distance to landing pad
      B. stability           – penalise high velocity and large attitude when close to pad
      C. landing_completion  – soft success bonus for two-leg contact with low speed
      D. fuel_penalty        – tiny penalty for using any engine
    """
    # --- unpack observations ---
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx_pos = next_obs[0]
    ny_pos = next_obs[1]

    # --- distances ---
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    velocity_norm = (x_vel**2 + y_vel**2) ** 0.5

    # =====================================================================
    # A. distance_progress: reward reduction in Euclidean distance to target
    # =====================================================================
    progress = (current_distance - next_distance) * 5.0   # scaling factor

    # =====================================================================
    # B. stability: smooth penalty for speed and attitude when near the pad
    #    gate: exp(-2 * distance)  →  close to 1 when distance small
    # =====================================================================
    gate = 2.718281828 ** (-2.0 * current_distance)     # approximate exp(-2d)
    w_vel = 0.1
    w_angle = 0.5
    w_angv = 0.5
    stability_penalty = - (w_vel  * velocity_norm**2 +
                           w_angle * angle**2 +
                           w_angv  * ang_vel**2) * gate

    # =====================================================================
    # C. landing_completion: soft success proxy – both legs on ground,
    #    near centre, with low velocity and stable attitude
    # =====================================================================
    both_feet = left_contact * right_contact   # 1.0 only if both are 1.0
    close_to_centre = current_distance < 0.3
    if both_feet > 0.5 and close_to_centre:
        speed_quality = max(0.0, 1.0 - velocity_norm / 0.5)
        angle_quality = max(0.0, 1.0 - abs(angle) / 0.5)
        angvel_quality = max(0.0, 1.0 - abs(ang_vel) / 0.5)
        landing_reward = 2.0 * speed_quality * angle_quality * angvel_quality
    else:
        landing_reward = 0.0

    # =====================================================================
    # D. fuel_penalty: minuscule cost for any engine use
    # =====================================================================
    fuel_penalty = -0.01 if action != 0 else 0.0

    total_reward = progress + stability_penalty + landing_reward + fuel_penalty

    components = {
        "distance_progress": progress,
        "stability": stability_penalty,
        "landing_completion": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## task_family / dynamics_subtype
- **task_family**: navigation_goal_reaching  
- **dynamics_subtype**: goal_approach_and_soft_contact (带双脚接触指示器的 2D 着陆器，离散推力控制)

## selected reward roles
1. **approach_to_target** (mandatory) → `distance_progress`  
2. **soft_landing_velocity_control + orientation_stabilization** (mandatory) → `stability`  
3. **stable_contact_termination_bonus** (mandatory) → `landing_completion`  
4. **fuel_efficiency_penalty** (mandatory) → `fuel_penalty`

## role‑to‑signal mapping & formula operators

| role | signals used | formula operator(s) |
|------|--------------|---------------------|
| approach_to_target | `x_position`, `y_position` (→ Euclidean distance) | `improvement_delta` (`current_dist - next_dist`) with scalar weight (5.0) |
| soft_landing_velocity_control | `x_velocity`, `y_velocity` (→ velocity norm) | `quadratic_penalty` multiplied by a `distance_gate` (via exponential approximation) |
| orientation_stabilization | `body_angle`, `angular_velocity` | `quadratic_penalty`, same distance-gate as above |
| stable_contact_termination_bonus | `left_support_contact`, `right_support_contact`, velocity norm, `body_angle`, `angular_velocity` | `joint_condition_proxy` (product of three bounded factors: speed_quality × angle_quality × angvel_quality, triggered only when both feet contact AND distance < 0.3) |
| fuel_efficiency_penalty | `action` (discrete) | `indicator_nonzero` (small constant penalty) |

## why these roles are included
- **distance_progress** is the main learning signal: it rewards every step that reduces the distance to the landing pad. It uses the delta form (`current_dist - next_dist`) to prevent the agent from “parking” at a good but suboptimal distance.  
- **stability** softly discourages high velocity and large tilting when the lander is near the pad. The gate `exp(-2*distance)` makes the penalty negligible far away but active during the final approach, preventing high‑speed crashes.  
- **landing_completion** provides a sparse but unambiguous success‑like bonus. It requires both leg‑contact indicators, proximity to centre, and nearly zero velocity/attitude. Its structure (product of bounded quality factors) gives continuous gradient *only when conditions are met*, encouraging the agent to simultaneously satisfy all landing criteria.  
- **fuel_penalty** is a minimal cost per engine firing, honouring the secondary objective of fuel efficiency without suppressing necessary actions.

## excluded roles and reasons
- **terminal_success_reward / terminal_failure_penalty**: Not used because `explicit_success_flag_available=false` and `explicit_failure_flag_available=false`; the `info` dict is empty. Hence we rely on a soft completion proxy instead of a hard terminal bonus.  
- **speed_modulation_in_freefall** (conditional): Not added because the distance‑progress signal naturally guides descent, and the stability penalty already tames lateral velocity when approaching the pad.  
- **survival_time_reward, exploration_bonus, multi_objective_energy_distance_balance**: Marked as *avoid* in the environment card due to missing signals or irrelevance. Not included.

## responsibilities deferred to later iterations
- **Stronger fuel efficiency optimisation** (e.g., higher penalty coefficient or dynamic scaling) – the current penalty is intentionally tiny to avoid hindering initial stability learning.  
- **Complex gate‑based dynamic curriculum** – the present gate is a simple exponential; more sophisticated schedules could be introduced once baseline landing is achieved.  
- **Action smoothness penalty** – no previous action is available, so it cannot be implemented now.

## expected failure modes to monitor after training
- **Flying off the viewport** → check if `x_position` diverges; may need a bounding penalty or a dedicated horizon loss signal.  
- **High‑speed crash** → despite the stability penalty and progress delta, the agent might accelerate to exploit the direction change reward; verify that the `landing_completion` bonus is sufficiently attractive to promote deceleration.  
- **Standing still or drifting on the pad without settling** → contact proxies could be partially met without full stop; the product‑based `landing_completion` requires near‑zero velocity, but if thresholds are too lax the agent may slide. Adjust `vel_max`, `angle_max` and `ang_max` thresholds if needed.  
- **Engine overuse** → if the fuel penalty is too small, the agent might fire engines constantly; it can be increased in later iterations once the core landing skill is stable.