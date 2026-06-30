# environment_contract

- env_id: Env_001
- function_signature: def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
- allowed observation signals:
  - obs[0], next_obs[0]: x_position relative to target
  - obs[1], next_obs[1]: y_position relative to target
  - obs[2], next_obs[2]: x_velocity
  - obs[3], next_obs[3]: y_velocity
  - obs[4], next_obs[4]: body_angle
  - obs[5], next_obs[5]: angular_velocity
  - obs[6], next_obs[6]: left_support_contact
  - obs[7], next_obs[7]: right_support_contact
- action: discrete engine command, usable only as current action
- info: no reliable fields available
- forbidden: original_reward, official_reward, fitness_score, individual_reward, info['success'], info['failure'], info['termination_reason']
- terminal_success_reward and terminal_failure_penalty remain blocked unless explicit signals are added later


# previous_reward.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # --- Component 1: Progress Delta Reward (main learning signal) ---
    # Reward moving closer to the target (0,0)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    # Increased coefficient from 10.0 to 15.0 to strengthen positive signal
    progress_reward = 15.0 * progress_delta
    
    # --- Component 2: Conditional Stability Penalty (reduced dominance) ---
    # Penalize high velocity, large angle, and high angular velocity
    # Only apply strongly when near target (dist < 1.0), otherwise reduce penalty
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.3 * abs(next_body_angle)  # reduced from 0.5
    angular_vel_penalty = 0.1 * abs(next_angular_vel)  # reduced from 0.2
    speed_penalty = 0.2 * speed  # reduced from 0.3
    
    # Apply distance-based gating: reduce penalty when far from target
    distance_factor = min(1.0, 1.0 / (0.5 * next_dist + 0.5))  # scales down when far
    stability_penalty = -distance_factor * (angle_penalty + angular_vel_penalty + speed_penalty)
    
    # --- Component 3: Soft Landing Proxy (increased bonus, continuous shaping) ---
    # Continuous bonus based on proximity, stability, and contact
    near_target_factor = max(0.0, 1.0 - next_dist / 0.5)  # 0 to 1 as dist goes 0.5 to 0
    stability_factor = max(0.0, 1.0 - abs(next_body_angle) / 0.3) * max(0.0, 1.0 - speed / 1.0)
    contact_factor = min(1.0, next_left_contact + next_right_contact)  # 0 to 2, capped at 1
    soft_landing_bonus = 1.0 * near_target_factor * stability_factor * contact_factor
    
    # --- Component 4: Small Distance Anchor (auxiliary) ---
    # Small negative reward proportional to distance to keep agent aware of goal
    distance_anchor = -0.1 * next_dist
    
    # --- Combine components ---
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + distance_anchor
    
    # --- Build components dict ---
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# best_reward.py (historical best, for reference)

This is the highest-scoring reward so far. Learn from it, do not make it worse.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # --- Component 1: Progress Delta Reward (main learning signal) ---
    # Reward moving closer to the target (0,0)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # --- Component 2: Stability Penalty (light constraint) ---
    # Penalize high velocity, large angle, and high angular velocity
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * abs(next_body_angle)
    angular_vel_penalty = 0.2 * abs(next_angular_vel)
    speed_penalty = 0.3 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # --- Component 3: Soft Landing Proxy (small bonus) ---
    # Bonus when near target, low speed, stable angle, and both supports contact
    near_target = next_dist < 0.3
    low_speed = speed < 0.5
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    soft_landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0
    
    # --- Component 4: Small Distance Anchor (auxiliary) ---
    # Small negative reward proportional to distance to keep agent aware of goal
    distance_anchor = -0.1 * next_dist
    
    # --- Combine components ---
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + distance_anchor
    
    # --- Build components dict ---
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# iteration_context.md

# Agent Context

- iteration: 3
- target_score: 200.000
- best_score: -110.396 (iter 1)
- current_score: -108.448
- trend: searching
- guidance: Continue refining based on evidence.
- suggested_action: tune or add

The analysis report and expert cards below provide more detailed diagnostic evidence.
Use them to decide your concrete action (tune/add/delete/mix/rebuild).

# Iteration Context for Reward Revision

## Agent Memory (history table)

| iter | score | best | skeleton_summary | trend |
|------|-------|------|------------------|-------|

## Diagnosis Guidance

### Analysis Summary
```json
{
  "failure_modes": [
    "stability_penalty_dominance",
    "early_failure_or_crash"
  ],
  "hacking_risks": [
    "stability_penalty_dominance"
  ],
  "component_analysis": {
    "distance_anchor": {
      "role": "anchor",
      "direction": "negative",
      "signal_strength": "moderate",
      "issue": "negative mean, but may be necessary for shaping"
    },
    "progress_reward": {
      "role": "progress",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "positive mean but low magnitude relative to penalties"
    },
    "soft_landing_bonus": {
      "role": "proxy",
      "direction": "positive",
      "signal_strength": "weak",
      "issue": "very low nonzero rate (0.9%), sparse signal"
    },
    "stability_penalty": {
      "role": "constraint",
      "direction": "negative",
      "signal_strength": "strong",
      "issue": "dominant negative component, mean -0.217, min -1.74"
    }
  },
  "skeleton_assessment": {
    "current_skeleton": [
      "distance_anchor",
      "progress_reward",
      "soft_landing_bonus",
      "stability_penalty"
    ],
    "iterations_on_this_skeleton": 1,
    "best_score_this_skeleton": -108.45,
    "stagnant": false,
    "skeleton_family": "anchor+progress+proxy+constraint"
  },
  "recommended_action": "tune",
  "reasoning": "Stability penalty dominates total reward (mean -0.217 vs progress 0.243), causing negative total. Soft landing bonus is too sparse (0.9% nonzero). External score is very low (-108). Recommend reducing stability penalty coefficient and increasing soft landing bonus frequency or magnitude.",
  "new_lessons": [
    "Stability penalty coefficient must be balanced to avoid dominating progress reward",
    "Soft landing bonus needs higher trigger rate or magnitude to provide useful signal"
  ]
}
```

### Expert Cards (compressed)
## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target
## early_failure_or_crash
- signal: negative external score and short episode length
- risk: reward does not guide stable control before termination
- fix: add smooth approach/landing signals; avoid relying on sparse terminal-like proxy

### KB Recommended Skeletons for task `navigation_goal_reaching`
- time_penalty, distance_reward, progress_delta_reward, potential_based_shaping, gated_reward
- Previously tried skeleton family: anchor+progress+proxy+constraint

## Training Feedback (raw evidence)

# Training Feedback

## External evaluation
- score: -108.448347
- episode_length: 74.100000 (mean)
- range: [-120.984102, -97.182870]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.097128 | 0.097128 | 1.000000 | -0.169501 | -0.000119 |
| progress_reward | 0.242793 | 0.256757 | 0.999993 | -0.621055 | 0.636037 |
| soft_landing_bonus | 0.004115 | 0.004115 | 0.009211 | 0.000000 | 0.914699 |
| stability_penalty | -0.216834 | 0.216834 | 1.000000 | -1.739710 | -0.000000 |
| total_reward | -0.067054 | 0.117649 | 1.000000 | -1.895030 | 0.912603 |
| generated_reward | -0.067054 | 0.117649 | 1.000000 | -1.895030 | 0.912603 |
| original_env_reward | -1.570631 | 2.392677 | 1.000000 | -100.000000 | 136.063634 |

## Signals
early_failure_or_crash; sparse_proxy:soft_landing_bonus; penalty_dominance:stability_penalty; penalty_dominance:original_env_reward
