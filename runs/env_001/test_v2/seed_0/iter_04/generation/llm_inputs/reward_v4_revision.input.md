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
    # Increased coefficient from 15.0 to 20.0 to strengthen positive signal
    progress_reward = 20.0 * progress_delta
    
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

- iteration: 4
- target_score: 200.000
- best_score: -110.396 (iter 1)
- current_score: -111.239
- trend: declining_from_best
- guidance: Investigate why score dropped from best. Consider reverting harmful changes.
- suggested_action: tune or rebuild

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
    "sparse_completion_proxy"
  ],
  "hacking_risks": [
    "stability_penalty_dominance",
    "sparse_completion_proxy"
  ],
  "component_analysis": {
    "distance_anchor": {
      "role": "anchor",
      "direction": "negative",
      "signal_strength": "moderate",
      "issue": "negative mean indicates agent is far from target; may need stronger shaping"
    },
    "progress_reward": {
      "role": "progress",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "positive mean but external score is very negative; progress not translating to success"
    },
    "soft_landing_bonus": {
      "role": "proxy",
      "direction": "positive",
      "signal_strength": "weak",
      "issue": "very low nonzero rate (0.9%); sparse and rarely triggered"
    },
    "stability_penalty": {
      "role": "constraint",
      "direction": "negative",
      "signal_strength": "strong",
      "issue": "large negative mean (-0.218) and always active; dominates total reward"
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
    "best_score_this_skeleton": -111.24,
    "stagnant": false,
    "skeleton_family": "anchor+progress+proxy+constraint"
  },
  "recommended_action": "tune",
  "reasoning": "外部得分-111.24远低于目标200，且original_env_reward均值-1.53表明环境惩罚大。stability_penalty均值-0.218且始终激活，主导总奖励，符合stability_penalty_dominance。soft_landing_bonus触发率仅0.9%，稀疏且贡献小，符合sparse_completion_proxy。progress_reward正向但未转化为成功，需调整系数。建议微调：降低stability_penalty系数，提高progress_reward系数，并增加soft_landing_bonus触发条件。",
  "new_lessons": [
    "stability_penalty coefficient should be reduced to avoid dominating total reward",
    "soft_landing_bonus needs more frequent triggering to provide useful shaping"
  ]
}
```

### Expert Cards (compressed)
## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target
## sparse_completion_proxy
- signal: completion/landing bonus trigger_rate < 1%
- risk: final bonus provides little early learning guidance
- fix: replace hard bonus with smoother landing-quality shaping

### KB Recommended Skeletons for task `navigation_goal_reaching`
- time_penalty, distance_reward, progress_delta_reward, potential_based_shaping, gated_reward
- Previously tried skeleton family: anchor+progress+proxy+constraint

## Training Feedback (raw evidence)

# Training Feedback

## External evaluation
- score: -111.238907
- episode_length: 74.100000 (mean)
- range: [-120.969771, -105.485038]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.096984 | 0.096984 | 1.000000 | -0.170105 | -0.000176 |
| progress_reward | 0.323339 | 0.341766 | 0.999995 | -0.807268 | 0.844599 |
| soft_landing_bonus | 0.003982 | 0.003982 | 0.009369 | 0.000000 | 0.914458 |
| stability_penalty | -0.218167 | 0.218167 | 1.000000 | -1.570842 | -0.000001 |
| total_reward | 0.012169 | 0.155207 | 1.000000 | -2.001005 | 0.919099 |
| generated_reward | 0.012169 | 0.155207 | 1.000000 | -2.001005 | 0.919099 |
| original_env_reward | -1.526222 | 2.404256 | 1.000000 | -100.000000 | 129.532942 |

## Signals
early_failure_or_crash; sparse_proxy:soft_landing_bonus; penalty_dominance:original_env_reward
