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
    # 提取状态变量
    x_pos = obs[0]
    y_pos = obs[1]
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    
    # 速度
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    
    # 姿态
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 接触标志
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # 计算距离
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    # 组件1: 主学习信号 - progress_delta_reward
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta
    
    # 组件2: 条件化稳定约束 - conditional_stability_penalty
    # 惩罚高速、大姿态角和大角速度，但在远距离时放宽约束
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # 根据距离调整惩罚强度：远距离时惩罚更轻
    distance_factor = min(1.0, next_dist / 0.5)  # 0~1, 距离越远因子越大(惩罚越轻)
    # 当距离远时，允许更大的速度和角度
    angle_penalty = 0.3 * abs(body_angle) * (1.0 - 0.5 * distance_factor)
    angular_vel_penalty = 0.1 * abs(angular_vel) * (1.0 - 0.5 * distance_factor)
    speed_penalty = 0.2 * speed * (1.0 - 0.5 * distance_factor)
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 组件3: 连续着陆质量塑形 - landing_quality_shaping
    # 替代稀疏的soft_landing_bonus，提供连续信号
    # 当接近目标时，奖励低速、稳定姿态和双支撑接触
    near_target = max(0.0, 1.0 - next_dist / 0.5)  # 0~1, 越近越大
    speed_quality = max(0.0, 1.0 - speed / 0.5)  # 0~1, 越慢越大
    angle_quality = max(0.0, 1.0 - abs(body_angle) / 0.3)  # 0~1, 越正越大
    contact_quality = 0.5 * (left_contact + right_contact)  # 0~1, 双支撑更好
    
    # 组合成连续信号，仅在接近目标时激活
    landing_quality = near_target * (0.3 * speed_quality + 0.3 * angle_quality + 0.4 * contact_quality)
    landing_quality_shaping = 3.0 * landing_quality
    
    # 组件4: 小权重距离锚点 - distance_anchor
    distance_anchor = -0.1 * next_dist
    
    # 总奖励
    total_reward = progress_delta_reward + stability_penalty + landing_quality_shaping + distance_anchor
    
    # 组件字典
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "landing_quality_shaping": landing_quality_shaping,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# iteration_context.md

# Agent Context

- iteration: 7
- target_score: 200.000
- best_score: 93.790 (iter 6)
- current_score: 93.790
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
    "goal_near_oscillation",
    "high_reward_without_success"
  ],
  "hacking_risks": [
    "goal_near_oscillation",
    "high_reward_without_success"
  ],
  "component_analysis": {
    "distance_anchor": {
      "role": "anchor",
      "direction": "negative",
      "signal_strength": "moderate",
      "issue": "negative mean indicates agent is far from target; may be too weak to pull agent closer"
    },
    "landing_quality_shaping": {
      "role": "proxy",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "low nonzero rate (0.32) suggests sparse activation; may not guide consistently"
    },
    "progress_delta_reward": {
      "role": "progress",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "positive mean but small magnitude; may be insufficient to drive progress"
    },
    "stability_penalty": {
      "role": "constraint",
      "direction": "negative",
      "signal_strength": "moderate",
      "issue": "negative mean but small magnitude; may be too weak to enforce stability"
    }
  },
  "skeleton_assessment": {
    "current_skeleton": [
      "distance_anchor",
      "landing_quality_shaping",
      "progress_delta_reward",
      "stability_penalty"
    ],
    "iterations_on_this_skeleton": 1,
    "best_score_this_skeleton": 93.79,
    "stagnant": false,
    "skeleton_family": "anchor+proxy+progress+constraint"
  },
  "recommended_action": "tune",
  "reasoning": "External score 93.79 is far below target 200. Progress_delta_reward mean 0.126 is low; distance_anchor negative (-0.082) indicates agent not reaching target. Landing_quality_shaping nonzero rate 0.32 suggests sparse guidance. No stagnation yet (only 1 iteration). Recommend tuning coefficients: increase progress_delta_reward coefficient, increase distance_anchor magnitude, and increase landing_quality_shaping activation frequency.",
  "new_lessons": [
    "progress_delta_reward coefficient may need to be increased to drive stronger progress",
    "distance_anchor negative mean suggests agent is not being pulled toward target; consider increasing its weight"
  ]
}
```

### Expert Cards (compressed)
## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward
## high_reward_without_success
- signal: generated_reward improves but external score stays poor
- risk: policy optimizes the custom reward but not the real task
- fix: reduce exploitable terms; add constraints tied to actual task progress and stable outcome

### KB Recommended Skeletons for task `navigation_goal_reaching`
- time_penalty, distance_reward, progress_delta_reward, potential_based_shaping, gated_reward
- Previously tried skeleton family: anchor+proxy+progress+constraint

## Training Feedback (raw evidence)

# Training Feedback

## External evaluation
- score: 93.789748
- episode_length: 433.800000 (mean)
- range: [-63.747210, 247.942875]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.082026 | 0.082026 | 1.000000 | -0.169501 | -0.000009 |
| landing_quality_shaping | 0.307413 | 0.307413 | 0.318349 | 0.000000 | 2.994347 |
| progress_delta_reward | 0.126131 | 0.135551 | 0.999793 | -0.377514 | 0.406659 |
| stability_penalty | -0.102975 | 0.102975 | 1.000000 | -1.296748 | -0.000000 |
| total_reward | 0.248544 | 0.353085 | 1.000000 | -1.313760 | 2.994109 |
| generated_reward | 0.248544 | 0.353085 | 1.000000 | -1.313760 | 2.994109 |
| original_env_reward | -0.671153 | 2.555039 | 1.000000 | -100.000000 | 125.946996 |

## Signals
partial_progress
