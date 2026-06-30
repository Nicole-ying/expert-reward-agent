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
    # 主学习信号：progress_delta_reward (保留，核心驱动)
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 2.0 * progress_delta

    # 稳定约束：stability_penalty (进一步削弱，避免主导progress)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.005 * abs(next_obs[4])  # 从0.01降至0.005
    angular_vel_penalty = 0.002 * abs(next_obs[5])  # 从0.005降至0.002
    speed_penalty = 0.01 * speed  # 从0.02降至0.01
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # 平滑接近引导：distance_reward (保留，作为小锚点)
    distance_reward = -0.1 * next_dist

    # 平滑接近引导：approach_shaping (新增，远距离时提供额外梯度)
    approach_shaping = 0.0
    if next_dist > 1.0:
        approach_shaping = -0.2 * next_dist  # 远距离时鼓励靠近

    # 平滑着陆引导：landing_shaping (替代硬阈值landing_bonus)
    near_target = next_dist < 1.0
    low_speed = speed < 1.5
    stable_angle = abs(next_obs[4]) < 0.5
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_shaping = 0.0
    if near_target and low_speed and stable_angle:
        landing_shaping = 0.5 * (1.0 - next_dist / 1.0)  # 距离越近奖励越大
        if both_contact:
            landing_shaping += 0.3  # 双脚着地额外奖励

    # 总奖励
    total_reward = progress_reward + stability_penalty + distance_reward + approach_shaping + landing_shaping

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "distance_reward": distance_reward,
        "approach_shaping": approach_shaping,
        "landing_shaping": landing_shaping,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# iteration_context.md

# Iteration Context for Reward Revision

This is the single compact context file for the next reward revision LLM.
Do not treat expert cards as templates; use them as diagnostic guidance.

## Previous Training Feedback

## 2. External evaluation
- mean_eval_reward: 176.998896
- mean_episode_length: 789.700000
- min_eval_reward: 106.054467
- max_eval_reward: 258.847001

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.006170, nonzero_rate: 0.999542
- stability_penalty mean: -0.002723, abs_mean: 0.002723

## 5. Preliminary failure hints

## Short Memory

## Stable Lessons

- Use external evaluation reward as the fitness signal; generated reward alone is not enough.
- Keep terminal_success_reward blocked until an explicit success signal is available.
- Keep terminal_failure_penalty blocked until failure reason is available.
- Contact flags are only usable inside a guarded landing proxy: near target + low speed + stable angle + contact.
- Avoid speed or stability penalties dominating the main progress signal.
- Avoid a hard sparse completion bonus as the only landing guidance.
- Keep memory short: record component structure, key evidence, diagnosis, and next action only.

## Latest Iter Detail

### iter_6

- reward_structure: approach_shaping + distance_reward + landing_shaping + progress_reward + stability_penalty
- external_score: 177.00
- mean_episode_length: 789.70
- reward_error_count: 0

#### component_evidence

- progress 0.006; stability -0.003

#### diagnosis

- needs_review

#### next_action

- inspect component balance

## Matched Expert Cards

- none

## Skeleton Revision Plan

### keep
- progress_delta_reward

### weaken
- none

### revise
- none

### consider_add
- distance_reward as a small anchor if progress-only guidance remains weak

### still_defer
- terminal_success_reward
- terminal_failure_penalty
- energy_penalty
- time_penalty
- gated_reward

## Reward Revision Boundary

- Revise the previous reward instead of generating from scratch.
- Keep the function signature unchanged.
- Do not use original_reward or unavailable info fields.
- Do not add terminal success/failure rewards without explicit signals.
- Prefer fewer components with clear roles over adding many skeletons.
