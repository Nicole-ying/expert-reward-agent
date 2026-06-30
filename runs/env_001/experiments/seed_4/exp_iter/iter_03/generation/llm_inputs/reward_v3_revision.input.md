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
    # 主学习信号：progress_delta_reward
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 2.0 * progress_delta

    # 稳定约束：stability_penalty (削弱)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.05 * abs(next_obs[4])  # 权重从0.1降至0.05
    angular_vel_penalty = 0.025 * abs(next_obs[5])  # 权重从0.05降至0.025
    speed_penalty = 0.1 * speed  # 权重从0.2降至0.1
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # 平滑接近引导：distance_reward (新增)
    # 当距离较远时，提供持续的正向引导，避免progress_delta在震荡时信号微弱
    distance_reward = -0.1 * next_dist  # 鼓励靠近目标，权重较小

    # 任务完成proxy：soft_landing_proxy (保留，但条件更宽松)
    near_target = next_dist < 0.5  # 阈值从0.3放宽到0.5
    low_speed = speed < 0.8  # 阈值从0.5放宽到0.8
    stable_angle = abs(next_obs[4]) < 0.3  # 阈值从0.2放宽到0.3
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_bonus = 1.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 总奖励
    total_reward = progress_reward + stability_penalty + distance_reward + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "distance_reward": distance_reward,
        "landing_bonus": landing_bonus,
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
- mean_eval_reward: -115.806024
- mean_episode_length: 71.200000
- min_eval_reward: -137.898140
- max_eval_reward: -95.703639

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.032432, nonzero_rate: 0.999992
- stability_penalty mean: -0.110041, abs_mean: 0.110041

## 5. Preliminary failure hints
- likely_failure_mode: early_failure_or_crash
- evidence: negative external reward and short episode length
- likely_issue: stability penalty may dominate progress signal

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

### iter_2

- reward_structure: distance_reward + landing_bonus + progress_reward + stability_penalty
- external_score: -115.81
- mean_episode_length: 71.20
- reward_error_count: 0

#### component_evidence

- progress 0.032; stability -0.110

#### diagnosis

- early_failure_or_crash

#### next_action

- add smoother approach/landing guidance

## Matched Expert Cards

## early_failure_or_crash

- signal: negative external score and short episode length
- risk: reward does not guide stable control before termination
- fix: add smooth approach/landing signals; avoid relying on sparse terminal-like proxy

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
