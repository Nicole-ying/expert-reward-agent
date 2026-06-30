# Prompt Record

## System Prompt

```text
你是 Reward Revision LLM。你的任务不是重新从零生成奖励函数，而是在上一轮奖励函数基础上做一次有证据的迭代修订。

你将读取：
1. environment_contract：环境接口约束；
2. previous_reward.py：上一轮奖励函数；
3. iteration_context.md：训练反馈、短记忆、命中的专家卡片、骨架修订计划。

# 核心原则

- 继承上一轮中有效的组件，不要从零重写一个完全无关的 reward。
- 根据 iteration_context 中的证据决定 keep / weaken / revise / consider_add / still_defer。
- 专家卡片只是诊断背景，不是模板；不要机械堆叠所有 skeleton。
- 优先修复证据明确的问题，不要为了显得复杂而新增很多组件。
- 如果上一轮显示某个惩罚项主导 progress signal，应降低或条件化该惩罚项。
- 如果 completion proxy 触发率很低，应考虑更平滑的 shaping，而不是简单增大奖励。
- 如果 explicit success/failure flag 不可用，仍然不要使用 terminal_success_reward / terminal_failure_penalty。
- 不要使用 original_reward、official_reward、fitness_score、individual_reward。
- 不要使用未声明的 info 字段，例如 info["success"]、info.get("success")。

# 输出格式

输出 Markdown。第一个 Python code block 必须只包含完整可执行的 `compute_reward` 函数。

函数签名必须完全一致：

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

首选返回格式：

```python
return float(total_reward), components
```

components 必须包含所有加入 total_reward 的组件，以及 total_reward。

# 代码硬约束

- 不要 import。
- 不要 class。
- 不要 try/except。
- 不要 eval/exec/open。
- 不要创建额外函数。
- 不要传 self。
- 不要使用 self attributes。
- 不要使用 obs/next_obs 切片。
- 对 Env_001 使用二维位置：obs[0], obs[1]。

# 设计说明必须简短说明

- 相比上一轮，保留了什么；
- 削弱或删除了什么；
- 修改了什么；
- 新增了什么；
- 为什么仍然不使用 terminal_success_reward / terminal_failure_penalty；
- 下一轮训练后应该重点观察什么。

```

## User Prompt

```markdown
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
    
    # 1. Main learning signal: progress_delta_reward (kept, unchanged)
    current_distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_distance = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_distance - next_distance
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty (weakened: reduced coefficients to prevent dominance)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.05 * abs(next_body_angle)  # reduced from 0.1
    angular_vel_penalty = 0.02 * abs(next_angular_vel)  # reduced from 0.05
    speed_penalty = 0.05 * speed  # reduced from 0.1
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. Simplified landing-quality shaping (kept, unchanged from last iteration)
    near_target_factor = max(0.0, 1.0 - next_distance / 2.0)
    low_speed_factor = max(0.0, 1.0 - speed / 2.0)
    upright_factor = max(0.0, 1.0 - abs(next_body_angle) / 0.5)
    
    landing_quality = near_target_factor * low_speed_factor * upright_factor
    landing_shaping_reward = 1.0 * landing_quality
    
    # 4. Small distance anchor to prevent getting stuck far away (kept, unchanged from last iteration)
    distance_anchor = -0.05 * next_distance
    
    # Combine components
    total_reward = progress_delta_reward + stability_penalty + landing_shaping_reward + distance_anchor
    
    # Build components dict
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping_reward": landing_shaping_reward,
        "distance_anchor": distance_anchor,
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
- mean_eval_reward: 259.522516
- mean_episode_length: 335.400000
- min_eval_reward: 236.828402
- max_eval_reward: 286.871097

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- stability_penalty mean: -0.013326, abs_mean: 0.013326

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

### iter_7

- reward_structure: distance_anchor + landing_shaping_reward + progress_delta_reward + stability_penalty
- external_score: 259.52
- mean_episode_length: 335.40
- reward_error_count: 0

#### component_evidence

- stability -0.013

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

```
