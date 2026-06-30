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
    # 提取观测变量
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
    
    # 1. 主学习信号：progress_delta_reward (保留)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. 稳定约束：stability_penalty (保留，但削弱)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.2 * abs(next_body_angle)
    angular_vel_penalty = -0.1 * abs(next_angular_vel)
    speed_penalty = -0.05 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. 着陆质量塑造：landing_quality_shaping (修订)
    # 使用连续信号替代稀疏奖励，根据接近目标、低速、姿态稳定和双接触的程度给予奖励
    near_target = max(0.0, 1.0 - next_dist / 0.5)
    low_speed = max(0.0, 1.0 - speed / 0.3)
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.2)
    both_contact = 1.0 if (next_left_contact > 0.5) and (next_right_contact > 0.5) else 0.0
    
    landing_quality = near_target * low_speed * stable_angle * both_contact
    landing_shaping_reward = 5.0 * landing_quality
    
    # 4. 小权重距离锚点：distance_anchor (保留)
    distance_anchor = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_shaping_reward + distance_anchor
    
    components = {
        "progress_reward": progress_reward,
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
- mean_eval_reward: -110.819870
- mean_episode_length: 69.100000
- min_eval_reward: -123.014656
- max_eval_reward: -95.244135

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.161265, nonzero_rate: 0.999991
- stability_penalty mean: -0.064739, abs_mean: 0.064739

## 5. Preliminary failure hints
- likely_failure_mode: early_failure_or_crash
- evidence: negative external reward and short episode length

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

### iter_4

- reward_structure: distance_anchor + landing_shaping_reward + progress_reward + stability_penalty
- external_score: -110.82
- mean_episode_length: 69.10
- reward_error_count: 0

#### component_evidence

- progress 0.161; stability -0.065

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

```
