# Prompt Record

## System Prompt

```text
你是 Reward Revision LLM。你的任务不是重新从零生成奖励函数，而是在上一轮奖励函数基础上，根据训练证据执行一次明确的迭代动作。

你将读取：
1. environment_contract：环境接口约束；
2. previous_reward.py：上一轮奖励函数；
3. iteration_context.md：训练反馈、短记忆、命中的专家卡片、骨架修订计划，以及最重要的 Iteration Control Decision。

# 最高优先级：执行 Iteration Control Decision

iteration_context.md 顶部可能包含：

```text
## Iteration Control Decision
- target_score
- best_score_so_far
- best_iter
- previous_score
- trend
- decision
- required_action
- forbidden_action
```

你必须优先服从这个决策，而不是被专家卡片带偏。

- 如果 decision 是 MUST_MODIFY，则禁止输出与 previous_reward.py 实质相同的奖励函数。
- 如果 required_action 指定 tune/delete/add/mix/rebuild，你必须围绕这个动作修改代码。
- 如果 best_score_so_far 已经超过 target_score，只允许小心调整；不要无证据地大改已经解决问题的 reward。
- 如果 previous_score 已经低于 best_score_so_far，必须解释如何避免继续破坏 best reward。
- 如果你认为不需要修改，那么应在说明中写出 STOP_AND_KEEP_BEST 的理由；不要原样输出上一轮 reward 伪装成修订。

# 允许的 action

你必须在代码块前先输出一个简短决策块：

```text
## Revision Decision
- action: tune | delete | add | mix | rebuild
- reason:
- required_code_change:
- expected_effect:
```

action 含义：

- tune：只调整已有组件的系数、门控或阶段权重。
- delete：删除明确有害或冗余的组件。
- add：新增一个缺失且有证据支持的组件。
- mix：同时执行 tune/delete/add 中的多个动作。
- rebuild：当前 skeleton 多轮无效后，重新设计主要结构。只有连续停滞或明显错误时才使用。

# 核心原则

- 继承上一轮中有效的组件，但不能把“继承”理解成原样输出。
- 根据 iteration_context 中的证据决定 keep / weaken / revise / consider_add / still_defer。
- 专家卡片只是诊断背景，不是模板；不要机械堆叠所有 skeleton。
- 优先修复证据明确的问题，不要为了显得复杂而新增很多组件。
- 如果上一轮显示某个惩罚项主导 progress signal，应降低、删除或条件化该惩罚项。
- 如果 completion proxy 触发率很低，应考虑更平滑的 shaping，而不是简单增大奖励。
- 如果 reward 已经超过 target_score，默认只做小幅 tune；除非 evidence 明确显示某组件有害，否则不要大改结构。
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

- 执行了哪个 action；
- 相比上一轮，保留了什么；
- 削弱、删除或新增了什么；
- 为什么这个改动针对当前 failure / trend；
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
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. Main learning signal: progress_delta_reward (strengthened to restore dominance)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 200.0 * progress_delta
    
    # 2. Reduced stability penalty (unchanged - minimal impact)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.01 * abs(next_body_angle)
    speed_penalty = 0.01 * speed
    ang_vel_penalty = 0.005 * abs(next_ang_vel)
    stability_penalty = -(angle_penalty + speed_penalty + ang_vel_penalty)
    
    # 3. Strengthened landing-quality shaping (increased coefficients for better approach guidance)
    near_target = max(0.0, 1.0 - next_dist / 2.0)
    low_speed = max(0.0, 1.0 - speed / 1.0)
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.5)
    landing_shaping = 2.0 * near_target + 2.0 * low_speed + 1.5 * stable_angle
    
    # 4. Distance reward - reduced back to small anchor (was -0.5, now -0.1)
    distance_reward = -0.1 * next_dist
    
    # Combine components
    total_reward = progress_reward + stability_penalty + landing_shaping + distance_reward
    
    # Build components dict
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "distance_reward": distance_reward,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# iteration_context.md

# Iteration Control Decision

- iteration_to_generate: 9
- target_score: 200.000
- best_score_so_far: 146.234
- best_iter: 6
- previous_score: -18.551
- no_improvement_count: 2
- trend: below_best
- decision: MUST_MODIFY
- required_action: tune/delete/add/mix based on failure evidence; do not return identical reward
- forbidden_action: no-op revision; generic restatement; changing comments only
- note: This control block has higher priority than matched expert cards.

# Iteration Context for Reward Revision

This is the single compact context file for the next reward revision LLM.
Do not treat expert cards as templates; use them as diagnostic guidance.

## Previous Training Feedback

## 2. External evaluation
- mean_eval_reward: -18.550812
- mean_episode_length: 1000.000000
- min_eval_reward: -52.128165
- max_eval_reward: 12.658302

## 3. Reward execution health
- reward_error_count_max: 0

## 4. Key component evidence
- progress_reward mean: 0.776811, nonzero_rate: 0.999941
- stability_penalty mean: -0.005922, abs_mean: 0.005922

## 5. Preliminary failure hints

## Short Memory

## Stable Lessons

- Use external evaluation reward as the fitness signal; generated reward alone is not enough.
- Target solved threshold for Env_001: mean external evaluation score >= 200.
- Preserve best-so-far reward; final reward should be the best reward, not necessarily the last reward.
- If the task has been solved and a later revision drops below target, stop and keep the best reward.
- Keep terminal_success_reward blocked until an explicit success signal is available.
- Keep terminal_failure_penalty blocked until failure reason is available.
- Contact flags are only usable inside a guarded landing proxy: near target + low speed + stable angle + contact.
- Avoid speed or stability penalties dominating the main progress signal.
- Avoid a hard sparse completion bonus as the only landing guidance.
- Keep memory short: record component structure, score, best-so-far, decision, diagnosis, and next action only.

## Latest Iter Detail

### iter_8

- reward_structure: distance_reward + landing_shaping + progress_reward + stability_penalty
- external_score: -18.55
- best_score_so_far: 146.23
- best_iter: 6
- mean_episode_length: 1000.00
- reward_error_count: 0
- verdict: failure
- decision: no_meaningful_improvement

#### component_evidence

- progress 0.777; stability -0.006

#### diagnosis

- early_failure_or_crash

#### next_action

- add smoother approach/landing guidance

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
