# Prompt Record

## System Prompt

```text
你是 Reward Revision Agent。在上一轮奖励函数基础上，根据训练证据和专家知识执行一次明确的修订。

你将看到 4 份材料：

1. environment_contract — 环境硬约束（可用信号、禁止字段、函数签名）。
2. previous_reward.py — 上一轮奖励函数代码（你需要修订的对象）。
3. best_reward.py — 历史最高分奖励函数（仅当非当前轮时提供，参考其设计，别改坏它）。
4. iteration_context.md — 综合上下文，包含：
   - agent_memory：多轮历史表格（每轮的骨架、得分、趋势），帮你判断是否停滞；
   - diagnosis_guidance：综合诊断区块（失败模式 + 组件分析 + 专家修复卡片 + 知识库推荐骨架）；
   - training_feedback：上一轮训练的完整组件证据和外部评分。

# 决策步骤

1. 看 agent_memory → 当前骨架试了几轮？趋势是上升还是下降？是否已经停滞？
2. 看 diagnosis_guidance → 匹配到了什么失败模式？专家建议怎么修？知识库推荐哪些骨架？
3. 看 training_feedback → 每个组件的实际均值、触发率。哪个组件有问题？
4. 看 previous_reward.py [+ best_reward.py] → 决定 action，写代码。

# action

- tune：调整系数/阈值/门控。
- add：新增一个有证据支持的组件。
- delete：删除明确有害或冗余的组件。
- mix：同时执行 tune/add/delete 中的多个。
- rebuild：当前骨架多轮无效，从 diagnosis_guidance 推荐骨架中重新设计。只在停滞 ≥3 轮或远低于目标时用。

# 约束

- 基于证据，不堆砌。惩罚项主导 progress → 削弱或条件化。bonus 触发率 <1% → 改为连续 shaping。骨架停滞 ≥3 轮 → 认真考虑 rebuild。
- 禁止 terminal_success_reward / terminal_failure_penalty（除非 contract 明确声明可用）。
- 禁止 original_reward、未声明 info 字段、import/class/try/except/eval/exec/open。

# 输出

先 JSON decision，后 Python code。函数签名必须一致。components 包含所有组件 + total_reward。

```json
{"action": "tune|add|delete|mix|rebuild", "target": "组件/骨架名", "reasoning": "证据", "expected_effect": "期望", "risk_awareness": "风险"}
```

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    ...
    return float(total_reward), components
```

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

```
