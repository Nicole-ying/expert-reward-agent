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
    # 提取状态变量
    # 位置（相对于目标）
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
    # 奖励每一步更接近目标
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta
    
    # 组件2: 轻量稳定约束 - stability_penalty
    # 惩罚高速、大姿态角和大角速度，鼓励稳定接近
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * abs(body_angle)
    angular_vel_penalty = 0.2 * abs(angular_vel)
    speed_penalty = 0.3 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 组件3: 软着陆代理奖励 - soft_landing_proxy
    # 当接近目标、低速、姿态稳定且双支撑接触时给予小奖励
    near_target = next_dist < 0.3
    low_speed = speed < 0.2
    stable_angle = abs(body_angle) < 0.2
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # 组件4: 小权重距离锚点 - distance_anchor
    # 辅助引导，防止在远距离时progress_delta信号太弱
    distance_anchor = -0.1 * next_dist
    
    # 总奖励
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + distance_anchor
    
    # 组件字典
    components = {
        "progress_delta_reward": progress_delta_reward,
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

- iteration: 6
- target_score: 200.000
- best_score: -110.396 (iter 1)
- current_score: -111.139
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
      "issue": "negative mean indicates agent is not staying near target; may be too weak to counteract stability penalty"
    },
    "progress_delta_reward": {
      "role": "progress",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "positive mean but insufficient to overcome negative components"
    },
    "soft_landing_bonus": {
      "role": "proxy",
      "direction": "positive",
      "signal_strength": "weak",
      "issue": "very low nonzero rate (0.46%) indicates sparse reward; agent rarely achieves soft landing"
    },
    "stability_penalty": {
      "role": "constraint",
      "direction": "negative",
      "signal_strength": "strong",
      "issue": "dominant negative component with large magnitude; likely causing agent to avoid movement"
    }
  },
  "skeleton_assessment": {
    "current_skeleton": [
      "distance_anchor",
      "progress_delta_reward",
      "soft_landing_bonus",
      "stability_penalty"
    ],
    "iterations_on_this_skeleton": 1,
    "best_score_this_skeleton": -111.139031,
    "stagnant": false,
    "skeleton_family": "anchor+progress+proxy+constraint"
  },
  "recommended_action": "tune",
  "reasoning": "Stability penalty dominates (mean -0.341, strong negative), causing agent to stay still to avoid penalty. Soft landing bonus is too sparse (0.46% nonzero rate) to guide learning. Progress reward is positive but insufficient. Recommend reducing stability penalty coefficient and increasing soft landing bonus frequency or magnitude.",
  "new_lessons": [
    "stability_penalty coefficient must be reduced to avoid dominance",
    "soft_landing_bonus needs denser shaping or higher reward to be effective"
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
- score: -111.139031
- episode_length: 74.100000 (mean)
- range: [-122.534771, -105.145711]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.097016 | 0.097016 | 1.000000 | -0.169501 | -0.000019 |
| progress_delta_reward | 0.160916 | 0.170245 | 0.999990 | -0.418539 | 0.425586 |
| soft_landing_bonus | 0.009273 | 0.009273 | 0.004637 | 0.000000 | 2.000000 |
| stability_penalty | -0.340556 | 0.340556 | 1.000000 | -2.804435 | -0.000000 |
| total_reward | -0.267383 | 0.284934 | 1.000000 | -3.049366 | 1.993564 |
| generated_reward | -0.267383 | 0.284934 | 1.000000 | -3.049366 | 1.993564 |
| original_env_reward | -1.538081 | 2.403107 | 1.000000 | -100.000000 | 126.195419 |

## Signals
early_failure_or_crash; sparse_proxy:soft_landing_bonus

```
