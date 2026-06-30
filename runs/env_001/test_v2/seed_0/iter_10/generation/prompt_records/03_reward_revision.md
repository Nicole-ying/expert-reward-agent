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
    
    # 组件1: 主学习信号 - progress_delta_reward (增强系数)
    progress_delta = current_dist - next_dist
    progress_delta_reward = 60.0 * progress_delta  # 从40.0增加到60.0
    
    # 组件2: 条件化稳定约束 - conditional_stability_penalty (保持原系数)
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # 根据距离调整惩罚强度：远距离时惩罚更轻
    distance_factor = min(1.0, next_dist / 0.5)  # 0~1, 距离越远因子越大(惩罚越轻)
    # 当距离远时，允许更大的速度和角度
    angle_penalty = 0.6 * abs(body_angle) * (1.0 - 0.5 * distance_factor)
    angular_vel_penalty = 0.2 * abs(angular_vel) * (1.0 - 0.5 * distance_factor)
    speed_penalty = 0.4 * speed * (1.0 - 0.5 * distance_factor)
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 组件3: 连续着陆质量塑形 - landing_quality_shaping (降低系数)
    # 当接近目标时，奖励低速、稳定姿态和双支撑接触
    near_target = max(0.0, 1.0 - next_dist / 0.8)  # 保持激活范围
    speed_quality = max(0.0, 1.0 - speed / 0.5)  # 0~1, 越慢越大
    angle_quality = max(0.0, 1.0 - abs(body_angle) / 0.3)  # 0~1, 越正越大
    contact_quality = 0.5 * (left_contact + right_contact)  # 0~1, 双支撑更好
    
    # 组合成连续信号，仅在接近目标时激活
    landing_quality = near_target * (0.3 * speed_quality + 0.3 * angle_quality + 0.4 * contact_quality)
    landing_quality_shaping = 4.0 * landing_quality  # 从6.0降低到4.0
    
    # 组件4: 增强距离锚点 - distance_anchor (降低权重)
    distance_anchor = -0.1 * next_dist  # 从-0.2降低到-0.1
    
    # 组件5: 新增 - 低速接近奖励 (减少振荡)
    # 当接近目标且速度较低时给予额外奖励，鼓励稳定接近
    near_target_smooth = max(0.0, 1.0 - next_dist / 0.5)  # 更窄的激活范围
    low_speed_bonus = near_target_smooth * max(0.0, 1.0 - speed / 0.3) * 2.0  # 0~2, 越慢越近越大
    
    # 总奖励
    total_reward = progress_delta_reward + stability_penalty + landing_quality_shaping + distance_anchor + low_speed_bonus
    
    # 组件字典
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "landing_quality_shaping": landing_quality_shaping,
        "distance_anchor": distance_anchor,
        "low_speed_bonus": low_speed_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# best_reward.py (historical best, for reference)

This is the highest-scoring reward so far. Learn from it, do not make it worse.

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
    
    # 组件1: 主学习信号 - progress_delta_reward (大幅增强系数)
    progress_delta = current_dist - next_dist
    progress_delta_reward = 40.0 * progress_delta  # 从20.0增加到40.0
    
    # 组件2: 条件化稳定约束 - conditional_stability_penalty (保持原系数)
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # 根据距离调整惩罚强度：远距离时惩罚更轻
    distance_factor = min(1.0, next_dist / 0.5)  # 0~1, 距离越远因子越大(惩罚越轻)
    # 当距离远时，允许更大的速度和角度
    angle_penalty = 0.6 * abs(body_angle) * (1.0 - 0.5 * distance_factor)
    angular_vel_penalty = 0.2 * abs(angular_vel) * (1.0 - 0.5 * distance_factor)
    speed_penalty = 0.4 * speed * (1.0 - 0.5 * distance_factor)
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 组件3: 连续着陆质量塑形 - landing_quality_shaping (微调系数)
    # 当接近目标时，奖励低速、稳定姿态和双支撑接触
    near_target = max(0.0, 1.0 - next_dist / 0.8)  # 保持激活范围
    speed_quality = max(0.0, 1.0 - speed / 0.5)  # 0~1, 越慢越大
    angle_quality = max(0.0, 1.0 - abs(body_angle) / 0.3)  # 0~1, 越正越大
    contact_quality = 0.5 * (left_contact + right_contact)  # 0~1, 双支撑更好
    
    # 组合成连续信号，仅在接近目标时激活
    landing_quality = near_target * (0.3 * speed_quality + 0.3 * angle_quality + 0.4 * contact_quality)
    landing_quality_shaping = 6.0 * landing_quality  # 从5.0增加到6.0
    
    # 组件4: 增强距离锚点 - distance_anchor (降低权重)
    distance_anchor = -0.2 * next_dist  # 从-0.5降低到-0.2
    
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

- iteration: 10
- target_score: 200.000
- best_score: 188.212 (iter 8)
- current_score: 19.510
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
    "goal_near_oscillation",
    "high_reward_without_success"
  ],
  "hacking_risks": [
    "goal_near_oscillation"
  ],
  "component_analysis": {
    "distance_anchor": {
      "role": "anchor",
      "direction": "negative",
      "signal_strength": "weak",
      "issue": "negative mean but small magnitude, may not be effective"
    },
    "landing_quality_shaping": {
      "role": "proxy",
      "direction": "positive",
      "signal_strength": "strong",
      "issue": "high mean but external score low, may be rewarding suboptimal behavior"
    },
    "low_speed_bonus": {
      "role": "efficiency",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "none"
    },
    "progress_delta_reward": {
      "role": "progress",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "mean positive but external score not improving, may be insufficient"
    },
    "stability_penalty": {
      "role": "constraint",
      "direction": "negative",
      "signal_strength": "weak",
      "issue": "small magnitude, may not enforce stability"
    }
  },
  "skeleton_assessment": {
    "current_skeleton": [
      "distance_anchor",
      "landing_quality_shaping",
      "low_speed_bonus",
      "progress_delta_reward",
      "stability_penalty"
    ],
    "iterations_on_this_skeleton": 1,
    "best_score_this_skeleton": 19.51,
    "stagnant": false,
    "skeleton_family": "progress+stability+landing_proxy+anchor"
  },
  "recommended_action": "tune",
  "reasoning": "External score 19.51 far below target 200. Landing quality shaping has strong positive signal but external score low, suggesting it may reward near-goal oscillation without success. Progress delta reward positive but insufficient. Recommend tuning coefficients: increase progress delta reward coefficient and reduce landing quality shaping coefficient to prioritize progress over proxy.",
  "new_lessons": [
    "Landing quality shaping can cause goal-near oscillation if not balanced with progress reward.",
    "Progress delta reward coefficient must be high enough to drive meaningful progress."
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
- Previously tried skeleton family: progress+stability+landing_proxy+anchor

## Training Feedback (raw evidence)

# Training Feedback

## External evaluation
- score: 19.509895
- episode_length: 868.300000 (mean)
- range: [-57.088469, 168.481450]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.048918 | 0.048918 | 1.000000 | -0.169501 | -0.000014 |
| landing_quality_shaping | 1.574789 | 1.574789 | 0.753924 | 0.000000 | 3.995997 |
| low_speed_bonus | 0.642045 | 0.642045 | 0.537667 | 0.000000 | 1.998023 |
| progress_delta_reward | 0.309660 | 0.341552 | 0.999046 | -1.969312 | 2.309510 |
| stability_penalty | -0.120864 | 0.120864 | 1.000000 | -2.705785 | -0.000002 |
| total_reward | 2.356712 | 2.398447 | 1.000000 | -3.039735 | 5.992805 |
| generated_reward | 2.356712 | 2.398447 | 1.000000 | -3.039735 | 5.992805 |
| original_env_reward | -0.138805 | 1.881180 | 1.000000 | -100.000000 | 120.737047 |

## Signals
partial_progress

```
