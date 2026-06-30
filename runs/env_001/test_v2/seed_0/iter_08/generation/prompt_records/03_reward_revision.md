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
    progress_delta_reward = 20.0 * progress_delta  # 从10.0增加到20.0
    
    # 组件2: 条件化稳定约束 - conditional_stability_penalty (增强系数)
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # 根据距离调整惩罚强度：远距离时惩罚更轻
    distance_factor = min(1.0, next_dist / 0.5)  # 0~1, 距离越远因子越大(惩罚越轻)
    # 当距离远时，允许更大的速度和角度
    angle_penalty = 0.6 * abs(body_angle) * (1.0 - 0.5 * distance_factor)  # 从0.3增加到0.6
    angular_vel_penalty = 0.2 * abs(angular_vel) * (1.0 - 0.5 * distance_factor)  # 从0.1增加到0.2
    speed_penalty = 0.4 * speed * (1.0 - 0.5 * distance_factor)  # 从0.2增加到0.4
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 组件3: 连续着陆质量塑形 - landing_quality_shaping (增强系数并调整激活阈值)
    # 替代稀疏的soft_landing_bonus，提供连续信号
    # 当接近目标时，奖励低速、稳定姿态和双支撑接触
    near_target = max(0.0, 1.0 - next_dist / 0.8)  # 从0.5放宽到0.8，增加激活范围
    speed_quality = max(0.0, 1.0 - speed / 0.5)  # 0~1, 越慢越大
    angle_quality = max(0.0, 1.0 - abs(body_angle) / 0.3)  # 0~1, 越正越大
    contact_quality = 0.5 * (left_contact + right_contact)  # 0~1, 双支撑更好
    
    # 组合成连续信号，仅在接近目标时激活
    landing_quality = near_target * (0.3 * speed_quality + 0.3 * angle_quality + 0.4 * contact_quality)
    landing_quality_shaping = 5.0 * landing_quality  # 从3.0增加到5.0
    
    # 组件4: 增强距离锚点 - distance_anchor (增强系数)
    distance_anchor = -0.5 * next_dist  # 从-0.1增加到-0.5
    
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

- iteration: 8
- target_score: 200.000
- best_score: 110.984 (iter 7)
- current_score: 110.984
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
    "goal_near_oscillation"
  ],
  "component_analysis": {
    "distance_anchor": {
      "role": "anchor",
      "direction": "negative",
      "signal_strength": "strong",
      "issue": "always negative, may discourage exploration"
    },
    "landing_quality_shaping": {
      "role": "proxy",
      "direction": "positive",
      "signal_strength": "strong",
      "issue": "none"
    },
    "progress_delta_reward": {
      "role": "progress",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "low mean, may need higher coefficient"
    },
    "stability_penalty": {
      "role": "constraint",
      "direction": "negative",
      "signal_strength": "moderate",
      "issue": "none"
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
    "best_score_this_skeleton": 110.98,
    "stagnant": false,
    "skeleton_family": "anchor+proxy+progress+constraint"
  },
  "recommended_action": "tune",
  "reasoning": "当前骨架仅运行1轮，得分110.98，未达目标200。progress_delta_reward均值低（0.147），建议增大其系数；distance_anchor始终为负，可能抑制探索，可降低其权重。landing_quality_shaping和stability_penalty表现正常。",
  "new_lessons": [
    "progress_delta_reward coefficient should be increased to drive more progress",
    "distance_anchor negative bias may hinder exploration; consider reducing its weight"
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
- score: 110.983632
- episode_length: 495.800000 (mean)
- range: [-45.099904, 265.964283]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.299615 | 0.299615 | 1.000000 | -0.847504 | -0.000019 |
| landing_quality_shaping | 1.313273 | 1.313273 | 0.667698 | 0.000000 | 4.994297 |
| progress_delta_reward | 0.146532 | 0.163949 | 0.999920 | -0.684212 | 0.794032 |
| stability_penalty | -0.152022 | 0.152022 | 1.000000 | -2.524427 | -0.000004 |
| total_reward | 1.008169 | 1.390079 | 1.000000 | -2.535247 | 4.993656 |
| generated_reward | 1.008169 | 1.390079 | 1.000000 | -2.535247 | 4.993656 |
| original_env_reward | -0.327450 | 3.209037 | 1.000000 | -100.000000 | 134.926482 |

## Signals
partial_progress

```
