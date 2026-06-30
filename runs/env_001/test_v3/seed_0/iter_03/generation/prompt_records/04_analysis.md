# Prompt Record

## System Prompt

```text
你是训练反馈分析模块。你的任务不是生成奖励函数，而是阅读训练结果，给出结构化的诊断报告。

你将读取：
1. training_feedback：上一轮 PPO 训练的完整反馈（外部评分、所有组件的均值和触发率、失败信号）；
2. agent_memory：最近多轮的历史记录（每轮的骨架、得分、趋势、组件证据）；
3. previous_reward.py：上一轮的完整奖励函数代码；
4. failure_mode_names：专家知识库中 10 种常见失败模式的名称列表；
5. hacking_risk_names：专家知识库中 8 种奖励黑客风险的名称列表。

你的任务：分析当前训练反馈，输出一份结构化诊断 JSON。

# 分析步骤

1. 阅读所有组件证据，判断每个组件的作用方向（正/负）和信号强度。
2. 对比 agent_memory 中的历史趋势：当前骨架是否在多轮中反复出现但未突破？
3. 从 failure_mode_names 中选出当前最匹配的 1-2 个失败模式。
4. 从 hacking_risk_names 中选出当前最匹配的 1-2 个奖励黑客风险。
5. 综合判断推荐动作：tune（微调系数）/ add（加新组件）/ delete（删除有损组件）/ mix（组合操作）/ rebuild（换骨架）。

# 匹配规则

- 只看证据，不猜测。如果没有明确信号匹配某个模式，不要强行选。
- 失败模式匹配时在 reasoning 中引用具体的组件证据和得分。
- 奖励黑客风险匹配时引用 generated_reward 与 external_score 的差距。
- 如果当前骨架已经在历史中连续出现 >= 3 轮且得分始终未突破 target_score 的 50%，应建议 rebuild。

# 输出格式

仅输出一个 JSON 对象，不要 Markdown，不要代码块，不要额外文字：

{
  "failure_modes": ["name1", "name2"],
  "hacking_risks": ["name1"],
  "component_analysis": {
    "component_name": {
      "role": "progress|constraint|proxy|efficiency|anchor",
      "direction": "positive|negative",
      "signal_strength": "strong|moderate|weak",
      "issue": "描述该组件可能的问题，没有则填 none"
    }
  },
  "skeleton_assessment": {
    "current_skeleton": ["comp1", "comp2", ...],
    "iterations_on_this_skeleton": 3,
    "best_score_this_skeleton": -10.5,
    "stagnant": true,
    "skeleton_family": "progress+stability+landing_proxy+anchor"
  },
  "recommended_action": "tune|add|delete|mix|rebuild",
  "reasoning": "简短的中文诊断推理，引用关键证据",
  "new_lessons": ["从本轮训练中学到的规律，如：progress_reward coefficient must be >= 50 to drive learning", "每条一条"]
}

```

## User Prompt

```markdown
# training_feedback
# Training Feedback

## External evaluation
- score: 158.822193
- episode_length: 728.400000 (mean)
- range: [93.826443, 208.689203]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| landing_shaping | 1.624923 | 1.624923 | 0.542595 | 0.000000 | 4.990018 |
| progress_reward | 0.203996 | 0.219075 | 0.998808 | -1.641093 | 2.296980 |
| stability_penalty | -0.037143 | 0.037143 | 1.000000 | -0.456914 | -0.000000 |
| total_reward | 1.791776 | 1.809284 | 1.000000 | -1.843835 | 4.990014 |
| generated_reward | 1.791776 | 1.809284 | 1.000000 | -1.843835 | 4.990014 |
| original_env_reward | -0.172914 | 1.689840 | 1.000000 | -100.000000 | 136.593785 |

## Signals
partial_progress; penalty_dominance:landing_shaping; penalty_dominance:generated_reward; penalty_dominance:original_env_reward


# agent_memory
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -111.26 | -111.26 | 0.00 | 74.10 | progress_reward=0.160 soft_landing_bonus=0.011 stability_penalty=-0.340 | new_best |
| 2 | progress_reward + soft_landing_bonus + stability_penalty | -111.79 | -111.26 | -0.53 | 74.10 | progress_reward=0.242 soft_landing_bonus=0.010 stability_penalty=-0.331 | no_meaningful_improvement |
| 3 | landing_shaping + progress_reward + stability_penalty | 158.82 | 158.82 | 0.00 | 728.40 | landing_shaping=1.625 progress_reward=0.204 stability_penalty=-0.037 | new_best |

## Stable Lessons

- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
- stability_penalty coefficient must be reduced to avoid penalty dominance
- progress_reward coefficient should be increased to drive learning
- progress_reward coefficient must be >= 50 to overcome stability_penalty dominance
- stability_penalty coefficient should be <= 0.1 to avoid penalty dominance
- soft_landing_bonus trigger rate <1% indicates sparse proxy; consider continuous shaping or relaxed conditions


# previous_reward.py
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state variables
    # Position (relative to target)
    x_pos = obs[0]
    y_pos = obs[1]
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    
    # Velocity
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    
    # Orientation
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # Contact flags
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # ========== Component 1: Progress Delta Reward (main learning signal) ==========
    # Distance to target at current step
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    # Distance to target at next step
    dist_next = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    # Progress: positive when moving closer to target
    progress_delta = dist_current - dist_next
    # Significantly increased coefficient to overcome penalty dominance
    progress_reward = 50.0 * progress_delta
    
    # ========== Component 2: Stability Penalty (light constraint, reduced) ==========
    # Penalize high speed, large angle, and high angular velocity
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # Reduced all penalties significantly to avoid dominance
    angle_penalty = 0.05 * abs(body_angle)
    angular_penalty = 0.02 * abs(angular_vel)
    speed_penalty = 0.1 * speed
    
    stability_penalty = -(angle_penalty + angular_penalty + speed_penalty)
    
    # ========== Component 3: Soft Landing Shaping (continuous, replaces sparse bonus) ==========
    # Continuous shaping that rewards being near target, low speed, stable angle, and both contacts
    # This provides gradient instead of sparse binary bonus
    near_target_score = max(0.0, 1.0 - dist_next / 0.5)  # 1.0 when dist=0, 0.0 when dist>=0.5
    low_speed_score = max(0.0, 1.0 - speed / 0.3)  # 1.0 when speed=0, 0.0 when speed>=0.3
    stable_angle_score = max(0.0, 1.0 - abs(body_angle) / 0.2)  # 1.0 when angle=0, 0.0 when angle>=0.2
    both_contact_score = 1.0 if (left_contact > 0.5) and (right_contact > 0.5) else 0.0
    
    # Combined continuous shaping signal (product ensures all conditions matter)
    landing_shaping = 5.0 * near_target_score * low_speed_score * stable_angle_score * (0.5 + 0.5 * both_contact_score)
    
    # ========== Total Reward ==========
    total_reward = progress_reward + stability_penalty + landing_shaping
    
    # ========== Components Dictionary ==========
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# expert_knowledge_context
# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。

## 1. 任务路由摘要
- navigation_goal_reaching：用密集过程引导；无 success flag 时禁用终点成功核心项；重点观察 goal_near_oscillation / high_reward_without_success / fast_crash_near_goal。

## 2. 相关奖励骨架摘要
### progress_delta_reward
- 角色: 主学习引导
- 数学形态: d(obs, goal) - d(next_obs, goal)
- 需要信号: obs[0], obs[1], next_obs[0], next_obs[1]
- 本轮建议: 推荐作为 v1 主信号：奖励每一步更接近目标。
- 风险: 目标附近震荡。
- 后续迭代: 可 clip；后续配合成功、时间或稳定信号。

### distance_reward
- 角色: 密集过程引导
- 数学形态: -d(obs, goal)
- 需要信号: obs[0], obs[1]
- 本轮建议: 可作为小权重 anchor；不要和 progress_delta_reward 同时大权重堆叠。
- 风险: 接近目标但不完成；不关心速度和姿态。
- 后续迭代: 训练后检查 high_reward_without_success。

### stability_penalty
- 角色: 轻量稳定约束
- 数学形态: -lambda_v*|velocity| - lambda_a*|angle| - lambda_w*|angular_velocity|
- 需要信号: next_obs[2], next_obs[3], next_obs[4], next_obs[5]
- 本轮建议: 如果任务要求稳定接近/着陆，v1 可以小权重加入。
- 风险: 过强会保守或不敢动。
- 后续迭代: 若高速撞击或姿态失稳，增大权重。

### soft_landing_proxy
- 角色: 任务完成近似信号
- 数学形态: small_bonus if near_target and low_speed and stable_angle and both_contact else 0
- 需要信号: position, velocity, angle, contact flags
- 本轮建议: 可选小权重；不能直接把 contact 当 success。
- 风险: 如果条件太宽，会变成 contact reward hacking。
- 后续迭代: 如果 high_reward_without_success，收紧条件或移除。

### terminal_success_reward
- 角色: 任务目标奖励
- 数学形态: R_success * I[success]
- 需要信号: 显式 success flag
- 本轮建议: 若 explicit_success_flag_available=false，不作为 v1 核心项。
- 风险: 会诱导 LLM 发明 info['success']。
- 后续迭代: 当 wrapper 明确暴露 success 后再加。

### terminal_failure_penalty
- 角色: 失败惩罚
- 数学形态: -R_failure * I[failure]
- 需要信号: 显式 failure flag 或 termination_reason
- 本轮建议: 若 explicit_failure_flag_available=false，不作为 v1 核心项。
- 风险: 误判终止原因。
- 后续迭代: 当能区分失败终止后再加。

### time_penalty
- 角色: 效率约束
- 数学形态: -lambda_time
- 需要信号: 每步调用
- 本轮建议: 通常后续加入，不建议 v1 太早加入。
- 风险: 可能导致冒险或快速失败。
- 后续迭代: 若能接近但拖太久，再小权重加入。

### energy_penalty
- 角色: 动作/能耗约束
- 数学形态: -lambda_action * engine_use(action)
- 需要信号: action
- 本轮建议: 通常后续加入，v1 太早加入可能不敢动。
- 风险: agent_afraid_to_move。
- 后续迭代: 能接近并稳定后再优化燃料。

### gated_reward
- 角色: 安全门控
- 数学形态: if unsafe then penalty else task_reward
- 需要信号: 明确 unsafe 条件
- 本轮建议: v1 不建议使用复杂门控。
- 风险: 门控过严导致学不到。
- 后续迭代: 若安全被进度奖励抵消，再加入。

### potential_based_shaping
- 角色: 势能塑形
- 数学形态: gamma*Phi(next_obs)-Phi(obs)
- 需要信号: 可定义 Phi
- 本轮建议: 不作为 v1 首选；比 progress_delta 更抽象。
- 风险: Phi 错误会误导学习。
- 后续迭代: 如果需要更标准的 shaping，再替换或补充。

## 3. reward_v1 生成要求
- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。
- 使用 role-based component budget，而不是固定组件数量。
- 推荐 2~4 个组件：1 个主学习信号 + 0~2 个稳定/安全约束 + 0~1 个任务完成 proxy。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty 作为 v1 核心项。
- 如果速度/姿态信号明确可用，且任务需要稳定接近或着陆，可以加入轻量 stability_penalty。
- 如果使用 contact，只能作为 soft_landing_proxy 的一部分，必须和 near_target、low_speed、stable_angle 组合，不要直接把 contact 当 success。
- energy_penalty、time_penalty、gated_reward 默认后续迭代再加入。
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。

# known_failure_modes
speed_penalty_dominance, stability_penalty_dominance, sparse_completion_proxy, early_failure_or_crash, high_reward_without_success, goal_near_oscillation, contact_reward_hacking, generated_reward_negative_average

# known_hacking_risks
speed_penalty_dominance, stability_penalty_dominance, sparse_completion_proxy, early_failure_or_crash, high_reward_without_success, goal_near_oscillation, contact_reward_hacking, generated_reward_negative_average

Based on the evidence above, output a diagnostic JSON.

```
