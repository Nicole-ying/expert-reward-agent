# Prompt Record

## System Prompt

```text
你是训练反馈分析模块。你的任务不是生成奖励函数，而是阅读训练结果，给出结构化的诊断报告。

你将读取：
1. training_feedback：上一轮训练的组件证据和得分；
2. agent_memory：历史演化表格（每轮的骨架、得分、趋势、组件信号）；
3. previous_reward.py：上一轮的完整奖励函数代码；
4. best_reward.py：历史最高分对应的奖励函数代码（可能与 previous 相同，也可能不同）；
5. expert_knowledge_context：专家知识库（任务类型、推荐骨架及数学形态、风险）；
6. failure_mode_names 和 hacking_risk_names：已知的失败模式和奖励黑客名称列表。

# 分析步骤

1. 阅读组件证据，判断每个组件的作用方向和信号强度。
2. **如果 best_reward.py 与 previous_reward.py 不同且 best 得分更高，逐行对比两段代码。**
   - 列出被修改的具体系数（如 progress=50→100, landing=5.0→2.0）。
   - 判断每个修改是导致了改善还是回归。
   - 如果 current 得分明显低于 best，推荐 revert（恢复到 best 的系数，只做小幅调整）。
3. 对比 agent_memory 历史：当前骨架试了几轮？趋势上升还是下降？
4. 从 failure_mode_names 中选出最匹配的 1-2 个失败模式。
5. 综合判断动作：revert（恢复到 best 配置）/ tune / add / delete / mix / rebuild。

# 动作含义

- revert：best_reward 得分显著高于 current，且代码差异可识别。恢复到 best_reward 的系数，只在此基础上做小幅修改。
- tune：调整系数/阈值/门控。
- add：新增一个有证据支持的组件。
- delete：删除明确有害的组件。
- mix：tune+add+delete 组合。
- rebuild：当前骨架已多轮无效，从 expert_knowledge_context 中选不同的数学形态。

# 匹配规则

- 只看证据，不猜测。
- 当 best_score > current_score 且差距 > 20 时，优先考虑 revert。
- 当前骨架连续 >= 3 轮且始终未突破 target 的 50%，应建议 rebuild。

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
  "recommended_action": "revert|tune|add|delete|mix|rebuild",
  "reasoning": "简短的中文诊断推理，引用关键证据",
  "new_lessons": ["从本轮训练中学到的规律，如：progress_reward coefficient must be >= 50 to drive learning", "每条一条"]
}

```

## User Prompt

```markdown
# training_feedback
# Training Feedback

## Training outcome
score=-113.704987, len=72.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| landing_shaping | 0.010606 | 0.010606 | 0.007734 | 0.000000 | 2.392725 |
| progress_reward | 1.290767 | 1.364724 | 0.999991 | -3.195675 | 3.387974 |
| stability_penalty | -0.237851 | 0.237851 | 1.000000 | -2.115911 | -0.000000 |
| total_reward | 1.063521 | 1.181778 | 1.000000 | -4.285132 | 3.004780 |
| generated_reward | 1.063521 | 1.181778 | 1.000000 | -4.285132 | 3.004780 |
| original_env_reward | -1.636045 | 2.433724 | 1.000000 | -100.000000 | 133.146437 |


# agent_memory
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -108.58 | -108.58 | 0.00 | 72.00 | progress_reward=0.160 soft_landing_bonus=0.012 stability_penalty=-0.342 | new_best |
| 2 | distance_anchor + landing_shaping + progress_reward + stability_penalty | -112.24 | -108.58 | -3.66 | 72.00 | distance_anchor=-0.485 landing_shaping=0.012 progress_reward=0.802 stability_penalty=-0.406 | no_meaningful_improvement |
| 3 | landing_shaping + progress_reward + stability_penalty | -101.17 | -101.17 | 0.00 | 72.00 | landing_shaping=0.013 progress_reward=0.160 stability_penalty=-0.338 | new_best |
| 4 | landing_shaping + progress_reward + stability_penalty | 67.07 | 67.07 | 0.00 | 841.00 | landing_shaping=2.049 progress_reward=0.206 stability_penalty=-0.079 | new_best |
| 5 | landing_shaping + progress_reward + stability_penalty | -113.70 | 67.07 | -180.78 | 72.00 | landing_shaping=0.011 progress_reward=1.291 stability_penalty=-0.238 | no_meaningful_improvement |

## Stable Lessons

- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
- distance_anchor can hurt performance when combined with strong progress_reward
- stability_penalty should not dominate total reward; keep mean magnitude below 0.2
- landing_shaping conditions should be relaxed to increase nonzero rate above 0.1
- progress_reward coefficient may need to be increased beyond 50 to achieve meaningful progress


# previous_reward.py (current, being analyzed)
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
    
    # --- Component 1: Progress Delta Reward (main learning signal, significantly increased) ---
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    # Increase coefficient from 50.0 to 80.0 to make progress dominate more
    progress_reward = 80.0 * progress_delta
    
    # --- Component 2: Stability Penalty (keep light, but slightly increase to prevent wild movement) ---
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.3 * abs(next_body_angle)  # was 0.2
    angular_vel_penalty = 0.15 * abs(next_angular_vel)  # was 0.1
    speed_penalty = 0.2 * speed  # was 0.15
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # --- Component 3: Landing Shaping (tighten conditions to prevent hacking, reduce coefficient) ---
    # Tighten near_target threshold from 1.0 to 0.6 to require being closer
    near_target = max(0.0, 1.0 - next_dist / 0.6)  # was 1.0
    # Tighten low_speed threshold from 1.0 to 0.5
    low_speed = max(0.0, 1.0 - speed / 0.5)  # was 1.0
    # Tighten stable_angle threshold from 0.5 to 0.3
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.3)  # was 0.5
    both_contact = 1.0 if (next_left_contact > 0.5 and next_right_contact > 0.5) else 0.0
    # Reduce shaping coefficient from 3.0 to 1.5 to prevent reward hacking
    landing_shaping = 1.5 * near_target * low_speed * stable_angle + 1.0 * both_contact * near_target * low_speed
    
    # --- Total Reward ---
    total_reward = progress_reward + stability_penalty + landing_shaping
    
    # --- Components dict ---
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# best_reward.py (historical highest score)
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
    
    # --- Component 1: Progress Delta Reward (main learning signal, increased) ---
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    # Increase coefficient from 10.0 to 50.0 to make progress dominate
    progress_reward = 50.0 * progress_delta
    
    # --- Component 2: Stability Penalty (reduced weights to avoid dominating) ---
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    # Reduce all penalties significantly so agent is not afraid to move
    angle_penalty = 0.2 * abs(next_body_angle)  # was 0.4
    angular_vel_penalty = 0.1 * abs(next_angular_vel)  # was 0.2
    speed_penalty = 0.15 * speed  # was 0.3
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # --- Component 3: Landing Shaping (relaxed conditions for higher activation) ---
    # Relax near_target threshold from 0.5 to 1.0 to activate earlier
    near_target = max(0.0, 1.0 - next_dist / 1.0)  # was 0.5
    # Relax low_speed threshold from 0.5 to 1.0
    low_speed = max(0.0, 1.0 - speed / 1.0)  # was 0.5
    # Relax stable_angle threshold from 0.3 to 0.5
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.5)  # was 0.3
    both_contact = 1.0 if (next_left_contact > 0.5 and next_right_contact > 0.5) else 0.0
    # Increase shaping coefficient to make it more impactful
    landing_shaping = 3.0 * near_target * low_speed * stable_angle + 2.0 * both_contact * near_target * low_speed
    
    # --- Total Reward ---
    total_reward = progress_reward + stability_penalty + landing_shaping
    
    # --- Components dict ---
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


# known_failure_modes
speed_penalty_dominance, stability_penalty_dominance, sparse_completion_proxy, early_failure_or_crash, high_reward_without_success, goal_near_oscillation, contact_reward_hacking, generated_reward_negative_average

# known_hacking_risks
speed_penalty_dominance, stability_penalty_dominance, sparse_completion_proxy, early_failure_or_crash, high_reward_without_success, goal_near_oscillation, contact_reward_hacking, generated_reward_negative_average

Based on the evidence above, output a diagnostic JSON.
If best_reward.py is provided and scored higher, compare its coefficients to previous_reward.py
and identify exactly which changes caused the regression. Recommend whether to revert.

```
