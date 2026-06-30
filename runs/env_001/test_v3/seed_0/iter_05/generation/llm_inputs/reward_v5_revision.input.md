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


# expert_reward_context.md

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
    
    # ========== Component 1: Progress Delta Reward (main learning signal, increased) ==========
    # Distance to target at current step
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    # Distance to target at next step
    dist_next = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    # Progress: positive when moving closer to target
    progress_delta = dist_current - dist_next
    # Increased coefficient to make progress the dominant signal
    progress_reward = 100.0 * progress_delta
    
    # ========== Component 2: Stability Penalty (light constraint, unchanged) ==========
    # Penalize high speed, large angle, and high angular velocity
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = 0.05 * abs(body_angle)
    angular_penalty = 0.02 * abs(angular_vel)
    speed_penalty = 0.1 * speed
    
    stability_penalty = -(angle_penalty + angular_penalty + speed_penalty)
    
    # ========== Component 3: Soft Landing Shaping (reduced, gentle nudge) ==========
    # Continuous shaping that rewards being near target, low speed, stable angle, and both contacts
    near_target_score = max(0.0, 1.0 - dist_next / 0.5)  # 1.0 when dist=0, 0.0 when dist>=0.5
    low_speed_score = max(0.0, 1.0 - speed / 0.3)  # 1.0 when speed=0, 0.0 when speed>=0.3
    stable_angle_score = max(0.0, 1.0 - abs(body_angle) / 0.2)  # 1.0 when angle=0, 0.0 when angle>=0.2
    both_contact_score = 1.0 if (left_contact > 0.5) and (right_contact > 0.5) else 0.0
    
    # Reduced coefficient from 5.0 to 2.0 to make it a gentle nudge
    landing_shaping = 2.0 * near_target_score * low_speed_score * stable_angle_score * (0.5 + 0.5 * both_contact_score)
    
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

# best_reward.py (historical best, for reference)

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

# iteration_context.md

# Iteration Context

## Recommended Action
**tune** — 本轮外部得分-142.80，远低于目标200。progress_reward均值1.599，信号强但不足以克服stability_penalty和原始环境惩罚。landing_shaping触发率仅1%，稀疏无效。历史中骨架迭代2次，最佳得分158.82（第3轮），但第4轮大幅下降，表明不稳定。建议：增大progress_reward系数至200以上，降低stability_penalty系数至0.05，提高landing_shaping系数至5.0并放宽条件以增加触发率。

## Agent Memory
| iter | score | best | skeleton_summary | trend |
|------|-------|------|------------------|-------|

## Expert Cards
## early_failure_or_crash
- signal: negative external score and short episode length
- risk: reward does not guide stable control before termination
- fix: add smooth approach/landing signals; avoid relying on sparse terminal-like proxy

## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## Training Evidence
# Training Feedback

## External evaluation
- score: -142.797815
- episode_length: 88.100000 (mean)
- range: [-206.654021, -3.831011]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| landing_shaping | 0.011850 | 0.011850 | 0.010386 | 0.000000 | 1.981661 |
| progress_reward | 1.598827 | 1.671059 | 0.999991 | -3.217669 | 5.221412 |
| stability_penalty | -0.122471 | 0.122471 | 1.000000 | -0.516735 | -0.000000 |
| total_reward | 1.488206 | 1.571583 | 1.000000 | -3.422873 | 4.793724 |
| generated_reward | 1.488206 | 1.571583 | 1.000000 | -3.422873 | 4.793724 |
| original_env_reward | -1.538719 | 3.915060 | 1.000000 | -100.000000 | 156.650270 |

## Signals
early_failure_or_crash; sparse_proxy:landing_shaping; penalty_dominance:generated_reward; penalty_dominance:original_env_reward
