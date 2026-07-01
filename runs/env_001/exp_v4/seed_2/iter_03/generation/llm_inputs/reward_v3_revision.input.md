# Environment

# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器/着陆器轨迹优化任务。一个飞行器从视口顶部中央附近开始，受到初始随机力的作用。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求"到达并稳定在中央目标着陆平台"，核心目标是导航到目标位置并稳定着陆，同时优化燃料消耗。这符合导航目标到达任务的核心特征——到达特定目标位置并保持稳定状态。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推测)
- obs[0]: x_position - 相对于目标着陆平台的水平坐标
- obs[1]: y_position - 相对于着陆平台高度的垂直坐标
- obs[2]: x_velocity - 水平线速度
- obs[3]: y_velocity - 垂直线速度
- obs[4]: body_angle - 机体姿态角
- obs[5]: angular_velocity - 角速度
- obs[6]: left_support_contact - 左侧支撑接触标志 (1.0 表示接触, 0.0 表示未接触)
- obs[7]: right_support_contact - 右侧支撑接触标志 (1.0 表示接触, 0.0 表示未接触)

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine - 不执行任何操作
- action 1: left_orientation_engine - 点火左侧姿态发动机
- action 2: main_engine - 点火主发动机
- action 3: right_orientation_engine - 点火右侧姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled - 机体不再运动或已稳定着陆，可能是成功终止
- failure-like termination: crash_or_body_contact - 坠毁或机体接触（非正常着陆接触），可能是失败终止
- ambiguous termination: horizontal_position_outside_viewport - 水平位置超出视口范围，可能是失败（飞出边界）
- truncation: 无显式截断（step返回的truncated始终为False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false - 没有明确的成功标志
- explicit_failure_flag_available: false - 没有明确的失败标志
- allowed_info_fields: {} - step返回的info为空字典，无可用info字段
- forbidden_or_uncertain_info_fields: 所有info字段均不可用

# environment_contract

- function_signature: def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
- allowed: obs[0..7], next_obs[0..7], action (0=noop,1=left,2=main,3=right), info (no reliable fields)
- forbidden: original_reward, official_reward, fitness_score, individual_reward, info['success'], info['failure'], info['termination_reason']
- terminal_success_reward and terminal_failure_penalty remain blocked (no explicit success/failure flag available).


# Expert Knowledge

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


# previous_reward.py (score: -112.242148)

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
    
    # --- Component 1: Progress Delta Reward (strong main signal) ---
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    # Increased coefficient from 10 to 50 to dominate learning
    progress_reward = 50.0 * progress_delta
    
    # --- Component 2: Distance Anchor (small continuous shaping to prevent oscillation) ---
    # Negative distance as a gentle pull toward goal; small weight to avoid dominating
    distance_anchor = -0.5 * next_dist
    
    # --- Component 3: Conditional Stability Penalty (reduced and near-target only) ---
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    # Only penalize stability when far from target; near target we want gentle behavior
    far_from_target = next_dist > 0.8
    if far_from_target:
        # Light penalty far away to allow movement
        angle_penalty = 0.2 * abs(next_body_angle)
        angular_vel_penalty = 0.1 * abs(next_angular_vel)
        speed_penalty = 0.2 * speed
        stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    else:
        # Near target: stronger stability penalty to encourage settling
        angle_penalty = 0.5 * abs(next_body_angle)
        angular_vel_penalty = 0.3 * abs(next_angular_vel)
        speed_penalty = 0.5 * speed
        stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # --- Component 4: Soft Landing Proxy (continuous shaping instead of sparse bonus) ---
    # Continuous reward for being near target with low speed and stable angle
    near_target = max(0.0, 1.0 - next_dist / 0.5)  # 1.0 when dist=0, 0.0 when dist>=0.5
    low_speed = max(0.0, 1.0 - speed / 0.5)  # 1.0 when speed=0, 0.0 when speed>=0.5
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.3)  # 1.0 when angle=0, 0.0 when angle>=0.3
    # Contact bonus only when already in good condition
    both_contact = 1.0 if (next_left_contact > 0.5 and next_right_contact > 0.5) else 0.0
    landing_shaping = 2.0 * near_target * low_speed * stable_angle + 1.0 * both_contact * near_target * low_speed
    
    # --- Total Reward ---
    total_reward = progress_reward + distance_anchor + stability_penalty + landing_shaping
    
    # --- Components dict ---
    components = {
        "progress_reward": progress_reward,
        "distance_anchor": distance_anchor,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# best_reward.py (score: -108.58)

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
    # Distance to goal (origin) at current step
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    # Distance to goal at next step
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    # Progress: positive when moving closer to goal
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # --- Component 2: Stability Penalty (light constraint) ---
    # Penalize high speed, large angle, and high angular velocity
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * abs(next_body_angle)
    angular_vel_penalty = 0.2 * abs(next_angular_vel)
    speed_penalty = 0.3 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # --- Component 3: Soft Landing Proxy (small bonus for good landing conditions) ---
    # Conditions: near target, low speed, stable angle, both supports contacting
    near_target = next_dist < 0.5
    low_speed = speed < 0.5
    stable_angle = abs(next_body_angle) < 0.3
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # --- Total Reward ---
    total_reward = progress_reward + stability_penalty + soft_landing_bonus
    
    # --- Components dict ---
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# iteration_context.md

# Iteration Context

## Recommended Action
**revert** — Best score (-108.58) is higher than current (-112.24). The main change was increasing progress_reward coefficient from 10 to 50 and adding distance_anchor. The increase likely caused oscillation (goal_near_oscillation) and the anchor added negative bias. Revert to best_reward.py coefficients (progress=10, no distance_anchor) and only tune stability_penalty or landing_shaping slightly.

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -108.58 | -108.58 | 0.00 | 72.00 | progress_reward=0.160 soft_landing_bonus=0.012 stability_penalty=-0.342 | new_best |
| 2 | distance_anchor + landing_shaping + progress_reward + stability_penalty | -112.24 | -108.58 | -3.66 | 72.00 | distance_anchor=-0.485 landing_shaping=0.012 progress_reward=0.802 stability_penalty=-0.406 | no_meaningful_improvement |

## Expert Cards
## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## high_reward_without_success
- signal: generated_reward improves but external score stays poor
- risk: policy optimizes the custom reward but not the real task
- fix: reduce exploitable terms; add constraints tied to actual task progress and stable outcome

## Stable Lessons (from previous iterations)
- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.485057 | 0.485057 | 1.000000 | -0.861736 | -0.000282 |
| landing_shaping | 0.011906 | 0.011906 | 0.007295 | 0.000000 | 2.817319 |
| progress_reward | 0.802278 | 0.848786 | 0.999995 | -2.037974 | 2.122748 |
| stability_penalty | -0.406204 | 0.406204 | 1.000000 | -3.496580 | -0.000000 |
| total_reward | -0.077077 | 0.457257 | 1.000000 | -5.329068 | 2.802108 |
| generated_reward | -0.077077 | 0.457257 | 1.000000 | -5.329068 | 2.802108 |
| original_env_reward | -1.598657 | 2.439755 | 1.000000 | -100.000000 | 139.652583 |
