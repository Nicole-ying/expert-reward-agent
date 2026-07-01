# Environment

# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器轨迹优化任务。一个飞行器从视口顶部中央附近开始，受到初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务明确要求"到达并稳定在中央目标平台"，核心是导航到目标位置并稳定停靠，同时优化燃料消耗，符合导航目标到达任务的定义。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推测)
- obs[0]: x_position - 相对于目标平台的水平坐标
- obs[1]: y_position - 相对于平台高度的垂直坐标
- obs[2]: x_velocity - 水平线速度
- obs[3]: y_velocity - 垂直线速度
- obs[4]: body_angle - 机体姿态角
- obs[5]: angular_velocity - 角速度
- obs[6]: left_support_contact - 左侧支撑接触标志 (0.0 或 1.0)
- obs[7]: right_support_contact - 右侧支撑接触标志 (0.0 或 1.0)

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine - 不执行任何操作
- action 1: left_orientation_engine - 启动左侧姿态引擎
- action 2: main_engine - 启动主引擎
- action 3: right_orientation_engine - 启动右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled - 当飞行器稳定停靠在目标平台上时触发，可能是成功终止
- failure-like termination: crash_or_body_contact - 发生碰撞或非预期机体接触，可能是失败终止
- failure-like termination: horizontal_position_outside_viewport - 水平位置超出视口范围，可能是失败终止
- ambiguous termination: 无明确区分成功/失败的单一终止标志
- truncation: False (step返回的truncated固定为False)

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无 (info字典为空)
- forbidden_or_uncertain_info_fields: 所有info字段均不可用，因为step返回空字典

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


# previous_reward.py (score: -108.495278)

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
    # Distance to goal (origin) at current and next step
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # --- Component 2: Stability Penalty (light constraint) ---
    # Penalize high speed, large angle, and high angular velocity
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = -0.5 * abs(body_angle)
    angular_vel_penalty = -0.3 * abs(angular_vel)
    speed_penalty = -0.2 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # --- Component 3: Soft Landing Proxy (task completion hint) ---
    # Small bonus when near target, low speed, stable angle, and both contacts
    near_target = next_dist < 0.5
    low_speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5 < 0.3
    stable_angle = abs(next_body_angle) < 0.2
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

# best_reward.py (score: 67.07)

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

# iteration_context.md

# Iteration Context

## Recommended Action
**revert** — 当前得分-108.5远低于历史最佳67.07（差距175.57）。对比best_reward.py，关键差异：progress_reward系数从50降为10，导致学习信号弱；stability_penalty权重过高（angle从0.2→0.5，angular从0.1→0.3，speed从0.15→0.2），导致惩罚主导；soft_landing_bonus条件过严且系数低（2.0 vs 3.0+2.0），激活率极低。建议恢复到best_reward的系数和组件结构（使用landing_shaping替代soft_landing_bonus），仅做小幅调整。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -108.58 | -108.58 | 0.00 | 72.00 | progress_reward=0.160 soft_landing_bonus=0.012 stability_penalty=-0.342 | new_best |
| 2 | distance_anchor + landing_shaping + progress_reward + stability_penalty | -112.24 | -108.58 | -3.66 | 72.00 | distance_anchor=-0.485 landing_shaping=0.012 progress_reward=0.802 stability_penalty=-0.406 | no_meaningful_improvement |
| 3 | landing_shaping + progress_reward + stability_penalty | -101.17 | -101.17 | 0.00 | 72.00 | landing_shaping=0.013 progress_reward=0.160 stability_penalty=-0.338 | new_best |
| 4 | landing_shaping + progress_reward + stability_penalty | 67.07 | 67.07 | 0.00 | 841.00 | landing_shaping=2.049 progress_reward=0.206 stability_penalty=-0.079 | new_best |
| 5 | landing_shaping + progress_reward + stability_penalty | -113.70 | 67.07 | -180.78 | 72.00 | landing_shaping=0.011 progress_reward=1.291 stability_penalty=-0.238 | no_meaningful_improvement |
| 6 | landing_shaping + progress_reward + stability_penalty | 67.07 | 67.07 | 0.00 | 841.00 | landing_shaping=2.049 progress_reward=0.206 stability_penalty=-0.079 | no_meaningful_improvement |
| 7 | landing_shaping + progress_reward + stability_penalty | -17.49 | 67.07 | -84.57 | 1000.00 | landing_shaping=1.116 progress_reward=0.407 stability_penalty=-0.089 | unsolved_stagnation_fresh_restart |
| 8 | progress_reward + soft_landing_bonus + stability_penalty | -108.50 | 67.07 | -175.57 | 72.00 | progress_reward=0.160 soft_landing_bonus=0.012 stability_penalty=-0.230 | no_meaningful_improvement |

## Expert Cards
## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## sparse_completion_proxy
- signal: completion/landing bonus trigger_rate < 1%
- risk: final bonus provides little early learning guidance
- fix: replace hard bonus with smoother landing-quality shaping

## Stable Lessons (from previous iterations)
- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
- distance_anchor can hurt performance when combined with strong progress_reward
- stability_penalty should not dominate total reward; keep mean magnitude below 0.2
- landing_shaping conditions should be relaxed to increase nonzero rate above 0.1
- progress_reward coefficient may need to be increased beyond 50 to achieve meaningful progress
- landing_shaping coefficient should be at least 3.0 to provide meaningful signal
- landing_shaping thresholds should be relaxed (dist<=1.0, speed<=1.0, angle<=0.5) to achieve nonzero rate above 0.1
- progress_reward coefficient of 50 is sufficient; increasing to 80 may cause instability

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| progress_reward | 0.160368 | 0.169634 | 0.999992 | -0.408113 | 0.424100 |
| soft_landing_bonus | 0.011723 | 0.011723 | 0.005861 | 0.000000 | 2.000000 |
| stability_penalty | -0.229804 | 0.229804 | 1.000000 | -3.068187 | -0.000147 |
| total_reward | -0.057713 | 0.084969 | 1.000000 | -3.244047 | 2.001660 |
| generated_reward | -0.057713 | 0.084969 | 1.000000 | -3.244047 | 2.001660 |
| original_env_reward | -1.557388 | 2.363114 | 1.000000 | -100.000000 | 129.798128 |
