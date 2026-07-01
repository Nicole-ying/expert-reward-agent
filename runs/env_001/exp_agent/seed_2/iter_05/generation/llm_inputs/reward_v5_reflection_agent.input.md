# 环境信息
# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器/着陆器轨迹优化任务。一个物体从视口顶部中央附近开始，带有初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务明确要求"到达并稳定在中央目标着陆平台"，具有明确的目标位置（着陆平台），同时涉及速度控制、姿态稳定和燃料效率优化，属于典型的导航到达任务。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（推断）
- obs[0]: x_position - 相对于目标着陆平台的水平坐标
- obs[1]: y_position - 相对于着陆平台高度的垂直坐标
- obs[2]: x_velocity - 水平线速度
- obs[3]: y_velocity - 垂直线速度
- obs[4]: body_angle - 机体姿态角
- obs[5]: angular_velocity - 角速度
- obs[6]: left_support_contact - 左侧支撑接触标志（1.0=接触，0.0=未接触）
- obs[7]: right_support_contact - 右侧支撑接触标志（1.0=接触，0.0=未接触）

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine - 不执行任何操作
- action 1: left_orientation_engine - 点火左侧姿态发动机
- action 2: main_engine - 点火主发动机
- action 3: right_orientation_engine - 点火右侧姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled - 机体停止运动并稳定，可能表示成功着陆
- failure-like termination: crash_or_body_contact - 坠毁或非正常机体接触；horizontal_position_outside_viewport - 水平位置超出视口边界
- ambiguous termination: 无
- truncation: 无（truncated 始终为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空，无显式成功标志）
- explicit_failure_flag_available: false（info 字典为空，无显式失败标志）
- allowed_info_fields: 无（info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用



# 环境契约
- function_signature: def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
- allowed: obs[0..7], next_obs[0..7], action (0=noop,1=left,2=main,3=right), info (no reliable fields)
- forbidden: original_reward, official_reward, terminal_success_reward, terminal_failure_penalty


# 专家知识骨架
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


# 当前奖励函数代码
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
    
    # 1. Main learning signal: progress_delta_reward (unchanged)
    # Agent is making progress toward target — keep this working signal
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty — INCREASED significantly
    # Agent crashes at ~72 steps because it falls over while moving toward target.
    # Current penalties (angle=-0.1, angular_vel=-0.05, speed=-0.03) are too weak.
    # Need stronger signal to teach upright posture during approach.
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -1.0 * abs(next_body_angle)       # was -0.1, 10x increase
    angular_vel_penalty = -0.5 * abs(next_angular_vel) # was -0.05, 10x increase
    speed_penalty = -0.2 * speed                        # was -0.03, ~7x increase
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. Soft landing proxy — INCREASED weight and relaxed conditions
    # nonzero_rate=1.8% is too low — agent almost never reaches landing state.
    # Increase bonus magnitude and make the factors less strict so agent gets
    # gradient signal earlier (before perfect landing conditions are met).
    proximity_factor = 1.0 / (1.0 + 3.0 * next_dist)   # was 5.0, now broader
    speed_factor = 1.0 / (1.0 + 3.0 * speed)            # was 5.0, now broader
    angle_factor = 1.0 / (1.0 + 5.0 * abs(next_body_angle))  # was 10.0, now broader
    angular_vel_factor = 1.0 / (1.0 + 3.0 * abs(next_angular_vel))  # was 5.0, now broader
    contact_factor = next_left_contact * next_right_contact  # continuous in [0,1]
    
    soft_landing_bonus = 20.0 * proximity_factor * speed_factor * angle_factor * angular_vel_factor * contact_factor  # was 10.0, now 20.0
    
    # 4. Small energy penalty for using engines (unchanged)
    energy_penalty = 0.0
    if action != 0:
        energy_penalty = -0.05
    
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# 历史最佳奖励函数代码
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
    
    # 1. Main learning signal: progress_delta_reward
    # Reward for moving closer to the target (origin in relative coordinates)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty: encourage low velocity, upright angle, low angular velocity
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.5 * abs(next_body_angle)
    angular_vel_penalty = -0.3 * abs(next_angular_vel)
    speed_penalty = -0.2 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. Soft landing proxy: small bonus when near target, stable, and both supports contact
    near_target = next_dist < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # 4. Small energy penalty for using engines (action != 0)
    energy_penalty = 0.0
    if action != 0:
        energy_penalty = -0.05
    
    # Combine rewards
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# 训练反馈
# Training Feedback

## Training outcome
score=-108.982652, len=72.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.015580 | 0.015580 | 0.311597 | -0.097896 |
| progress_delta_reward | 0.159147 | 0.168636 | 0.999993 | 1.000000 |
| soft_landing_bonus | 0.085496 | 0.085496 | 0.017304 | 0.537213 |
| stability_penalty | -0.270811 | 0.270811 | 1.000000 | -1.701641 |
| total_reward | -0.041748 | 0.205102 | 0.999999 | -0.262323 |
| generated_reward | -0.041748 | 0.205102 | 0.999999 | -0.262323 |
| original_env_reward | -1.506394 | 2.371567 | 1.000000 | -9.465418 |

## Distribution
- score: mean=-108.982652, min=-124.079149, max=-95.059093
- episode_length: mean=72.000000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -110.68 | -110.68 | 0.00 | 72.00 | energy_penalty=-0.008 progress_delta_reward=0.160 soft_landing_bonus=0.012 stability_penalty=-0.242 | new_best |
| 2 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -118.44 | -110.68 | -7.75 | 71.90 | energy_penalty=-0.005 progress_delta_reward=0.162 soft_landing_bonus=0.014 stability_penalty=-0.006 | no_meaningful_improvement |
| 3 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -111.05 | -110.68 | -0.37 | 72.00 | energy_penalty=-0.009 progress_delta_reward=0.160 soft_landing_bonus=0.033 stability_penalty=-0.039 | no_meaningful_improvement |
| 4 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -108.98 | -110.68 | 1.70 | 72.00 | energy_penalty=-0.016 progress_delta_reward=0.159 soft_landing_bonus=0.085 stability_penalty=-0.271 | no_meaningful_improvement |
