# Prompt Record

## System Prompt

```text
你是 Reward Revision Agent。根据训练证据和专家知识执行一次明确的修订。

你将看到：

1. environment_contract — 环境硬约束（可用信号、禁止字段、函数签名）。
2. expert_reward_context.md — 专家知识库上下文（任务类型 + 推荐骨架及其数学形态 + 风险）。
3. previous_reward.py — 上一轮奖励函数代码。
4. best_reward.py — 历史最高分奖励函数（仅当非当前轮时提供）。
5. iteration_context.md — 包含：
   - Recommended Action（分析 LLM 的建议动作和理由）
   - Agent Memory（历史表格）
   - Expert Cards（匹配到的失败模式修复卡片）
   - Training Evidence（组件证据表格和信号检测）

# 决策步骤

1. 看 Recommended Action — 分析 LLM 建议 tune / mix / rebuild？为什么？
2. 看 Agent Memory — 当前骨架试了几轮？趋势？
3. 看 Expert Cards — 专家建议怎么修？
4. 看 Training Evidence — 每个组件的实际均值和触发率。
5. 看 expert_reward_context.md — 知识库推荐哪些骨架？有没有数学形态更适合的？
6. 看 previous_reward.py [+ best_reward.py] → 决定 action，写代码。

# action

- revert：best_reward 得分明显更高时，恢复到 best 的系数配置，仅做小幅改动。
- tune：调系数/阈值/门控。
- add：加新组件。
- delete：删除有害/冗余组件。
- mix：tune+add+delete 组合。
- rebuild：换骨架。从 expert_reward_context.md 中选一个不同的数学形态。

# 约束

- 证据驱动，不堆砌。惩罚项主导 progress → 削弱或条件化。bonus 触发率 <1% → 改为连续 shaping。
- 如果 Recommended Action 是 rebuild，必须选不同骨架，不能返回同骨架的系数变体。
- 禁止 terminal_success_reward / terminal_failure_penalty（除非 contract 声明可用）。
- 禁止 original_reward、未声明 info 字段、import/class/try/except/eval/exec/open。

# 输出

直接输出 Python code。可以在代码前用注释简短说明改动。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    ...
    return float(total_reward), components
```
函数签名必须一致。components 含所有组件 + total_reward。不 import/class/try/except。

```

## User Prompt

```markdown
# Environment

# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器轨迹优化任务。一个飞行器从视口顶部中央附近开始，受到初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务明确要求到达并稳定在中央目标平台，属于典型的导航到达任务。同时包含速度控制、姿态稳定和燃料效率优化，但核心目标是到达目标位置。

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
- action 1: left_orientation_engine - 点火左侧姿态发动机
- action 2: main_engine - 点火主发动机
- action 3: right_orientation_engine - 点火右侧姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled - 机体停止运动并稳定在平台上，可能是成功着陆
- failure-like termination: crash_or_body_contact - 坠毁或非正常机体接触；horizontal_position_outside_viewport - 水平位置超出视口边界
- ambiguous termination: 无
- truncation: 无 (truncated 始终为 False)

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false - 没有明确的成功标志
- explicit_failure_flag_available: false - 没有明确的失败标志
- allowed_info_fields: 无 (info 字典为空 {})
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

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


# previous_reward.py (score: -110.358684)

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
    # Reward for moving closer to the target (0,0)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 5.0
    progress_reward = progress_delta * progress_scale
    
    # --- Component 2: Stability Penalty (light constraint) ---
    # Penalize high speed, large angle, and high angular velocity
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.5
    angular_vel_penalty = abs(next_angular_vel) * 0.3
    speed_penalty = speed * 0.2
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # --- Component 3: Soft Landing Proxy (small bonus) ---
    # Small bonus when near target, low speed, stable angle, and both contacts
    near_target = next_dist < 0.3
    low_speed = speed < 0.5
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # --- Component 4: Action Penalty (small efficiency cost) ---
    # Penalize using any engine (actions 1, 2, 3)
    action_penalty = 0.0
    if action != 0:
        action_penalty = -0.05
    
    # --- Total Reward ---
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + action_penalty
    
    # --- Components dict ---
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "action_penalty": action_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# best_reward.py (score: -108.86)

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 计算当前位置到目标(0,0)的距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist  # 正值表示更接近目标
    progress_delta_reward = 10.0 * progress_delta  # 主信号权重较大

    # ========== 稳定约束：stability_penalty ==========
    # 惩罚速度、姿态角和角速度，鼓励稳定接近
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = abs(next_obs[4])  # 姿态角偏离0的绝对值
    angular_vel_penalty = abs(next_obs[5])  # 角速度绝对值
    stability_penalty = -0.5 * speed - 0.3 * angle_penalty - 0.1 * angular_vel_penalty

    # ========== 任务完成 proxy：soft_landing_proxy ==========
    # 当接近目标、低速、姿态稳定且双支撑接触时给予小奖励
    near_target = next_dist < 0.3
    low_speed = speed < 0.2
    stable_angle = abs(next_obs[4]) < 0.2
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    soft_landing_proxy = 2.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # ========== 动作代价：energy_penalty（小权重） ==========
    # 使用引擎（action != 0）时给予小惩罚
    energy_penalty = -0.1 if action != 0 else 0.0

    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy + energy_penalty

    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# iteration_context.md

# Iteration Context

## Recommended Action
**revert** — Current score (-110.36) is worse than best (-111.64) but best_reward.py (score -111.64) uses progress_delta_reward with coefficient 10.0 and energy_penalty -0.1, while current uses progress_reward with coefficient 5.0 and action_penalty -0.05. The best configuration achieved higher score despite similar skeleton. Revert to best_reward.py coefficients and components, then consider tuning stability_penalty weights to reduce dominance.

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -111.64 | -111.64 | 0.00 | 73.60 | energy_penalty=-0.011 progress_delta_reward=0.161 soft_landing_proxy=0.008 stability_penalty=-0.549 | new_best |
| 2 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -120.28 | -111.64 | -8.64 | 73.40 | energy_penalty=-0.009 progress_delta_reward=0.324 soft_landing_proxy=0.015 stability_penalty=-0.222 | no_meaningful_improvement |
| 3 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -115.99 | -111.64 | -4.35 | 73.60 | energy_penalty=-0.010 progress_delta_reward=0.161 soft_landing_proxy=0.010 stability_penalty=-0.549 | no_meaningful_improvement |
| 4 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -114.12 | -111.64 | -2.48 | 73.60 | energy_penalty=-0.010 progress_delta_reward=0.193 soft_landing_proxy=0.008 stability_penalty=-0.550 | unsolved_stagnation_fresh_restart |
| 5 | action_penalty + progress_reward + soft_landing_bonus + stability_penalty | -110.80 | -111.64 | 0.84 | 73.60 | action_penalty=-0.006 progress_reward=0.081 soft_landing_bonus=0.010 stability_penalty=-0.344 | no_meaningful_improvement |
| 6 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -111.64 | -111.64 | 0.00 | 73.60 | energy_penalty=-0.011 progress_delta_reward=0.161 soft_landing_proxy=0.008 stability_penalty=-0.549 | no_meaningful_improvement |
| 7 | energy_penalty + landing_shaping + progress_delta_reward + stability_penalty | -108.86 | -111.64 | 2.79 | 73.60 | energy_penalty=-0.010 landing_shaping=0.085 progress_delta_reward=0.324 stability_penalty=-0.219 | unsolved_stagnation_fresh_restart |
| 8 | action_penalty + progress_reward + soft_landing_bonus + stability_penalty | -110.36 | -111.64 | 1.28 | 73.60 | action_penalty=-0.007 progress_reward=0.081 soft_landing_bonus=0.010 stability_penalty=-0.243 | no_meaningful_improvement |

## Expert Cards
## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## Stable Lessons (from previous iterations)
- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
- soft_landing_proxy with strict conditions may be too sparse to provide useful signal
- Increasing progress_delta_reward coefficient without adjusting stability can cause oscillation and score drop.
- Relaxing soft_landing_proxy conditions did not increase its trigger rate; it remains too sparse.
- Reducing stability penalty weight can lead to oscillation and score regression.
- Increasing soft_landing_proxy reward does not improve trigger rate if conditions remain strict.
- stability_penalty weights should prioritize speed over angle to avoid dominance
- stability_penalty coefficients must be balanced to avoid dominating progress signal
- soft_landing_proxy with strict conditions is too sparse; consider relaxing conditions or using a continuous proxy

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| action_penalty | -0.007419 | 0.007419 | 0.148378 | -0.050000 | 0.000000 |
| progress_reward | 0.080831 | 0.085478 | 0.999990 | -0.203761 | 0.211999 |
| soft_landing_bonus | 0.009917 | 0.009917 | 0.004959 | 0.000000 | 2.000000 |
| stability_penalty | -0.242780 | 0.242780 | 1.000000 | -3.219305 | -0.000001 |
| total_reward | -0.159451 | 0.178346 | 1.000000 | -3.362454 | 2.000603 |
| generated_reward | -0.159451 | 0.178346 | 1.000000 | -3.362454 | 2.000603 |
| original_env_reward | -1.577038 | 2.307488 | 1.000000 | -100.000000 | 139.173759 |

```
