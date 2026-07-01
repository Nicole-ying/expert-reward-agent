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
reason: 任务描述明确要求"到达并稳定在中央目标着陆平台"，涉及位置到达、速度控制、姿态稳定，属于典型的导航到达任务。同时包含燃料效率优化（使用尽可能少的引擎推力），但核心目标是到达目标位置。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（推测）
- obs[0]: x_position - 相对于目标着陆平台的水平坐标
- obs[1]: y_position - 相对于着陆平台高度的垂直坐标
- obs[2]: x_velocity - 水平线速度
- obs[3]: y_velocity - 垂直线速度
- obs[4]: body_angle - 机体姿态角
- obs[5]: angular_velocity - 角速度
- obs[6]: left_support_contact - 左侧支撑接触标志（1.0表示接触，0.0表示未接触）
- obs[7]: right_support_contact - 右侧支撑接触标志（1.0表示接触，0.0表示未接触）

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine - 不执行任何操作
- action 1: left_orientation_engine - 点火左侧姿态发动机
- action 2: main_engine - 点火主发动机
- action 3: right_orientation_engine - 点火右侧姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（机体稳定/静止在目标区域，可能表示成功着陆）
- failure-like termination: crash_or_body_contact（坠毁或机体接触地面，可能表示失败）
- ambiguous termination: horizontal_position_outside_viewport（水平位置超出视口，可能是失败或边界条件）
- truncation: 无显式截断（step返回的truncated固定为False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info字典为空{}，无显式成功标志）
- explicit_failure_flag_available: false（info字典为空{}，无显式失败标志）
- allowed_info_fields: 无（info为空字典）
- forbidden_or_uncertain_info_fields: 所有info字段（因为info为空）

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


# previous_reward.py (score: -119.870233)

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 1. 主学习信号：progress_delta_reward ==========
    # 目标位置为 (0, 0)，因为 obs[0] 和 obs[1] 是相对于目标着陆平台的坐标
    # 当前距离平方
    current_dist_sq = obs[0]**2 + obs[1]**2
    # 下一步距离平方
    next_dist_sq = next_obs[0]**2 + next_obs[1]**2
    
    # 距离减少为正奖励，距离增加为负奖励
    progress_delta = current_dist_sq - next_dist_sq
    # 缩放因子，使奖励值在合理范围
    progress_scale = 2.0
    progress_delta_reward = progress_delta * progress_scale
    
    # ========== 2. 稳定/安全约束：stability_penalty ==========
    # 惩罚高速、大姿态角和大角速度
    # 使用 next_obs 因为这是动作执行后的状态
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 速度惩罚（鼓励减速）
    speed = (x_vel**2 + y_vel**2) ** 0.5
    speed_penalty_weight = 0.1
    speed_penalty = -speed_penalty_weight * speed
    
    # 姿态角惩罚（鼓励保持水平）
    angle_penalty_weight = 0.05
    angle_penalty = -angle_penalty_weight * abs(body_angle)
    
    # 角速度惩罚（鼓励稳定）
    angular_vel_penalty_weight = 0.02
    angular_vel_penalty = -angular_vel_penalty_weight * abs(angular_vel)
    
    stability_penalty = speed_penalty + angle_penalty + angular_vel_penalty
    
    # ========== 3. 任务完成 proxy：soft_landing_proxy ==========
    # 当接近目标、低速、姿态稳定且双支撑接触时给予小奖励
    near_target_threshold = 0.3  # 距离阈值
    low_speed_threshold = 0.2    # 速度阈值
    stable_angle_threshold = 0.1  # 角度阈值
    
    near_target = next_dist_sq < near_target_threshold**2
    low_speed = speed < low_speed_threshold
    stable_angle = abs(body_angle) < stable_angle_threshold
    both_contact = next_obs[6] > 0.5 and next_obs[7] > 0.5
    
    soft_landing_bonus_weight = 1.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_proxy = soft_landing_bonus_weight
    else:
        soft_landing_proxy = 0.0
    
    # ========== 4. 动作代价（小权重） ==========
    # 轻微惩罚使用引擎，鼓励燃料效率
    # action 0 是无引擎，其他动作使用引擎
    action_penalty_weight = 0.01
    if action == 0:
        action_penalty = 0.0
    else:
        action_penalty = -action_penalty_weight
    
    # ========== 组合总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy + action_penalty
    
    # ========== 构建 components dict ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "action_penalty": action_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# iteration_context.md

# Iteration Context

## Recommended Action
**tune** — 当前骨架仅运行1轮，得分-119.87远低于目标200。best_reward与previous_reward完全相同，无需revert。主要问题是stability_penalty和speed_penalty权重过大（-0.1086和-0.1056），压制了progress_delta_reward的正信号（0.054）。soft_landing_proxy触发率极低（0.0043），条件过严。建议降低speed_penalty_weight和angle_penalty_weight，放宽soft_landing_proxy条件，并增加progress_scale以增强学习信号。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -119.87 | -119.87 | 0.00 | 73.90 | action_penalty=-0.002 angle_penalty=-0.002 angular_vel_penalty=-0.001 progress_delta_reward=0.054 soft_landing_proxy=0.004 | new_best |

## Expert Cards
## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## Stable Lessons (from previous iterations)
- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| action_penalty | -0.001973 | 0.001973 | 0.197301 | -0.010000 | 0.000000 |
| angle_penalty | -0.001591 | 0.001591 | 1.000000 | -0.208482 | -0.000000 |
| angular_vel_penalty | -0.001434 | 0.001434 | 0.999825 | -0.167778 | -0.000000 |
| progress_delta_reward | 0.053915 | 0.057292 | 0.999991 | -0.096340 | 0.104236 |
| soft_landing_proxy | 0.004321 | 0.004321 | 0.004321 | 0.000000 | 1.000000 |
| speed_penalty | -0.105574 | 0.105574 | 0.999853 | -0.236266 | -0.000000 |
| stability_penalty | -0.108599 | 0.108599 | 1.000000 | -0.520032 | -0.000000 |
| total_reward | -0.052336 | 0.061855 | 1.000000 | -0.616372 | 0.999728 |
| generated_reward | -0.052336 | 0.061855 | 1.000000 | -0.616372 | 0.999728 |
| original_env_reward | -1.569260 | 2.337791 | 1.000000 | -100.000000 | 125.034054 |

```
