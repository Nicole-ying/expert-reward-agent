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
这是一个2D飞行器/着陆器轨迹优化任务。一个物体从视口顶部中央附近开始，带有初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、降低速度、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求"到达并稳定在中央目标着陆平台"，核心目标是导航到目标位置并稳定停靠，同时优化燃料消耗（引擎推力使用）。这符合导航到达任务的核心特征：目标位置到达 + 速度/姿态控制。

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
- success-like termination: body_not_awake_or_settled - 当机体停止运动并稳定在目标位置时触发，可能是成功终止
- failure-like termination: crash_or_body_contact - 坠毁或机体接触地面（非目标平台），可能是失败终止
- ambiguous termination: horizontal_position_outside_viewport - 水平位置超出视口边界，可能是失败或截断
- truncation: 无显式截断条件，但视口边界外可视为截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false - step返回的info为空字典{}，无显式成功标志
- explicit_failure_flag_available: false - step返回的info为空字典{}，无显式失败标志
- allowed_info_fields: 无 - info字典为空，无可用字段
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


# previous_reward.py (score: -118.814912)

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    # 当前位置
    x_pos = obs[0]
    y_pos = obs[1]
    # 下一步位置
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    # 当前速度
    x_vel = obs[2]
    y_vel = obs[3]
    # 姿态
    body_angle = obs[4]
    angular_vel = obs[5]
    # 接触标志
    left_contact = obs[6]
    right_contact = obs[7]
    
    # 下一步速度（用于稳定性惩罚）
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    
    # 计算到目标(0,0)的距离
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    # ========== 组件1: 主学习信号 - progress_delta_reward ==========
    # 奖励每一步更接近目标
    progress_delta = current_dist - next_dist
    progress_scale = 10.0
    progress_reward = progress_delta * progress_scale
    
    # ========== 组件2: 稳定约束 - stability_penalty ==========
    # 惩罚高速、大姿态角和大角速度
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle)
    angular_vel_penalty = abs(next_angular_vel)
    
    stability_scale = 0.5
    speed_scale = 0.3
    angle_scale = 0.2
    angular_vel_scale = 0.1
    
    stability_penalty = -(
        speed_scale * speed +
        angle_scale * angle_penalty +
        angular_vel_scale * angular_vel_penalty
    ) * stability_scale
    
    # ========== 组件3: 任务完成proxy - soft_landing_proxy ==========
    # 当接近目标、低速、小角度且双接触时给予小奖励
    near_target_threshold = 0.3
    low_speed_threshold = 0.2
    stable_angle_threshold = 0.2
    
    is_near_target = next_dist < near_target_threshold
    is_low_speed = speed < low_speed_threshold
    is_stable_angle = abs(next_body_angle) < stable_angle_threshold
    is_both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if is_near_target and is_low_speed and is_stable_angle and is_both_contact:
        soft_landing_bonus = 2.0
    
    # ========== 组件4: 动作代价 - energy_penalty (小权重) ==========
    # 使用引擎（动作1,2,3）时给予小惩罚
    engine_actions = [1, 2, 3]
    energy_penalty_scale = 0.1
    energy_penalty = 0.0
    if action in engine_actions:
        energy_penalty = -energy_penalty_scale
    
    # ========== 总奖励 ==========
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    # 构建组件字典
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# best_reward.py (score: -18.56)

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 1. 主学习信号：progress_delta_reward（大幅增强） ==========
    current_dist_sq = obs[0]**2 + obs[1]**2
    next_dist_sq = next_obs[0]**2 + next_obs[1]**2
    progress_delta = current_dist_sq - next_dist_sq
    progress_scale = 80.0  # 从5.0大幅提升到80.0，增强学习信号
    progress_delta_reward = progress_delta * progress_scale
    
    # ========== 2. 稳定/安全约束：stability_penalty（保持低权重） ==========
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    speed = (x_vel**2 + y_vel**2) ** 0.5
    speed_penalty_weight = 0.02
    speed_penalty = -speed_penalty_weight * speed
    
    angle_penalty_weight = 0.01
    angle_penalty = -angle_penalty_weight * abs(body_angle)
    
    angular_vel_penalty_weight = 0.005
    angular_vel_penalty = -angular_vel_penalty_weight * abs(angular_vel)
    
    stability_penalty = speed_penalty + angle_penalty + angular_vel_penalty
    
    # ========== 3. 任务完成 proxy：soft_landing_proxy（收紧条件，改为连续塑形） ==========
    # 收紧条件以解决contact_reward_hacking，同时保持连续塑形
    near_target_threshold = 0.3  # 从0.5收紧到0.3
    low_speed_threshold = 0.3    # 从0.5收紧到0.3
    stable_angle_threshold = 0.2  # 从0.3收紧到0.2
    
    near_target = next_dist_sq < near_target_threshold**2
    low_speed = speed < low_speed_threshold
    stable_angle = abs(body_angle) < stable_angle_threshold
    both_contact = next_obs[6] > 0.5 and next_obs[7] > 0.5
    
    # 连续塑形：条件越严格满足，奖励越大
    soft_landing_bonus_weight = 3.0  # 从2.0提升到3.0，补偿收紧条件
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_proxy = soft_landing_bonus_weight
    elif near_target and low_speed and stable_angle:
        soft_landing_proxy = soft_landing_bonus_weight * 0.5
    elif near_target and low_speed:
        soft_landing_proxy = soft_landing_bonus_weight * 0.2
    else:
        soft_landing_proxy = 0.0
    
    # ========== 4. 动作代价（保持小权重） ==========
    action_penalty_weight = 0.005
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
**revert** — 当前得分-118.81远低于best得分-18.56（差距>100）。对比代码：best使用progress_scale=80.0，当前仅10.0；best的stability_penalty权重总和约0.035，当前约0.5（speed_scale=0.3, angle_scale=0.2, angular_vel_scale=0.1乘以stability_scale=0.5），导致stability_penalty主导；best的soft_landing_proxy使用连续塑形（多级奖励），当前仅单一阈值2.0且触发率极低。建议恢复到best配置，仅做小幅调整。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -119.87 | -119.87 | 0.00 | 73.90 | action_penalty=-0.002 angle_penalty=-0.002 angular_vel_penalty=-0.001 progress_delta_reward=0.054 soft_landing_proxy=0.004 | new_best |
| 2 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -27.09 | -27.09 | 0.00 | 1000.00 | action_penalty=-0.004 angle_penalty=-0.002 angular_vel_penalty=-0.001 progress_delta_reward=0.031 soft_landing_proxy=0.957 | new_best |
| 3 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -18.56 | -18.56 | 0.00 | 800.60 | action_penalty=-0.004 angle_penalty=-0.004 angular_vel_penalty=-0.002 progress_delta_reward=1.416 soft_landing_proxy=0.511 | new_best |
| 4 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -48.55 | -18.56 | -29.99 | 1000.00 | action_penalty=-0.004 angle_penalty=-0.003 angular_vel_penalty=-0.001 progress_delta_reward=0.915 soft_landing_proxy=1.793 | no_meaningful_improvement |
| 5 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -18.56 | -18.56 | 0.00 | 800.60 | action_penalty=-0.004 angle_penalty=-0.004 angular_vel_penalty=-0.002 progress_delta_reward=1.416 soft_landing_proxy=0.511 | no_meaningful_improvement |
| 6 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -48.55 | -18.56 | -29.99 | 1000.00 | action_penalty=-0.004 angle_penalty=-0.003 angular_vel_penalty=-0.001 progress_delta_reward=0.915 soft_landing_proxy=1.793 | unsolved_stagnation_fresh_restart |
| 7 | energy_penalty + progress_reward + soft_landing_bonus + stability_penalty | -118.81 | -18.56 | -100.26 | 74.00 | energy_penalty=-0.008 progress_reward=0.162 soft_landing_bonus=0.008 stability_penalty=-0.167 | no_meaningful_improvement |

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
- stability_penalty权重过大（-0.1086）会主导总奖励，抑制progress信号
- soft_landing_proxy触发率过低（0.0043）需放宽条件或增加权重
- soft_landing_proxy thresholds should be tightened to avoid contact_reward_hacking
- soft_landing_proxy触发率低（0.21），需放宽阈值或增加权重以提升学习信号
- progress_delta_reward系数80.0有效驱动学习，可保持
- soft_landing_proxy thresholds must be tight (<=0.3) to avoid contact_reward_hacking
- soft_landing_proxy weight should not exceed 3.0 when thresholds are tight

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| energy_penalty | -0.008259 | 0.008259 | 0.082590 | -0.100000 | 0.000000 |
| progress_reward | 0.161521 | 0.170844 | 0.999994 | -0.418984 | 0.421667 |
| soft_landing_bonus | 0.007932 | 0.007932 | 0.003966 | 0.000000 | 2.000000 |
| stability_penalty | -0.166909 | 0.166909 | 1.000000 | -0.776104 | -0.000000 |
| total_reward | -0.005714 | 0.078491 | 1.000000 | -1.064015 | 2.005625 |
| generated_reward | -0.005714 | 0.078491 | 1.000000 | -1.064015 | 2.005625 |
| original_env_reward | -1.640570 | 2.317459 | 1.000000 | -100.000000 | 128.534317 |

```
