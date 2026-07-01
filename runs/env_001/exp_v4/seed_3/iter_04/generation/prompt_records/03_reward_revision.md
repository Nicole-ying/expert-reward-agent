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
reason: 任务明确要求"到达并稳定在中央目标着陆平台"，核心是导航到目标位置并稳定停靠，同时优化燃料消耗。这符合导航目标到达任务的定义。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推断)
- obs[0]: x_position — 相对于目标着陆平台的水平坐标
- obs[1]: y_position — 相对于平台高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体朝向角度
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左侧支撑接触标志（0或1）
- obs[7]: right_support_contact — 右侧支撑接触标志（0或1）

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine — 不做任何操作
- action 1: left_orientation_engine — 启动左侧姿态引擎
- action 2: main_engine — 启动主引擎
- action 3: right_orientation_engine — 启动右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled — 机体停止运动并稳定在平台上，可能是成功着陆的标志
- failure-like termination: crash_or_body_contact — 坠毁或非正常机体接触；horizontal_position_outside_viewport — 水平位置超出视口范围
- ambiguous termination: 无
- truncation: 无（step返回的truncated始终为False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false — 没有明确的success标志
- explicit_failure_flag_available: false — 没有明确的failure标志
- allowed_info_fields: 无（info始终为空字典{}）
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


# previous_reward.py (score: -110.508541)

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. 主学习信号：progress_delta_reward（恢复到best的2.0系数）
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 2.0
    progress_reward = progress_delta * progress_scale
    
    # 2. 稳定约束：stability_penalty（恢复到best的系数）
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.5
    angular_vel_penalty = abs(next_ang_vel) * 0.3
    speed_penalty = speed * 0.2
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 任务完成proxy：soft_landing_proxy（恢复到best的严格条件，奖励从1.0小幅提升到1.5）
    near_target = next_dist < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        landing_bonus = 1.5  # 从1.0小幅提升到1.5，鼓励完成着陆
    
    # 4. 动作代价：energy_penalty（保持不变）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -engine_use * 0.1
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# best_reward.py (score: -110.51)

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. 主学习信号：progress_delta_reward
    # 计算当前位置到目标(0,0)的距离
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 2.0
    progress_reward = progress_delta * progress_scale
    
    # 2. 稳定约束：stability_penalty
    # 惩罚速度、姿态角和角速度
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.5
    angular_vel_penalty = abs(next_ang_vel) * 0.3
    speed_penalty = speed * 0.2
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 任务完成proxy：soft_landing_proxy
    # 当接近目标、低速、姿态稳定且双接触时给予小奖励
    near_target = next_dist < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        landing_bonus = 1.0
    
    # 4. 动作代价：energy_penalty（小权重）
    # 惩罚使用引擎（action != 0）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -engine_use * 0.1
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# iteration_context.md

# Iteration Context

## Recommended Action
**tune** — 当前骨架已运行3轮，得分稳定在-110左右，无显著改善。best_reward.py与previous_reward.py几乎相同（仅landing_bonus从1.0提升到1.5），但best得分-111.98略低于当前-110.51，因此无需revert。主要问题是stability_penalty系数过高（0.5,0.3,0.2）导致其主导总奖励，建议降低系数（如0.2,0.1,0.1）以鼓励探索；同时progress_reward系数2.0可能过低，可尝试提升至5.0或10.0以增强学习信号；landing_bonus条件可适当放宽（如dist<0.8, speed<0.5, angle<0.3）以提高触发率。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -111.98 | -111.98 | 0.00 | 70.60 | energy_penalty=-0.012 landing_bonus=0.005 progress_reward=0.032 stability_penalty=-0.243 | new_best |
| 2 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -112.65 | -111.98 | -0.67 | 70.60 | energy_penalty=-0.010 landing_bonus=0.014 progress_reward=0.161 stability_penalty=-0.120 | no_meaningful_improvement |
| 3 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -110.51 | -111.98 | 1.47 | 70.60 | energy_penalty=-0.013 landing_bonus=0.008 progress_reward=0.032 stability_penalty=-0.242 | no_meaningful_improvement |

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
- stability_penalty coefficients should be tuned to avoid dominating the total reward
- progress_reward coefficient increase from 2.0 to 10.0 did not improve score, likely due to goal_near_oscillation
- stability_penalty coefficients reduction from (0.5,0.3,0.2) to (0.2,0.1,0.1) reduced penalty but did not improve overall performance
- landing_bonus condition relaxation (dist<0.8, speed<0.5, angle<0.3) and reward increase to 2.0 did not increase landing rate (nonzero rate still very low)

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| energy_penalty | -0.012977 | 0.012977 | 0.129772 | -0.100000 | -0.000000 |
| landing_bonus | 0.008281 | 0.008281 | 0.005521 | 0.000000 | 1.500000 |
| progress_reward | 0.032392 | 0.034250 | 0.999992 | -0.080574 | 0.084193 |
| stability_penalty | -0.241722 | 0.241722 | 1.000000 | -3.432804 | -0.000000 |
| total_reward | -0.214026 | 0.229844 | 1.000000 | -3.572376 | 1.499441 |
| generated_reward | -0.214026 | 0.229844 | 1.000000 | -3.572376 | 1.499441 |
| original_env_reward | -1.581722 | 2.299603 | 1.000000 | -100.000000 | 128.588050 |

```
