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

# Env_002 环境理解卡片

## 1. 任务目标
这是一个2D双足行走任务。智能体需要控制一个双足机器人在不平坦地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。智能体必须协调两条腿的髋关节和膝关节，产生稳定的双足步态。如果身体摔倒则终止回合。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 任务描述明确为"2D locomotion task"，目标是行走前进，动作空间是连续关节力矩控制，观察空间包含身体姿态、关节角度、速度等信息，属于典型的连续控制运动任务。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32 (推测)
- obs[0]: hull_angle - 主体相对于竖直方向的角度
- obs[1]: hull_angular_velocity - 主体的角速度
- obs[2]: horizontal_velocity - 前后线性速度
- obs[3]: vertical_velocity - 上下线性速度
- obs[4]: hip1_angle - 腿1髋关节角度
- obs[5]: hip1_speed - 腿1髋关节角速度
- obs[6]: knee1_angle - 腿1膝关节角度
- obs[7]: knee1_speed - 腿1膝关节角速度
- obs[8]: hip2_angle - 腿2髋关节角度
- obs[9]: hip2_speed - 腿2髋关节角速度
- obs[10]: knee2_angle - 腿2膝关节角度
- obs[11]: knee2_speed - 腿2膝关节角速度
- obs[12]: leg1_contact - 腿1地面接触标志（1.0=接触，0.0=无接触）
- obs[13]: leg2_contact - 腿2地面接触标志（1.0=接触，0.0=无接触）
- obs[14..23]: lidar_0..lidar_9 - 10个LIDAR测距仪沿前方地形的距离测量值

## 4. 动作空间 action_space
- type: Box (连续)
- shape: (4,)
- bounds: [-1.0, 1.0] 每个关节
- action 0: hip_torque_leg1 - 施加在腿1髋关节上的力矩
- action 1: knee_torque_leg1 - 施加在腿1膝关节上的力矩
- action 2: hip_torque_leg2 - 施加在腿2髋关节上的力矩
- action 3: knee_torque_leg2 - 施加在腿2膝关节上的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain - 到达地形终点，可视为成功完成
- failure-like termination: body_fallen_over - 身体摔倒，可视为失败
- ambiguous termination: 无
- truncation: 无（step返回的truncated始终为False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info字典为空{}，无显式成功标志）
- explicit_failure_flag_available: false（info字典为空{}，无显式失败标志）
- allowed_info_fields: 无（info始终为{}）
- forbidden_or_uncertain_info_fields: 所有info字段（因为info为空字典）

# environment_contract

- function_signature: def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
- allowed: obs[0..7], next_obs[0..7], action (0=noop,1=left,2=main,3=right), info (no reliable fields)
- forbidden: original_reward, official_reward, fitness_score, individual_reward, info['success'], info['failure'], info['termination_reason']
- terminal_success_reward and terminal_failure_penalty remain blocked (no explicit success/failure flag available).


# Expert Knowledge

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。

## 1. 任务路由摘要
- locomotion_continuous_control：按该任务类型选择主学习信号，并先检查接口可用性。

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


# previous_reward.py (score: -76.012264)

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward（大幅增强） ==========
    # 使用水平速度变化作为前进驱动力
    current_horizontal_vel = obs[2]
    next_horizontal_vel = next_obs[2]
    progress_delta = next_horizontal_vel - current_horizontal_vel
    # 系数从5.0提升至15.0，增强驱动力（证据显示均值仅0.012，远不足以驱动学习）
    progress_delta_reward = 15.0 * progress_delta

    # ========== 稳定约束：stability_penalty（适度增强） ==========
    # 惩罚身体倾斜角度过大和角速度过大，防止摔倒
    hull_angle = next_obs[0]  # 身体倾斜角度
    hull_angular_vel = next_obs[1]  # 身体角速度
    # 角度惩罚：仅当角度>0.2时开始惩罚，线性增长（阈值从0.3收紧至0.2）
    angle_penalty = -0.3 * max(0.0, abs(hull_angle) - 0.2)
    # 角速度惩罚：仅当角速度>0.3时开始惩罚，线性增长（阈值从0.5收紧至0.3）
    angular_vel_penalty = -0.2 * max(0.0, abs(hull_angular_vel) - 0.3)
    stability_penalty = angle_penalty + angular_vel_penalty

    # ========== 效率约束：energy_penalty（保持小权重） ==========
    action_magnitude = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.01 * action_magnitude

    # ========== 任务完成proxy：soft_landing_proxy（收紧条件，降低权重） ==========
    leg1_contact = next_obs[12]
    leg2_contact = next_obs[13]
    both_contact = 1.0 if (leg1_contact > 0.5 and leg2_contact > 0.5) else 0.0
    # 收紧直立条件：角度<0.2，角速度<0.5（防止contact_reward_hacking）
    is_upright = 1.0 if (abs(hull_angle) < 0.2 and abs(hull_angular_vel) < 0.5) else 0.0
    # 收紧速度条件：水平速度在0.5到3.0之间（防止低速震荡）
    is_good_speed = 1.0 if (0.5 < next_horizontal_vel < 3.0) else 0.0
    # 降低权重从0.5至0.2，减少稀疏bonus的干扰
    soft_landing_bonus = 0.2 * both_contact * is_upright * is_good_speed

    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + energy_penalty + soft_landing_bonus

    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# best_reward.py (score: 11.95)

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward（大幅增强） ==========
    # 使用水平速度变化作为前进驱动力
    current_horizontal_vel = obs[2]
    next_horizontal_vel = next_obs[2]
    progress_delta = next_horizontal_vel - current_horizontal_vel
    # 权重从2.0提升至5.0，增强驱动力
    progress_delta_reward = 5.0 * progress_delta

    # ========== 稳定约束：stability_penalty（大幅削弱） ==========
    # 惩罚身体倾斜角度过大和角速度过大，防止摔倒
    hull_angle = next_obs[0]  # 身体倾斜角度
    hull_angular_vel = next_obs[1]  # 身体角速度
    # 使用线性惩罚替代指数惩罚，避免极端负值
    # 角度惩罚：仅当角度>0.3时开始惩罚，线性增长
    angle_penalty = -0.1 * max(0.0, abs(hull_angle) - 0.3)
    # 角速度惩罚：仅当角速度>0.5时开始惩罚，线性增长
    angular_vel_penalty = -0.05 * max(0.0, abs(hull_angular_vel) - 0.5)
    stability_penalty = angle_penalty + angular_vel_penalty

    # ========== 效率约束：energy_penalty（保持小权重） ==========
    action_magnitude = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.01 * action_magnitude

    # ========== 任务完成proxy：soft_landing_proxy（放宽条件） ==========
    leg1_contact = next_obs[12]
    leg2_contact = next_obs[13]
    both_contact = 1.0 if (leg1_contact > 0.5 and leg2_contact > 0.5) else 0.0
    # 放宽直立条件：角度<0.4，角速度<1.0
    is_upright = 1.0 if (abs(hull_angle) < 0.4 and abs(hull_angular_vel) < 1.0) else 0.0
    # 放宽速度条件：水平速度在0.2到4.0之间
    is_good_speed = 1.0 if (0.2 < next_horizontal_vel < 4.0) else 0.0
    soft_landing_bonus = 0.5 * both_contact * is_upright * is_good_speed

    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + energy_penalty + soft_landing_bonus

    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# iteration_context.md

# Iteration Context

## Recommended Action
**revert** — 当前得分-76.01远低于best得分11.95，差距87.96。对比代码，current将progress_delta_reward系数从5.0提升至15.0，但信号仍弱且出现大幅负值；stability_penalty系数和阈值收紧但惩罚过弱；soft_landing_bonus权重降低且条件收紧导致几乎不触发。这些修改均导致性能下降。建议恢复到best_reward的系数（progress_delta_reward=5.0, stability_penalty系数0.1/0.05, soft_landing_bonus=0.5且条件放宽），仅在此基础上小幅调整。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -102.55 | -102.55 | 0.00 | 33.00 | energy_penalty=-0.017 progress_delta_reward=0.003 soft_landing_bonus=0.000 stability_penalty=-0.137 | new_best |
| 2 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | 11.95 | 11.95 | 0.00 | 863.30 | energy_penalty=-0.015 progress_delta_reward=0.012 soft_landing_bonus=0.073 stability_penalty=-0.004 | new_best |
| 3 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -76.01 | 11.95 | -87.96 | 104.40 | energy_penalty=-0.016 progress_delta_reward=0.051 soft_landing_bonus=0.008 stability_penalty=-0.006 | no_meaningful_improvement |

## Expert Cards
## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## high_reward_without_success
- signal: generated_reward improves but external score stays poor
- risk: policy optimizes the custom reward but not the real task
- fix: reduce exploitable terms; add constraints tied to actual task progress and stable outcome

## contact_reward_hacking
- signal: contact-related reward triggers without good external score
- risk: agent exploits contact flags without safe task completion
- fix: require near target, low speed, stable angle, and both supports before using contact

## Stable Lessons (from previous iterations)
- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
- progress_delta_reward coefficient should be >= 5 to drive learning
- stability_penalty should be stronger to prevent instability

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| energy_penalty | -0.015598 | 0.015598 | 1.000000 | -0.040000 | -0.000004 |
| progress_delta_reward | 0.051131 | 0.273950 | 0.999999 | -10.290904 | 4.781192 |
| soft_landing_bonus | 0.008465 | 0.008465 | 0.042326 | 0.000000 | 0.200000 |
| stability_penalty | -0.005643 | 0.005643 | 0.142118 | -0.889759 | -0.000000 |
| total_reward | 0.038355 | 0.279512 | 1.000000 | -10.375505 | 4.740382 |
| generated_reward | 0.038355 | 0.279512 | 1.000000 | -10.375505 | 4.740382 |
| original_env_reward | -0.991363 | 1.235989 | 1.000000 | -100.000000 | 0.795828 |

```
