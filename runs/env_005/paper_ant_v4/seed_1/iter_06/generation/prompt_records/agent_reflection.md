# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。先用训练证据解释失败，再选择最小且可验证的干预。你的目标不是匹配某个已知环境或骨架名称，而是改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- feedback来自训练后固定策略的同一批评估轨迹。`episode_sum_mean`表示每回合有符号累计量，`magnitude_share`表示绝对累计量份额，`signed_share`保留净方向，`active_rate`表示非零触发率（旧日志中的`nonzero_rate`）。
- 组件统计是观察证据，不是因果贡献。必须结合score、episode_length、terminated/truncated、历史修改及其结果判断。
- 不同时间语义不可直接比较：逐步差分、持续状态值、惩罚和稀疏事件bonus不能套同一个比例阈值。
- 不得仅因任务描述出现“跳跃、着陆、抓取”等语义，就断言对应状态量是缺失奖励职责。新增职责必须有轨迹行为、终止分布、组件激活或历史干预结果支持；证据不足时明确写“未知”，优先保持best主结构并提出最小可证伪修改。
- episode达到时间上限且失败终止很少时，首先判断现有主信号是否已经实现稳定行为、剩余差距是否来自效率或主目标强度；没有行为证据时，不为动作过程本身添加proxy。

# 唯一决策流程

按顺序完成，不能因知识库或工具命中某个变换就跳级。

## 1. 行为与历史诊断

拿到训练反馈后，必须先明确回答原有三个问题：

1. **这个agent发生了什么？** episode相对正常长度明显偏短，可能在快速失败；episode很长但得分差，可能在徘徊；得分已经好但某组件尺度异常，可能存在exploit，必须用行为证据验证。
2. **哪个组件最值得干预？** 结合组件数学形态、episode_sum_mean、signed_share、magnitude_share、active_rate、外部score和episode_length判断，不得把数值占比直接写成因果贡献。
3. **我之前改了什么？** 从Agent Memory检查上一轮动作、预测和实际效果。如果上次改了A但得分没有实质变化，这次不要再次修改A。

随后确认当前失败是否来自该组件或某个缺失职责，一次只选择一个干预目标。current明显差于best时，以best代码为基础，但必须做一个新的、有证据的修改，不能原样复制best。

## 2. 信号完备性检查

检查当前奖励是否具有任务所需且可达的职责，而不是要求固定组件名称：

- 任务进展或可学习的过程引导；
- 必要的稳定、安全或动作约束；
- 当过程最优不等于任务完成时，能区分联合满足或完成状态的信号。

如果必要职责缺失、active_rate接近0、数学形态使反馈塌缩，或proxy与外部任务明显错位，进入Level 2。若职责基本完备、符号与数学形态合理，只是相对尺度异常，先进入Level 1。

## 3. 选择干预层级

### Level 1：尺度修复

适用条件：组件职责、符号和数学形态合理，证据主要表明单个组件过强或过弱。

- 只调整一个组件的系数，其他组件保持不变。
- 对同为逐步稠密信号、且progress明确承担主引导职责的早期学习阶段，旧实验中的`ratio_to_progress`思想仍然适用：`|penalty/progress| > 0.5`可作为优先检查并尝试降系数的经验触发器；`penalty/progress ≈ 0.1`可作为轻约束起点。
- 上述比例是v6实验中的有效经验先验，不是因果结论或通用硬阈值；事件bonus、持续状态奖励和正负抵消严重的差分均值不能直接套用。
- 若一次明确的尺度修复后，尺度异常已经消失但外部行为没有实质改善，不继续反复调同一系数，转Level 2。

### Level 2：有方向的数学结构变换

适用条件：必要信号缺失/不可达，或证据直接否定当前数学形态，或Level 1已修复尺度但策略仍失败。每轮只改变一个目标组件；改变该组件形态时同步设置与新值域匹配的系数，仍算一次组件干预。

| 证据模式 | 结构变换 | 下一轮验证 |
|---|---|---|
| 任务事件几乎不触发，缺少局部反馈 | sparse_to_dense：稀疏事件→连续过程证据 | active_rate及外部表现改善，且不产生proxy徘徊 |
| 极端值支配奖励 | unbounded_to_bounded：无界→归一化有界 | 极端轨迹支配下降，得分方差下降 |
| 占据好状态即可持续获奖 | state_to_improvement：状态值→状态改善量/有效势能差 | 停留不再积累收益，任务进展改善 |
| 约束在无关阶段妨碍探索 | global_to_local_gate：全局→阶段相关/局部门控 | 早期探索与局部约束同时改善；先确认不是单纯尺度过强 |
| 独立目标可互相补偿 | independent_to_joint：加权和→联合满足 | 单项刷分减少，必要条件共同改善 |
| 多个小因子相乘导致塌缩 | product_to_noncollapsing_joint：乘积→几何平均/软最小/门控和 | 非零反馈增多且联合约束保留 |
| 持续事件被重复领取 | persistent_to_transition_event：持续状态→有效状态转移 | 重复积累下降，外部完成保持或改善 |
| proxy提高但外部任务不升 | proxy_to_completion_alignment：代理目标→任务完成对齐 | proxy与外部分数重新同向 |
| 复杂耦合无法诊断 | coupled_to_diagnostic_components：耦合→少量直接组件 | 组件可解释并形成单一干预假设 |
| 稠密proxy形成中等分平台 | dense_to_task_event：全程proxy→局部/转移任务信号 | 刷新best，完成相关行为增加 |

常用数学性质：二值稀疏条件信用分配困难；连续乘积表达联合满足但可能塌缩；加权和反馈稠密但允许目标补偿；bounded函数限制极端值但输入仍需按环境尺度归一化；门控只应在证据表明“作用阶段错误”时使用。

### Level 3：重建主骨架

满足任一条件时停止局部修补：

- 同一骨架家族已迭代2轮以上，且历史最佳得分仍未超过target的25%；
- 同一结构家族连续2轮以上未刷新best，且至少做过一次Level 2；
- Level 2改变数学形态后没有实质改善；
- 同一结构家族连续3轮未刷新best且仍未达到target，即使已超过target的25%也要警惕中等分平台。

Level 3可以更换主信号框架或重新组合少量组件。expert_reward_context中的骨架是设计原语和风险提示，不是封闭候选列表；可以采用、组合、变形或基于环境事实创建新结构。

# 代码约束

- 禁止terminal_success_reward、terminal_failure_penalty、original_reward。
- 只能使用环境事实摘要声明的obs、next_obs、action和info字段，不得发明字段、切片维度或新输入。
- 第一个Python code block只能包含一个完整的`compute_reward`函数；不要写import、class、try/except或额外函数，不要使用self。
- 禁止eval/exec/open，禁止使用original_reward或原始环境reward。
- 需要平方根时使用`** 0.5`，禁止import numpy。需要指数形式时使用`2.718281828 ** exponent`，或改用`1/(1+k*x)`、`max(0,1-x/D)`等无需库的bounded表达式。
- 除Level 3重建外，每轮只修改一个目标组件，不顺带调整其他组件。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回`(float(total_reward), components)`；components只放总公式中直接出现的奖励组件，不放total_reward和中间调制器。

# 输出

先用以下固定字段各写一句，不复述输入表格：

1. `evidence`：支持判断的外部结果、组件证据和上一轮结果；
2. `behavior_diagnosis`：策略当前的失败行为；
3. `signal_completeness`：必要职责是否完备、可达；
4. `selected_level`：Level 1、Level 2或Level 3及触发条件；
5. `selected_intervention`：唯一目标组件及具体修改；
6. `falsifiable_hypothesis`：为什么该修改应改善策略；
7. `expected_next_round`：下一轮哪些指标应如何变化；
8. `main_risk`：最可能引入的新漏洞。

然后立即输出完整Python代码。预期必须能被下一轮反馈证伪。

```

## User Prompt

```markdown
# Search objective
- target_score: 2000.000000
- current_score: -710.052195
- gap_to_target: 2710.052195
- target_achievement_ratio: -35.503%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -710.052195）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- extract relevant observations ----
    body_z = obs[0]
    # quaternion components
    quat_x = obs[2]
    quat_y = obs[3]
    # forward (world x) velocity
    body_x_vel = obs[13]

    # ---- compute health indicators ----
    # target body height (midpoint of healthy range)
    target_height = 0.6
    height_error = body_z - target_height

    # body up vector z-component in world frame (derived from quaternion)
    # body_up = [2*(qx*qz + qw*qy), 2*(qy*qz - qw*qx), 1 - 2*(qx^2 + qy^2)]
    body_up_z = 1.0 - 2.0 * (quat_x * quat_x + quat_y * quat_y)

    # tilt error: 0 when perfectly upright, >0 when tilting
    tilt_error = 1.0 - body_up_z

    # combine height and tilt errors into a single posture error
    posture_error = (height_error * height_error + tilt_error * tilt_error) ** 0.5

    # exponential health gate: 1 when posture_error = 0, decays towards 0 as error grows
    k_gate = 5.0
    health_gate = 2.718281828 ** (-k_gate * posture_error)

    # ---- forward progress signal ----
    # only reward moving forward (positive x velocity)  
    forward_speed = body_x_vel if body_x_vel > 0.0 else 0.0
    # bounded mapping to avoid unbounded credit for extremely high speeds
    forward_bounded = forward_speed / (1.0 + forward_speed)

    # ---- total reward ----
    # multiplicative combination: agent must maintain good posture to earn forward progress credit
    w_forward = 8.0
    total_reward = w_forward * forward_bounded * health_gate

    # components dict for logging (total_reward itself is the only additive component)
    components = {
        'progress': total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-710.052195, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-975.869558, -165.916531]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 440.985061 | 100.0% | 100.0% | 35.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
学习一个连续控制的四足机器人（8个扭矩控制关节）在3D空间中稳定向前行走或跑动。任务的核心是产生持续的前向平移，同时保持身体高度在健康区间（0.2~1.0）且姿态基本直立。任何因为倒地、弹飞或数值异常导致的提前终止都视为失败。策略不能仅仅保持静止不倒，必须展现出有效的前进运动。

## 3. 观察空间 observation_space
- type: Box  
- shape: [27]  
- dtype: float32（推断）  
- 各维度含义与奖励可用性：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | body_z | 躯干质心的世界 z 坐标（高度） | true |
| 1 | quat_w | 方向四元数实部 w | true（用于计算姿态） |
| 2 | quat_x | 方向四元数虚部 x | true |
| 3 | quat_y | 方向四元数虚部 y | true |
| 4 | quat_z | 方向四元数虚部 z | true |
| 5 | joint_1_angle | 第1髋关节角 | true（位置信号） |
| 6 | joint_2_angle | 第1踝关节角 | true |
| 7 | joint_3_angle | 第2髋关节角 | true |
| 8 | joint_4_angle | 第2踝关节角 | true |
| 9 | joint_5_angle | 第3髋关节角 | true |
| 10 | joint_6_angle | 第3踝关节角 | true |
| 11 | joint_7_angle | 第4髋关节角 | true |
| 12 | joint_8_angle | 第4踝关节角 | true |
| 13 | body_x_velocity | 躯干在世界 x 方向（前方）的速度 | true **← 核心前进信号** |
| 14 | body_y_velocity | 躯干在世界 y 方向（侧向）的速度 | true |
| 15 | body_z_velocity | 躯干在 z 方向的速度 | true |
| 16 | body_roll_velocity | 绕 x 轴角速度（翻滚） | true |
| 17 | body_pitch_velocity | 绕 y 轴角速度（俯仰） | true |
| 18 | body_yaw_velocity | 绕 z 轴角速度（偏航） | true |
| 19 | joint_1_velocity | 第1髋关节角速度 | true |
| 20 | joint_2_velocity | 第1踝关节角速度 | true |
| 21 | joint_3_velocity | 第2髋关节角速度 | true |
| 22 | joint_4_velocity | 第2踝关节角速度 | true |
| 23 | joint_5_velocity | 第3髋关节角速度 | true |
| 24 | joint_6_velocity | 第3踝关节角速度 | true |
| 25 | joint_7_velocity | 第4髋关节角速度 | true |
| 26 | joint_8_velocity | 第4踝关节角速度 | true |

额外可计算信号：body_up_z = 1 - 2*(quat_x² + quat_y²)，表示躯干 z 轴在世界 z 轴的投影（1.0 为完全直立）。

## 4. 动作空间 action_space
- type: Box  
- shape: [8]  
- continuous: true  
- 界限: [-1.0, 1.0]（每个关节扭矩）  

| 动作维度 | 名称 | 含义 |
|----------|------|------|
| 0 | hip_1_torque | 第1髋关节扭矩 |
| 1 | ankle_1_torque | 第1踝关节扭矩 |
| 2 | hip_2_torque | 第2髋关节扭矩 |
| 3 | ankle_2_torque | 第2踝关节扭矩 |
| 4 | hip_3_torque | 第3髋关节扭矩 |
| 5 | ankle_3_torque | 第3踝关节扭矩 |
| 6 | hip_4_torque | 第4髋关节扭矩 |
| 7 | ankle_4_torque | 第4踝关节扭矩 |

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: **无明确成功终止**。正常存活到时间上限（truncation）可视为一种成功存活信号，但未保证前进距离。
- failure-like termination: terminated=True 由以下条件触发：
  - body_height_outside_healthy_range: body_z ≤ 0.2 或 body_z ≥ 1.0（倒塌/弹飞）
  - state_value_outside_finite_range: 任何状态值变为 NaN 或无穷大
- ambiguous termination: 无。
- truncation: 时间上限到达，truncated=True。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 被屏蔽，无法获取任何标志）
- explicit_failure_flag_available: false （同上，但可从 terminated 原因推断失败）
- allowed_info_fields: [] （空列表，意味着 **info 完全不可用**）
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止，包括 reward_forward、reward_ctrl、reward_contact、reward_survive、x_position、y_position、distance_from_origin 等。

## 7. 可用于奖励函数的信号
- position: body_z（高度），关节角度（8个），身体姿态四元数（可计算 body_up_z）
- velocity: body_x_velocity（前向速度），body_y_velocity（侧向漂移），body_z_velocity（上下），body_roll/pitch/yaw_velocity（姿态变化率），8个关节速度
- orientation: 通过四元数计算身体直立度（body_up_z），或惩罚较大滚动/俯仰角速度
- action/engine: 8维扭矩动作（可用于动作幅度惩罚或平滑性约束）
- other: 从姿态推导的 body_up_z，身体高度距离健康边界的距离

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_progress + height_penalty + lateral_drift_penalty + upright_stability | 434.48 | 434.48 | 0.00 | 1000.00 | forward_progress=0.054 height_penalty=-0.311 lateral_drift_penalty=-0.044 upright_stability=0.037 | new_best |
| 2 | forward_progress + height_penalty + lateral_drift_penalty + upright_stability | -224.85 | 434.48 | -659.32 | 1000.00 | forward_progress=1.057 height_penalty=-0.374 lateral_drift_penalty=-0.044 upright_stability=-0.146 | no_meaningful_improvement |
| 3 | forward_progress + height_penalty + lateral_drift_penalty + upright_stability | 265.02 | 434.48 | -169.46 | 852.65 | forward_progress=2.571 height_penalty=-0.377 lateral_drift_penalty=-0.045 upright_stability=-0.171 | no_meaningful_improvement |
| 4 | forward_progress + height_penalty + lateral_drift_penalty + upright_stability | 86.94 | 434.48 | -347.53 | 950.60 | forward_progress=2.747 height_penalty=-1.059 lateral_drift_penalty=-0.049 upright_stability=-0.145 | same_skeleton_oscillation_fresh_restart |
| 5 | progress | -710.05 | 434.48 | -1144.53 | 1000.00 | progress=0.576 | no_meaningful_improvement |

```
