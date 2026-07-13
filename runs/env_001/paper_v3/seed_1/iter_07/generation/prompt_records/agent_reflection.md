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

# 工具

核心Level判断必须依靠本Prompt完成。仅在根因不确定、多个Level 2变换难以区分或需要骨架细节时调用一次最相关工具：

- `search_reward_design_knowledge(query)`：检索相似失败模式和经验修复。
- `get_skeleton_detail(skeleton_name)`：查看数学形态、原理和陷阱。
- `get_reward_transformation(query)`：查看结构变换的原理、风险和验证目标。

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
- target_score: 200.000000
- current_score: 37.653049
- gap_to_target: 162.346951
- target_achievement_ratio: 18.827%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 37.653049）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    x = obs[0]            # horizontal position relative to target pad
    y = obs[1]            # vertical position relative to target pad
    vx = obs[2]           # horizontal velocity
    vy = obs[3]           # vertical velocity
    angle = obs[4]        # body angle
    ang_vel = obs[5]      # angular velocity
    left_contact = obs[6] # left support leg contact (0 or 1)
    right_contact = obs[7]# right support leg contact (0 or 1)

    # Hyperparameters
    w_goal = 1.0
    alpha_proximity = 5.0
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_landing = 3.0           # weight for continuous landing quality reward
    beta_speed = 10.0          # speed quality decay
    beta_angle = 10.0          # angle quality decay

    # Distance to target center (squared)
    dist_sq = x**2 + y**2

    # Soft proximity weight: ~1 when close to target, ~0 when far
    proximity = 1.0 / (1.0 + alpha_proximity * dist_sq)

    # 1. Main progress: drive toward target center (dense quadratic penalty on distance)
    goal_proximity = -w_goal * dist_sq

    # 2. Soft landing velocity penalty: active only near the target
    velocity_penalty = -w_vel * (vx**2 + vy**2) * proximity

    # 3. Orientation stability penalty: penalize tilt and spin everywhere (light weight)
    orientation_penalty = -w_angle * (angle**2) - w_angvel * (ang_vel**2)

    # 4. Continuous landing quality reward: high when close, slow, and upright
    speed_sq = vx**2 + vy**2
    speed_quality = 1.0 / (1.0 + beta_speed * speed_sq)
    angle_sq = angle**2
    angle_quality = 1.0 / (1.0 + beta_angle * angle_sq)
    contact_reward = w_landing * proximity * speed_quality * angle_quality

    # Total reward
    total_reward = goal_proximity + velocity_penalty + orientation_penalty + contact_reward

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    x = obs[0]            # horizontal position relative to target pad
    y = obs[1]            # vertical position relative to target pad
    vx = obs[2]           # horizontal velocity
    vy = obs[3]           # vertical velocity
    angle = obs[4]        # body angle
    ang_vel = obs[5]      # angular velocity
    left_contact = obs[6] # left support leg contact (0 or 1)
    right_contact = obs[7]# right support leg contact (0 or 1)

    # Hyperparameters (tunable in later iterations)
    w_goal = 1.0
    alpha_proximity = 5.0          # controls the activation radius for proximity-based terms
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_contact = 2.0

    # Distance to target center (squared)
    dist_sq = x**2 + y**2

    # Soft proximity weight: ~1 when close to target, ~0 when far
    proximity = 1.0 / (1.0 + alpha_proximity * dist_sq)

    # 1. Main progress: drive toward target center (dense quadratic penalty on distance)
    goal_proximity = -w_goal * dist_sq

    # 2. Soft landing velocity penalty: active only near the target
    velocity_penalty = -w_vel * (vx**2 + vy**2) * proximity

    # 3. Orientation stability penalty: penalize tilt and spin everywhere (light weight)
    orientation_penalty = -w_angle * (angle**2) - w_angvel * (ang_vel**2)

    # 4. Contact reward: encourage both legs touching, gated by proximity
    both_legs_contact = left_contact * right_contact  # 1 only if both are 1
    contact_reward = w_contact * both_legs_contact * proximity

    # Total reward
    total_reward = goal_proximity + velocity_penalty + orientation_penalty + contact_reward

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=37.653049, len=895.900000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-141.735094, 138.633443]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 1960.931640 | 94.0% | 94.0% | 100.0% |
| goal_proximity | -109.445230 | -5.2% | 5.2% | 100.0% |
| velocity_penalty | -13.647435 | -0.7% | 0.7% | 100.0% |
| orientation_penalty | -3.071650 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标：控制一个 2D 飞行器从顶部中央区域安全、快速地到达并稳定停靠在中央目标垫上。  
次目标：尽可能节省引擎推力。  
不应混淆的目标：纯粹的飞行姿态控制不是最终目标；省燃料不能影响成功着陆。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: float32 (推测，其中的接触标志为 0.0 或 1.0)  
- 字段含义与可奖励性：

| 索引 | 名称 | 含义 | 可用于奖励？ |
|------|------|------|--------------|
| 0 | x_position | 相对目标垫的水平坐标 | 是 |
| 1 | y_position | 相对目标垫高度的垂直坐标 | 是 |
| 2 | x_velocity | 水平线速度 | 是 |
| 3 | y_velocity | 垂直线速度 | 是 |
| 4 | body_angle | 机体角度 (orientation) | 是 |
| 5 | angular_velocity | 角速度 | 是 |
| 6 | left_support_contact | 左支撑腿接触标志 (0/1) | 是 |
| 7 | right_support_contact | 右支撑腿接触标志 (0/1) | 是 |

所有 8 个量均可在奖励函数中安全使用；全部源自环境本身，无隐藏信息。

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：

| 动作 ID | 名称 | 含义 |
|---------|------|------|
| 0 | no_engine | 无任何引擎推力 |
| 1 | left_orientation_engine | 启动左姿态引擎（产生旋转力矩） |
| 2 | main_engine | 启动主引擎（产生主推力，可能同时影响姿态） |
| 3 | right_orientation_engine | 启动右姿态引擎（与左引擎方向相反） |

## 5. step 与终止条件分析
### 5.1 终止模式
- 可能表示成功的终止模式：
  - `body_not_awake_or_settled`：如果同时满足位置接近目标、速度极小、双支撑腿着地且角度稳定，可视为成功着陆。
- 可能表示失败的终止模式：
  - `crash_or_body_contact`：机体非腿部分与障碍物/地面碰撞。
  - `horizontal_position_outside_viewport`：超出水平边界。
- 模糊终止模式：
  - `body_not_awake_or_settled`：在偏离目标或仅有单腿接触时也可能触发，此时可能是失败（例如跌落后停止）。
- 截断：未提及任何时间截断或步骤限制。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] (info 固定为空字典)
- forbidden_or_uncertain_info_fields: 所有 info 字段均不存在，不得使用。终止原因不可从 info 获得。

## 7. 可用于奖励函数的信号
- **位置**：`x_position`, `y_position`（相对目标垫，可直接做距离度量）
- **速度**：`x_velocity`, `y_velocity`（控制着陆时的冲击力与稳定性）
- **姿态**：`body_angle`, `angular_velocity`（稳定竖直）
- **接触**：`left_support_contact`, `right_support_contact`（两条支撑腿是否接触）
- **动作/引擎**：`action`（离散动作索引，可用于燃料消耗度量）
- **其他**：无

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D vehicle/lander
  actuator_type: discrete (one main engine, two orientation engines)
  contact_structure: two support legs (binary contacts)
primary_objectives:
  - 到达并停留在目标垫上（位置接近、双支撑腿接触）
  - 低速、稳定着陆（速度小、角度竖直）
secondary_objectives:
  - 尽量少使用引擎推力（节省燃料）
  - 尽快完成着陆（非强制时间最小化）
main_failure_risks:
  - 机体非腿部分与地面碰撞
  - 水平漂出视口
  - 角度过大导致不稳定着陆或接触失败
  - 过早切断引擎导致粗猛着陆
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity_reward
  purpose: 驱动机体向目标垫移动并停留在中心
  why_required: 任务核心是到达目标位置，没有此职责 agent 将无方向
  usable_signals: [x_position, y_position]
  risks: 单纯距离奖励可能引导高速冲撞，需配合速度控制

- role_id: soft_landing_velocity_penalty
  purpose: 在接近目标时抑制过大的水平与垂直速度
  why_required: 成功要求“settle”，高速冲击会导致失败或无法稳定
  usable_signals: [x_velocity, y_velocity, x_position, y_position]
  risks: 如果对所有时间惩罚速度，可能阻止探索；应只在靠近目标时激活

- role_id: orientation_stability_reward
  purpose: 保持机体角度稳定（接近 0 或竖直）
  why_required: 着陆需要平稳接触，大角度倾斜可能只接触单腿甚至倾覆
  usable_signals: [body_angle, angular_velocity]
  risks: 角度控制权重过大可能抑制必要的姿态调整；可通过绝对角度惩罚实现

- role_id: contact_reward
  purpose: 奖励双支撑腿同时接触
  why_required: 着陆成功的明确信号；能帮助 agent 理解“settle”状态
  usable_signals: [left_support_contact, right_support_contact]
  risks: 若只奖励接触而不要求位置，agent 可能学会到处触碰；必须与目标位置结合

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency_penalty
  purpose: 减少不必要的引擎使用
  condition_to_use: 当 agent 已经能稳定到达目标附近时，可以逐步引入动作惩罚；或在任意时刻加入极小权重，但确保不压倒主目标
  usable_signals: [action]
  risks: 过早或过大惩罚会使 agent 不敢移动，完全不做探索

- role_id: settlement_bonus
  purpose: 在完全满足着陆条件时给予一次性高奖励
  condition_to_use: 仅在极靠近目标、速度极小、双支撑腿接触且角度稳定的状态下给予；相当于“成功”奖励的代理
  usable_signals: [x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact]
  risks: 如果阈值过严可能过于稀疏，难以学习；过松则误触

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_crash_penalty
  reason: 缺少可靠的失败/碰撞标志，终止原因不可从 info 获得；若强行用 next_obs 推断崩溃（例如检测到全零）可能不可靠且引入噪声
  forbidden_or_missing_signals: [collision_flag, termination_reason]

- role_id: time_optimality_penalty
  reason: 任务描述要求“尽快”，但未提供时间步数度量或截断信息；主动针对步数设计奖励可能干扰主要目标，可作为次要考虑，不宜作为强约束
  forbidden_or_missing_signals: [step_count, truncation_flag]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity_reward | x_position, y_position | — | dense_state_signal, euclidean_distance, gaussian_mixture, potential_based | 可用 -distance^2 或 exp(-dist) 等；可配合 shaping |
| soft_landing_velocity_penalty | x_velocity, y_velocity, x_position, y_position | — | quadratic_penalty, masked_by_proximity | 仅在靠近目标时生效，设为平方惩罚 |
| orientation_stability_reward | body_angle, angular_velocity | — | quadratic_penalty, absolute_penalty | 可直接使用 -angle^2, -ang_vel^2 或线性惩罚 |
| contact_reward | left_support_contact, right_support_contact | — | boolean_and, discrete_bonus | 当两个接触标志均为 1 时给予正奖励；与位置耦合 |
| fuel_efficiency_penalty | action | fuel_consumption_per_action (隐含) | action_cost_lookup, scaling | 离散动作：非零动作给予小额惩罚；逐渐引入 |
| settlement_bonus | 全部位置/速度/角度/接触信号 | — | thresholded_bonus | 所有条件逻辑与，成功后给较大常数 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 离目标很远就停止（过早认为任务完成） | 终止时的 x_position, y_position 大，或单腿/无接触 | 加强 goal_proximity_reward，或降低 settlement_bonus 阈值 |
| 高速冲撞目标垫（crash 终止） | 终止前速度大，x,y 速度未衰减，或仅有极短时接触 | 加强 soft_landing_velocity_penalty 或在较远距离就施加减速 |
| 持续悬停不下降（不敢靠近或燃料恐惧） | y_position 缓慢变化，长期不终止，动作多为 0 | 减弱或延迟 fuel_efficiency_penalty，增加目标垫附近的吸引 |
| 角度剧烈摇晃无法稳定 | body_angle 和 angular_velocity 振幅大，接触断续 | 增加 orientation_stability_reward 权重，考虑角度变化率惩罚 |
| 单腿着地反复弹跳 | 仅一侧接触标志为 1，且 x,y 位置略偏 | 提升接触奖励，并对单腿接触给予轻微负奖励（需安全引入） |
| 燃料使用过度，但成功着陆 | 动作序列中大量非零动作 | 在训练后期逐步引入 fuel_efficiency_penalty，进行策略微调 |
| 水平漂出视口 | x_position 绝对值过大触发终止 | 早期强化位置惩罚，或用较大的位置奖励梯度防止漂移 |
```
