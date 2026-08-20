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
- current_score: 89.065223
- gap_to_target: 110.934777
- target_achievement_ratio: 44.533%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 89.065223）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft_contact, nright_contact = next_obs

    # ----------------------------------------------------------------
    # 1. Goal proximity: potential-based delta shaping
    # ----------------------------------------------------------------
    distance = (x**2 + y**2)**0.5
    next_distance = (nx**2 + ny**2)**0.5
    proximity_reward = 2.0 * (distance - next_distance)

    # ----------------------------------------------------------------
    # 2. Orientation penalty: keep body upright and stable
    # ----------------------------------------------------------------
    orientation_penalty = -0.1 * (angle**2) - 0.1 * (angvel**2)

    # ----------------------------------------------------------------
    # 3. Soft landing: punish high velocities when near the pad
    # ----------------------------------------------------------------
    proximity_gate = 1.0 / (1.0 + 5.0 * distance)
    speed_sq = vx**2 + vy**2
    speed_penalty_gated = -0.5 * speed_sq * proximity_gate

    # ----------------------------------------------------------------
    # 4. Settlement improvement bonus (TRANSITION-BASED)
    #    Rewards the delta in settled_score, not persistent occupation.
    #    settled_score = avg_contact * proximity_gate * stillness
    #    Only positive improvements are rewarded; regression is not penalized
    #    (speed_penalty and orientation_penalty already cover regression).
    # ----------------------------------------------------------------
    # Current settled score (from obs)
    current_avg_contact = (left_contact + right_contact) / 2.0
    current_prox_gate = 1.0 / (1.0 + 5.0 * distance)
    current_speed_sq = vx**2 + vy**2
    current_stillness = 1.0 / (1.0 + 10.0 * (current_speed_sq + angvel**2))
    current_settled = current_avg_contact * current_prox_gate * current_stillness

    # Next settled score (from next_obs)
    next_avg_contact = (nleft_contact + nright_contact) / 2.0
    next_prox_gate = 1.0 / (1.0 + 5.0 * next_distance)
    next_speed_sq = nvx**2 + nvy**2
    next_stillness = 1.0 / (1.0 + 10.0 * (next_speed_sq + nangvel**2))
    next_settled = next_avg_contact * next_prox_gate * next_stillness

    # Reward only improvement: becoming more settled than before
    settlement_bonus = 400.0 * max(0.0, next_settled - current_settled)

    # ----------------------------------------------------------------
    # Combine
    # ----------------------------------------------------------------
    total_reward = (
        proximity_reward +
        orientation_penalty +
        speed_penalty_gated +
        settlement_bonus
    )

    components = {
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "speed_penalty_gated": speed_penalty_gated,
        "settlement_bonus": settlement_bonus
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft_contact, nright_contact = next_obs

    # ----------------------------------------------------------------
    # 1. Goal proximity: potential-based delta shaping
    #    Positive when moving toward target, negative when moving away
    # ----------------------------------------------------------------
    distance = (x**2 + y**2)**0.5
    next_distance = (nx**2 + ny**2)**0.5
    proximity_reward = 2.0 * (distance - next_distance)

    # ----------------------------------------------------------------
    # 2. Orientation penalty: keep body upright and stable
    # ----------------------------------------------------------------
    orientation_penalty = -0.1 * (angle**2) - 0.1 * (angvel**2)

    # ----------------------------------------------------------------
    # 3. Soft landing: punish high velocities when near the pad
    # ----------------------------------------------------------------
    proximity_gate = 1.0 / (1.0 + 5.0 * distance)
    speed_sq = vx**2 + vy**2
    speed_penalty_gated = -0.5 * speed_sq * proximity_gate

    # ----------------------------------------------------------------
    # 4. Settlement bonus: continuous reward for settled state.
    #    CHANGED from product contact (binary 0/1) to average contact
    #    (continuous 0.0/0.5/1.0) so single-foot contact gets partial
    #    credit, eliminating the reward cliff between one and two feet.
    # ----------------------------------------------------------------
    next_proximity_gate = 1.0 / (1.0 + 5.0 * next_distance)
    next_speed_sq = nvx**2 + nvy**2
    stillness = 1.0 / (1.0 + 10.0 * (next_speed_sq + nangvel**2))
    # Average contact: 0.0 none, 0.5 one foot, 1.0 both feet
    avg_contact = (nleft_contact + nright_contact) / 2.0
    settlement_bonus = 2.0 * avg_contact * next_proximity_gate * stillness

    # ----------------------------------------------------------------
    # Combine
    # ----------------------------------------------------------------
    total_reward = (
        proximity_reward +
        orientation_penalty +
        speed_penalty_gated +
        settlement_bonus
    )

    components = {
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "speed_penalty_gated": speed_penalty_gated,
        "settlement_bonus": settlement_bonus
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=89.065223, len=805.900000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[35.028495, 212.743853]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settlement_bonus | 16726.003158 | 99.9% | 99.9% | 35.4% |
| speed_penalty_gated | -5.804138 | -0.0% | 0.0% | 100.0% |
| proximity_reward | 2.569321 | 0.0% | 0.0% | 100.0% |
| orientation_penalty | -1.970223 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是一个 2D 飞行器 / 着陆器轨迹优化问题。  
主体从视野顶部中央附近出发，并带有随机初速度。核心目标是**尽快到达并稳定停靠在画面中央的目标平台上、且保持姿态平稳**；次要目标是在此过程中尽量减少引擎推力消耗。需要关键控制能力：接近目标、减速、维持直立姿态并通过左右支撑点与平台安全接触。

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: (8,)  
- **dtype**: `float32`  

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|----------------|
| 0 | x_position | 相对于目标平台的水平坐标 | true |
| 1 | y_position | 相对于平台高度的高度坐标 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 主体倾角（方向角） | true |
| 5 | angular_velocity | 角速度 | true |
| 6 | left_support_contact | 左支撑接触标志（0/1） | true |
| 7 | right_support_contact | 右支撑接触标志（0/1） | true |

注：接触标志虽然在环境中可能是 `1.0` 或 `0.0`，但该维度对奖励设计有用。

## 4. 动作空间 action_space
- **type**: Discrete  
- **n**: 4  

| 动作编号 | 名称 | 含义 | 说明 |
|----------|------|------|------|
| 0 | no_engine | 不点火 | 被动滑行 |
| 1 | left_orientation_engine | 左侧姿控引擎 | 产生角度变化 / 姿态调整 |
| 2 | main_engine | 主引擎 | 产生主要向下推力，影响垂直速度 |
| 3 | right_orientation_engine | 右侧姿控引擎 | 产生相反侧姿态调整 |

本离散动作空间没有幅度控制，每次执行一个动作施加固定强度的冲量。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` —— 当主体运动停止且可能稳定在平台上时触发。该条件**并非明确只包含成功**，可能包含因其他原因静止（如坠毁后静止），需要结合位置与接触状态判断。
- **failure-like termination**:  
  `crash_or_body_contact` —— 主体与非目标地面/结构物接触导致坠毁；  
  `horizontal_position_outside_viewport` —— 主体水平飘出视野边界。
- **ambiguous termination**:  
  `body_not_awake_or_settled` 本身未直接区分成功与失败。
- **truncation**:  
  来源于 `step` 源码中的 `terminated` 标志，`truncated` 始终为 `False`，没有超时截断。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
- **allowed_info_fields**: 空（`step` 返回的 `info` 为空字典 `{}`）  
- **forbidden_or_uncertain_info_fields**: 所有 `info` 内未声明的字段，包括假设的 `success`、`failure`、`termination_reason` 等均不可用。

## 7. 可用于奖励函数的信号
- **position**: `x_position`, `y_position`（相对平台坐标，可用以引导接近与保持在目标上方）
- **velocity**: `x_velocity`, `y_velocity`（减速靠近平台）
- **orientation**: `body_angle`, `angular_velocity`（维持姿态平稳）
- **contact**: `left_support_contact`, `right_support_contact`（稳定接触检测）
- **action/engine**: 可以针对推力动作施加惩罚（如 `action != 0`）以鼓励节省燃料
- **other**: 无额外信号。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with two support legs/skids
  actuator_type: discrete main engine + two side orientation engines
  contact_structure: left and right touch sensors, platform at center
primary_objectives:
  - reach and settle on the central target pad (i.e., x≈0, y≈0, stopped)
  - maintain upright orientation and stable contact on the pad
secondary_objectives:
  - minimize engine thrust usage (fuel efficiency)
  - minimize time to reach the goal (speed, implicitly via fast settling)
main_failure_risks:
  - crashing into ground or non-pad surfaces (body contact other than pad)
  - drifting out of the horizontal viewport
  - failing to kill velocity, leading to bounce or slide off the pad
  - over-correction of orientation causing tilt and crash
  - early termination while not actually on the pad (misinterpreted settle)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `goal_proximity`  
  **purpose**: 驱动主体向目标平台靠近并保持在平台附近。  
  **why_required**: 没有到达目标的引导，任务无法完成。  
  **usable_signals**: `x_position`, `y_position`（距平台的距离）  
  **risks**: 若只奖励接近而忽略速度，可能导致高速冲过或撞毁。

- **role_id**: `soft_landing_and_settling`  
  **purpose**: 在接近目标后降低速度并稳定停留在平台上。  
  **why_required**: 仅到达位置不足以成功，还需减速并稳定，否则会飞出或弹跳。  
  **usable_signals**: `x_velocity`, `y_velocity`, `x_position`, `y_position`, contact flags  
  **risks**: 过度强调低速可能使智能体过早悬停，浪费燃料与时间。

- **role_id**: `orientation_penalty`  
  **purpose**: 惩罚主体倾斜，鼓励保持垂直姿态。  
  **why_required**: 翻倒会使接触传感器失效、直接导致坠毁，且稳定平台接触需要直立。  
  **usable_signals**: `body_angle`, `angular_velocity`  
  **risks**: 若权重过大，可能阻碍必要的姿态调整机动。

### 10.2 条件职责 conditional_roles
- **role_id**: `engine_usage_penalty`  
  **purpose**: 鼓励在完成任务的前提下尽可能少使用引擎。  
  **condition_to_use**: 在任务的主要完成阶段，当接近平台后或全局均可加入，但应在到达目标前不压制动作为好；可分段加权。  
  **usable_signals**: `action` (离散动作是否为非零)  
  **risks**: 过早惩罚可能使智能体不探索；过重可能导致不敢使用引擎进行修正而坠毁。

- **role_id**: `safe_contact_bonus`  
  **purpose**: 当双腿均与平台接触且姿态良好时给予额外正奖励，强化稳定着陆。  
  **condition_to_use**: 当观察到 `left_support_contact` 和 `right_support_contact` 同时为 1 且位置在平台附近时。  
  **usable_signals**: `left_support_contact`, `right_support_contact`, `x_position`, `y_position`  
  **risks**: 可能在平台外意外触发接触（如果环境存在其他可接触物），需结合位置判断。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `time_penalty`  
  **reason**: 环境没有提供时间步计数信号，且虽可通过累计步数替代，但“尽快”已隐含在 shaping（如逼近目标）中，不需要显式时间段惩罚，反而可能引起探索不足。  
  **forbidden_or_missing_signals**: 无原生的时间步索引；可用额外内建计数器实现，但无必要且容易冲突。

- **role_id**: `progress_export_constant`  
  **reason**: 环境未提供成功率或进度反馈的额外 info 字段，无法直接依据“是否成功”给予稀疏奖励，需要依赖观察重构。  
  **forbidden_or_missing_signals**: info 中无可信 success/failure 信号。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | x_position, y_position | — | dense_state_signal, distance_to_target, gaussian_like | 可用距离平滑奖励 |
| soft_landing_and_settling | x_velocity, y_velocity, x_position, y_position, contacts | — | dense_state_signal, bounded_signal, quadratic_penalty, contact-weighted | 结合距离与速度制作“着陆区”奖励 |
| orientation_penalty | body_angle, angular_velocity | — | dense_state_signal, quadratic_penalty | 对倾角与角速度施罚 |
| engine_usage_penalty | action | — | discrete_action_penalty, linear_penalty | 当 action ≠ 0 时给予小幅负奖励 |
| safe_contact_bonus | left_support_contact, right_support_contact, x_position, y_position | — | conditional_bonus, gating (both contact and proximity) | 双接触且 x,y 接近零时给正向奖励 |
| (avoid) time_penalty | — | elapsed steps in episode | — | 避免引入，容易与速度 shaping 冲突 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 主体经常越过平台后飘出视野 | `x_position` 绝对值持续增大，`body_angle` 大，`terminated` 因 horizontal outside viewport | 增加惩罚侧向速度和倾角，引入横向引导力 |
| 到达平台上方但弹跳后坠毁 | `y_velocity` 在接触瞬间过高，接触信号一闪即过，`terminated` 因 crash | 加强软着陆速度惩罚，或基于高度与速度的二次项 |
| 智能体只开主引擎垂直悬停而不移动 | `x_position` 偏差大，但 `y_position` 始终小，动作几乎只有主引擎 | 增加目标引导的奖励，或对无横向动作给予微小惩罚 |
| 双腿接触但姿态不稳定最终倾倒 | `body_angle` 和 `angular_velocity` 波动大，接触后仍终止 | 提高姿势惩罚，以及鼓励接触后静止的奖励 |
| 由于过度节省燃料而未能到达平台 | 奖励高但任务失败，`x_position`/`y_position` 远，`action` 很少非零 | 降低燃油惩罚权重，或改在末期才启用 |
| 在平台附近盘旋但永不降落 | `y_position` 小但不触发 settle（始终有点速度或轻微移动） | 加入基于位置与速度的“dock”奖励，当位置近、速度低时给出强正奖励 |

此环境卡片提供了完整的事实依据与奖励设计思路，后续 Reward Generator 可据此生成符合契约的奖励函数，Reflection Agent 也能用于诊断训练行为。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_bonus + orientation_penalty + proximity_reward + speed_penalty_gated | -65.97 | -65.97 | 0.00 | 72.00 | contact_bonus=0.001 orientation_penalty=-0.006 proximity_reward=-0.971 speed_penalty_gated=-0.166 | new_best |
| 2 | contact_bonus + orientation_penalty + proximity_reward + speed_penalty_gated | 126.73 | 126.73 | 0.00 | 749.25 | contact_bonus=0.276 orientation_penalty=-0.004 proximity_reward=0.004 speed_penalty_gated=-0.012 | new_best |
| 3 | orientation_penalty + proximity_reward + settlement_bonus + speed_penalty_gated | 187.04 | 187.04 | 0.00 | 715.90 | orientation_penalty=-0.004 proximity_reward=0.005 settlement_bonus=0.934 speed_penalty_gated=-0.016 | new_best |
| 4 | attitude_penalty + direction_reward + event_bonus + fuel_penalty + soft_landing_proxy | 168.77 | 187.04 | -18.27 | 517.55 | attitude_penalty=-0.013 direction_reward=0.045 event_bonus=0.808 fuel_penalty=-0.007 soft_landing_proxy=0.373 | no_meaningful_improvement |
| 5 | orientation_penalty + proximity_reward + settlement_bonus + speed_penalty_gated | 197.60 | 197.60 | 0.00 | 669.50 | orientation_penalty=-0.004 proximity_reward=0.005 settlement_bonus=0.942 speed_penalty_gated=-0.016 | new_best |
| 6 | orientation_penalty + proximity_reward + settlement_bonus + speed_penalty_gated | 89.07 | 197.60 | -108.54 | 805.90 | orientation_penalty=-0.005 proximity_reward=0.007 settlement_bonus=15.403 speed_penalty_gated=-0.023 | no_meaningful_improvement |

```
