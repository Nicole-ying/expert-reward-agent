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
- current_score: -56.089203
- gap_to_target: 256.089203
- target_achievement_ratio: -28.045%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -56.089203）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Next state (post-action)
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Component 1: State-based proximity reward ---
    # Continuous per-step reward that grows as distance to pad shrinks.
    # Bounded above at 1.0 (when dist=0), provides gradient at all distances.
    curr_dist = (x**2 + y**2) ** 0.5
    w_approach = 1.0
    comp_approach = w_approach / (1.0 + curr_dist)

    # --- Component 2: Soft landing velocity penalty (distance-gated) ---
    # Gate: full strength only when close to pad.
    gate_vel = 1.0 / (1.0 + curr_dist)
    vel_penalty = (vx**2 + vy**2) * gate_vel
    w_vel = 0.2  # increased from 0.1 to strengthen soft-landing guidance
    comp_soft_landing = -w_vel * vel_penalty

    # --- Component 3: Upright stabilization ---
    angle_penalty = angle**2
    angvel_penalty = ang_vel**2
    w_angle = 0.1
    w_angvel = 0.1
    comp_stabilization = -w_angle * angle_penalty - w_angvel * angvel_penalty

    # --- Component 4: Successful contact reward ---
    w_contact = 3.0
    comp_contact = w_contact * (left_contact * right_contact)

    total_reward = comp_approach + comp_soft_landing + comp_stabilization + comp_contact
    components = {
        "approach_proximity": comp_approach,
        "soft_landing_velocity": comp_soft_landing,
        "upright_stabilization": comp_stabilization,
        "contact_reward": comp_contact
    }
    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Goal: Guide a 2D lander to softly touch down on the target pad with both feet,
    staying upright and near zero velocity.
    """
    # Unpack next_obs (post-action state)
    x = next_obs[0]          # horizontal offset from pad
    y = next_obs[1]          # vertical offset (positive = above pad)
    vx = next_obs[2]         # horizontal velocity
    vy = next_obs[3]         # vertical velocity
    angle = next_obs[4]      # body angle
    ang_vel = next_obs[5]    # angular velocity
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Component 1: Approach and soft landing ---
    # Distance to target (the pad is at (0,0))
    dist = (x**2 + y**2) ** 0.5

    # Positive reward that increases as distance decreases (bounded)
    approach_reward = 1.0 / (1.0 + dist)

    # Velocity penalty gated by proximity: heavy only when close
    gate_vel = 1.0 / (1.0 + dist)
    vel_penalty = (vx**2 + vy**2) * gate_vel

    w_approach = 1.0
    w_vel      = 0.1
    comp_approach_landing = w_approach * approach_reward - w_vel * vel_penalty

    # --- Component 2: Upright stabilization ---
    angle_penalty = angle**2
    angvel_penalty = ang_vel**2
    w_angle  = 0.1
    w_angvel = 0.1
    comp_stabilization = - w_angle * angle_penalty - w_angvel * angvel_penalty

    # --- Component 3: Successful contact reward ---
    # Give a clear signal when both landing legs touch simultaneously
    w_contact = 3.0
    comp_contact = w_contact * (left_contact * right_contact)

    total_reward = comp_approach_landing + comp_stabilization + comp_contact
    components = {
        "approach_and_soft_landing": comp_approach_landing,
        "upright_stabilization": comp_stabilization,
        "contact_reward": comp_contact
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-56.089203, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-96.142958, -22.078098]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_proximity | 584.898317 | 99.4% | 99.4% | 100.0% |
| soft_landing_velocity | -2.874669 | -0.5% | 0.5% | 100.0% |
| upright_stabilization | -0.375690 | -0.1% | 0.1% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
任务主目标：控制一个二维飞行器从顶部中心区域出发，快速、平稳地降落到画面中央的目标着陆垫上，并在着陆瞬间保持稳定的姿态和速度，使左右支撑腿都与垫面接触。次目标：在满足成功着陆的前提下，尽可能缩短到达时间和节省发动机推力（燃料）。不应混淆为目标只是悬停或单纯到达上空，必须实际软着陆并稳定停靠。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推断)
- obs[0]: x_position，飞行器相对于目标着陆垫的水平坐标，reward_usable: true
- obs[1]: y_position，飞行器相对于着陆垫高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，水平线性速度，reward_usable: true
- obs[3]: y_velocity，垂直线性速度，reward_usable: true
- obs[4]: body_angle，机体朝向角度，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（0.0 或 1.0），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志（0.0 或 1.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，不激活任何发动机，无推力输出
- action 1: left_orientation_engine，激活左侧姿态调整发动机（产生角加速度和可能的侧向力）
- action 2: main_engine，激活主发动机（产生向上的推力，同时可能影响姿态）
- action 3: right_orientation_engine，激活右侧姿态调整发动机（与左姿态发动机对称）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（机体静止/稳定），如果此时 x,y 位置非常接近目标、速度近乎零且左右接触标志均为 1.0，则大概率是成功着陆。
- failure-like termination: crash_or_body_contact 和 horizontal_position_outside_viewport 明确对应坠毁、接触不良或飞出视口。
- ambiguous termination: body_not_awake_or_settled 也可能发生在未到达目标但外界原因导致静止的情况，需要结合位置和接触进一步判断。
- truncation: 未明确给出，但可能存在最大步数截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （info 字典为空，没有任何可用字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均被禁止使用，包括假想的 "success"、"failure"、"termination_reason"

## 7. 可用于奖励函数的信号
- position: 
  - `obs[0]` (x_position, 水平偏差)
  - `obs[1]` (y_position, 垂直偏差)
- velocity: 
  - `obs[2]` (x_velocity)
  - `obs[3]` (y_velocity)
- orientation: 
  - `obs[4]` (body_angle)
  - `obs[5]` (angular_velocity)
- contact: 
  - `obs[6]` (left_support_contact)
  - `obs[7]` (right_support_contact)
- action/engine: 
  - `action` 本身（0~3），可区分不同发动机使用，用于计算燃料代价
- other: 
  - 从上述信号可构造综合接近程度、着陆稳定性、姿态水平等派生信号

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D 双足/双支撑着陆飞行器 (lander-like)
  actuator_type: 主推力发动机 + 两个姿态调整发动机
  contact_structure: 左右两个着陆支撑腿，接触时产生 1.0 标志
primary_objectives:
  - 到达并稳定停留在目标着陆垫（位置偏差极小，速度近零）
  - 着陆时同时产生左右支撑腿接触
secondary_objectives:
  - 尽可能短的时间到达（间接通过步数惩罚实现）
  - 节省燃料（减少发动机使用）
main_failure_risks:
  - 坠毁或机体部分异常接触（crash_or_body_contact）
  - 水平飞出视口边界
  - 过早“稳定”但未在目标区域（误触发 body_not_awake_or_settled）
  - 着陆时姿态过倾斜导致单腿接触或翻倒
  - 过度使用主发动机导致燃料耗尽或剧烈震荡
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: approach_target
  purpose: 驱动飞行器向目标点移动，缩小水平与垂直偏差。
  why_required: 到达目标是主任务，必须提供持续的趋近信号。
  usable_signals: [obs[0], obs[1]] （位置偏差）
  risks: 仅用距离可能导致高速冲撞，需要配合减速。

- role_id: soft_landing_velocity
  purpose: 在接近目标时降低速度，实现软着陆。
  why_required: 着陆必须速度近零，否则即使位置正确也会失败或坠毁。
  usable_signals: [obs[2], obs[3]] （线速度），可与位置距离联合塑造（速度惩罚随距离衰减）
  risks: 全局惩罚速度会抑制初期快速接近，需要逐步激活或距离门控。

- role_id: upright_stabilization
  purpose: 保持机体姿态水平，避免倾斜过度。
  why_required: 着陆时姿态稳定是成功接触的前提，左右接触要求姿态接近零。
  usable_signals: [obs[4], obs[5]] （角度、角速度）
  risks: 过度惩罚轻微摆动可能导致保守控制，可结合即将着陆的信号加强。

- role_id: successful_contact_reward
  purpose: 奖励两腿同时接触的状态。
  why_required: 最终着陆成功的硬性指标，无此信号无法判断任务完成。
  usable_signals: [obs[6], obs[7]] （接触标志乘积或和）
  risks: 仅靠接触奖励可能导致飞行器提前摆出接触姿势而不真正到达目标，必须与位置和速度联合使用。

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency
  purpose: 惩罚不必要的发动机使用，节省燃料。
  condition_to_use: 在整个 episode 中持续生效，但权重不宜过高以免干扰主目标学习。
  usable_signals: [action] （通过区分无动作、主引擎、姿态引擎赋予不同惩罚）
  risks: 早期训练可能因惩罚导致 agent 不敢使用推力，可采用线性衰减或很小系数。

- role_id: time_penalty
  purpose: 鼓励尽快完成任务。
  condition_to_use: 每步给予小的负奖励（活代价），或在成功着陆时给予一次性的与步数成反比的奖励。
  usable_signals: 隐式，可通过在 episode 结束时测量步数实现，但 compute_reward 单步接口无法直接获取步数；可以用每步小的固定负奖励模拟。
  risks: 固定步惩罚过大可能使 agent 急功近利而坠毁。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: orientation_engine_usage_penalty_alone (单独惩罚姿态发动机)
  reason: 姿态发动机是必要控制手段，一味惩罚会阻止 agent 学习稳定姿态，应与姿态误差配合使用。
  forbidden_or_missing_signals: 无可直接判断是否“浪费”姿态发动机的信号，仅靠动作类型无法区分有效调整与无意义乱喷。

- role_id: exact_contact_sequence_bonus
  reason: 缺少阶段标签或脚力传感器，无法推断正确的着陆顺序（如先左后右），强行设计可能导致脆弱奖励。
  forbidden_or_missing_signals: 无接触时序信息，仅有当前帧的 bool 标志。

- role_id: success_flag_based_reward
  reason: 环境未提供显式的 success 标志，info 为空，根本不可用。
  forbidden_or_missing_signals: info 中无任何字段。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_target | obs[0], obs[1] | 无 | dense_state_signal (基于距离函数，如 -sqrt(x^2+y^2) 或 shaped exponential) | 距离最小化是基础 |
| soft_landing_velocity | obs[2], obs[3] | 无 | bounded_signal (可结合距离门控，如 -||v|| * f(dist) ) | 防止高速撞击 |
| upright_stabilization | obs[4], obs[5] | 无 | quadratic_penalty (角度平方 + 角速度平方) | 姿态保持水平 |
| successful_contact_reward | obs[6], obs[7] | 无 | logical_and_or_sum ( reward when both are 1, maybe only when also near target ) | 着陆成功标志 |
| fuel_efficiency | action | 无 | discrete_action_penalty (不同 action 赋予不同负值) | 节省燃料 |
| time_penalty | 无直接信号 | 真实步数 | fixed_constant_penalty (每步 -small_value) | 加速完成 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 飞行器悬停在空中不下降 | y_position 长期为正，y_velocity 约等于零，主发动机使用频繁 | 降低燃料惩罚权重或增加下降推力引导信号，如对负 y_velocity 给予奖励 |
| 高速砸向目标然后反弹或坠毁 | 终止前速度范数极大，终止时 crash_or_body_contact 触发 | 加强软着陆速度惩罚，尤其是低高度时；可以引入速度上限惩罚 |
| 到达目标但姿态大幅倾斜，单腿接触 | obs[4] 绝对值大，且 obs[6] + obs[7] < 2.0 时终止 | 增加姿态稳定权重，或在接近目标时放大姿态误差的惩罚 |
| 长时间左右摆动，无法稳定 | angular_velocity 持续振荡，接触不断变化 | 对角速度施加平滑惩罚，或结合速度/姿态联合塑造更稳定的下降动力学 |
| 一味节省燃料而不用主发动机，导致缓慢漂离 | 距离增加，y_velocity 缓慢正向，很少使用 action 2 | 提高主任务趋近奖励的权重，或者允许适度燃料使用，或动态调整燃料惩罚系数 |
| 在目标附近稳定但未触发终止（可能被 truncation 截断） | episode 结束但 position error 很小，接触标志不为全1 | 检查环境终止条件是否对“稳定”判定过于严格；可考虑在接近目标时略微增大接触奖励，激励 agent 完成最后压触动作 |
```
