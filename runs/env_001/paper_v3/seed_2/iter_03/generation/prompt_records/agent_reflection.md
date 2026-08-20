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
- current_score: -85.225247
- gap_to_target: 285.225247
- target_achievement_ratio: -42.613%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -85.225247）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # weights and sensitivities
    w_progress = 5.0
    w_angle_penalty = 0.2
    w_landing = 1.0
    a_v = 10.0       # sensitivity for vertical speed
    b_angle = 10.0   # sensitivity for body angle in landing quality

    # current and next distances to target (0,0)
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 1. Main progress signal: improvement towards target
    progress = w_progress * (dist - next_dist)

    # 2. Orientation stability: penalty for body angle to prevent tumbling and crash
    orientation_penalty = -w_angle_penalty * (next_obs[4] ** 2)

    # 3. Landing quality bonus: soft success proxy when contacts are active
    contact = max(next_obs[6], next_obs[7])  # any leg contact
    vertical_speed_factor = 1.0 / (1.0 + a_v * (next_obs[3] ** 2))
    angle_factor = 1.0 / (1.0 + b_angle * (next_obs[4] ** 2))
    landing_quality = w_landing * contact * vertical_speed_factor * angle_factor

    total_reward = progress + orientation_penalty + landing_quality

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # weights and sensitivities
    w_progress = 1.0
    w_angle_penalty = 0.2
    w_landing = 1.0
    a_v = 10.0       # sensitivity for vertical speed
    b_angle = 10.0   # sensitivity for body angle in landing quality

    # current and next distances to target (0,0)
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 1. Main progress signal: improvement towards target
    progress = w_progress * (dist - next_dist)

    # 2. Orientation stability: penalty for body angle to prevent tumbling and crash
    orientation_penalty = -w_angle_penalty * (next_obs[4] ** 2)

    # 3. Landing quality bonus: soft success proxy when contacts are active
    contact = max(next_obs[6], next_obs[7])  # any leg contact
    vertical_speed_factor = 1.0 / (1.0 + a_v * (next_obs[3] ** 2))
    angle_factor = 1.0 / (1.0 + b_angle * (next_obs[4] ** 2))
    landing_quality = w_landing * contact * vertical_speed_factor * angle_factor

    total_reward = progress + orientation_penalty + landing_quality

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-85.225247, len=69.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-117.831261, -56.370675]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 5.590425 | 85.2% | 88.4% | 100.0% |
| landing_quality | 0.751702 | 11.5% | 11.5% | 3.5% |
| orientation_penalty | -0.008576 | -0.1% | 0.1% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 2D 飞行器软着陆任务。目标是从初始位置（接近视图顶部中央，具有随机初始力）飞到中央的目标着陆平台，**平稳着陆并停稳**。任务主要目标是**成功、安全地到达目标并停靠**，附属目标包括**尽快完成**和**尽量节省燃料**（即尽量少用引擎）。智能体需要学会接近目标、减速、保持稳定姿态，并以低速、小角度接触平台。不要把“省燃料”或“快速”当作主目标，它们不能以牺牲着陆安全为代价。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: float32（默认）  
- 各维度含义与奖励可用性：
  - obs[0]: x_position（相对目标平台的水平偏移），reward_usable: true  
  - obs[1]: y_position（相对平台高度的垂直偏移），reward_usable: true  
  - obs[2]: x_velocity（水平线速度），reward_usable: true  
  - obs[3]: y_velocity（垂直线速度），reward_usable: true  
  - obs[4]: body_angle（机体倾斜角），reward_usable: true  
  - obs[5]: angular_velocity（角速度），reward_usable: true  
  - obs[6]: left_support_contact（左支撑腿接触标志，1.0 表示接触，0.0 表示未接触），reward_usable: true  
  - obs[7]: right_support_contact（右支撑腿接触标志），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 各动作含义：
  - action 0: no_engine（不点火，无推力）  
  - action 1: left_orientation_engine（点燃左姿态引擎，产生侧向力矩或推力）  
  - action 2: main_engine（点燃主引擎，提供主要上升/下降推力）  
  - action 3: right_orientation_engine（点燃右姿态引擎）

## 5. step 与终止条件分析
### 5.1 终止模式
根据环境源码，terminated 由以下条件之一触发：
- crash_or_body_contact：机体发生碰撞或非正常接触（例如翻滚导致机身触地），通常为失败。  
- horizontal_position_outside_viewport：水平位置超出可视边界，视为失控失败。  
- body_not_awake_or_settled：机体不再活跃或已稳定停靠。该终止可能意味着成功（已经停稳在平台上），但也可能因陷入静止失败状态而终止。  

**因此**：
- success-like termination：当 body_not_awake_or_settled 触发，且同时满足 left/right support contact 均为 1、位置靠近中心、速度极低、姿态接近水平。  
- failure-like termination：crash_or_body_contact 或 horizontal_position_outside_viewport 触发；或者虽然 settled 但状态不满足安全着陆条件（如姿态过大、偏离平台）。  
- ambiguous termination：body_not_awake_or_settled 本身需要结合观测状态才能判断是否成功。没有显式的“success”或“failure”标志。  
- truncation：环境中未观察到 episode 截断（步数限制未体现），但假设存在最大步数可能属于 truncation。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: {}（info 为空字典，根本不可用）  
- forbidden_or_uncertain_info_fields: original_reward, 以及任何未在 step 源码中显式返回的 info 字段。

## 7. 可用于奖励函数的信号
- position: x_position, y_position（相对目标平台的偏移）  
- velocity: x_velocity, y_velocity  
- orientation: body_angle, angular_velocity  
- contact: left_support_contact, right_support_contact  
- action/engine: 动作编号（0,1,2,3）  
- other: 无其他显式信号

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2d_lander_with_two_legs
  actuator_type: thrusters (main engine + orientation engines)
  contact_structure: two_point_contacts (left and right legs)
primary_objectives:
  - 平稳降落在目标平台上（left 和 right 腿接触，机体接近水平，速度极小）
  - 避免碰撞、翻滚或飞出视口
secondary_objectives:
  - 尽快完成着陆（时间效率）
  - 最少燃料消耗（尽量不用引擎）
main_failure_risks:
  - 姿态失控导致翻滚或机身直接撞地
  - 水平方向偏离过大飞出边界
  - 着陆时垂直速度过大导致硬着陆
  - 过早关闭引擎导致悬停失败
  - 过度使用引擎浪费燃料或过度调整姿态
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_landing_success
  purpose: 鼓励成功、稳定地着陆在平台上
  why_required: 任务核心；没有它智能体可能不学习着陆，或停留在空中不动
  usable_signals: [x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact]
  risks: 只能用接触和位置/速度推断成功，可能存在伪成功（如卡在平台边缘也触发接触）

- role_id: crash_and_out_of_bounds_prevention
  purpose: 惩罚碰撞、翻滚或飞出视口的行为
  why_required: 安全约束；必须让智能体避免致命失败
  usable_signals: [body_angle, x_position, y_position, left_support_contact, right_support_contact]
  risks: 角度阈值和位置边界需谨慎设定，过于严格可能阻止正常机动

- role_id: soft_landing_condition
  purpose: 接触平台时确保低速、姿态平稳
  why_required: 即使接触也会因为速度过快导致失败（被环境检测为 crash）
  usable_signals: [y_velocity, body_angle, left_support_contact, right_support_contact]
  risks: 过早惩罚速度可能阻碍接近目标，必须结合接触标志

- role_id: orientation_stability
  purpose: 保持机体尽量接近水平，防止翻滚
  why_required: 姿态过大易引发 crash，且是成功着陆条件之一
  usable_signals: [body_angle]
  risks: 过度惩罚正常转向可能妨碍姿态调整，需与角速度结合

- role_id: progress_towards_target
  purpose: 在飞行阶段引导智能体向目标平台移动
  why_required: 没有引导可能随机漂浮浪费时间
  usable_signals: [x_position, y_position, x_velocity, y_velocity]
  risks: 仅靠位置奖励可能导致智能体高速冲向平台，必须配合软着陆约束

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency
  purpose: 尽量减少引擎使用次数
  condition_to_use: 当智能体已具备基本着陆能力（成功率 > 阈值）后加入，或在训练后期逐步增强
  usable_signals: [action]
  risks: 过早加入会抑制必要的引擎使用，导致无法完成着陆

### 10.3 慎用/禁用职责 avoid_roles
- role_id: time_optimization
  reason: 观察空间中没有时间或步数信号，无法可靠测量时间效率；且容易与燃料节约冲突。
  forbidden_or_missing_signals: [time, step_count]

- role_id: exact_center_landing
  reason: 只要在平台上且稳定就行，无需极致中心对齐；过分强调可能使探索变得困难。
  related_signals: 无，但可用 x_position 过度约束

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_landing_success | x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact | explicit success flag | dense_state_signal (基于距离和速度的阈值判断) + bounded_signal (当条件满足时给予固定奖励) | 需要区分“在空中”和“已着陆”，使用接触标志与速度阈值结合 |
| crash_and_out_of_bounds_prevention | body_angle, x_position, y_position | explicit failure flag | quadratic_penalty (角度/位置超出安全范围) 或 bounded_signal (一旦超出给予负奖励) | 边界和角度阈值需调参 |
| soft_landing_condition | y_velocity, body_angle, left_support_contact, right_support_contact | 无 | dense_state_signal (接触时垂直速度和角度绝对值惩罚) | 仅在接触时激活 |
| orientation_stability |
```
