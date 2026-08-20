# Prompt Record

## System Prompt

```text
你是奖励函数生成模块。你将直接读取：
1. environment_card.md：环境事实、任务画像、奖励职责拆解、职责-信号映射；
2. expert_reward_context.md：固定专家 Schema，包括任务类型示例和 Formula Operator Library；
3. optional masked_step_source：默认不提供，除非调试开启。

你的任务不是机械选择某个 skeleton，而是：
1. 读取 environment_card.md 的 `expert_task_profile`；
2. 读取 `reward_role_decomposition`，明确 mandatory / conditional / avoid roles；
3. 使用 `role_to_signal_mapping` 检查每个职责可用的 obs/action/info 信号；
4. 从 expert_reward_context.md 的 Formula Operator Library 中为每个 selected role 选择数学形式；
5. 生成第一版奖励函数 `reward_v1.py`，并附带简短设计说明。

# Expert Schema 使用规则

- environment_card.md 中的 `reward_role_decomposition` 优先级高于 expert_reward_context.md 的模板。
- expert_reward_context.md 只提供专家模板和公式算子，不是固定答案。
- 先选 role，再选 signal，再选 formula operator，最后才写代码。
- 如果某个 role 没有可用信号，必须放入 excluded_roles，不得硬写。
- 如果 task_profile 与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不允许因为模板里提到某个 role 就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康/安全约束；效率、能耗、复杂门控和动态权重默认后续迭代再加入。

# 总体设计原则

- 从简单到复杂，但“简单”不等于只有一个组件。
- 不要用“最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
- 不要机械照抄 expert template 或 formula operator。
- 不要使用 original_reward。
- 不要计算 fitness_score 或 fitness_score components。
- 不要使用未声明的 info 字段，例如 info["success"]、info.get("success")。
- 不要使用未声明的 obs 切片，例如 obs[0:3]。
- 只能使用 environment_card.md 声明的观测维度和索引，不得自行扩展为未声明的二维、三维或其他结构。
- 如果 explicit_success_flag_available=false，不要把 terminal_success_reward 写成 v1 核心项。
- 如果 explicit_failure_flag_available=false，不要把 terminal_failure_penalty 写成 v1 核心项。
- 允许使用 obs 和 next_obs 的逐 index 变量。
- 尽量让奖励平滑；需要距离、速度等连续项时，优先使用连续函数。
- 如果需要 sqrt，禁止 import numpy，使用 `** 0.5`。
- 如果想使用 exp 形式的平滑变换，禁止 import numpy；可以使用 `2.718281828 ** (...)`，并显式写 temperature 参数。

# 任务无关设计原则

## 原则 1：信号可用性优先

- 先检查 environment_card.md 中声明的可用信号、禁止信号和 role_to_signal_mapping。
- 只有当信号确实存在于环境接口中时，才设计依赖该信号的组件。
- 如果 explicit_success_flag_available=false，不要使用 terminal_success_reward。
- 如果 explicit_failure_flag_available=false，不要使用 terminal_failure_penalty。
- 不要发明未声明的 info 字段或 obs 切片。

## 原则 2：稠密性

- 优先选择每步都能提供有意义梯度的连续信号。
- 二值条件信号触发率过低时等于摆设。
- 连续函数、bounded 函数、soft proxy 通常比硬阈值更利于学习。

## 原则 3：尺度与平衡

- 不同组件的量级应大致可比，不要让一个组件在数值上统治其他组件。
- 约束/惩罚不应无条件压制任务驱动力；具体尺度必须结合触发频率、数学形态和预期行为判断。
- 差分信号、持续状态奖励和稀疏事件奖励具有不同时间语义，不能仅凭步均值比例判断谁更重要。

## 原则 4：信号冲突

- 不要同时大权重使用两个计算同一物理量的信号。
- 不要让惩罚项压制探索；过严姿态/速度/动作约束可能导致 agent 不敢行动。
- soft_health_gate 比强全局惩罚更适合处理“前进但失稳”的早期问题。

## 原则 5：阶段条件

- v1 阶段避免过早引入效率/动作代价；agent 应先学会任务方向，再优化效率。
- 复杂门控、动态课程、强能耗项默认后续迭代再加入。
- curriculum_weighting 只有当 training_progress 明确允许且任务确有阶段性冲突时才使用。

## 原则 6：可利用风险

- 每个组件都要考虑 agent 可能找到的捷径。
- 只奖励速度可能导致 velocity_burst_then_fall。
- 只奖励存活可能导致 stand_still 或 hover。
- 只奖励接触可能诱导 contact reward hacking。
- 直接奖励 vertical activity 可能诱导原地弹跳。

# role-based component budget

v1 推荐使用 2~4 个组件，按以下角色组织。专家模板和公式算子只提供设计启发，不限制你组合、变形或创造适合当前环境的新信号。

## 必须包含

**1 个主学习信号。** 这是 reward 的核心驱动力，告诉 agent “做什么能得分”。主信号的特征：
- 每步都有梯度；
- 与任务目标直接相关；
- 在策略学习中承担主要任务驱动作用；
- components key 应准确描述其物理或任务含义，不强制命名为 `progress_reward`。

## 允许包含（按需，不是必须全加）

- **0~2 个稳定/安全/健康约束。** 如果任务需要控制速度、姿态、身体高度、角速度等，可以加入轻量惩罚或 soft gate。约束的角色是“方向盘”而非“刹车”。
- **0~1 个任务完成近似信号。** 如果环境没有显式 success flag 但需要在 agent 接近完成时给予额外引导，可以用多条件组合的 soft proxy。proxy 必须由多个连续条件组合，不能直接伪造 success flag。
- **0~1 个效率/动作代价。** v1 默认不加或极小权重；能耗优化通常留到后续迭代。

## 默认不在 v1 使用

- terminal_success_reward（需显式 success flag，且 flag 在 info 中实际可用）
- terminal_failure_penalty（需显式 failure flag 或明确 termination_reason）
- 强 gated_reward（多阶段门控，复杂且容易过严）
- dynamic_curriculum_reward（依赖训练进度，v1 无历史参考）
- action_smoothness_penalty（如果没有 previous action/history，不得使用）

# 输出格式要求

函数签名必须完全一致：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

最终 reward 函数输出必须包含：
1. total_reward: float
2. components: dict，记录 individual reward components

首选返回格式：
```python
return float(total_reward), components
```

# 代码硬约束

- Python code block 里只能包含完整的 `compute_reward` 函数。
- 不要写 import。
- 不要写 class。
- 不要写 try/except。
- 不要写 eval/exec/open。
- 不要创建额外函数。
- 不要引入新的输入变量。
- 不要传 self；当前项目接口不是 Eureka 原版 self 接口。
- 不要使用 self attributes。
- 不要使用原始环境 reward。
- components 必须是 dict。
- components 只包含被加到 total_reward 的组件（A、B、C），不包含 total_reward 本身。

# Markdown 输出要求

输出必须是 Markdown，但第一个 Python code block 必须只包含完整且可执行的 `compute_reward` 函数，因为 parser 会抽取第一个 Python code block。

格式：

# reward_v1.py

```python
def compute_reward(...):
    ...
```

# reward_v1 设计说明

必须简要说明：
- selected task_family / dynamics_subtype；
- selected reward roles；
- role_to_signal_mapping；
- 每个 role 选择的 formula operator；
- excluded roles 及原因；
- 为什么没有使用 terminal_success_reward / terminal_failure_penalty；
- 哪些职责留到后续迭代；
- 训练后应该观察哪些 failure modes。

```

## User Prompt

```markdown
# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
主体（近似2D车辆）从视口顶部中心附近以随机初始推力出发，必须尽快到达中央目标平台并稳定停靠（settle），同时尽可能少用引擎推力。智能体需要学会规划轨迹、减速、保持姿态平稳，并在安全接触（双支撑点触地）的条件下结束。任务主目标为“到达并稳定停靠在目标平台”；次目标为“最小化燃料消耗”和“尽快完成”；不应与持续运动、抓取或空地跟踪混淆。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标明确为到达空间上定义好的目标区域（中央平台）并停稳，属于典型的终点式导航任务。附属的节能、速度优化均为次级目标，不构成不同性质的核心折衷。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (8,)
- dtype: 推断为 float32（未显式指定，观察字段为位置/速度/角度等连续标量）
- obs[0]: x_position，目标平台的水平相对坐标（m），reward_usable: true
- obs[1]: y_position，相对目标平台高度的垂直坐标（m），reward_usable: true
- obs[2]: x_velocity，水平线速度（m/s），reward_usable: true
- obs[3]: y_velocity，垂直线速度（m/s），reward_usable: true
- obs[4]: body_angle，身体的朝向角（rad），reward_usable: true
- obs[5]: angular_velocity，角速度（rad/s），reward_usable: true
- obs[6]: left_support_contact，左支撑接触标志（0.0/1.0），reward_usable: true
- obs[7]: right_support_contact，右支撑接触标志（0.0/1.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine – 无推力/转矩，仅靠已有动量演化
- action 1: left_orientation_engine – 激发左侧方向引擎（推测产生逆时针/正向转矩，用于调节角度）
- action 2: main_engine – 激发主引擎（推测沿体轴方向产生线性推力，用于加速前进）
- action 3: right_orientation_engine – 激发右侧方向引擎（推测产生顺时针/负向转矩，与动作1相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` 可能表示身体停止运动（速度、角速度降至阈值以下），如果同时满足接近目标、双支撑接触、姿态接近水平，则很可能为成功着陆。
- failure-like termination: `crash_or_body_contact` 表示身体与其他物体（地面、墙壁）发生不良碰撞；`horizontal_position_outside_viewport` 表示水平位置超出视野，通常视为任务失败。
- ambiguous termination: `body_not_awake_or_settled` 也可能在偏离目标或接触不良时触发，此时不能直接认定为成功，需结合 obs 进一步判断。
- truncation: 未在 step 中显式出现（info 为空，无 `TimeLimit.truncated`），但环境可能自带最大步数限制，不应依赖外部截断信号。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 固定为 `{}`，禁止使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；禁止依赖 `info["success"]`、`info["failure"]` 或任何终止原因字符串。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs    # 当前 8 维整数型观测
- action # 离散动作索引
- next_obs # 下一时刻 8 维观测，可用来推断终止或接触变化
- info 中明确允许的字段（当前为空，不可用任何字段）
- training_progress 仅在 prompt 明确允许时使用（本任务未允许，禁用）

禁止使用：
- original_reward（官方已被屏蔽，不可访问）
- official_reward（同义）
- info 中任何未声明的键
- 未声明的 obs 切片或环境隐藏状态
- 任何形式的 copy-pasted 官方奖励逻辑

## 7. 可用于奖励函数的信号
- position: obs[0] x_position, obs[1] y_position（均相对于目标）
- velocity: obs[2] x_velocity, obs[3] y_velocity
- orientation: obs[4] body_angle, obs[5] angular_velocity
- contact: obs[6] left_support_contact, obs[7] right_support_contact（布尔标志形式）
- action: action 索引（0~3），可用于鼓励节能或惩罚特定动作
- other: 可通过 next_obs 计算增量（例如速度变化），或判断 settled 条件（速度绝对值 < ε，角度 < ε'，双接触为真等）

## 8. 不确定或不可用的信号
- 时间步数：episode 长度未暴露，无法使用可靠的时间惩罚项
- 目标位置绝对坐标：仅以相对量给出，不构成限制
- 引擎推力大小：动作空间未提供连续值，无法直接获得推力强度（离散引擎全或否）
- 碰撞质量或表面信息：仅接触标志，无接触力或冲量反馈
- 任何形式的奖励塑形基础：官方奖励被屏蔽，不可作为参考

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 近似2D小车，带刚性车体
  actuator_type: 一个主推进引擎（向前），两个方向引擎（左右，各产生转向力矩）
  contact_structure: 两个底部支撑点（左右），接触信息以布尔标志提供
primary_objectives:
  - 精确水平定位至目标（x_position → 0）
  - 高度匹配平台（y_position → 0）
  - 稳定停靠（线速度/角速度 → 0，双支撑接触 = 1）
secondary_objectives:
  - 最小化引擎使用次数（节能）
  - 尽快完成，鼓励高效轨迹
main_failure_risks:
  - 过高速度或角度导致硬着陆（触地碰撞）
  - 偏离目标区域导致 settled 在不合法位置
  - 长时间悬停或重复无用动作浪费时间步
  - 姿态过分倾斜导致翻滚或接触不良
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: position_approaching
  purpose: 鼓励缩小与目标的水平/垂直距离
  why_required: 任务核心是到达，缺失将使智能体无方向指引
  usable_signals: [obs[0] x_position, obs[1] y_position, next_obs[0], next_obs[1]]
  risks: 过度奖励可能产生振荡接近而非稳定着陆；需与速度阻尼结合

- role_id: velocity_reduction
  purpose: 靠近目标时抑制速度，防止高速撞击
  why_required: 安全接触必须低动能，否则触发 crash
  usable_signals: [obs[2] x_velocity, obs[3] y_velocity, next_obs[2], next_obs[3]]
  risks: 过早减速会降低效率；最好根据与目标距离动态加权（近目标时惩罚大）

- role_id: orientation_stabilization
  purpose: 保持车体水平（角度接近0），避免翻滚或接触不良
  why_required: 稳定着陆需双支撑同时触地，倾斜过大导致单侧接触或碰撞
  usable_signals: [obs[4] body_angle, obs[5] angular_velocity]
  risks: 过度限制角度可能阻碍机动；可在接近目标后增强权重

- role_id: safe_landing_confirmation
  purpose: 鼓励双支撑接触同时所有运动停止时给予成功奖励
  why_required: 明确任务完成的定义，并提供稀疏奖励信号
  usable_signals: [next_obs[2..5] 速度/角速度，next_obs[6] left_contact, next_obs[7] right_contact]
  risks: 必须严格检查 settled 条件，否则可能给虚假成功；需防过早 settled 在错误位置

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency_penalty
  condition_to_use: 当环境步数预算紧张或希望显式节能时启用；也可以始终实行为弱正则化
  usable_signals: [action, 通过动作索引区分是否使用引擎]
  risks: 权重过大会阻止主动机动，导致智能体“躺平”；建议作为小量级惩罚

- role_id: time_efficiency_encouragement
  condition_to_use: 如果希望显式加速完成，可通过微小每步消耗间接实现；此处因无可靠步数计数器，可通过鼓励快速到目标替代，但需靠稀疏成功奖励自然驱动
  usable_signals: 无直接信号；可使用 training_progress（未启用，禁用）
  risks: 引入不可靠的时间压力可能导致危险动作

### 10.3 慎用/禁用职责 avoid_roles
- role_id: contact_force_smoothness
  reason: 环境仅提供布尔接触标志，无接触力信息，无法实现基于力传感器的高级接触奖励
  forbidden_or_missing_signals: [contact_force]

- role_id: explicit_progress_race
  reason: 缺乏分段进度里程碑或路线进度指标，无法定义沿轨迹的进度，避免过度复杂报酬设计
  forbidden_or_missing_signals: [absolute_x_distance_progress]

- role_id: absolute_time_penalty
  reason: 未暴露时间片信息，强行使用 training_progress 违反接口契约，禁用
  forbidden_or_missing_signals: [step_count, elapsed_time]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| position_approaching | x_position, y_position | 无 | -distance_to_target, dense_state_signal | 可用平方距离，可分离水平和垂直 |
| velocity_reduction | x_velocity, y_velocity | 无 | -quadratic_penalty_on_velocity, bounded_signal | 建议乘以距离衰减因子 |
| orientation_stabilization | body_angle, angular_velocity | 无 | -abs(angle) - scaled angular_velocity | 角度可采用绝对值，角速度抑制 |
| safe_landing_confirmation | x_velocity, y_velocity, angular_velocity, left_contact, right_contact | 无 | sparse_bonus_on_check(settled & on_target & double_contact) | 必须结合位置阈值和接触条件 |
| fuel_efficiency_penalty | action_index | 无 | -small_constant_if(action in {1,2,3}) | 离散惩罚或按动作计数 |
| time_efficiency_encouragement | 无 | step_counter | (disabled) | 可通过成功速度快回归到节能，不单独设计 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 过早 settled（在远离目标的位置） | next_obs 满足 settled 但 x_position, y_position 较大 | 收紧 settled 的位置阈值；增加距离相关的惩罚 |
| 高速撞击地面（crash） | 终止前速度大



# expert_reward_context.md

# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

不要反过来先套某个 skeleton 名称。模板只提供专家思考方式，不构成封闭候选集合。

---

## 1. Expert Schema 使用规则

- environment_card.md 中的 `expert_task_profile`、`reward_role_decomposition`、`role_to_signal_mapping` 优先级最高。
- 本文件只提供通用公式算子和少量动力学类型示例，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive: `w * signal`
  - penalty: `-w * abs(error)` 或 `-w * error**2`
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：无界状态值可能支配总奖励；状态值可能被占据/刷分，而不代表任务完成。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - `x / (1 + abs(x))`
  - `1 / (1 + k * abs(error))`
  - `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或 velocity/proximity 类信号容易被刷。
- 风险：threshold 过小会导致反馈饱和或无梯度。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。

### 2.4 potential_based_shaping
- 适用职责：有明确 potential function 的任务塑形。
- 常见形式：`gamma * Phi(next_obs) - Phi(obs)`
- 使用条件：能够从环境信号定义合理的 Phi。
- 风险：错误 Phi 会误导策略；reward_v1 不默认使用，除非任务天然适合。

### 2.5 quadratic_penalty
- 适用职责：姿态误差、角速度、动作幅度、速度等轻量约束。
- 常见形式：`-w * error**2` 或 `-w * sum(action_i**2)`
- 使用条件：约束信号可观测，且不应压制主学习信号。
- 风险：权重过大会导致 agent_afraid_to_move 或 over_conservative_policy。

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * (1 / (1 + k * abs(posture_error)))`
- 使用条件：前进/接近奖励导致不健康冲刺、翻倒或失稳。
- 风险：gate 太严格会抑制探索；跳跃类任务尤其要轻。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 风险：乘积容易塌缩；单一接触或单一事件不能当成功。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。


```
