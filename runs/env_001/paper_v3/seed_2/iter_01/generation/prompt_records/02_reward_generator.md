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
这是一个 2D 飞行器软着陆任务。目标是从初始位置（接近视图顶部中央，具有随机初始力）飞到中央的目标着陆平台，**平稳着陆并停稳**。任务主要目标是**成功、安全地到达目标并停靠**，附属目标包括**尽快完成**和**尽量节省燃料**（即尽量少用引擎）。智能体需要学会接近目标、减速、保持稳定姿态，并以低速、小角度接触平台。不要把“省燃料”或“快速”当作主目标，它们不能以牺牲着陆安全为代价。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是让飞行器到达指定的目标着陆平台并稳定停靠（goal reaching），速度与燃料消耗只是附属优化项，不存在多个权重相当且冲突的核心目标。

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

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（8维向量，当前步观察）
- action（当前步采取的动作）
- next_obs（8维向量，一步后的观察）
- info（但该环境 info 为空字典，实际没有可用字段）
- training_progress：**本次任务未明确允许，禁止使用**（默认禁止）

禁止使用：
- original_reward（已屏蔽，不可用）
- official_reward（同理）
- 任何未在信息空间列表里声明的 info 字段
- 任何未在观察空间中声明的 obs 切片
- 假设 terminated 或 reason 等额外信息

## 7. 可用于奖励函数的信号
- position: x_position, y_position（相对目标平台的偏移）  
- velocity: x_velocity, y_velocity  
- orientation: body_angle, angular_velocity  
- contact: left_support_contact, right_support_contact  
- action/engine: 动作编号（0,1,2,3）  
- other: 无其他显式信号

## 8. 不确定或不可用的信号
- original_reward：完全屏蔽，不可用。  
- 环境的成功/失败标志：不存在。  
- 燃料量：观察空间中无燃料信息，只能通过动作间接推测。  
- 时间惩罚信号：无步数和时间信息。  
- 目标平台的精确尺寸：未知，只能从观察空间中位置信号推断接触成功与否。  
- 视口边界：未提供，但可从位置异常判断。

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
