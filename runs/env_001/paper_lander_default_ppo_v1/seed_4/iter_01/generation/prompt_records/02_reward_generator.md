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
本任务是一个 2D 飞行器 / 着陆器轨迹优化问题。  
主体从视野顶部中央附近出发，并带有随机初速度。核心目标是**尽快到达并稳定停靠在画面中央的目标平台上、且保持姿态平稳**；次要目标是在此过程中尽量减少引擎推力消耗。需要关键控制能力：接近目标、减速、维持直立姿态并通过左右支撑点与平台安全接触。  

## 2. 任务类型选择
- **selected_route_id**: `navigation_goal_reaching`
- **confidence**: high
- **reason**:  
  任务的首要目标是到达一个指定的固定目标位置（中央平台）并稳定停留。速度与能耗均为从属优化项，不存在目标间权重相当且不可调和的情况，因此归为“导航/到达目标”大类，而非多目标或存活类。  

- **dynamics_subtype**: `goal_approach_and_soft_contact`  
  主体需要接近平台、减速、保持姿态并最终以低速与平台柔性接触，符合“接近目标并低速、稳定接触/停靠”的特征。  

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

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：
- `obs` —— 当前观测
- `action` —— 执行的动作
- `next_obs` —— 下一步观测（含变化后的位置、速度、姿态、接触信息）
- `info` 中**明确允许**的字段（当前为无字段可用）
- `training_progress` —— 仅在 prompt 明确允许时使用（当前未提及，暂不用）

**禁止使用**：
- `original_reward`（已掩码）
- `official_reward` 类似的任何内部奖励信号
- 未声明的 `info` 字段（如 `success`、`failure`）
- 未在任务说明中给出的 `obs` 切片语义

## 7. 可用于奖励函数的信号
- **position**: `x_position`, `y_position`（相对平台坐标，可用以引导接近与保持在目标上方）
- **velocity**: `x_velocity`, `y_velocity`（减速靠近平台）
- **orientation**: `body_angle`, `angular_velocity`（维持姿态平稳）
- **contact**: `left_support_contact`, `right_support_contact`（稳定接触检测）
- **action/engine**: 可以针对推力动作施加惩罚（如 `action != 0`）以鼓励节省燃料
- **other**: 无额外信号。

## 8. 不确定或不可用的信号
- 明确的成功/失败标志（`info` 中不存在）
- 平台绝对位置（仅提供相对位置，隐含平台位置固定于原点）
- 剩余的燃料量（未提供）
- 绝对的引擎推力值（动作仅为离散开关）
- 平台的形状、宽度信息（未显式给出，但可通过接触推断）

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

---

## 3. Expert Task Template A: planar_monoped_hopping

### 适用任务线索
- 平面单腿或少腿跳跃式前进；
- 观测中有 torso_height、torso_angle、forward_velocity、vertical_velocity、joint angles/speeds；
- 动作为连续关节力矩；
- 终止通常与高度、躯干角度或状态非法有关；
- 任务要求 sustained forward progress，而不是只保持直立。

### 主职责 mandatory reward roles
1. sustained_forward_progress
   - 目的：鼓励持续向前运动，而不是短时间冲刺。
   - 可用信号：forward_velocity。
   - 推荐算子：dense_state_signal、bounded_signal、soft_health_gate。
   - 风险：velocity_burst_then_fall、shuffling_without_real_hop。

2. healthy_posture_or_height_constraint
   - 目的：保持身体高度和躯干姿态在健康范围附近，同时允许必要跳跃动态。
   - 可用信号：torso_height、torso_angle、torso_angular_velocity。
   - 推荐算子：quadratic_penalty、bounded_signal、soft_health_gate。
   - 风险：约束过强会抑制跳跃，使 agent 不敢探索。

### 条件职责 conditional reward roles
1. light_energy_regularization
   - 条件：只有当策略已经能产生前进后再加入。
   - 可用信号：action。
   - 推荐算子：quadratic_penalty。
   - 风险：过早加入会导致 energy_freeze。

2. vertical_dynamics_regularization
   - 条件：如果策略只是滑行、乱跳或原地弹跳，再考虑轻量垂直动态约束。
   - 可用信号：vertical_velocity、torso_height。
   - 推荐算子：bounded_signal、quadratic_penalty。
   - 风险：直接奖励 vertical activity 可能导致原地弹跳。

### 慎用/禁用 avoid roles
- bipedal_alternating_contact_reward：单腿任务不适配；
- contact_reward_without_contact_signal：没有接触信号时不能使用；
- unconditional_alive_bonus：容易站着不动或拖时间；
- strong_vertical_activity_reward：容易学成原地弹跳。

---

## 4. Expert Task Template B: multi_legged_body_locomotion

### 适用任务线索
- 多足身体或高维关节身体；
- 动作为连续 torque；
- 目标是沿某一方向持续前进；
- 观测可能包含 body orientation、body velocity、joint positions/speeds、actions；
- 失败通常表现为翻滚、侧向漂移、腿部乱动、能耗过高或原地抖动。

### 主职责 mandatory reward roles
1. directional_forward_progress
   - 目的：鼓励沿目标方向前进。
   - 可用信号：forward_velocity 或对应方向速度。
   - 推荐算子：dense_state_signal、bounded_signal。
   - 风险：sideways_drift、velocity_farming、thrashing_forward。

2. body_orientation_health
   - 目的：避免躯干翻滚、侧翻或极端姿态。
   - 可用信号：torso_orientation、body_angle、angular_velocity。
   - 推荐算子：quadratic_penalty、bounded_signal、soft_health_gate。
   - 风险：过强会导致 over_conservative_policy。

### 条件职责 conditional reward roles
1. light_action_energy_regularization
   - 条件：策略已有明显前进后再加入。
   - 可用信号：action。
   - 推荐算子：quadratic_penalty。
   - 风险：过强会导致 agent_afraid_to_move。

2. lateral_drift_control
   - 条件：只有当侧向速度或姿态信号明确可用时使用。
   - 可用信号：side_velocity、body_orientation。
   - 推荐算子：quadratic_penalty、bounded_signal。
   - 风险：如果没有明确信号，不得伪造。

### 慎用/禁用 avoid roles
- bipedal_alternating_gait：多足身体不一定需要双足交替步态；
- monoped_vertical_hopping：多足行走不应被引导成跳跃；
- contact_reward_without_contact_signal：没有接触信号时不能使用；
- unconditional_alive_bonus：容易导致站立或原地抖动。

---

## 5. reward_v1 生成要求

- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。
- 使用 role-based component budget：每个组件必须有明确职责，不能为了显得完整而堆叠。
- 以 environment_card.md 的 reward_role_decomposition 为主，本文件模板为辅。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类职责、复杂门控和 curriculum_weighting 默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 `return float(total_reward), components`；components 必须是 dict。

```
