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
任务主目标：控制一个二维飞行器从顶部中心区域出发，快速、平稳地降落到画面中央的目标着陆垫上，并在着陆瞬间保持稳定的姿态和速度，使左右支撑腿都与垫面接触。次目标：在满足成功着陆的前提下，尽可能缩短到达时间和节省发动机推力（燃料）。不应混淆为目标只是悬停或单纯到达上空，必须实际软着陆并稳定停靠。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 核心目的是到达并稳定停留在指定目标点（着陆垫），属于“导航到目标”任务族。附属目标（快速、省燃料、姿态稳定）是为了提升着陆质量，不改变任务主目标，因此不属于多目标任务。动力学子类型选为 goal_approach_and_soft_contact，因为要求接近目标、减速、控制姿态并安全接触。

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

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```
允许使用：
- obs：当前观察向量（8维）
- action：当前执行的动作（0~3）
- next_obs：执行动作后的下一观察向量（8维）
- info：空字典 {}，无可用字段
- training_progress：暂时无需使用（prompt 未明确要求）

禁止使用：
- original_reward（被屏蔽的内部奖励）
- 任何未在 obs/next_obs 中明确声明的信号
- 任何 info 字段

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

## 8. 不确定或不可用的信号
- 目标位置不是观测的一部分，而是通过 obs[0] 和 obs[1] 的相对值体现；无法直接获得目标的绝对坐标或是否达到目标区域的 bool 标志。
- 没有显式的 "success" 或 "landed" 布尔信号。
- 没有燃料剩余量或时间步数的直接信息（除非环境提供，但这里没有）。
- 没有外部 reward 或 info 中的阶段指示。

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
