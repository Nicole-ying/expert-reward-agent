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
该匿名环境是一个 2D 飞行器轨迹优化任务。飞行器从视口顶部中央附近出发，初始带有随机作用力。任务核心目标是使飞行器尽快到达并稳定停靠在中央目标平台上，同时尽可能少地使用引擎推力。智能体需要在接近目标的过程中降低速度、保持姿态稳定，并以低速度、小角度实现双腿安全触地。次要目标是节省燃料，避免不必要的引擎动作。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心是到达指定目标平台并稳定停靠，具有明确的空间目标（目标垫中央），而保持姿态、节油等均为辅助要求，并非并列或多个冲突目标。尽管有接触条件，但本质仍是状态空间中的目标到达问题，因此属于导航/目标到达族。

动力学子类型：goal_approach_and_soft_contact（接近目标并低速、稳定接触/停靠）

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: 推断为 float32（未明确给出，但通常如此）
- 各维度含义：
  - obs[0]: x_position，飞行器相对于目标垫的水平坐标，reward_usable: true
  - obs[1]: y_position，飞行器相对于垫面高度的垂直坐标，reward_usable: true
  - obs[2]: x_velocity，水平线速度，reward_usable: true
  - obs[3]: y_velocity，垂直线速度，reward_usable: true
  - obs[4]: body_angle，机体姿态角，reward_usable: true
  - obs[5]: angular_velocity，角速度，reward_usable: true
  - obs[6]: left_support_contact，左支撑腿接触标志（0.0 或 1.0），reward_usable: true
  - obs[7]: right_support_contact，右支撑腿接触标志（0.0 或 1.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：
  - action 0: no_engine，不做任何推进
  - action 1: left_orientation_engine，点燃左侧姿态引擎
  - action 2: main_engine，点燃主引擎
  - action 3: right_orientation_engine，点燃右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- **body_not_awake_or_settled**：飞行器因稳定停靠而进入休眠或被判定为已安定，环境终止。此条件是**可能的成功终止**，但需要根据终止时的状态（位置接近目标、双腿接触、低速度、小角度）进一步判别。
- **crash_or_body_contact**：飞行器与地面或障碍物发生剧烈碰撞（可能倾覆或过猛接触），视为**失败终止**。
- **horizontal_position_outside_viewport**：飞行器水平飞出可视区域，视为**失败终止**。

（注：未给出 truncation 条件，无最大步数截断信息，但可以预见训练中可能设置时间上限，但环境源代码中未体现。）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 空（step 返回 info 为 {}，无可用的额外字段）
- forbidden_or_uncertain_info_fields: 禁止使用任何 info 字段，因为均为空。

成功与否必须完全从观测序列和终止时的状态（next_obs）中推断。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用：**
- `obs`：当前观测，8 维数值向量
- `action`：当前执行的动作，0~3 整数
- `next_obs`：下一观测，结构同 obs
- `info`：但当前环境 info 为空字典，实际不可用任何字段
- `training_progress`：仅在 prompt 明确允许或训练框架要求时使用，当前环境描述未明确允许，保持谨慎（可声明为“若允许则可使用”）

**禁止使用：**
- `original_reward`（已遮蔽，禁止重构）
- 任何未声明的 info 字段
- 任何未声明的 obs 切片或外部状态

## 7. 可用于奖励函数的信号
- **位置信号**：obs[0] (x_position), obs[1] (y_position)；next_obs 对应维度。
- **速度信号**：obs[2] (x_velocity), obs[3] (y_velocity)；next_obs 对应维度。
- **姿态信号**：obs[4] (body_angle), obs[5] (angular_velocity)；next_obs 对应维度。
- **接触信号**：obs[6] (left_support_contact), obs[7] (right_support_contact)；next_obs 对应维度。
- **动作/引擎使用信号**：action 值本身（0 为无推力，其他为使用引擎）。
- **其他**：终止信号 `terminated` 未直接进入 reward 函数，但可通过 `next_obs` 后是否结束来间接感知（训练循环外部，通常 reward 函数不接收 terminated 标志）。若环境提供 `training_progress`，可用于调度课程。

## 8. 不确定或不可用的信号
- 任何形如 `info["success"]`、`info["failure"]`、`info["landing_quality"]` 等官方成功/失败标签：**不可用**（info 为空）。
- 剩余燃料量、引擎推力强度等内部物理量：**未在观测中提供**，不可直接获取。
- 全局时间步数或剩余时间：**未提供**（除非通过 training_progress 间接获得）。
- 环境重置时的随机初始力大小：**不可使用**（仅在初始时影响，奖励函数不应依赖初始状态）。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete (4 actions)
morphology:
  body_type: rigid_body_with_two_legs
  actuator_type: impulse_based_main_and_orientation_engines
  contact_structure: left_and_right_leg_contact_sensors
primary_objectives:
  - 尽快到达目标平台中央并稳定停靠
  - 实现安全软着陆（低速度、小角度、双腿触地）
secondary_objectives:
  - 最小化引擎使用（省油）
main_failure_risks:
  - 飞行器硬撞击地面或倾覆
  - 水平漂出视野范围
  - 未能及时减速导致越标或错过平台
  - 姿态角过大导致触地后倾翻
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: goal_reaching_and_safe_contact**
  - purpose: 激励飞行器靠近目标平台并最终在满足成功条件时稳定停靠。
  - why_required: 这是任务的核心目标，无此奖励则学习无方向。
  - usable_signals: x_position, y_position, left_support_contact, right_support_contact, velocity/angle（用于判定成功状态）。
  - risks: 若只考虑位置而不考虑速度/接触，可能导致高速撞击；需与着陆质量奖励配合。

- **role_id: soft_landing_quality**
  - purpose: 惩罚触地时过大的速度、过大的姿态角以及角速度，鼓励双腿同时接触。
  - why_required: 保证安全着陆，避免硬着陆和倾翻，是任务的关键质量要求。




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
