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

- 从简单到复杂，但”简单”不等于只有一个组件。
- 不要用”最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
- 写完 reward 后自检：① 每个终止条件是否有前兆软信号？② 任务目标是否有直接的进度信号？③ 动作维度 ≥ 6 时，是否缺少效率约束（即使权重很小）？
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
本环境是一个 2D 飞行器轨迹优化任务。初始时，飞行器从视口顶部中心附近以随机初速度开始运动。核心目标是**尽快且平稳地降落在中央目标垫上**，同时**尽量减少发动机使用**。学习者必须学会接近目标、主动减速、保持机身姿态稳定，并在所有支撑腿（或接触点）安全触垫后停止动作。失败的着陆（碰撞、飞出水平边界）应彻底避免。注意：节能是次目标，不应凌驾于成功着陆之上。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**
confidence: high
reason: 核心是到达并停靠在指定目标位置（目标垫），属于典型的导航-目标到达任务。节能是附属优化，不是等价多目标。

dynamics_subtype: **goal_approach_and_soft_contact**  
说明：飞行器在重力（隐含）与推进力作用下运动，需主动控制姿态与主推力，实现**接近 → 减速 → 稳定姿态 → 软接触**的连串过程，符合接近目标并低冲击接触的动力学特征。

## 3. 观察空间 observation_space
- type: Box（连续向量）
- shape: (8,)
- dtype: float32（推断）
- obs[0]: x_position —— 飞行器中心相对目标垫中心的水平坐标。reward_usable: true
- obs[1]: y_position —— 飞行器底部（或重心）相对垫面的垂直高度。reward_usable: true
- obs[2]: x_velocity —— 水平线速度。reward_usable: true
- obs[3]: y_velocity —— 垂直线速度。reward_usable: true
- obs[4]: body_angle —— 机身倾斜角。reward_usable: true
- obs[5]: angular_velocity —— 角速度。reward_usable: true
- obs[6]: left_support_contact —— 左支撑/腿触地标志（0/1）。reward_usable: true
- obs[7]: right_support_contact —— 右支撑/腿触地标志（0/1）。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine（无任何推力，仅靠惯性/重力）  
- action 1: left_orientation_engine（启动左侧姿态发动机，产生逆时针/向右的力矩）  
- action 2: main_engine（启动主发动机，向上产生推力）  
- action 3: right_orientation_engine（启动右侧姿态发动机，产生顺时针/向左的力矩）  

主发动机提供垂直方向推力，姿态发动机用于调整倾斜角度从而改变水平推力方向，典型的**主推力+姿态控制**模式。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  终止条件中的 `body_not_awake_or_settled` 极可能对应成功着陆后的稳定状态：飞行器停在目标垫上，速度降至零，物理引擎判定其静止休眠。
- failure-like termination:  
  - `crash_or_body_contact`：飞行器主体（非支撑腿）触地或与障碍物碰撞。  
  - `horizontal_position_outside_viewport`：水平坐标超出视口范围。
- ambiguous termination:  
  理论上 `body_not_awake_or_settled` 也可能因其它原因（如卡在边界外）触发，但实际环境中通常与成功着陆强关联。
- truncation: 本环境无时间截断（step 返回 truncated 固定为 False）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空字典）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空）
- forbidden_or_uncertain_info_fields: 无任何 info 字段
- 成功/失败只能**间接推断**：  
  当 episode 终止时，通过最终观测（位置、速度、角度、接触标志）综合判断是否成功着陆：
  - 推断成功条件：`|x_position| < ε_x`，`|y_position| < ε_y`，`√(vx²+vy²) < ε_vel`，`|body_angle| < ε_ang`，且至少一个接触标志为 1。
  - 否则为失败。  
  该推断路径标记为 **derived_possible**。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`（动作执行前的状态）
- `action`（选择的动作索引）
- `next_obs`（动作执行后的状态）
- `info` 中明确允许的字段（当前为无，因此不可用 info）
- `training_progress` 仅有明确指示时才使用（本例无需课程式奖励，不使用）

禁止使用：
- `original_reward` / `official_reward`（已屏蔽）
- 任何未在 obs 说明中声明的 obs 切片
- 任何未在允许列表中的 info 字段

## 7. 可用于奖励函数的信号
**位置与接近度**：
- `next_obs[0]`（x 偏移）：越靠近 0 越好。
- `next_obs[1]`（y 偏移）：越靠近 0 越好，且应在安全范围内。
- 可直接计算距离 `dist = sqrt(x² + y²)` 或步间距离减少量 `delta_dist`。

**速度**：
- `next_obs[2]`（vx）、`next_obs[3]`（vy）：着陆时需极低速度，飞行中可适度引导向下减速。

**姿态**：
- `next_obs[4]`（body_angle）：应保持在安全区间内（例如 ±0.3 rad），防止侧翻。
- `next_obs[5]`（angular_velocity）：应趋近 0。

**接触**：
- `next_obs[6]`（左触地）、`next_obs[7]`（右触地）：成功着陆通常需要双接触（或至少一腿触地且速度达标）。

**动作**：
- `action` 可用来惩罚发动机使用（尤其 punishment for engine actions），但禁止奖励“无动作”。

**终端事件（derived_possible）**：
- 由最终 `next_obs` 推断的成功奖励（大正分）或失败惩罚（负分）。

## 8. 不确定或不可用的信号
- 燃料/能量消耗量：未提供。
- 目标垫的绝对坐标：未提供，只有相对位置。
- 地面高度/垫高度：未直接给出，但 y_position 已是相对垫面高度，可认为垫面在 y=0 处。
- 环境时间步长：未知，但可假设固定，无需使用。
- 任何 info 字段：不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_with_two_contacts (left/right support)
  actuator_type: main_engine_and_orientation_thrusters
  contact_structure: two_legs_or_pads_contact_with_ground
primary_objectives:
  - land_on_center_pad: 飞行器最终静止在目标垫上方，双接触点触垫，姿态近乎水平。
secondary_objectives:
  - minimize_engine_usage: 减少总动作次数或推力积分。
  - smooth_and_stable_approach: 避免剧烈摆动、高速冲击。
main_failure_risks:
  - crashing_body: 主体接触地面或障碍物导致坠毁。
  - drifting_out_of_viewport: 水平移动超出可观测区域。
  - hard_landing: 速度过大或角度倾斜导致侧翻，即使接触垫也视为失败（可通过身体接触垫判定失败？注意 crash_or_body_contact 可能包含了除腿以外的身体触地，所以触垫后若腿部接触而身体未触则为安全，身体触垫触发 crash）。
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: approach_progress  
  purpose: 引导飞行器向目标垫移动，每步因为距离减小而获得正反馈。  
  why_required: 要到达目标，必须持续缩短与垫的相对距离。若只有稀疏终端奖励，探索困难。  
  usable_signals: [x_position, y_position] (计算 distance)  
  risks: 若使用状态值（proximity）可能造成悬停收割（停在远处但距离固定，仍能得分）。因此应采用**步间距离减少量（delta distance）作为主信号**，鼓励每一步更近。当距离很小时可切换为接近 0 的距离接近奖励以保证接触。  
  候选算子：`delta(distance)`, `improvement`。

- role_id: soft_landing_condition  
  purpose: 确保最终接触时满足低速度、小倾斜角、双接触（或至少一腿触地）。  
  why_required: 任务要求 “make safe contact”，单纯到达位置不足以成功。  
  usable_signals: [x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact]  
  risks: 在飞行过程中过早接触应该给予惩罚（如飞行中无意触地）。因此需将该职责与“非终止阶段”区分，仅当检测到接触且速度/角度符合安全着陆条件时给予正奖励，否则如果接触状态出现但未满足条件则视作 crash 惩罚。  
  候选算子：`hinge_penalty(velocity)`, `bounded_signal(body_angle)`, `sparse_terminal_bonus`。

- role_id: orientation_stability  
  purpose: 维持机身水平，防止旋转失控。  
  why_required: 大角度可能导致侧翻或触地失败，且影响主发动机推力方向。  
  usable_signals: [body_angle, angular_velocity]  
  risks: 过强限制可能限制必要的姿态调整。可采用 **hinge penalty**：只在 angle 超过安全范围（如 ±0.3 rad）时施加惩罚，或在角速度过高时惩罚。  
  候选算子：`hinge_penalty`, `angular_velocity_penalty`。

### 10.2 条件职责 conditional_roles
- role_id: engine_efficiency  
  condition_to_use: 当任务明确要求“省燃料”或 agent 开始出现频繁无效点火时启用。  
  usable_signals: [action]  
  risks: 若过早加入会抑制探索，导致 agent 不敢使用主发动机。建议在训练初期权重很低或延迟引入（curriculum）。  
  候选算子：`per_step_action_cost`（对非零动作施加小惩罚）。

- role_id: terminal_success_bonus (derived_possible)  
  condition_to_use: 仅在 episode 终止且能可靠推断成功时使用。  
  usable_signals: [final x_position, y_position, velocity, angle, contacts]  
  risks: 推断错误会奖励失败行为。必须设置严格的阈值且配合调试输出验证准确率。  
  候选算子：`sparse_terminal_bonus`, `conditional_bonus`.

- role_id: crash_penalty (derived_possible)  
  condition_to_use: 仅在 episode 终止且推断为失败（crash 或 out of bounds）时施加。  
  usable_signals: [final x_position, y_position, contacts, and any rule for crash]  
  risks: 同上，需谨慎推断。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: survival_time_reward  
  reason: 任务不是存活类型，停滞不动不能得分，时间奖励会鼓励不降落、反复悬停消耗步数。禁用。  
- role_id: constant_forward_velocity  
  reason: 没有固定前向轴，下降速度过快反而是危险。本环境应减速，而非维持高速前进。禁用。  
- role_id: path_smoothness_penalty (action rate)  
  reason: 发动机切换是离散动作，但可作为 optional 燃料效率的一部分，但非必需，且可能阻碍探索。暂不入围。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_progress | x_position, y_position (derived distance) | – | delta



# expert_reward_context.md

# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

---

## 1. Expert Schema 使用规则

- environment_card.md 中的任务画像和可用信号优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role（任务需要什么类型的奖励信号），再选 signal（哪个观测维度承载这个 role），再选 formula operator（用什么数学形式表达），最后写代码。
- 如果某个 role 需要的信号在观测空间中不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- reward_v1 以主学习信号和必要的稳定/安全约束为重点。效率、能耗、复杂门控和动态权重可以在后续迭代中按需加入，但不应因"模板没列"而排除合理的设计。

---

## 2. 信号完备性自查清单

在完成初始设计后，逐一检查以下信号类型是否被覆盖——不是每个任务都需要全部，但每一项的缺失应是有意选择：

- **主进展信号**：agent 朝任务目标前进时是否获得正向反馈？该信号是否每步都有梯度？
- **灾难性失败信号**：是否存在明确的终止惩罚（如摔倒、飞出边界）？如果观测中可推断失败状态，是否给予了足够强的负向信号？
- **效率/代价信号**：连续动作空间中是否有能量消耗或控制代价约束？离散动作空间中是否有不必要的动作惩罚？
- **任务完成信号**：终止条件中是否包含 success-like 条件？相应的观测是否可被用来构造任务完成的软近似信号？
- **健康/稳定约束**：agent 是否因缺少姿态/速度/位置约束而产生不安全行为？

---

## 3. Formula Operator Library

每个算子包含：数学形式、使用条件、适用证据。

### 3.1 dense_state_signal
数学形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2`
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
使用条件：该状态信号每步可观测，且与某项任务职责直接相关。
适用证据：
  - 凸化 → episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小（agent 满足于低水平稳态）。
  - hinge → 约束组件的 active_rate≈100%（全时惩罚）但 terminated 率仍高，说明 agent 在安全范围内也被持续惩罚，需要只在越界时生效的 hinge。
风险：线性正奖励在信号平台期无梯度；凸化权重过大可能诱导极端行为；hinge 的 threshold 需根据环境卡片的观测范围设定。

### 3.2 improvement_delta
数学形式：`old_measure - new_measure`（期望减少时）或 `next_value - current_value`（期望增加时）
使用条件：obs 和 next_obs 中存在可比较的标量度量，该度量沿最优路径应单调变化。
适用证据：有明确的进展度量（位置、距离、高度、角度等），且该度量的变化比瞬时速率更能反映真实进展。
与 dense_state_signal 的选择：如果要鼓励"处于某种好状态"，用 `w * signal`。如果要鼓励"朝好方向改变"，用 delta。delta 的优势是 agent 无法在好状态上停滞不前，必须持续改善。适合：agent 当前的绝对状态值不能完全反映进展（如位置——站在原点不动 vs. 走到终点但位置绝对值可能相同）。
注意：对观测中直接给出的速度信号（如 `horizontal_velocity`）不要做 delta——速度本身已经是变化率。对观测中的位置/角度/距离类信号优先考虑 delta。

### 3.3 potential_based_shaping
数学形式：`potential(next_obs) - potential(obs)`
使用条件：(1) 任务有一个可量化的进展度量（如位置、距离、高度）；(2) 该度量沿最优路径应单调变化；(3) 能从观测中构造一个标量的 potential function。
如何构造 potential：从观测中选择一个在任务完成时达到极值、且沿最优路径单调变化的信号（或信号组合）。potential 的计算只能依赖观测，不能依赖环境内部状态。
与 improvement_delta 的关系：两者数学上等价。potential_based_shaping 的优势在于允许将多个信号编码到一个 potential 中（如同时考虑位置和姿态），而 improvement_delta 通常用于单个度量。
风险：potential 若与任务目标不一致会系统性地误导策略。reward_v1 中如果存在天然的进展度量，优先使用 improvement_delta 的简单形式；当需要组合多个信号构造进展度量时，使用 potential_based_shaping。

### 3.4 quadratic_penalty
数学形式：`-w * error**2` 或 `-w * sum(action_i**2)`
使用条件：约束信号连续可观测，惩罚不应压制主学习信号。用于轻量抑制——需要约束但不至于触发终止的行为。
适用证据：某维度出现高频大幅波动或极端值但未触发终止。
与 hinge 的选择：如果约束有明确的安全边界（如身体倾角超过 X 度必摔），用 hinge（3.1）。如果只是希望"越小越好"没有硬边界（如控制代价、小幅抖动），用 quadratic。
风险：权重过大导致 agent 不敢行动。

### 3.5 soft_health_gate
数学形式：`main_reward * gate_factor`，gate_factor ∈ [0, 1] 在身体状态恶化时平滑衰减。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (safe_bound - current) / margin))`
使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
适用证据：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍 >0——agent 在"先冲后死"，需要在健康恶化时切断主奖励而非额外加罚。
风险：gate 太严格抑制探索；衰减区间应设在"接近危险但尚未终止"的范围内。

### 3.6 terminal_event
数学形式：`if failure_condition: reward = -PENALTY`（硬覆盖 per-step 奖励），或 `if success_condition: reward = +BONUS`
使用条件：(1) 存在可从观测推断的灾难性失败状态（如身体倾角超过阈值 + 接触地面）或任务完成状态；(2) 环境 info 为空因此无法直接读取终止原因。
如何构造：不要依赖 info 字段判断终止原因。可从观测推断：摔倒 → hull_angle 突然偏转 + 身体位置急剧下降；到达终点 → 持续前进中 episode 突然终止（truncated）；出界 → 位置坐标超出有效范围。
适用证据：agent 频繁触发某种终止模式，但当前奖励没有针对该模式提供差异化信号——比如所有终止回合 reward 都一样，agent 无法区分成功和失败。
与 hinge/gate 的区别：hinge 在越界前提供连续梯度，gate 在恶化时衰减主信号。terminal_event 在事件发生的那一刻提供硬信号——没有梯度，但语义明确（"这就是你应该避免/追求的结果"）。

### 3.7 action_efficiency
数学形式：`-w * sum(|action_i|)` 或 `-w * sum(action_i**2)`
使用条件：动作空间 ≥ 2 维连续控制，且任务包含隐含的效率需求（如 locomotion、manipulation）。
适用证据：agent 学会完成任务但动作幅度异常大、能耗高——说明缺效率约束。通常系数较小（主信号 per-step 的 1-5%），避免压制探索。
注意：离散动作空间通常不需要此算子，因为离散动作的选择隐含了代价。首次迭代可不加入，后续迭代若观察到无效动作频繁出现再考虑。

### 3.8 joint_condition_proxy
数学形式：`factor_1 * factor_2 * ...`（每个 factor 为连续 bounded 形式）或 `(f1 + f2 + ...) / n` 或 `(f1 * f2 * ...) ** (1/n)`
使用条件：没有显式 success flag，但有连续信号可构造任务完成的软近似。
适用证据：agent 能在各子条件分别取得进展但无法同时满足。
风险：乘积塌缩（一个 factor→0 则整体→0）；用几何平均或算术平均可缓解。

### 3.9 bounded_signal
数学形式：`x / (1 + abs(x))` 或 `1 / (1 + k * abs(error))` 或 `max(0, 1 - abs(error) / threshold)`
使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。用于压缩极端值而非施加约束。
与 hinge 的区别：bounded 是从两端压缩信号范围，hinge 是只在超出阈值时施加惩罚。如果目标是"值不应超过 X"，用 hinge；如果目标是"值不应该爆炸但无所谓具体范围"，用 bounded。

### 3.10 preview_conditioned_reward
数学形式：`main_reward * preview_factor`，preview_factor 基于观测中能反映**未来状态**的信号（如距离传感器、高度采样、前方地形探测），在不利前景下从 1 平滑衰减到下限。
使用条件：(1) 观测中存在提供前方/未来信息的维度；(2) 该维度可以映射到"前景好/坏"的连续度量；(3) agent 的失败模式与"无法提前调整行为以应对即将到来的状态变化"相关。
如何构造：从提供未来信息的观测中选择一个标量信号，设计一个在安全前景下接近 1、危险前景下接近下限（如 0.3-0.5）的衰减函数。下限不为零以避免完全抑制探索。
适用证据：agent 在相似的瞬时状态下表现差异大（同样的速度/姿态，有时成功有时失败），说明当前状态本身不足以区分好坏——缺少关于"接下来会发生什么"的信息。
与 soft_health_gate 的区别：gate 用当前的**身体状态**乘主奖励（"我已经歪了，别冲了"——被动响应）。preview 用**未来信息**乘主奖励（"前面是坑，别冲了"——主动预判）。两者可以共存：`main_reward * health_gate * preview_factor`。
风险：preview 信号若有噪声会导致主奖励波动；衰减下限设太低会抑制必要探索。

---

## 4. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"定位到合适的算子变换。
以数学语义和训练表现证据为准，不要求组件名完全匹配。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2`，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |
| 缺少灾难性失败信号 | 终止率高且失败回合 reward 非负 | terminal_event | 从观测推断失败状态，加入硬覆盖惩罚 |
| 缺少任务完成信号 | agent 持续前进但 episode 在无摔倒情况下终止 | terminal_event 或 improvement_delta | 用位置 delta 做正向奖励，或在确认可达终点时加入软完成 bonus |


```
