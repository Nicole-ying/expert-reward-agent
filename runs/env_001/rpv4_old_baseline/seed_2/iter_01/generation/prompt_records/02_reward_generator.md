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
```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code

**2. components: dict**，记录 individual reward components。

首选返回格式：
```python
return float(total_reward), components
```

# 代码硬约束

- Python code block 里只能包含完整的 `compute_reward` 函数。
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
主目标：控制一个从画面顶部中央附近出发的飞行器，安全、稳定地降落在画面中央的目标平台上。要求着陆时速度接近于零、姿态接近竖直，且所有支脚平稳接触平台。

次要目标：在确保主目标达成的前提下，尽量缩短飞行时间，并尽量减少主引擎和姿态引擎的使用（即节省燃料）。

不可混淆的目标：不应将“快速到达”或“节省燃料”凌驾于“安全着陆”之上；也不能将“悬停”或“保持在目标上方”当作成功条件。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 核心目标是到达指定的目标位置并稳定停留，属于典型的导航到达类任务。附属的燃料、时间要求均为次要，不改变主目标的定性。observations 直接提供相对于目标的坐标，符合目标导向的导航范式。

dynamics_subtype: goal_approach_and_soft_contact（接近目标并实现低速、稳定接触）

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测）
- obs[0]: x_position (相对目标平台的水平坐标), 可直接用于距离/接近奖励，reward_usable: true
- obs[1]: y_position (相对目标平台高度的垂直坐标), 同上，reward_usable: true
- obs[2]: x_velocity (水平线速度), 可用于着陆软度控制，reward_usable: true
- obs[3]: y_velocity (垂直线速度), 同上，reward_usable: true
- obs[4]: body_angle (机体朝向角), 可用于姿态奖励，reward_usable: true
- obs[5]: angular_velocity (角速度), 可用于姿态稳定性惩罚，reward_usable: true
- obs[6]: left_support_contact (左侧支脚接触标志，1.0 表示接触), 可用于着陆状态判断，reward_usable: true
- obs[7]: right_support_contact (右侧支脚接触标志，1.0 表示接触), 同上，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine (无推力，仅受重力/物理影响)
- action 1: left_orientation_engine (点燃左侧姿态推进器，产生旋转力矩)
- action 2: main_engine (点燃主推进器，提供向上推力并可能产生力矩)
- action 3: right_orientation_engine (点燃右侧姿态推进器，产生反方向旋转力矩)

## 5. step 与终止条件分析
### 5.1 终止模式
- success‑like termination: body_not_awake_or_settled 如果发生在飞行器已接触地面且速度/角速度极低时，极可能意味着成功着陆；但如果发生在半空中或刚碰撞后，则可能是早期终止。
- failure‑like termination: crash_or_body_contact（与地面或障碍的异常碰撞）、horizontal_position_outside_viewport（水平飞出边界）明确为失败。
- ambiguous termination: body_not_awake_or_settled 本身不区分成功/失败，需要结合观察判断。
- truncation: 代码中未出现 episode length 截断，但实际部署时可能通过外部 wrapper 实现，当前源中未见。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info = {} 或未返回任何键）
- forbidden_or_uncertain_info_fields: 任何未在以上列出的字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前 step 的观察）
- action（当前 step 执行的动作）
- next_obs（下一 step 的观察，可用于检测变化或下一时刻状态）
- info 中明确且安全可用的字段（当前为空，故禁止使用任何 info 内容）
- training_progress（仅当 prompt 明确允许使用进度信息时才可用；此处未声明，禁止使用）

禁止使用：
- original_reward（官方奖励已遮罩）
- official_reward（同上）
- 任何未在观察空间声明中列出的 obs 切片
- 任何未在允许列表中的 info 字段

## 7. 可用于奖励函数的信号
- position: x_position, y_position（可直接计算到目标的距离、高度差）
- velocity: x_velocity, y_velocity（可衡量接近速度、着陆软度）
- orientation: body_angle（偏离竖直的角度），angular_velocity（旋转速度）
- contact: left_support_contact, right_support_contact（着陆脚是否触地，可判断着陆状态）
- action/engine: action 索引可映射到是否使用主引擎、姿态引擎，用于推力/燃料惩罚
- other: 可通过 (obs, next_obs) 的组合构造微分信号，如速度变化、角速度变化等

## 8. 不确定或不可用的信号
- 明确的连续接触力/碰撞力（没有）
- 燃料余量或推力大小（没有直接测量，仅能通过动作间接推测）
- 成功标志位（info 中无）
- 平台检测区域（仅通过相对坐标隐含，没有显式“目标区域”标记）
- 任何与“官方奖励”相关的隐式信息

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 带支脚的垂直起降飞行器 (lander-like)
  actuator_type: 一个主推进器 + 两个对称姿态推进器
  contact_structure: 两个独立支脚，可分别检测接触
primary_objectives:
  - 到达目标平台中心（x_position ≈ 0，y_position ≈ 0）
  - 实现稳定、低速着陆（速度 ≈ 0，且两支脚同时触地）
  - 保持直立姿态（body_angle ≈ 0）
secondary_objectives:
  - 尽量缩短到达时间（在安全前提下）
  - 最小化推进器使用（动作非零即燃料消耗）
main_failure_risks:
  - 高速撞击地面导致 crash_or_body_contact
  - 水平方向漂移超出视口
  - 着陆时姿态严重倾斜并侧翻
  - 长时间悬停不降落（可能导致超时截断，若加装 wrapper）
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: proximity_to_target
  purpose: 引导飞行器向目标平台移动
  why_required: 是任务的核心到达需求，无此则难以收敛到目标
  usable_signals: [x_position, y_position]
  risks: 如果仅用距离奖励，可能导致高速撞地；必须与速度/姿态约束结合

- role_id: soft_landing_conditions
  purpose: 确保着陆时速度接近于零，姿态竖直，支脚平稳接触
  why_required: 防止以危险方式完成任务（高速冲击、侧翻）
  usable_signals: [x_velocity, y_velocity, body_angle, angular_velocity, left_support_contact, right_support_contact]
  risks: 如果权重过高，可能导致飞行器不敢接近地面（奖励悬崖）

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency
  purpose: 按需惩罚不必要的推进器使用，以节省燃料
  condition_to_use: 当飞行器已经足够接近目标或已经完成着陆时，可加大惩罚；在远离目标时不应过度惩罚，以免阻碍探索
  usable_signals: [action]
  risks: 过早或过强的燃料惩罚会阻碍飞行器学习上升和移动

- role_id: terminal_landing_bonus
  purpose: 在成功着陆（接触且稳定）时给予一次性奖励，强化最终行为
  condition_to_use: 仅当 next_obs 显示两支脚同时接触且速度/角速度都接近零时授予
  usable_signals: [next_obs 的 contact、velocity、angular_velocity]
  risks: 如果阈值设置不当，可能把不稳定着陆也判为成功

### 10.3 慎用/禁用职责 avoid_roles
- role_id: time_bonus_or_penalty
  reason: 鼓励快速到达容易导致飞行器以危险方式高速撞击，与安全着陆冲突。当前环境没有显式的时间步惩罚接口，且 safety 高于 speed。
  forbidden_or_missing_signals: [无可用全局时间步数限制，且 time 信号未显式提供（依赖外部进度）]

- role_id: exact_position_shape_reward
  reason: 环境只需到达中心，不存在复杂的形状奖励（如走廊、栅栏），不需要分段引导。
  forbidden_or_missing_signals: [无子目标点]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| proximity_to_target | x_position, y_position | – | distance_reward, shaped_sq_distance, bounded_progress | 可结合指数衰减奖励靠近过程 |
| soft_landing_conditions | x_velocity, y_velocity, body_angle, angular_velocity, left_contact, right_contact | – | velocity_penalty, angle_penalty, quadratic_penalty | 仅在有接近行为时激活，否则奖励为零 |
| fuel_efficiency | action | – | action_count_penalty, engine_usage_penalty | 可对 action≠0 施加轻量惩罚 |
| terminal_landing_bonus | next_obs[2:8] (速度、角度、接触) | – | threshold_bonus | 需要严格的成功条件判定，防止 false positive |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 学习到稳定悬停在空中而不下降 | y_position 均值远大于 0，且长时间未触发终止 | 减小距离奖励在远距离时的强度，增加“接近即下降”的引导；或增加对 y_position > 0 的轻微惩罚 |
| 高速撞击目标平台 | 终止前的 y_velocity 很大，奖励曲线在终止前突然上升但失败率高 | 大幅提高 soft_landing_conditions 的权重，尤其是 y_velocity 惩罚；加入速度上限逐级惩罚 |
| 使用主引擎一直向上冲，飞出视口 | y_position 异常大，然后 horizontal_position_outside 终止 | 确保 proximity 奖励使用相对坐标的绝对值，避免正反馈溢出；对 out_of_bounds 给予极重惩罚 |
| 着陆时向一侧大幅倾斜并侧翻 | body_angle 在终止时显著偏离 0，单侧接触 | 增强角度惩罚，且令 terminal bonus 要求两脚同时接触；可配合 angular_velocity 惩罚 |
| 反复点火但不移动（燃料浪费） | action 多非零，但位移很小 | 检查燃料或推力惩罚是否过弱；适当加入动作平滑惩罚 (action_change) 或对无效推力惩罚 |



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





# Validation repair

具体错误：Reward v1 failed validation: 出现禁止模式: import ; 修复要求：删除所有import和numpy调用；平方根使用**0.5，有界函数使用max/min或不需要库的代数表达式 (record: runs/env_001/rpv4_old_baseline/seed_2/iter_01/generation/validations/reward_v1.validation.json)

只修复代码合规问题，不重新分析环境，不改变原定奖励设计。直接输出完整合规的compute_reward函数。

# Invalid previous draft

# reward_v1.py

```python
import math

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state variables from next_obs (post-action state)
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    body_angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Proximity reward (main learning signal)
    # Drive the agent toward the target platform (x=0, y=0)
    dist_sq = x * x + y * y
    proximity_reward = -0.5 * dist_sq

    # 2. Velocity penalty (soft landing constraint)
    # Penalise excessive horizontal and vertical velocity
    v_thresh = 0.2
    v_penalty = 0.0
    if abs(vx) > v_thresh:
        v_penalty += abs(vx) - v_thresh
    if abs(vy) > v_thresh:
        v_penalty += abs(vy) - v_thresh
    velocity_penalty = -1.0 * v_penalty

    # 3. Body angle penalty (stability constraint)
    # Penalise tilting away from upright
    angle_thresh = 0.1  # radians
    if abs(body_angle) > angle_thresh:
        angle_penalty = -1.0 * (abs(body_angle) - angle_thresh)
    else:
        angle_penalty = 0.0

    # 4. Landing bonus (task completion proxy)
    # Strong bonus when both legs touch with low speed, angle, and angular velocity
    k_v = 5.0
    k_angle = 10.0
    k_angvel = 5.0

    vx_factor = 1.0 / (1.0 + k_v * abs(vx))
    vy_factor = 1.0 / (1.0 + k_v * abs(vy))
    angle_factor = 1.0 / (1.0 + k_angle * abs(body_angle))
    angvel_factor = 1.0 / (1.0 + k_angvel * abs(angvel))
    contact_factor = left_contact * right_contact  # only if both feet touch

    landing_bonus = 20.0 * contact_factor * vx_factor * vy_factor * angle_factor * angvel_factor

    total = proximity_reward + velocity_penalty + angle_penalty + landing_bonus
    components = {
        "proximity_reward": proximity_reward,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "landing_bonus": landing_bonus
    }
    return total, components
```

# reward_v1 设计说明

**任务家族与动力学子类型**  
- `task_family`: `navigation_goal_reaching`  
- `dynamics_subtype`: `goal_approach_and_soft_contact` (带支脚的垂直起降飞行器，离散推力控制)

**选中的奖励职责 (selected reward roles)**  
1. **proximity_to_target** (mandatory) – 核心驱动力，引导飞行器向目标平台中心移动。  
2. **soft_landing_conditions** (mandatory) – 通过速度和姿态约束确保低速、直立着陆，避免撞击和侧翻。  
3. **terminal_landing_bonus** (conditional) – 当双支脚平稳接触且状态良好时给予强正向信号，作为隐式成功完成的近似。

**职责‑信号映射**  
| 职责 | 信号来源 (next_obs) | 使用的公式算子 |
|------|---------------------|----------------|
| proximity_to_target | `x_position`, `y_position` | `dense_state_signal` (二次惩罚: `-0.5 * distance²`) |
| soft_landing (速度) | `x_velocity`, `y_velocity` | `dense_state_signal` 的 hinge 变体: `-1.0 * max(0, |v| - threshold)` |
| soft_landing (姿态) | `body_angle` | 同上 hinge: `-1.0 * max(0, |angle| - threshold)` |
| terminal_landing_bonus | `left_contact`, `right_contact`, `vx`, `vy`, `angle`, `angvel` | `joint_condition_proxy` (因子乘积) + `bounded_signal` (`1/(1+k*|error|)`) |

**排除的职责及原因**  
- `terminal_success_reward` / `terminal_failure_penalty`: 环境 `info` 为空，无显式成功/失败标志，无法安全实现。  
- `fuel_efficiency`: 离散动作的效率惩罚在 v1 暂不加入，避免阻碍飞行器学习必要的上升和姿态控制；留待后续迭代。  
- `time_bonus_or_penalty` (avoid role): 与安全着陆冲突，禁用。  
- `exact_position_shape_reward`: 无子目标分段需求，禁用。

**为什么未使用 terminal_success/failure 奖励**  
因为 `explicit_success_flag_available` 和 `explicit_failure_flag_available` 均为 false，`info` 字典不提供任何终止原因。强行推断会引入不可靠信号，因此 v1 完全依赖连续状态信号驱动学习，并用 `landing_bonus` 作为软完成信号。

**推迟到后续迭代的职责**  
- 动作代价/燃料惩罚 (`fuel_efficiency`)：待着陆行为稳定后加入轻量惩罚以优化能耗。  
- 基于观测推断的 terminal failure penalty：如果能可靠地从 `obs` 推断 crash 或飞出边界，可加入硬惩罚。  
- `soft_health_gate`：若观察到 agent “高速冲向目标后 crash” 的模式，可以在速度/角度恶化时逐渐衰减主奖励，而不是简单 hinge。

**训练后应重点观察的 failure modes**  
- **悬停不降落**：`y_position` 长期远离零，末端未触发着陆奖励；需检查 `proximity` 权重是否过大或 `landing_bonus` 吸引力不足。  
- **高速撞击**：`vy` 过大且终止时未双接触；可通过增大速度 hinge 阈值斜率或在后续迭代引入速度上限惩罚缓解。  
- **着陆倾斜侧翻**：`body_angle` 大且仅单脚接触；可能需要加大 `angle_penalty` 系数或结合 `angular_velocity` 惩罚。  
- **利用着陆奖励刷分**（理论上较难）：若 agent 重复接触又脱离，`landing_bonus` 乘积会剧烈波动，但不会持续获得高分；若出现，可降低 `w_terminal` 或加入腿接触的持续性要求。
```
