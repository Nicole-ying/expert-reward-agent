# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
主体是一个2D车体（类似着陆器），起始于画面上方中心附近，带有随机初始力。智能体需要尽快飞到画面中央的目标着陆垫，稳定降落并停止，同时尽量少使用引擎推力。核心目标是**安全到达目标垫并稳定停靠**，次要目标是最小化时间和燃料消耗。不应混淆的是：着陆成功不等于单纯到达垫子上方，还需低速、姿态垂直、接触平稳。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务主目标明确为“到达中央目标垫并停靠”，符合导航到目标点的定义，附属目标（快速、省燃料）为性能优化，不构成同等权重冲突的多目标，因此不属于multi_objective_task。动力学表现为接近目标、减速、姿态稳定后软接触，属于goal_approach_and_soft_contact子类型。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 连续浮点（具体由环境决定，通常float32）
- obs[0]: x_position（相对目标垫的水平位置），reward_usable: true
- obs[1]: y_position（相对垫高度的垂直位置，0表示与垫齐平，正为上方），reward_usable: true
- obs[2]: x_velocity（水平线速度），reward_usable: true
- obs[3]: y_velocity（垂直线速度），reward_usable: true
- obs[4]: body_angle（机体姿态角），reward_usable: true
- obs[5]: angular_velocity（角速度），reward_usable: true
- obs[6]: left_support_contact（左支撑腿接触标志，1.0接触，0.0未接触），reward_usable: true
- obs[7]: right_support_contact（右支撑腿接触标志，1.0接触，0.0未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，无任何引擎工作，仅靠惯性/重力演化
- action 1: left_orientation_engine，点燃左侧姿态引擎，主要用于纠正或产生逆时针旋转倾向
- action 2: main_engine，点燃主引擎，产生主要推力（通常向上方向，抵抗重力/减速）
- action 3: right_orientation_engine，点燃右侧姿态引擎，产生顺时针旋转倾向

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination: 身体稳定垫上停靠（由body_not_awake_or_settled可能触发），暗示成功着陆。
- failure-like termination: crash_or_body_contact（剧烈碰撞）、horizontal_position_outside_viewport（飞出视野，位置丢失）。
- ambiguous termination: body_not_awake_or_settled 仅说明身体不再移动或进入休眠，未明确是成功还是失败，但结合任务目标，大概率是成功，然而无法直接判断具体原因。
- truncation: 提供的step源码中未出现时间截断，但真实环境可能有步数上限，此处未暴露。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info字典为空，无任何字段可用）
- forbidden_or_uncertain_info_fields:
  - 任何未在allowed_info_fields中声明的字段
  - 官方奖励信号 original_reward 禁止直接使用或重构
  - 环境内部终止原因代码不可见

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs: 当前步观测
- action: 当前步动作
- next_obs: 下一步观测（可用于计算差分、判断着陆事件等）
- info: 仅允许使用明确声明的字段（当前为空映射）
- training_progress: 仅在任务额外说明允许时用于退火调度，此处允许使用以可能实现课程学习，但不可直接作为时间进度指标

禁止使用：
- original_reward（官方奖励被屏蔽）
- 任何未在允许列表中的info字段
- 未标记的obs切片（但全部obs均标记可用，故无此问题）
- 环境内部步数或时钟（未提供）

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])
- velocity: x_velocity (obs[2]), y_velocity (obs[3])
- orientation: body_angle (obs[4]), angular_velocity (obs[5])
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])
- action/engine: action (0-3) → 可判别是否使用主引擎、姿态引擎
- other: 可派生量如与目标的距离（√(x²+y²)）、是否接触垫（任意contact flag>0）、垂直速度符号、角度绝对值等。

## 8. 不确定或不可用的信号
- 碰撞/坠毁事件：crash_or_body_contact 只在终止时产生，且无法通过obs直接判断是否发生碰撞（除非结合位置/速度急剧变化，但无可靠阈值），因此不能用作奖励函数中的确定性信号。
- 任务完成标志：没有显式成功标志，body_not_awake_or_settled 是终止理由之一，但在step中未被传递为信息字段。
- 时间/步数计数：无可用变量，无法在奖励函数内获知当前episode已进行的步数，因此不能奖励“快速到达”。
- 视口边界值：不知道horizontal_position_outside_viewport的具体判定阈值，无法安全地在奖励中基于位置硬判定。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: vehicle/lander
  actuator_type: one main vertical engine + two lateral attitude engines
  contact_structure: two support legs, each with a binary contact flag
primary_objectives:
  - 到达目标垫上方并稳定降落（软接触）
secondary_objectives:
  - 最小化燃料消耗（减少引擎使用）
  - 保持姿态稳定性（避免翻滚）
  - 在可行前提下尽快到达（此处因缺少步数信号不能奖励时间）
main_failure_risks:
  - 高速撞击垫子（crash）
  - 飞出视口丢失
  - 着陆姿态过大导致倾覆
  - 过度使用引擎浪费燃料且姿态不稳
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- role_id: proximity_to_target  
  purpose: 引导机体向目标垫移动，奖励接近目标的行为  
  why_required: 任务核心是到达目标点，无此奖励无法完成导航  
  usable_signals: [x_position, y_position, 可选派生距离]  
  risks: 过度奖励靠近但高速撞击，需结合速度信号抑制

- role_id: soft_landing_on_pad  
  purpose: 当任一支撑腿接触垫时，要求低速且姿态垂直，鼓励安全着陆  
  why_required: 仅到达目标垫上方不足以成功，需要稳定接触停靠  
  usable_signals: [left_support_contact, right_support_contact, y_velocity, x_velocity, body_angle]  
  risks: 易导致过早接触奖励，需结合接触触发条件；可能使智能体贪图接触但忽略周围姿态

- role_id: orientation_stability  
  purpose: 惩罚过大的倾斜角度或角速度，防止翻滚失控  
  why_required: 姿态不稳定会导致无法精确着陆甚至崩溃  
  usable_signals: [body_angle, angular_velocity]  
  risks: 过度惩罚可能阻碍必要的姿态调整机动，需适度放宽

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency  
  condition_to_use: 始终可用，但系数可随 training_progress 增大以鼓励后期更省燃料  
  usable_signals: [action]（惩罚主动使用引擎，特别是主引擎）  
  risks: 初期过度惩罚引擎使用会抑制探索和必要的减速机动，需渐进加重或仅在接近目标时激活

- role_id: time_pressure_soft  
  condition_to_use: 若环境未来提供步数指标或允许通过训练步数计算“剩余时间”信号时可用，当前**因无可用信号，暂无法实现**  
  usable_signals: [缺失episode步数信息]  
  risks: 无信号直接导致无法实现

### 10.3 慎用/禁用职责 avoid_roles
- role_id: crash_penalty  
  reason: 缺少显式碰撞指示器，且无法从obs可靠推断碰撞事件；强行使用易造成错误负奖励  
  forbidden_or_missing_signals: [crash_event, 明确失败标志]

- role_id: out_of_bounds_penalty_early  
  reason: 视口边界值未知，无法安全地在奖励函数中基于位置硬判定越界风险  
  forbidden_or_missing_signals: [boundary limits]

- role_id: explicit_success_bonus  
  reason: 无显式成功标志可用，无法安全发放成功奖励，容易错误奖励非成功终止状态  
  forbidden_or_missing_signals: [success_flag]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| proximity_to_target | x_position, y_position | - | dense_state_signal (e.g., negative distance, bounded_signal), gaussian | 可从next_obs计算新距离给予差分奖励 |
| soft_landing_on_pad | contact flags, y_velocity, x_velocity, body_angle | - | conditional_reward (if contact), bounded_signal, threshold_gate | 要求低y速度，小x速度，小角度 |
| orientation_stability | body_angle, angular_velocity | - | quadratic_penalty, bounded_signal | 可同时对角度绝对值与角速度施加惩罚 |
| fuel_efficiency | action | - | action_penalty (counter the use of engine actions) | 惩罚action=1,2,3，或仅惩罚main_engine(2) |
|



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





# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -64.690

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| angle_reward + landing_bonus + proximity_reward + speed_reward | 1 | -64.690 | -64.690 | unsolved |
| angvel_penalty + contact_reward + descent_reward + horiz_penalty + landing_bonus + orient_penalty | 1 | -72.180 | -72.180 | unsolved |
| angle_reward + contact_reward + height_reward + landing_bonus + proximity_reward + speed_reward | 1 | -80.090 | -80.090 | unsolved |
| contact_reward + landing_bonus + shaping_reward + time_penalty | 1 | -91.710 | -91.710 | unsolved |
| contact_reward + engine_penalty + landing_bonus + shaping_reward + time_penalty | 1 | -107.510 | -107.510 | unsolved |
| angle_penalty + landing_bonus + proximity_reward + velocity_penalty | 1 | -108.860 | -108.860 | unsolved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
