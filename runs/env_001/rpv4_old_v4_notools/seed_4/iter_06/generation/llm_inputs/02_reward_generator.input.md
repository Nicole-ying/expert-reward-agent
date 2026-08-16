# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
任务主目标：控制一个具有两个支撑接触点的二维飞行器（类似着陆器）从视野顶部中心附近出发，尽快到达并稳定停靠在中央目标平台上，同时尽量节省发动机燃料。代理需学会准确接近目标（水平与垂直位置归零）、降低速度、保持竖直姿态并实现双腿同时平稳触地。  
不被混淆的目标：纯粹最小化燃料消耗或最大程度悬停都不是目标，核心是“快速、安全、节能地完成停靠”。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标为到达并停靠在明确的目标位置，附属有速度、姿态、能耗等约束，无其他权重相当的冲突目标，符合导航型任务定义。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（至少部分值为浮点，接触标志为实数 0.0/1.0）
- obs[0]: x_position – 飞行器相对于目标平台的水平偏移，reward_usable: true
- obs[1]: y_position – 飞行器相对于平台高度的高度差（0表示正好在平台高度），reward_usable: true
- obs[2]: x_velocity – 水平线速度，reward_usable: true
- obs[3]: y_velocity – 垂直线速度，reward_usable: true
- obs[4]: body_angle – 飞行器俯仰角/倾斜角，reward_usable: true
- obs[5]: angular_velocity – 角速度，reward_usable: true
- obs[6]: left_support_contact – 左支撑接触标志（1.0接触，0.0脱离），reward_usable: true
- obs[7]: right_support_contact – 右支撑接触标志，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action/action_dim 0: no_engine – 关闭所有发动机，无推力
- action/action_dim 1: left_orientation_engine – 点燃左侧姿态发动机（产生旋转力矩及可能少量平移力）
- action/action_dim 2: main_engine – 点燃主发动机（主要提供垂直向上推力）
- action/action_dim 3: right_orientation_engine – 点燃右侧姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 若发生在飞行器位于目标平台中心且接触双腿着地时，可视为成功停靠。
- failure-like termination: crash_or_body_contact（异常碰撞）、horizontal_position_outside_viewport（水平出界）、body_not_awake_or_settled 若发生在远离平台或仅单腿接触等不安全区域，也视为失败。
- ambiguous termination: 仅凭 termination 标志无法区分成功/失败，需结合最终状态判断。
- truncation: 无显式 episode 截断信息，step source 中 truncated 固定为 False。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（返回空字典 {})
- forbidden_or_uncertain_info_fields: 一切 info 字段，因无任何明确声明。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前时刻观察）
- action（当前动作）
- next_obs（下一时刻观察）
- info（当前为空，不允许假设任何字段）
- training_progress（此处 prompt 未明确允许使用，默认禁用）

禁止使用：
- original_reward（官方奖励已屏蔽）
- 任何未在 3. 节中声明的 obs 切片含义以外的信号
- 任何未从 step 源码中明确出现的 info 字段

## 7. 可用于奖励函数的信号
- position: next_obs[0] (x), next_obs[1] (y) —— 可用于计算距目标平台中心的距离或高度差
- velocity: next_obs[2] (vx), next_obs[3] (vy) —— 可用于惩罚高速着陆或水平漂移
- orientation: next_obs[4] (angle) —— 可用于鼓励竖直姿态
- angular_velocity: next_obs[5] —— 可用于惩罚剧烈旋转
- contact: next_obs[6], next_obs[7] —— 可用于奖励双腿同时接触平台
- action/engine: action（0～3） —— 可用于惩罚使用主发动机及姿态发动机，鼓励少用推力

## 8. 不确定或不可用的信号
- 无官方 success/failure 标志
- 无触地冲击力或触地速度阈值信息
- 无环境是否处于 “settled” 状态的明确中间信号（只有环境内部判断）
- 无实时燃料消耗数值，只能通过动作类别间接反映

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_with_landing_legs
  actuator_type: discrete_thrusters (main + two orientation)
  contact_structure: two-point_support (left/right)
primary_objectives:
  - reach_target_pad (minimize final distance to pad center, x→0, y→0)
  - achieve_stable_settle (zero relative velocity, upright, both legs contact)
  - minimize_time (implicitly via density of positive rewards)
secondary_objectives:
  - minimize_fuel_or_engine_usage
main_failure_risks:
  - crash_or_body_contact (hard landing, hitting ground/edge)
  - horizontal_out_of_bounds
  - unstable_oscillation (excessive rotation, bouncing)
  - fuel-inefficient hovering or overshoot
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: approach_target_position
  purpose: 驱使飞行器水平与垂直位置接近目标平台中心 (0,0)
  why_required: 主任务“到达目标”的直接量化
  usable_signals: [next_obs[0], next_obs[1]] → 距离/平方距离
  risks: 可能鼓励高速撞击，需与速度惩罚配合

- role_id: stabilize_orientation_and_angular_velocity
  purpose: 鼓励竖直姿态、抑制旋转
  why_required: 安全软著陆的关键前提
  usable_signals: [next_obs[4], next_obs[5]] → 角度绝对值、角速度平方
  risks: 若权重过高，可能让飞行器悬停过久

- role_id: penalize_engine_usage
  purpose: 节省燃料 / 最小化发动机使用
  why_required: 任务描述的节能要求
  usable_signals: [action] → 是否使用主发动机 / 姿态发动机
  risks: 与快速到达目标冲突，需适当平衡权重

### 10.2 条件职责 conditional_roles
- role_id: soft_landing_velocity_penalty
  condition_to_use: 当接触标志从无到有或持续接触时，惩罚过大的垂直或水平速度
  usable_signals: [next_obs[2], next_obs[3], next_obs[6], next_obs[7]] → 速度大小 × 接触标志
  risks: 延迟给予成功奖励，可能阻碍接近阶段正常速度

- role_id: dual_leg_contact_bonus
  condition_to_use: 只在接近目标位置且低速时，额外奖励双腿同时接触
  usable_signals: [next_obs[0], next_obs[1], next_obs[6], next_obs[7]] → 同时为1且位置接近0
  risks: 可能误导代理优先追求接触而忽略位置精度

### 10.3 慎用/禁用职责 avoid_roles
- role_id: early_termination_success_bonus
  reason: 无显式成功标志；仅凭 body_not_awake_or_settled 不可靠，可能奖励在半途停稳（非目标位置）的局部静止
  forbidden_or_missing_signals: [explicit success flag, allowed info fields]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_target_position | next_obs[0], next_obs[1] | — | -distance_to_target, -squared_distance, bounded_signal | 可用欧氏距离或各分量平方和，离目标越近奖励越高 |
| stabilize_orientation_and_angular_velocity | next_obs[4], next_obs[5] | — | -abs(angle), -angle^2, -ang_vel^2, quadratic_penalty | 角度接近0获得最高奖励 |
| penalize_engine_usage | action (0–3) | 准确燃料消耗值 | -I(action!=0), -per_action_cost | 不同动作可设不同代价（主引擎>姿态>空闲） |
| soft_landing_velocity_penalty | next_obs[2], next_obs[3], next_obs[6], next_obs[7] | 实际接触力 | -contact * velocity_magnitude | 仅在接触时乘速度惩罚，避免正常接近阶段被罚 |
| dual_leg_contact_bonus | next_obs[6], next_obs[7], next_obs[0], next_obs[1] | — | use contact_and_position_gate | 要求同时接触且位置靠近0才给正奖励 |
| early_termination_success_bonus | — | explicit success info | avoid | 禁止使用，缺乏可靠信号 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 悬停不降或下降过慢 | y位置长时间不减小，主引擎使用率低 | 加强高度下降的奖励形状，或时间衰减调整权重 |
| 高速撞击目标平台 | 终止前瞬间速度很大，crash_or_body_contact | 提升速度惩罚权重，或增加接触时速度惩罚 |
| 单腿先触地导致倾倒 | 左右接触标志不同步，角度大幅变化后终止 | 增加双腿同步奖励，接触时必须角度接近0 |
| 水平漂移出界 | x位置超出范围前无纠正动作 | 强化水平位置接近0的奖励，或引入水平出界惩罚梯度 |
| 往复摆动或振荡 | 动作频繁切换 1/3，角速度幅度大 | 引入动作平滑代价，或加大 angular_velocity 惩罚 |
| 过度使用主发动机保安全 | 主引擎几乎全程开启，燃料理论消耗大 | 加强动作代价，或只在接近目标时允许适量使用 |



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
- best_score_so_far: -107.730

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + speed_penalty | 4 | -107.730 | -107.730 | unsolved |
| attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + settling_stability | 1 | -121.510 | -121.510 | unsolved |

## Previous interventions

- iter 2 (score=-110.570, structure=attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + speed_penalty): selected_level: L3: 得分始终接近负值，远低于目标 200，且主信号框架将距离改进错误地导向跌落，符合“score 始终接近零或负→重写整个 reward，换完全不同的主信号框架”的规则。第一轮即发现根本错误，必须重构。 | selected_intervention: 完全重写奖励函数，采用经典着陆器势能差框架：
- iter 3 (score=-110.090, structure=attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + speed_penalty): selected_level**：**L2: 理由：仅调系数（L1）无法解决“远距离速度被错误惩罚”的结构性问题，必须改变 `speed_penalty` 的数学形态，使其依赖距离。 | selected_intervention**：
- iter 4 (score=-107.730, structure=attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + speed_penalty): `selected_level`：**L2: 理由：增减系数（L1）无法解决着陆奖励稀疏的本质问题；必须修改 landing_contact 组件的数学形态，将其从离散终态事件变为连续函数。 | `selected_intervention`：

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
