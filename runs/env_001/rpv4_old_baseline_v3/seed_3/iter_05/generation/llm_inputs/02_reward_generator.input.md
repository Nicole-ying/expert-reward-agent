# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本任务是一个 2D 车辆轨迹优化问题。智能体从视口上方中心附近出发，需要尽可能快地飞到目标平台（中央垫板）并稳定停靠，同时尽量节省发动机推力。  
**主要目标**：准确到达目标位置（x≈0, y≈0）、软着陆（低线速度、低角速度、双支撑接触并稳定）。  
**次要目标**：燃料经济性（减少不必要的发动机使用），但仅在不牺牲着陆成功的情况下才考虑。  
**不应混淆的目标**：不是单纯的导航到达，必须在接触阶段满足姿态和速度约束，否则视为失败。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达一个指定的空间位置（目标平台），并附带姿态控制、速度抑制和接触安全性要求；该类问题的主干属于目标到达，附属约束可在奖励函数中作为条件角色加入。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推断)
- obs[0] (x_position): 相对于目标平台的水平坐标，可正可负 (reward_usable: true)
- obs[1] (y_position): 相对于垫板高度的垂直坐标 (reward_usable: true)
- obs[2] (x_velocity): 水平线速度 (reward_usable: true)
- obs[3] (y_velocity): 垂直线速度 (reward_usable: true)
- obs[4] (body_angle): 机体朝向角 (reward_usable: true)
- obs[5] (angular_velocity): 角速度 (reward_usable: true)
- obs[6] (left_support_contact): 左支撑腿接触标志 (0.0 或 1.0) (reward_usable: true)
- obs[7] (right_support_contact): 右支撑腿接触标志 (0.0 或 1.0) (reward_usable: true)

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0 (no_engine): 不启动任何发动机，让机体自由飘移/受重力影响
- action 1 (left_orientation_engine): 启动左侧姿态发动机，产生旋转力矩（改变朝向）
- action 2 (main_engine): 启动主发动机，产生向机体上方的推力（主要进行高度/速度控制）
- action 3 (right_orientation_engine): 启动右侧姿态发动机，产生反向旋转力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled — 当机体休眠时触发，可能暗示已稳定着地，但需要结合接触和速度进一步确认；也可能是其他原因导致不再活动。
- failure-like termination: crash_or_body_contact（机身与地面/其它物体异常碰撞）或 horizontal_position_outside_viewport（水平飞出视野）明确为失败。
- ambiguous termination: body_not_awake_or_settled 不能单独作为成功标志，必须检查观测中的双接触状态和速度。
- truncation: 无，或未声明，环境未提供步长限制。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典）
- explicit_failure_flag_available: false （info 未提供故障原因字段）
- allowed_info_fields: 无 （info = {}，禁止使用原始奖励或任何未声明的 info）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs （当前观测，所有维度均可）
- action （当前动作）
- next_obs （下一状态，用于检测是否达成的着陆条件或进度变化）
- 上述信号中已声明的字段

禁止使用：
- original_reward（已屏蔽）
- info（为空，不可用）
- training_progress（任务提示中未明确允许使用，不可用）
- 任何未在观察空间中显式声明的外部信号

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position)、obs[1] (y_position) — 相对目标位置，可构建距离、高度误差等
- velocity: obs[2] (x_velocity)、obs[3] (y_velocity) — 代表接近时的速度，可惩罚高速接触
- orientation: obs[4] (body_angle) — 应接近直立/目标姿态
- contact: obs[6] (left_support_contact)、obs[7] (right_support_contact) — 用于判断着陆完成
- action/engine: action 本身可判断是否使用了发动机（用于燃料惩罚）
- other: obs[5] (angular_velocity) 用于姿态稳定

## 8. 不确定或不可用的信号
- 无明确的“成功”标志位（如 terminal_info 中的 success），只能从 next_obs 的接触+速度+位置组合推断
- 无燃料或推力直接测量，只能通过是否执行动作 1/2/3 来近似能耗
- 无外部参考目标轨迹或距离理想路径的偏差信号

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: single_body_lander
  actuator_type: thruster (main + two orientation)
  contact_structure: two_leg_contact
primary_objectives:
  - reach_target_position (x→0, y→0)
  - stable_landing (low velocities, upright orientation, both legs contact)
secondary_objectives:
  - fuel_efficiency (minimize engine usage)
  - rapid_arrival (implicitly through discounting)
main_failure_risks:
  - crash_into_ground_or_pad_due_to_excessive_velocity
  - drift_out_of_viewport_horizontally
  - tumbling_or_uncontrolled_rotation_preventing_landing
  - never_activating_engines_and_missing_target
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: progress_to_target
  purpose: 指引智能体向目标平台移动，减小与目标的水平与垂直距离。
  why_required: 没有此职责，智能体没有方向性，难以找到着陆点。
  usable_signals: [x_position, y_position, 上一帧位置（通过 obs/next_obs 差分）]
  risks: 如果仅用距离惩罚，可能在目标附近徘徊而不会减速着陆。

- role_id: soft_landing
  purpose: 在接近目标区域时强制降低线速度、角速度，并最终确保双支撑稳定接触。
  why_required: 任何仅依赖位置奖励的策略都可能以高速撞击平台导致失败。
  usable_signals: [x_velocity, y_velocity, angular_velocity, body_angle, left_support_contact, right_support_contact, x_position, y_position]
  risks: 如果权重过高，可能导致智能体不敢下降；需配合进度奖励使用。

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency
  purpose: 最小化发动机使用次数或时长，从而节省燃料。
  condition_to_use: 在训练后期或当智能体已能稳定着陆时开启；或使用极小的负系数仅起辅助作用。初期不建议使用，以免抑制必要的推力探索。
  usable_signals: [action (0=无动作, 1/3=姿态, 2=主发动机)]
  risks: 若过早或过强，会造成智能体选择无动作而错过着陆窗口，甚至因重力坠落而失败。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_success_bonus_from_info
  reason: info 中没有 success 标志，无法使用。
  forbidden_or_missing_signals: [info.success]

- role_id: trajectory_smoothing (动作连续性)
  reason: 动作为离散 4 选 1，连续动作序列的平滑性难以定义，强行惩罚动作切换可能导致发动机频繁抖动（振荡），反而降低性能。
  forbidden_or_missing_signals: 无自然动作平滑度量，且可能与该离散系统的最佳控制模式冲突。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| progress_to_target | x_position, y_position (current & last) | 无 | dense_state_signal (欧氏距离), potential-based reward (距离的减少) | 避免仅用绝对距离，否则终点奖励过强 |
| soft_landing | x_velocity, y_velocity, angular_velocity, body_angle, left/right contacts, proximity (x,y) | 无 | bounded_signal (速度幅度), quadratic_penalty (超限部分), threshold event reward (双接触且速度<阈值) | 需根据接近程度动态开启，防止远处减速 |
| fuel_efficiency | action | 真实的推力积分 | sparse_action_penalty (每个非零动作给负值) | 极小的负值，或仅在着陆成功后回溯性奖励节能行为时使用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 什么都不做，自由落体坠毁 | 平均奖励低，动作直方图 action=0 占绝对主导 | 增加 progress_to_target 奖励权重，或加入很小的动作多样性奖励（仅用于早期探索） |
| 持续大推力水平高速飞出视口 | x_velocity 最终极大，x_position 超出阈值 | 增加水平速度惩罚，或增加位置边界接近时的警告奖励 |
| 到达目标上方但旋转失控无法稳定 | body_angle/angular_velocity 不停振荡，最终 crash | 增加姿态稳定奖励，特别是接近目标时加大角速度惩罚 |
| 频繁开关发动机，消耗高且轨迹不稳定 | 动作变化率高，双接触次数少 | 加入软着陆的强事件奖励，宁可稍慢也要降速，必要时降低燃料效率职责的影响 |
| 以单腿或倾斜方式撑着平台但不稳定 | 只有一条腿接触，或 body_angle 非零但 termination 发生 | 更严格定义双接触 + 小角度才算成功，使用事件奖励强调这些条件 |



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
- best_score_so_far: -32.350

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| contact_bonus + goal_proximity + orientation_penalty + velocity_penalty | 1 | -32.350 | -32.350 | unsolved |
| contact_reward + fuel_cost + landing_reward + orient_cost + progress + proximity_bonus | 1 | -105.480 | -105.480 | unsolved |
| contact_reward + descent_reward + fuel_cost + landing_bonus + orient_cost + progress | 2 | -121.570 | -363.300 | unsolved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
