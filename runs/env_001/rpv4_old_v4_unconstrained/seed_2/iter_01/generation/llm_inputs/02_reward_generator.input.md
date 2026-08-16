# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 飞行器/着陆器轨迹优化任务。智能体（飞行器）起始于视口顶部中央附近，需要通过学习控制其主引擎与姿态引擎，**到达并稳定停靠在中央目标平台上**。
**主目标：尽可能快地到达目标平台并实现稳定、安全的接触（着陆）。**
**重要次目标（效率约束）：在完成主目标的过程中，尽量减少引擎推力使用（节约燃料）。**
**不该混淆的目标：** 此任务的核心是导航到达，而非单纯的存活平衡或持续前进。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确指向“到达并稳定在中央目标平台”这个核心目标，速度与能耗要求是完成主任务时的效率约束，而非独立冲突的多目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (推理假设，基于连续物理量)
- obs[0]: x_position, 飞行器相对于目标平台的水平坐标，reward_usable: true
- obs[1]: y_position, 飞行器相对于目标平台高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity, 飞行器的水平线性速度，reward_usable: true
- obs[3]: y_velocity, 飞行器的垂直线性速度，reward_usable: true
- obs[4]: body_angle, 飞行器的姿态角度，reward_usable: true
- obs[5]: angular_velocity, 飞行器的角速度，reward_usable: true
- obs[6]: left_support_contact, 左侧支撑腿/结构接触标志 (0.0 或 1.0)，reward_usable: true
- obs[7]: right_support_contact, 右侧支撑腿/结构接触标志 (0.0 或 1.0)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0 (no_engine): 不点火，依靠惯性滑行
- action 1 (left_orientation_engine): 点燃一个侧向/方向引擎（原文为 orientation engine），产生逆时针/顺时针力矩，用于调整姿态
- action 2 (main_engine): 点燃主引擎，产生向上的推力，用于减速和悬停
- action 3 (right_orientation_engine): 点燃另一个侧向/方向引擎，与 action 1 力矩方向相反

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled`。这可能在飞行器以低速平稳地着陆在目标平台后触发。这是最接近成功的终止条件，但不能排除因卡住或静止在错误位置而触发的情况。
- failure-like termination: `crash_or_body_contact` (任何非任务期望的身体接触/坠毁)、`horizontal_position_outside_viewport`（飞出水平边界）。
- ambiguous termination: 上述"body_not_awake_or_settled"本质上带有模糊性，可能因不稳定的卡死而非成功着陆触发。
- truncation: prompt未提及，假设为`False`。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false (无明确info["success"])
- explicit_failure_flag_available: false (无明确info["failure"])
- allowed_info_fields: {} (step source返回空info)
- forbidden_or_uncertain_info_fields: 任何未在step source中声明的info字段，如“success”, “failure”, “termination_reason”, “landed” 等。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs: 形状为 (8,) 的上一状态向量
- action: 上一动作 (0~3)
- next_obs: 形状为 (8,) 的当前状态向量
- info: 仅能使用空字典 {}，无可用字段
- training_progress: 仅在 prompt 明确需要课程学习时使用

禁止使用：
- original_reward（已屏蔽）
- 官方或任何仿真器内置奖励
- 任何未在 step source 的 info 中声明的字段

## 7. 可用于奖励函数的信号
- position: `next_obs[0]` (x坐标), `next_obs[1]` (y坐标)
- velocity: `next_obs[2]` (x速度), `next_obs[3]` (y速度)
- orientation: `next_obs[4]` (角度), `next_obs[5]` (角速度)
- contact: `next_obs[6]` (左脚/支撑接触), `next_obs[7]` (右脚/支撑接触)
- action/engine: `action` (可设计引擎惩罚，但必须服务于主目标，即鼓励着陆而非停滞)

## 8. 不确定或不可用的信号
- 成功/失败标志：无。
- 目标平台具体坐标/范围：未知，只能从观察的零位中心推断 `x=0` 为目标水平中心，但垂直目标高度未知。`y_position` 为零可能代表接触平台高度。
- 燃料消耗量：无直接观测。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body
  actuator_type: idealized_thrusters
  contact_structure: two_landing_legs_or_support_points
primary_objectives:
  - reach_and_settle_on_target_platform (minimize final distance to target, zero velocity, stable angle, safe contact)
secondary_objectives:
  - fuel_efficiency (minimize total engine usage)
  - time_to_land (minimize episode length, implicitly driven by early termination)
main_failure_risks:
  - crashing into ground or other structures
  - flying out of viewport boundaries
  - getting stuck in an unstable state that triggers premature termination
  - using excessive main engine and oscillating without landing
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity
  purpose: 驱动飞行器靠近目标平台。
  why_required: 核心任务，没有此职责智能体将无明确目标。
  usable_signals: [next_obs[0], next_obs[1]]
  risks: 可能导致高速飞向目标并撞毁。

- role_id: velocity_management
  purpose: 在接近目标时（或全程）抑制速度，实现软着陆。
  why_required: `body_not_awake_or_settled` 终止条件暗示需要极低速度才能触发成功着陆。
  usable_signals: [next_obs[2], next_obs[3], next_obs[5]]
  risks: 过早抑制速度可能导致飞行器悬停、不下降或油耗极高。

- role_id: orientation_stabilization
  purpose: 保持飞行器姿态水平（零角度或小角度），并抑制角速度。
  why_required: 平稳着陆需要双腿接触 (`next_obs[6]`, `[7]`)，机身倾斜可能导致坠毁。
  usable_signals: [next_obs[4], next_obs[5]]
  risks: 若接触时对角度要求过于苛刻，可能阻碍着陆。

- role_id: safe_contact
  purpose: 鼓励以两侧支撑腿接触目标平台。
  why_required: 这是成功着陆的最终标志。
  usable_signals: [next_obs[6], next_obs[7]]
  risks: 可能诱导智能体在任意位置强行接触地面以获取奖励，因此必须与 `goal_proximity` 结合，仅在接近目标区域时给予奖励。

### 10.2 条件职责 conditional_roles
- role_id: sparse_success_reward
  condition_to_use: 当且仅当`terminated`为True，且我们能高度确信是成功着陆时（例如，通过组合检查：双脚接触、位置近零、速度近零、角度近零）。由于没有明确的成功标志，此职责的实现需要高度保守。
  usable_signals: [terminated (从 gym step返回), next_obs[6], next_obs[7], next_obs[0], next_obs[1], next_obs[3]]
  risks: 判断标准设置不当会导致非成功终止被错误奖励，导致奖励作弊。必须极其谨慎。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: engine_penalty
  reason: 禁止作为独立需求，只能作为其他职责的辅助项（例如在接近目标且速度已够低时，轻度惩罚非必要引擎使用）。过度惩罚引擎会使智能体倾向于不做动作，导致无法飞向目标或硬着陆。
  forbidden_or_missing_signals: [engine_fuel_consumption]

- role_id: time_penalty
  reason: 容易与存活压力混淆，可能导致智能体过早采取不安全的着陆行为（如一头扎向地面），冲击安全和稳定职责。仅在训练后期可作为极小因子引入。
  risks: 导致崩溃式着陆。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | next_obs[0], next_obs[1] | target_absolute_coordinates | distance_based_shaping, gaussian_potential, linear_quadratic_penalty | 结合速度进行 shaping |
| velocity_management | next_obs[2], next_obs[3], next_obs[5] | - | quadratic_penalty, bounded_signal | 可设计为目标位置相关的自适应速度上限 |
| orientation_stabilization | next_obs[4], next_obs[5] | - | quadratic_penalty, absolute_penalty | 对角速度和绝对角度进行惩罚 |
| safe_contact | next_obs[6], next_obs[7] | - | conditional_bonus, step_signal | 仅当位置和速度条件满足时给予密集或稀疏奖励 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 不飞向目标，无目的徘徊 | 各episode回报不增长，`x_position` 和 `y_position` 无明显向0收敛趋势。 | 增大 goal_proximity 奖励相对 engine_penalty 的权重。 |
| 高速撞击地面 (crash) | 大量 episode 以 crash 终止，`y_velocity` 在撞击前为高负值。 | 增强 velocity_management，特别是在 `y_position` 接近0时的速度惩罚。 |
| 水平飞出边界 | 大量 episode 因 `horizontal_position_outside_viewport` 终止。 | 增强对 `x_position` 偏差和 `x_velocity` 的控制，降低姿态引擎使用时的角速度副作用。 |
| 在目标上空振荡，无法着陆 | `y_position` 在略高于0的位置震荡，`y_velocity` 在零附近正负切换。 | 调整 velocity_management 的权重或形式，允许一定下沉速度；或在接近目标时降低悬停奖励密度，增加接触奖励引导。 |
| 着陆时姿态失控倾倒 | 双脚接触时 `body_angle` 或 `angular_velocity` 较大，导致后续 body contact。 | 增大 orientation_stabilization 权重，或将其与 safe_contact 奖励关联。 |
| 过早触发“settled”导致停在平台外 | 在远离目标的位置双脚着地并静止，导致“settled”终止，但任务未成功。 | 严格化 safe_contact 奖励的发放条件，必须结合对 `x_position` 和 `y_position` 的严格检查。 |



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

