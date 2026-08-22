# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器着陆任务。飞行器从画面顶部中心附近以初始随机推力开始运动。核心目标是 **尽可能快且稳定地降落在中心目标垫上**（位置归零、速度很低、姿态竖直、双腿安全接触地面）。次要目标是 **减少引擎燃料消耗**。智能体必须学习靠近目标点、减速、保持竖直、安全触地，同时避免坠毁、飞出屏幕或不稳定着陆。

## 2. 任务类型选择
selected_route_id: `navigation_goal_reaching`  
confidence: `high`  
reason: 任务的核心是到达指定的固定目标垫（位置和姿态确定），能耗和速度是次要优化项。符合 goal-reaching 定义，不具有存活式平衡、驾驶安全、多目标冲突等特征。

dynamics_subtype: `goal_approach_and_soft_contact`  
说明：飞行器需要接近目标、减速、姿态归正，并让两条支撑腿与垫子发生安全低碰撞接触，属于典型的“接近+软接触”动力学。  

## 3. 观察空间 observation_space
- type: `Box`
- shape: `(8,)`
- dtype: 推测为 `float32`
- 各维度均可用作奖励信号（reward_usable: true）

| 索引 | 名称                  | 含义                              | reward_usable |
|------|-----------------------|-----------------------------------|---------------|
| 0    | x_position            | 水平坐标，相对于目标垫中心         | true          |
| 1    | y_position            | 垂直坐标，相对于目标垫高度         | true          |
| 2    | x_velocity            | 水平方向线速度                     | true          |
| 3    | y_velocity            | 垂直方向线速度                     | true          |
| 4    | body_angle            | 机体倾角                           | true          |
| 5    | angular_velocity      | 角速度                             | true          |
| 6    | left_support_contact  | 左支撑腿是否接触垫子（1.0/0.0）    | true          |
| 7    | right_support_contact | 右支撑腿是否接触垫子（1.0/0.0）   | true          |

## 4. 动作空间 action_space
- type: `Discrete`
- n: `4`
- 动作含义：

| 动作值 | 名称                      | 含义                                     |
|--------|---------------------------|------------------------------------------|
| 0      | no_engine                 | 无推力，仅受重力与惯性                   |
| 1      | left_orientation_engine   | 点燃左侧姿态引擎（产生逆时针旋转力矩）     |
| 2      | main_engine               | 点燃主引擎（产生向上的推力）               |
| 3      | right_orientation_engine  | 点燃右侧姿态引擎（产生顺时针旋转力矩）     |

## 5. step 与终止条件分析
### 5.1 终止模式
- **crash_or_body_contact**：身体与地面/障碍物发生严重碰撞（坠毁），一般为失败。
- **horizontal_position_outside_viewport**：水平位置超出视口边界，失败。
- **body_not_awake_or_settled**：身体进入休眠或稳定停止。这既可能是因为成功着陆稳定，也可能因坠毁后静止，属于**模糊终止**。
- **无主动截断（truncation）**（任务环境未明确定义时间限制）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: `false`
- explicit_failure_flag_available: `false`
- allowed_info_fields: `{}` （info 为空字典）
- forbidden_or_uncertain_info_fields: 任何 info 内的字段均不可用

> 智能体无法直接通过信息字典获知成功/失败；必须从观测状态（尤其是位置、速度、接触标志）与终止标志的组合推断。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    ...
```

**允许使用**：
- `obs`、`action`、`next_obs` 的全部内容
- `info` 中由本卡片明确声明为可用的字段（当前为**无**）
- `training_progress` 仅当需要时可用这里没有要求，暂不使用

**严格禁止**：
- `original_reward`（官方奖励已被隐藏）
- 任何未在本卡片第3节/第5.2节中声明的 `obs` 额外切片或 `info` 字段

## 7. 可用于奖励函数的信号
- **位置**：`x_position`（希望接近 0）、`y_position`（希望接近 0，表示在垫子上）
- **速度**：`x_velocity`、`y_velocity`（着陆时希望接近 0）
- **角度**：`body_angle`（着陆时希望接近 0，即竖直）
- **角速度**：`angular_velocity`（希望接近 0）
- **接触**：`left_support_contact`、`right_support_contact`（双撑脚同时为 1 表示安全着陆）
- **动作**：离散动作可为惩罚省燃料的依据（0 为无推力，1/2/3 为使用某种引擎）

## 8. 不确定或不可用的信号
- 任何 `info` 中的字段：不存在，无法使用
- `original_reward`：被屏蔽，禁止参考
- 显式的成功/失败标志：无
- 任务完成后的稀疏 terminal bonus：无（因 compute_reward 中不传入 `done`）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D_lander
  actuator_type: discrete_thrusters (main, left/right_orientation)
  contact_structure: two_support_legs
primary_objectives:
  - reach_target_pad (x->0, y->0)
  - stable_soft_landing (low velocity, upright angle, both legs touching)
secondary_objectives:
  - minimal_engine_usage (fuel efficiency)
  - fast_settlement (time-to-land)
main_failure_risks:
  - crash_into_ground_or_obstacle
  - fly_out_of_viewport
  - excessive_oscillation (rollover)
  - landing_on_one_leg_or_with_large_tilt
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `approach_and_landing_shaping`  
  **purpose**: 提供连续稠密信号，引导飞行器靠近目标垫、减速、保持竖直并触发双腿接触。  
  **why_required**: 任务成功的全部物理条件（位置、速度、姿态、双撑脚接触）均需通过此信号塑造，否则智能体无法学到安全着陆。  
  **usable_signals**:  
    - `x_position`, `y_position`（距离目标远近）  
    - `x_velocity`, `y_velocity`（控制着陆冲击）  
    - `body_angle`, `angular_velocity`（保持竖直）  
    - `left_support_contact`, `right_support_contact`（安全着陆确认）  
  **risks**: 若各分量权重不平衡，可能导致智能体只顾位置而忽略姿态，或过度减速导致悬停过久。

### 10.2 条件职责 conditional_roles
- **role_id**: `action_penalty`  
  **purpose**: 惩罚非零引擎动作，以鼓励节省燃料。  
  **condition_to_use**: 始终可加入，但惩罚幅度不能大到阻碍必要的飞行控制。  
  **usable_signals**: 离散动作（0 为无推力，1-3 为使用引擎）  
  **risks**: 过强惩罚会导致智能体不敢点火，无法抵消重力、飞出边界；过弱则对燃料节省无效。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `terminal_success_bonus`  
  **reason**: 奖励函数接口不提供 `done` 或 `terminated` 标志，无法仅在一局结束时发放一次性的成功奖励。可用稠密 shaping 替代。  
  **forbidden_or_missing_signals**: done 标志  
- **role_id**: `info_based_failure_penalty`  
  **reason**: `info` 为空字典，不包含任何失败指标。  
  **forbidden_or_missing_signals**: info 字段

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_and_landing_shaping | x_position, y_position, x_velocity, y_velocity, body_angle, angular_velocity, left_support_contact, right_support_contact | （无） | distance_to_target, bounded_signal, quadratic_penalty, contact_bonus (when y near 0) | 核心奖励，所有信号均可用 |
| action_penalty | action (discrete 0/1/2/3) | （无） | discrete_action_cost (e.g. 0 for no thrust, penalty for others) | 配合使用需控制强度 |
| terminal_success_bonus | （无） | done/terminated flag | — | 禁用 |
| info_based_failure_penalty | （无） | info failure field | — | 禁用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 自由落体撞击 | y_velocity 很大且 y_position 快速减小，最终 crash | 增加靠近垫子时的速度惩罚，强化减速需求 |
| 飞出水平视口 | abs(x_position) 持续增大，最终终止 | 增加 x 偏离惩罚，或 reward x 向中心的吸引 |
| 姿态失控翻转 | abs(body_angle) 或 angular_velocity 剧增 | 加大对角度和角速度的惩罚，使用二次或绝对值 |
| 悬停犹豫不决 | episode 过长，y_position 徘徊在中高度 | 引入时间惩罚或降低悬停奖励，甚至给予负奖励促进下降 |
| 单腿着陆失败 | 最终只有一只脚接触，另一只悬空或撞击 | 在 y 接近 0 时显著奖励双腿接触，惩罚仅单腿接触 |
| 过度使用引擎 | 连续非零动作，燃料消耗高 | 适度增强 action_penalty，或按使用频率衰减奖励 |



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
- best_score_so_far: -5.350

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| angle_penalty + approach_reward + boundary_penalty + fuel_penalty + landing_bonus + proximity_warmth | 1 | -5.350 | -5.350 | unsolved |
| angle_penalty + approach_reward + boundary_penalty + contact_reward + fuel_penalty + landing_potential | 1 | -39.070 | -39.070 | unsolved |
| angle_penalty + approach_reward + boundary_penalty + contact_reward + fuel_penalty + landing_readiness | 1 | -45.740 | -45.740 | unsolved |
| attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + settling_stability | 4 | -65.000 | -109.730 | unsolved |
| angle_penalty + approach_reward + boundary_penalty + landing_bonus + velocity_penalty | 1 | -98.300 | -98.300 | unsolved |
| angle_penalty + approach_reward + boundary_penalty + close_proximity + contact_reward + fuel_penalty | 1 | -115.610 | -115.610 | unsolved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
