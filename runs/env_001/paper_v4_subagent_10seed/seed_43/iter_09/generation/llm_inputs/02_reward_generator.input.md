# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 刚体着陆/停靠任务。agent 从视口上方中央附近出发，带有一个随机初始力。主目标是尽快飞到中央的目标着陆垫，并在垫上稳定停靠（安全接触）。次要目标是在完成主目标的前提下，尽量缩短飞行时间和减少发动机推力使用。agent 需要学会接近目标、降低速度、保持姿态稳定、并以双腿接触垫面实现软着陆。
不该混淆的目标：不能把存活时间当作主要奖励信号（任务不是为了活得久，而是尽快到达并停稳）；不能把能耗降低到影响到达任务的程度，能耗只是附属优化。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 核心目标是到达指定的目标位置（中央着陆垫）并实现稳定接触，属于导航到达类任务。附属有时长和能耗优化，但不是多目标冲突中的多个同等重要目标，因此不选 multi_objective_task。动力学子类型进一步细分为 goal_approach_and_soft_contact（接近目标 + 低速稳定接触）。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: [8]
- dtype: float32 (以实际环境为准，通常为 float)
- obs[0]: x_position，含义：水平坐标（相对目标着陆垫中心），reward_usable: true
- obs[1]: y_position，含义：垂直坐标（相对垫面高度），reward_usable: true
- obs[2]: x_velocity，含义：水平线速度，reward_usable: true
- obs[3]: y_velocity，含义：垂直线速度，reward_usable: true
- obs[4]: body_angle，含义：机体倾斜角度，reward_usable: true
- obs[5]: angular_velocity，含义：角速度，reward_usable: true
- obs[6]: left_support_contact，含义：左支撑腿接触标志（0.0 或 1.0），reward_usable: true
- obs[7]: right_support_contact，含义：右支撑腿接触标志（0.0 或 1.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，不做任何推力，相当于滑行/自由落体
- action 1: left_orientation_engine，启动左侧姿态发动机（通常产生逆时针力矩）
- action 2: main_engine，启动主发动机（通常产生向上推力，可能同时影响姿态）
- action 3: right_orientation_engine，启动右侧姿态发动机（通常产生顺时针力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功标记。推测成功情况为：双腿接触着陆垫、机体接近垫中心、速度极低，此时可能触发 `body_not_awake_or_settled` 终止。
- failure-like termination: 
  - `crash_or_body_contact`: 除支撑腿之外的身体部分接触地面或障碍物（撞击/侧翻等）。
  - `horizontal_position_outside_viewport`: 水平飞出画面边界。
  - 因姿态或位置异常导致的无效停稳（例如单腿接触、翻倒后静止）也会触发 `body_not_awake_or_settled`，但不应视为成功。
- ambiguous termination: `body_not_awake_or_settled` 在成功软着陆和失败后静止时均可能触发，仅凭该事件无法区分。
- truncation: 无，只有终止 (terminated) 模式。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 空字典 `{}`（无任何 info 字段可用）
- forbidden_or_uncertain_info_fields: 所有未声明的字段均不可用；不存在 success、failure 等语义标志。

成功/失败只能通过 episode 结束时的观测信号间接推断：
- 推断成功：`left_support_contact == 1.0 and right_support_contact == 1.0`，`abs(x_position)` 很小，`abs(y_position)` 很小（接近垫面），线速度和角速度接近于零。
- 推断失败：上述条件不满足，或者观测到极端速度/位置值后终止。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（当前步观测）
- action（当前步动作）
- next_obs（下一步观测）
- info 中明确允许的字段（当前为空，不可用任何字段）
- training_progress 只有 prompt 明确允许时才用，此处未允许，禁止使用

禁止使用：
- original_reward 或任何官方奖励
- 未声明的 info 字段
- 未声明的 obs 切片或 any 环境私有状态

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position)，可计算到目标垫的距离（欧氏距离或加权距离），可计算相邻两步的距离变化。
- velocity: obs[2] (x_velocity), obs[3] (y_velocity)，可用于惩罚急停撞击，或在接近目标时鼓励减速。
- orientation: obs[4] (body_angle)，obs[5] (angular_velocity)，可鼓励保持竖直、减少旋转。
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact)，可给予双腿着陆奖励；单腿或不正常接触时不予奖励。
- action/engine: action 索引 1,2,3 表示使用引擎，0 为无推力，可做能耗惩罚；也可结合动力学判断推力强度。
- 间接成功信号 derived_possible: episode 终止时，通过位置、接触和速度组合推断是否成功软着陆，可用于终端奖励。

## 8. 不确定或不可用的信号
- 无直接的成功/failure 标志，需依赖 derived_possible 推断。
- 无剩余时间或步数信息（若需时间惩罚，需自己计时，但本任务未强制要求优化时长，不建议添加时间微分奖励以免过度复杂化）。
- 无风力、初始随机力等环境的隐藏状态。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_with_two_legs
  actuator_type: main_engine_and_two_orientation_engines
  contact_structure: two_support_contacts
primary_objectives:
  - reach the target landing pad (minimize horizontal/vertical distance)
  - make safe soft contact with both legs on the pad
  - stabilize body (zero/low velocity and angular velocity at touchdown)
secondary_objectives:
  - minimize time-to-land (implicitly through progress bonus)
  - minimize engine usage (fuel efficiency)
main_failure_risks:
  - overshooting or crashing outside viewport
  - landing with body tilt/one leg causing crash
  - excessive fuel consumption with no progress
  - premature engine cutoff leading to hard landing
  - over-controlling and oscillation
```

## 10. 奖励职责拆解 reward_role_decomposition

### 骨架选择推理小结
根据任务核心“离目标更近了吗”，主信号算子族选定为 **delta(distance)**（朝目标每一步的距离减少量），以避免纯 proximity 奖励造成的悬停陷阱。当距离很小、双腿接触时，配合接触成功奖励和稳定奖励；当偏离目标时，不因单纯存活而获得正分。

### 10.1 主职责 mandatory_roles
- role_id: approach_progress
  purpose: 鼓励每一步向目标着陆垫靠近。
  why_required: 核心目标为到达指定位置，必须提供密集的进度信号引导 agent 学习轨迹走向。
  usable_signals: [x_position, y_position] (计算



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





# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -18.800

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| landing_approach_reward + progress + soft_landing_penalty | 1 | -18.800 | -18.800 | unsolved |
| contact_success_reward + landing_approach_reward + progress | 1 | -55.780 | -55.780 | unsolved |
| action_cost + angle_penalty + landing_soft_reward + progress + safety_penalty | 1 | -80.850 | -80.850 | unsolved |
| action_cost + gate_factor + shaping + success_bonus | 1 | -95.670 | -95.670 | unsolved |
| contact_success_reward + progress + soft_landing_penalty | 1 | -112.840 | -112.840 | unsolved |
| landing_bonus + progress + soft_landing_penalty | 1 | -115.300 | -115.300 | unsolved |
| contact_success_reward + landing_gate + progress | 1 | -115.490 | -115.490 | unsolved |
| action_cost + angle_penalty + boundary_penalty + landing_soft_reward + progress | 1 | -117.780 | -117.780 | unsolved |

## Previous interventions

- iter 7 (score=-80.850, structure=action_cost + angle_penalty + landing_soft_reward + progress + safety_penalty): 本轮修改 **Level 2 结构变换**，将永不死触发（active_rate=0）的 `boundary_penalty` 组件替换为一个新的 `safety_penalty` 组件，填补“碰撞前兆安全约束”的信号缺口。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
