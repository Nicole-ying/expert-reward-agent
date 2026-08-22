# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器（带有主引擎与两个方位引擎）从视口顶部中心附近出发，尽可能快地飞抵并稳定在中央目标平台上。主要目标是到达目标位置并保持安全、低速、姿态水平的着陆；次要目标是在过程中尽量少使用引擎推力（节约能耗）。不应将快速到达、姿态稳定或省燃料视作与主目标等同的多个独立目标——它们都是为主着陆服务的附属要求。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心是到达指定目标位置（中央目标平台），能耗最小化和速度要求是附属优化项，不存在多个权重相当且冲突的核心目标，因此归入导航目标到达类。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64（推测为标准连续值，也可能是 float32）
- 各维含义：
  - obs[0]: x_position，水平坐标（相对目标垫横向偏移），reward_usable: true
  - obs[1]: y_position，垂直坐标（相对垫面高度），reward_usable: true
  - obs[2]: x_velocity，水平线速度，reward_usable: true
  - obs[3]: y_velocity，垂直线速度，reward_usable: true
  - obs[4]: body_angle，机体方向角（可能以竖直为 0），reward_usable: true
  - obs[5]: angular_velocity，角速度，reward_usable: true
  - obs[6]: left_support_contact，左支撑腿接触标志（0.0 或 1.0），reward_usable: true
  - obs[7]: right_support_contact，右支撑腿接触标志（0.0 或 1.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：
  - action 0: no_engine —— 不做任何推力输出
  - action 1: left_orientation_engine —— 点燃左侧方位引擎（产生逆时针力矩）
  - action 2: main_engine —— 点燃主引擎（产生向前或向上的推力，具体方向需结合机体角度）
  - action 3: right_orientation_engine —— 点燃右侧方位引擎（产生顺时针力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  - `body_not_awake_or_settled`：机体进入静止或稳定状态，很可能是在目标平台上成功着陆后触发。此条件在观测未提供直接标志，但表现为所有速度、角速度归零且接触标志为真的稳态。
- failure-like termination:
  - `crash_or_body_contact`：机体非支撑部分碰撞地面或障碍物，导致损毁。
  - `horizontal_position_outside_viewport`：水平坐标超出允许范围（飞离视口）。
- ambiguous termination:
  - 无。
- truncation:
  - 根据代码，`terminated` 在三种条件之一触发时设为 True，无其他截断，`truncated=False` 恒成立，因此不存在超时截断（除非底层实现有限定最大步数，但未在 spec 中体现，视为无）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

注：成功与失败的推断只能通过观测信号间接进行。例如，当 `left_support_contact` 与 `right_support_contact` 均为 true，且 `x_position`、`y_position` 接近 0，速度、角速度近乎 0 时，可推定为成功着陆终止（derived_possible）。撞毁可能伴随接触信号突变或速度极大，但无可靠单步信号，因此不推荐直接用于奖励，可转为避免边界和冲击的策略。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（前一步观测）
- action（当前步执行的动作）
- next_obs（执行动作后、终止前的观测）
- info（但当前环境 info 始终为空，实际不可用）
- training_progress 仅当 prompt 明确允许时才使用（当前未明确允许，慎用）

禁止使用：
- original_reward（被屏蔽的官方奖励）
- 任何未在 observation_space 中声明的 obs 切片
- 任何 info 字典字段（因其为空）

## 7. 可用于奖励函数的信号
- position：`next_obs[0]` (x)，`next_obs[1]` (y) — 相对目标垫的位置
- velocity：`next_obs[2]` (vx)，`next_obs[3]` (vy)
- orientation：`next_obs[4]` (angle)，`next_obs[5]` (angular_velocity)
- contact：`next_obs[6]` (left_contact)，`next_obs[7]` (right_contact)
- action/engine：当前动作 `action`（0~3）可用于推断引擎使用
- derived_possible：可通过连续观测检测出界（|x| 过大 → failure precursor）、冲击（速度突变结合接触变化）、接近成功着陆（低速度 + 双接触 + 角度小 + 位置近零）等推断，用于构建条件奖励或惩罚

## 8. 不确定或不可用的信号
- 明确的终止原因字段（如 `'crash'`、`'landed'`）不可用
- 燃料消耗量或推力大小（动作仅表示引擎类型，未给出推力值）
- 身体其他部分碰撞信息（仅有两条支撑腿的接触标志）
- 目标是否已达到的布尔标志
- 视口边界的具体数值（需从采样或经验中推导）
- 任何步数或时间剩余信息（无 `truncation` 或 `steps_remaining`）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D 飞行器/着陆器，具有左右两条支撑腿
  actuator_type: 三个独立离散引擎（主引擎、左方位引擎、右方位引擎）+ 无操作
  contact_structure: 两条支撑腿分别提供左、右接触布尔信号，其他身体部分接触可能导致 crash
primary_objectives:
  - 到达并停留在目标平台上（位置、速度、姿态三点同时满足）
secondary_objectives:
  - 最小化引擎使用（总动作中非零动作次数）
  - 尽快完成着陆（隐含时间惩罚）
main_failure_risks:
  - 水平飞出视口
  - 撞毁（非支撑部位触地）
  - 过度震荡或在目标上空不停盘旋，无法进入稳定状态
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity
  purpose: 驱动飞行器向目标位置移动，基于到目标垫的欧氏距离或单向距离。
  why_required: 这是导航任务的核心，没有它无法学会向目标靠近。
  usable_signals: [next_obs[0], next_obs[1]]
  risks: 纯距离奖励可能导致飞行



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
- best_score_so_far: -110.220

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| orientation_penalty + proximity_delta + velocity_danger | 1 | -110.220 | -110.220 | unsolved |
| landing_bonus + orientation_penalty + proximity_delta + velocity_danger | 2 | -111.880 | -111.880 | unsolved |
| orientation_penalty + proximity_delta + soft_approach_bonus + velocity_danger | 1 | -115.170 | -115.170 | unsolved |
| orientation_penalty + proximity_delta + velocity_penalty | 1 | -116.460 | -116.460 | unsolved |

## Previous interventions

- iter 4 (score=-111.880, structure=landing_bonus + orientation_penalty + proximity_delta + velocity_danger): 修改方案属于 **Level 1 尺度修复**：保持连续乘积形式，但大幅松弛门控阈值（距离衰减系数 0.3 → 1.0，速度/角度截止 0.3 → 0.5，使因子能在常见的着陆前状态中被激活），并将权重从 80.0 降为 20.0，避免单步奖励过度支配总回报（校准：预计激活时 per‑step ≤ 20 × 0.1~0.3 = 2~6，不超过主信号 proximity_delta 的 2~3 倍，可接受）。其他组件保持不变。
- iter 5 (score=-115.170, structure=orientation_penalty + proximity_delta + soft_approach_bonus + velocity_danger): 本轮修改一个组件：将 `landing_bonus`（僵尸组件，active_rate=0%）替换为 `soft_approach_bonus`。原组件依赖腿接触标志与严格速度阈值，从未被激活；agent 的终止模式表明它在高速撞击中 crash，没有机会产生腿接触或满足窄阈值。新组件去除腿接触依赖，使用连续的 y 高度 gate 与速度、角度因子，在接近着陆垫低高度且速度、姿态良好时给予正奖励，从而提供可学习的软着陆梯度。数学形式为三

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
