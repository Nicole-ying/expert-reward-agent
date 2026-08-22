# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个2D着陆器从视野顶部中央区域出发，借助初始随机速度，以最短时间和最少燃料消耗完成向中央着陆垫的精确软着陆。要求最终稳定停靠在目标垫上：位置接近垫中心，速度趋近于零，姿态保持垂直，且两条支撑腿与地面发生安全接触。避免任何形式的撞击、侧翻、越界或持续不稳定振荡。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 核心目标是让着陆器到达并稳定在指定的固定目标垫上，属于导航-目标到达任务族。燃料最小化和快速到达是性能优化，非冲突性多目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64 (默认，观测中 contact 是 0.0/1.0)
- obs[0]: `x_position` – 相对于目标垫的水平距离，reward_usable: true
- obs[1]: `y_position` – 相对于着陆垫高度（垫平面）的垂直距离，reward_usable: true
- obs[2]: `x_velocity` – 水平线速度，reward_usable: true
- obs[3]: `y_velocity` – 垂直线速度，reward_usable: true
- obs[4]: `body_angle` – 机体相对于垂直方向的偏转角度，reward_usable: true
- obs[5]: `angular_velocity` – 机体角速度，reward_usable: true
- obs[6]: `left_support_contact` – 左支撑腿接触标志 (1.0 表示接触)，reward_usable: true
- obs[7]: `right_support_contact` – 右支撑腿接触标志 (1.0 表示接触)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` – 无引擎推力，仅靠惯性飞行
- action 1: `left_orientation_engine` – 点燃左侧姿态调节引擎，产生顺时针力矩和少量侧推力
- action 2: `main_engine` – 点燃主发动机，产生向上推力，同时消耗燃料
- action 3: `right_orientation_engine` – 点燃右侧姿态调节引擎，产生逆时针力矩和少量侧推力

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` – 着陆器进入静止/休眠状态通常意味着两条腿牢固着陆且已稳定，可视为成功着陆。
- failure-like termination:
  - `crash_or_body_contact` – 机体任何非支撑腿部分触地、猛烈撞击或侧翻，导致坠毁。
  - `horizontal_position_outside_viewport` – 水平漂移超出可接受边界（离开视口），代表任务失败。
- ambiguous termination: 无。
- truncation: 返回 `False`，无额外截断限制；但实际环境中可能存在最大步数限制，但 env 未透露，本卡片不采用。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （空字典）
- forbidden_or_uncertain_info_fields: `original_reward`, `official_reward`, `success`, `failure`, 任何未在 info 中声明的字段

注意：不能根据 `terminated` 的真假直接判断成功/失败，因为有两种失败和一种成功都会触发终止，但 `terminated` 本身不区分原因。必须从 `next_obs` 和 `done` 中推断，或者整合观测信号（如两条腿是否接触、速度大小等）来构建奖励。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs` – 当前步观测数组 (8,)
- `action` – 当前执行的动作 (int)
- `next_obs` – 下一步观测数组 (8,)
- `info` – 仅限空字典，不得依赖任何字段

禁止使用：
- `original_reward` （官方奖励被遮蔽，严禁以任何方式引用或重构）
- `training_progress` 除非本提示明确声明允许（此处未声明）
- 任何未在 observation_space 描述中明确列出的 `obs` 切片
- 任何未在 info 约束中列出的字段

## 7. 可用于奖励函数的信号
- position: `x_position`, `y_position` （均为相对目标垫）
- velocity: `x_velocity`, `y_velocity`
- orientation: `body_angle`, `angular_velocity`
- contact: `left_support_contact`, `right_support_contact` （0.0/1.0 浮点）
- action/engine: 动作编号（0～3），可据此构造燃料消耗罚项或推力鼓励
- other: 可通过 `next_obs` 观察变化量（如速度变化、角度变化），但每一步间隔固定，近似微分可用

## 8. 不确定或不可用的信号
- 任何显式成功/失败标志 (info 中无) 。
- 燃料余量或耗量 （观察空间中未提供，只能通过动作使用情况间接推断）。
- 目标位置绝对坐标 （因为观测本身就是相对于目标垫的偏移量，但目标垫位置固定未给出，只能假设垫中心为 (0,0) 参考系）。
- 地形高度、风力扰动等环境隐藏变量。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: symmetric two-legged lander (lander)
  actuator_type: thrusters (main engine + 2 orientation engines)
  contact_structure: two support legs with binary contact sensors at the feet
primary_objectives:
  - Achieve soft landing on the target pad: x_pos ≈ 0, y_pos ≈ 0, low velocity, near-upright orientation, both legs in contact.
secondary_objectives:
  - Minimize engine usage (fuel) and time to landing.
main_failure_risks:
  - Crashing by high vertical speed or body part contact.
  - Drifting out of horizontal bounds.
  - Over-rotating and failing to stabilize.
  - Inefficient hover or oscillation leading to never settling.
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: `goal_approach_and_soft_landing`
  purpose: 鼓励着陆器靠近目标垫、减速、保持直立，并最终用双腿接触地面。
  why_required: 这是任务的核心成功条件，没有它智能体无法学习到达并稳定着陆。
  usable_signals:
    - 位置误差：`x_position`, `y_position`（越小越好）
    - 速度幅值：`x_velocity`, `y_velocity`（越小越好）
    - 姿态误差：`|body_angle|` 或 `body_angle^2`
    - 接触奖励：`left_support_contact` 与 `right_support_contact` 同时为 1 时给予奖励
  risks:
    - 单纯位置接近奖励可能导致高速撞击；必须配合速度惩罚。
    - 姿态奖励需要 careful 设计，以免过早将着陆器锁死在直立状态而妨碍必要的倾角调整。

### 10.2 条件职责 conditional_roles
- role_id: `fuel_efficiency`
  condition_to_use: 可在任何阶段加入，但权重需平衡，避免在降落关键阶段过度抑制主发动机使用。
  usable_signals:
    - 当前动作是否为 `main_engine` (动作2) 或姿态引擎 (动作1/3) 触发惩罚；或者给予恒定的位置无关奖励，并对任何非零推力动作施加小惩罚。
  risks:
    - 过度惩罚燃料使用可能导致智能体拒绝点火，永远无法着陆; 建议用小的负奖励或在成功着陆后给予更大的一次性奖励来抵消。
    - 若结合位置速度惩罚，燃料惩罚需适度。

- role_id: `time_pressure` (快速到达)
  condition_to_use: 如果希望在有限时间内完成任务，可加入每一步微小的负奖励，但会加剧燃料惩罚压力。通常不需要显式实现，因为步数限制自然产生压力。
  usable_signals: 每步恒定负值（如-0.05），但需谨慎。
  risks: 可能导致匆忙撞击，必须伴随强力安全约束。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: `explicit_termination_reward`
  reason: 环境不提供 info 中的成功/失败标志，且 `terminated` 无法区分成功与失败，若直接根据 `terminated` 给予大奖励极其危险（可能把失败也当作成功奖赏）。任何依赖终止原因分发的奖励都不可用。
  forbidden_or_missing_signals: 缺失 `success`/`failure` 字段。

- role_id: `shaping_based_on_original_reward`
  reason: original_reward 被禁止使用，不能作为参考或差值奖励。
  forbidden_or_missing_signals: original_reward 被遮蔽。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_approach_and_soft_landing | x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact | 无 | `bounded_signal( (x_pos^2 + y_pos^2) )`, `quadratic_penalty(velocity)`, `cosine_proximity(angle)`, `logical_and(left_contact, right_contact)` | 接触信号可作为步骤内奖励，但应仅在两条腿都接触并且速度都接近0时给予大奖励，防止提前奖励。 |
| fuel_efficiency | action (0-4) | 燃料消耗量 | `discrete_action_penalty([0, -0.03, -0.3, -0.03])` 或类似 | 主发动机的惩罚应显著高于姿态引擎，因为它的脉冲更大。注意平衡。 |
| time_pressure | （每步常数） | 无 | `stepwise_constant_penalty(-0.005)` | 可以省略，由环境截断时间自然施压。 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 高速撞击坠毁 | 训练曲线奖励不升，episode 长度短，最终 `y_velocity` 很大负值且终止 | 增加速度惩罚权重，特别是在低高度时；引入高度相关速度上限惩罚。 |
| 悬停不降或无限等待 | episode 长度达到环境最大值但不终止成功，位置接近但仍有速度，双腿未同时接触 | 增加高度奖励（对接近地面给予小奖励）或引入 soft landing bonus，仅在双腿接触且速度极小时给予较大奖励。 |
| 越界漂出视口 | 水平位置超出边界导致终止，x_position 绝对值很大 | 加重水平偏差惩罚，或者在奖励中施加平方惩罚，让智能体更用力修正。 |
| 过早起火导致燃料耗尽 | 每步燃料惩罚已存在但智能体仍大量使用主发，最终熄灭后坠落 | 检查燃料惩罚是否过小，或引入燃料总预算感知（但观测无燃料信息），可模拟采用动作熵惩罚或增加主发惩罚。 |
| 振荡不稳，永远不 sleep | 身体角速度持续非零，腿交替接触但不稳定 | 增加角速度惩罚，或者接触后给予小量额外固定奖励以鼓励快速镇定。 |
| 只有一条腿接触就停 | 左或右腿接触为1而另一个为0，终止由于body_not_awake? 需确认是否只有完整着陆才触发 sleep | 确保接触奖励要求双腿都接触，避免单腿接触产生奖励。 |



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
- best_score_so_far: -84.020

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| approach_reward + landing_bonus + stability_penalty + thrust_cost | 1 | -84.020 | -84.020 | unsolved |
| contact_reward + landing_bonus + pos_attract + progress_reward + stability_penalty + step_cost | 1 | -113.100 | -113.100 | unsolved |
| approach_reward + landing_bonus + stability_penalty + thrust_cost + unbalanced_penalty + vel_penalty | 1 | -114.200 | -114.200 | unsolved |
| altitude_reward + approach_reward + landing_bonus + stability_penalty + thrust_cost + vel_penalty | 1 | -185.720 | -185.720 | unsolved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
