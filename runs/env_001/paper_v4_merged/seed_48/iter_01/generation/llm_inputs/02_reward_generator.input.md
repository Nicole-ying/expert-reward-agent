# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
任务目标是控制一个受随机初始力作用的二维飞行器，使其从视野顶部中央出发，尽可能快地飞抵并稳定停靠于中央目标平台上。核心目标是“到达并停靠”（reaching and settling），附属优化目标包括“尽可能少用引擎推力”和“保持姿态稳定、安全接触”。不应将其混淆为单纯的存活任务、无限制漫游任务或多目标博弈任务。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求“reach and settle at a central target pad”，核心目标是到达指定目标位置；附属目标为速度、能耗和姿态质量，符合导航到达的主任务范式。观测空间提供相对于目标的坐标，进一步证实其到达属性。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float64 (推测连续量)
- obs[0]: x_position (水平方向相对于目标平台的坐标)，reward_usable: true
- obs[1]: y_position (垂直方向相对于平台高度的坐标)，reward_usable: true
- obs[2]: x_velocity (水平线速度)，reward_usable: true
- obs[3]: y_velocity (垂直线速度)，reward_usable: true
- obs[4]: body_angle (机体朝向角)，reward_usable: true
- obs[5]: angular_velocity (角速度)，reward_usable: true
- obs[6]: left_support_contact (左支撑腿接触标志, 0/1)，reward_usable: true
- obs[7]: right_support_contact (右支撑腿接触标志, 0/1)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine (无推力)，即不激活任何引擎
- action 1: left_orientation_engine (左姿态引擎)，产生顺时针或逆时针力矩中的一种；具体方向需在交互中推断，但用于调整朝向
- action 2: main_engine (主引擎)，沿机体纵轴提供推力，用于平移/减速/抵抗重力
- action 3: right_orientation_engine (右姿态引擎)，产生与左姿态引擎相反的力矩，用于反方向姿态修正

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止；任务期望通过“到达并稳定停靠”后 episode 结束，这很可能通过 **timeout/truncation** 或在目标区域达到低速度、双腿接触、小角度等条件后被环境内部判定为 settled 而终止。
- failure-like termination:
  - *crash_or_body_contact*: 机体部分（非支撑腿）碰撞地面/平台以外区域，或姿态严重偏离导致翻倒。
  - *horizontal_position_outside_viewport*: 机体飞出水平边界，视为严重失控。
  - *body_not_awake_or_settled*: 可能是检测到速度/加速度极小但未达成着陆条件，或进入睡眠状态的超时机制。
- ambiguous termination: 支撑腿接触目标平台但未满足所有稳定条件，被 terminated 可能属于部分成功/硬着陆，不能直接视为完美成功。
- truncation: 任务可能包含 episode 长度上限，届时会直接截断。该截断不携带成功/失败固有语义。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 根据 source，step 返回空字典 `{}` ，因此 `info` 无任何可用字段。
- forbidden_or_uncertain_info_fields: info字典为空，无字段可用；不得依赖任何隐式 info 键。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs: 包含步骤执行前的状态（8维）
- action: 执行的动作（0~3）
- next_obs: 执行动作后的状态（8维）
- info: 必须为空字典，禁止访问任何字段
- training_progress: 本次 prompt 并未明确允许或禁止；保守做法是仅在绝对必要时使用，且不得作为唯一主信号

禁止使用：
- original_reward
- official_reward
- 任何未在 info 中声明的字段
- 任何未在 obs 空间中声明的隐藏状态

## 7. 可用于奖励函数的信号
- position: `next_obs[0:2]` 表示相对于目标平台的位置。可计算当前距离、距离变化量。
- velocity: `next_obs[2:4]` 线速度。可用于接近速度、稳定着陆时趋零、水平漂移控制。
- orientation: `next_obs[4]` 机体角度。可用于姿态维护、着陆时接近水平的奖励/惩罚。
- contact:
  - `next_obs[6]` 左支撑腿接触
  - `next_obs[7]` 右支撑腿接触
  - derived_possible: 双腿同时接触（legs_contact = left & right）是成功着陆的关键条件，可直接从观测构造。
- action/engine: `action` 可以用于对引擎使用施加惩罚。
- other:
  - angular_velocity `next_obs[5]` 可用于控制姿态抖动的阻尼惩罚。
  - derived_possible: settled 成功事件可间接推断：如果 episode 未因 crash/越界终止而截断，且最后几步保持双腿接触、低速度、小角度，则很可能为成功着陆。可在最终奖励中使用 sparse terminal success bonus，但必须标注为 derived_possible，且需在策略中小心处理以避免误判。

## 8. 不确定或不可用的信号
- 精确的成功标记 (info-based): 不存在。
- 燃料/推力能量消耗绝对值: 未直接提供推力大小，仅知动作选择，因此只能对“使用引擎”这一动作本身进行惩罚，无法得知实际推力大小。
- 地面/平台的法向量或确切碰撞位置: 不可用。
- 风向/随机力强度: 未暴露，无法直接补偿。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body (vehicle/lander)
  actuator_type: discrete thrusters (1 main + 2 orientation)
  contact_structure: two support legs (left/right contacts)
primary_objectives:
  - reach target pad (minimize distance to goal)
  - achieve soft landing (low velocity, near-zero angle, both legs in contact)
secondary_objectives:
  - minimize engine usage (actions 1,2,3 penalized)
  - minimize time-to-land (implicit in fast approach)
main_failure_risks:
  - crashing body parts other than legs
  - exiting horizontal bounds
  - excessive angle / angular velocity leading to instability
  - hard landing with high linear velocity
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: delta_distance_to_goal
  purpose: 推动 agent 向目标平台移动，是“更快到达”的核心驱动力。
  why_required: agent 初始位于顶部中央，必须克服随机初始力抵达中央。若无此信号，agent 可能悬停或随意飘浮。
  usable_signals: [next_obs[0], next_obs[1], obs[0], obs[1]]
  risks: 仅依赖 proximity (当前距离) 会导致 agent 停在半空不好不坏的平衡点而不着陆；因此必须使用 delta 形式 (distance_curr - distance_next) 或 improvement 来消除悬停陷阱。

- role_id: soft_landing_terminal_bonus
  purpose: 在 episode 结束时给予成功软着陆的明确奖励，将“到达”与“安全停靠”绑定。
  why_required: delta_distance 仅保证接近，不能保证减速、姿态对齐和双腿接触。必须用 terminal bonus 将优化引向最终的稳定状态。
  usable_signals: [next_obs[6], next_obs[7], next_obs[2:4], next_obs[4]]; derived_possible 判定 terminal 是否为成功 (非 crash/越界)。
  risks: bonus 过大会扭曲飞行阶段的决策；应设为适度值。判定成功需组合多个条件，误判会导致奖励污染。

### 10.2 条件职责 conditional_roles
- role_id: health_constraint
  condition_to_use: 当 agent 接近危险状态时激活，防止在接近目标时 crash。不应全程施加，避免初期探索受挫。
  usable_signals: [next_obs[4] 大角度, next_obs[0] 出界风险, derived_possible crash 接触 (可能来自角度突变或速度骤降)]
  risks: 过强的姿态/出界惩罚会阻碍早期探索，导致 agent 不敢移动或不敢调整姿态。应采用 hinge loss 仅在超出安全阈值时惩罚。

- role_id: engine_usage_penalty
  condition_to_use: 仅在任务明确要求“尽可能少用引擎”时加入，且通常作为弱正则项。不建议在到达目标前给予过大权重，否则 agent 可能选择不点火而飘离目标。
  usable_signals: [action == 1,2,3]
  risks: 过度惩罚会与“尽快到达”冲突，使 agent 行动迟缓。权重应远小于 delta_distance 奖励。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: constant_survival_bonus
  reason: 生存不是主任务；给予持续存活奖励会制造悬停收割陷阱，使 agent 满足于不坠毁而不着急着陆，与“尽快”冲突。
  forbidden_or_missing_signals: 无独立存活信号。

- role_id: angular_velocity_smoothness
  reason: 虽然姿态平稳是好事，但单独的成分在多数抵达目标环境中会与“快速接近”构成不必要的折衷。若观测发现姿态振荡严重，再作为微弱阻尼添加。初始版本禁用。
  forbidden_or_missing_signals: 无。

## 11. role_to_signal_mapping
| role_id                     | usable signals                                              | missing signals | candidate formula operators                                        | notes                                                                                              |
| --------------------------- | ----------------------------------------------------------- | --------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| delta_distance_to_goal      | obs[0:1], next_obs[0:1]                                    | —               | delta(distance), bounded_signal, improvement                       | 绝对距离作为 fallback 信号，但主信号必须是距离减量或类似进步度量。                   |
| soft_landing_terminal_bonus | next_obs[6], next_obs[7], next_obs[2:4], next_obs[4]       | —               | sparse terminal indicator, success_condition                       | 需 derived_possible 推断成功终止。组合双腿接触、低速度、小角度判定。                 |
| health_constraint           | next_obs[4], next_obs[0], derived_possible crash 推测      | —               | hinge_penalty, bounded_penalty                                     | 只在超出阈值时启用；不要用 quadratic penalty 在安全区内也惩罚。                       |
| engine_usage_penalty        | action (1,2,3)                                              | —               | constant_penalty                                                   | 权重必须很小，防止抑制探索。                                                                   |
| survival_bonus (avoid)      | —                                                           | —               | —                                                                  | 与“尽快”冲突。                                                                                 |
| angular_smoothness (avoid)  | next_obs[5]                                                 | —               | —                                                                  | 避免过早引入多目标权衡；若抖动严重再作为弱阻尼加入。                                          |

## 12. 初始训练后应观察的 failure modes
| failure_mode                         | evidence_to_check                                                              | possible_intervention                                                                           |
| ------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| 悬停不降落                            | episode 时间耗尽；agent 停留在目标上方但距离不再下降                           | 检查 delta_distance 是否被正确实现为 improvement；禁用生存奖励或降低生存相关权重。             |
| 快速冲向地面但 crash                  | 高速垂直接地、双腿未接触、episode 以 crash 终止                                 | 提高 soft_landing_terminal_bonus 的条件严格度；引入 hinge penalty 限制最大允许接地速度。       |
| 只用一个支撑腿接触，姿态歪斜          | 一侧接触为 1 另一侧为 0，body_angle 非零                                      | 在 terminal bonus 中要求双腿同时接触；或者在接近目标时添加姿态对齐的微弱辅助奖励。              |
| 大量使用引擎但进展缓慢                | 频繁使用 main_engine 但 Δdistance 小；燃料消耗高但前进少                       | 确认 delta_distance 信号是否足够强；考虑增加成功着陆 bonus 的吸引力，超过引擎惩罚。             |
| 胆小不点火，被风吹出视野              | 动作始终为 0，next_obs[0] 快速增大 (出界)                                      | 暂时降低 engine_penalty；确认 delta_distance 能提供足够梯度鼓励向目标移动；增加出界 penalty。    |
| 姿态失控旋转                          | angular_velocity 或 body_angle 一直增大至 crash                                | 先不修改奖励，检查策略探索是否得到足够的姿态动作；若持续，再引入 angular_velocity hinge penalty。 |



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

