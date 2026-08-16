# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
任务是在 2D 视口中控制一个带引擎和双腿的飞行器，从顶部中心附近出发，尽快且平稳地降落到位于中央的目标平台上，并保持稳定姿势。核心目标是**到达目标平台并安全停靠**；次要目标包括：**最小化引擎推力（燃料消耗）**、**缩短飞行时间**、**保持着陆时姿态水平**、**两腿平稳触地**。任务不要求连续行进，也不涉及抓取物体。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达指定位置（目标垫）并稳定停止，非连续行走或生存平衡。次要目标（时间、燃料、姿态）服从主目标，不属于权重相当的多目标冲突任务。

动力学子类型 dynamics_subtype: goal_approach_and_soft_contact  
（接近目标并低速、稳定接触/停靠）

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（推断）
- obs[0]: x_position（相对目标垫水平坐标），reward_usable: true
- obs[1]: y_position（相对平台高度垂直坐标），reward_usable: true
- obs[2]: x_velocity（水平线速度），reward_usable: true
- obs[3]: y_velocity（垂直线速度），reward_usable: true
- obs[4]: body_angle（机体倾斜角度），reward_usable: true
- obs[5]: angular_velocity（角速度），reward_usable: true
- obs[6]: left_support_contact（左腿触地标志），值 0.0 或 1.0，reward_usable: true
- obs[7]: right_support_contact（右腿触地标志），值 0.0 或 1.0，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine（不做任何推进）
- action 1: left_orientation_engine（点火左侧姿态引擎，产生一个方向的旋转力矩）
- action 2: main_engine（点火主引擎，产生垂直向上的推力）
- action 3: right_orientation_engine（点火右侧姿态引擎，产生相反方向的旋转力矩）

（注：动作空间为离散，每个动作代表同时启用特定引擎的组合，但无法同时启用多个。未提供引擎推力、力矩大小或燃料消耗速率的具体数值，只能从观测变化间接感知。）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（飞行器静止在目标平台上且两腿平稳接触，可视为成功着陆）
- failure-like termination: crash_or_body_contact（本体与地面或其他物体非法接触，或角度/速度过大导致坠毁），horizontal_position_outside_viewport（飞出水平边界）
- ambiguous termination: 无（依据三个条件区分）
- truncation: 未提供，假设没有时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典，无显式标志）
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空，不可使用任何 info 键）
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止使用，因为 step 返回的是空字典

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测，shape (8,)），可使用全部 8 维
- action（当前动作，0~3 整数）
- next_obs（下一时刻观测）
- training_progress（仅当环境定义明确说明可用时，此处未提供，应避免使用）

禁止使用：
- original_reward（官方奖励已掩蔽，不可使用）
- info（因为为空，任何尝试访问 info 字段均不可用）
- 未在观测空间中声明的任何其他信息

## 7. 可用于奖励函数的信号
- position: obs[0]（x坐标）, obs[1]（y坐标）
- velocity: obs[2]（x速度）, obs[3]（y速度）
- orientation: obs[4]（角度）, obs[5]（角速度）
- contact: obs[6]（左腿触地）, obs[7]（右腿触地），二值信号
- action/engine: 可获取当前动作，推断使用了哪个引擎（0=无，1=左姿态，2=主引擎，3=右姿态）。可用于最小化推力使用。
- other: 无其他明确信号

## 8. 不确定或不可用的信号
- 显式着陆成功或失败标志：不存在
- 累计燃料消耗或剩余燃料：不可用
- 时间步数或计时器：未提供（但可根据轨迹长度间接推断，不过非显式信号）
- 平台中心精确坐标：通过 obs[0] 和 obs[1] 接近 0 时表示位于目标上方，平台高度定义为某固定值（但未提供具体数值，可从观测范围估计）
- 安全着陆所需的速度/角度阈值：环境未给出，需通过尝试确定合理范围（如 |angle| < 阈值, |velocity| < 阈值）
- 地面高度或地形信息：不可用

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: planar_vehicle_with_two_legs
  actuator_type: discrete_thrusters (main engine, two orientation thrusters)
  contact_structure: two_point_contact_on_legs
primary_objectives:
  - reach_target_pad (obs[0]≈0, obs[1]≈0)
  - soft_landing (low velocity, near-zero angle, both legs in contact)
secondary_objectives:
  - minimize_engine_usage (sparse action 2, 1, 3)
  - minimize_time_to_landing (implicit: quick settling)
  - keep_stable_orientation_during_flight (penalize large angles)
main_failure_risks:
  - crashing_body (excessive speed or angle when touching ground)
  - going_out_of_bounds (horizontal position too large)
  - landing_on_side_or_tipping_over (one leg contact, large angle)
  - excessive_fuel_use_or_oscillation (alternating actions frequently)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_reaching_progress  
  purpose: 鼓励向目标接近，减少水平与垂直距离  
  why_required: 任务是到达目标垫，直接衡量进度  
  usable_signals: [obs[0], obs[1], next_obs[0], next_obs[1]]  
  risks: 单纯距离奖励可能让飞行器高速撞向目标，需配合减速和姿态约束

- role_id: soft_landing_on_both_legs  
  purpose: 促进两腿同时触地，保持身体水平且低速  
  why_required: 成功着陆的定义，否则视为坠毁  
  usable_signals: [obs[4], obs[6], obs[7], obs[2], obs[3]]  
  risks: 不恰当的正奖励可能导致提前终止或奖励消失

- role_id: action_sparsity  
  purpose: 惩罚主引擎和姿态引擎的使用，鼓励无动作滑翔  
  why_required: 明确要求“尽可能少用推力”  
  usable_signals: [action]  
  risks: 过度惩罚可能阻碍必要调整，需要平衡

### 10.2 条件职责 conditional_roles
- role_id: terminal_success_bonus  
  condition_to_use: 只有通过观测判断成功着陆时（如 next_obs 满足 contact both, low speed, near-zero angle, position near zero），并且 done=True（但 done 信号不可直接获取，需外部传入或根据 next_obs 推断？由于无法获取 done 标志，可能需要从 last observation 判断）  
  usable_signals: [next_obs[6], next_obs[7], next_obs[2], next_obs[3], next_obs[4]]  
  risks: 错误识别成功会污染训练；若无 done 标志，可仅在最后一帧给予奖励（需环境提供 done 参数，但 reward 函数规范中未显式包含 done，需注意：调用时可能传入额外参数如 terminated。如果未提供，则无法使用。根据接口，只有 compute_reward(obs, action, next_obs, original_reward, info, training_progress)，所以 done 不可用，此条件职责可能无法安全实现，除非用 next_obs 的接触+零速度推测后由外部逻辑给予，但 reward 函数本身无法知道 episode 是否结束。因此，慎用）

- role_id: orientation_stabilization_during_flight  
  condition_to_use: 当飞行器尚未着陆时（根据接触标志和高度判断）  
  usable_signals: [obs[4], obs[5]]  
  risks: 着陆时角速度惩罚可能妨碍最后姿态调整

### 10.3 慎用/禁用职责 avoid_roles
- role_id: binary_crash_penalty  
  reason: 无法获取可靠 crash 信号（无 info 标志）；通过观测阈值硬判 crash 可能导致误判（如腿触碰但角度大，可能被环境视为 crash，但 reward 无法区分 crash 与正常着陆终止）。应依赖环境返回的终止和 next_obs 综合判断，但在仅 reward 函数中无法获得 done 和真实终止原因，故应避免用巨大负奖励直接惩罚 crash，改为通过 soft landing 引导

- role_id: explicit_success_flag_dependent_bonus  
  reason: 缺少任何 info 标志，无法使用

- role_id: time_penalty  
  reason: 没有时间步计数器信号，无法可靠施加，仅能通过总步数间接影响，但不建议在单步 reward 中硬编码

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_reaching_progress | obs[0], obs[1], next_obs[0], next_obs[1] | 目标位置具体值（但知道目标在0,0） | distance_change_reward, dense_state_signal, linear_progress | 使用 -（当前位置到目标距离）或距离缩减量 |
| soft_landing_on_both_legs | obs[6], obs[7], obs[4], obs[3], next_obs相应值 | 安全阈值（需实验定义） | bounded_signal, and_combination, product | 当双腿接触且角度速度都在允许范围内时给予正向奖励；可作为 done 条件外的连续塑造 |
| action_sparsity | action | 引擎推力大小（无法量化燃料消耗） | discrete_action_penalty, step_cost | 对 action 1/2/3 施加固定小惩罚，action 0 不罚 |
| terminal_success_bonus | next_obs[6], next_obs[7], next_obs[2], next_obs[3], next_obs[4] | done 标志，真实成功/失败状态 | sparse_event_bonus | 依赖外部在 episode 结束时传递额外标志或使用最后一帧观测，但奖励函数若无 done 参数则无法可靠实施，可能需降级为条件启用 |
| orientation_stabilization_during_flight | obs[4], obs[5] | 飞行/着陆分割 | quadratic_penalty on angle, angular velocity | 仅在 contact==0 时激活，避免干扰着陆 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 高速坠毁 | episode 长度极短，最终 y_velocity 很大，或角度严重倾斜 | 增加 soft landing 惩罚权重，降低 progress 权重 |
| 悬停不下降或过度盘旋 | 大量使用姿态引擎，y_position 不变或增加，燃料耗尽 | 调整 goal_reaching 信号优先竖直接近，加强主引擎鼓励 |
| 落入边界外 | x_position 超限终止 | 增加水平位置误差惩罚 |
| 仅单腿着陆并翻倒 | 终止时一条腿接触、角度大 | 强化双侧接触奖励，增加角度惩罚，甚至奖励 landing_legs_both |
| 引擎使用过于频繁/振荡 | action 频繁在 1/3 间切换 | 增大 action sparsity 惩罚，或加入动作变化平滑损失（但需连续观测前一个动作，此处未提供，可考虑使用动作保留奖励） |
| 未学会快速下降，拖延时间 | episode 很长但不 crash，最终依靠长时间滑翔才着陆 | 加入隐含时间惩罚（如每步微小负奖励），或整体 shaping 势能下降优先 |



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
- best_score_so_far: -124.550

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| contact_reward + orientation_penalty + progress_reward + vel_damping | 4 | -124.550 | -159.200 | unsolved |

## Previous interventions

- iter 2 (score=-168.020, structure=contact_reward + orientation_penalty + progress_reward + vel_damping): selected_level**：L2: 这里不是单纯的权重问题，vel_damping 的数学条件需要修改才能从“死组件”变为有效信号。因此选择 L2 修改其公式形态。 | selected_intervention: 目标组件：`vel_damping`（将其重定义为“接近地面时的速度惩罚”）
- iter 3 (score=-124.550, structure=contact_reward + orientation_penalty + progress_reward + vel_damping): selected_level**：**L2**（改动数学形态） | selected_intervention**：

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
