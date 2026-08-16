# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
让一个 2D 飞行器/着陆器从初始随机扰动开始，尽快到达并稳定在中心目标平台上，同时尽量少用引擎推力。核心目标是实现安全、低速的接触与稳定停泊；次要目标是减少燃料消耗和缩短耗时。不应将“稳定停泊”与单纯的“到达目标位置”混淆：最终必须让左右支撑腿都接触地面（左/右接触标志为真），速度与角速度足够小，且身体角度接近水平，才算成功。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 核心目标是到达指定目标平台并稳定停泊，具有明确的终点状态（位置、接触），虽附带节能与时间要求，但这些是明确的附属优化项，不足以构成多目标冲突，因此归入“目标到达”任务族。

动力学更细子类：goal_approach_and_soft_contact，因为任务强调低速接近、稳定接触与角度控制，具有明显的接近目标并实现软接触的特征。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: 连续浮点（float32/float64），后两维为 0/1 但编码为连续值
- obs[0]: name=`x_position`, meaning=水平坐标相对于目标平台中心的偏移，reward_usable: true
- obs[1]: name=`y_position`, meaning=垂直坐标相对于平台高度的偏移，reward_usable: true
- obs[2]: name=`x_velocity`, meaning=水平线速度，reward_usable: true
- obs[3]: name=`y_velocity`, meaning=垂直线速度（向上为正或向下为正），reward_usable: true
- obs[4]: name=`body_angle`, meaning=本体方向角（弧度或度），reward_usable: true
- obs[5]: name=`angular_velocity`, meaning=角速度，reward_usable: true
- obs[6]: name=`left_support_contact`, meaning=左支撑腿接触地面时为 1.0，否则 0.0，reward_usable: true
- obs[7]: name=`right_support_contact`, meaning=右支撑腿接触地面时为 1.0，否则 0.0，reward_usable: true

所有维度在奖励函数中均可使用，因为它们描述了任务目标直接相关的状态。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine —— 不开启任何引擎，无推力消耗
- action 1: left_orientation_engine —— 开启左侧定向引擎，可能产生逆时针旋转和/或横向推力
- action 2: main_engine —— 开启主引擎，通常产生向上的推力（用于减速/悬停）
- action 3: right_orientation_engine —— 开启右侧定向引擎，可能产生顺时针旋转和/或横向推力

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` —— 当物体处于休眠状态或已稳定在接触表面时触发。这通常发生在着陆并满足静态稳定条件后，应被理解为成功。但需注意：该终止也可能在远离目标的位置意外休眠时发生，此时并非真正成功。
- failure-like termination: `horizontal_position_outside_viewport` —— 水平位置超出视口范围，明确失败。
- ambiguous termination: `crash_or_body_contact` —— 任何身体与环境几何体发生严重碰撞或任意本体部分接触地面。若为高速碰撞、翻滚或仅部分接触，则为失败；但若为期望的左右支撑腿同时接触平台，则可能同时被 `body_not_awake_or_settled` 捕获。单独该终止可能意味着硬着陆或非稳定接触，因此单独不可直接判定为成功。
- truncation: 源代码中 `truncated` 参数固定为 `False`，环境不会因为步数上限截断，但训练中通常会有外部时间限制，在此环境卡片中忽略。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 字典为空，无任何标志）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info = {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止使用，因为环境不提供任何额外信息。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前观察（所有维度）
- action: 当前动作
- next_obs: 下一时刻观察（所有维度）
- info: 空字典，不允许依赖任何键
- training_progress: 本任务描述未提及启用该参数，故应假定禁止使用

禁止使用：
- original_reward（已掩码的官方奖励）
- official_reward 或其任何变体
- 未在允许列表中出现的任何附加数据

## 7. 可用于奖励函数的信号
- position: next_obs[0] (x_position), next_obs[1] (y_position) —— 用于判断是否靠近目标并处于平台高度
- velocity: next_obs[2] (x_velocity), next_obs[3] (y_velocity) —— 用于鼓励低速接触
- orientation: next_obs[4] (body_angle), next_obs[5] (angular_velocity) —— 用于保持水平姿态与稳定
- contact: next_obs[6] (left_support_contact), next_obs[7] (right_support_contact) —— 用于确认两腿着地
- action/engine: action 值 —— 可用来惩罚使用引擎（非零动作），以体现节能
- other: 无

## 8. 不确定或不可用的信号
- 是否已成功：无明确 success flag；需要根据终止时刻的 next_obs 接触状态、速度、位置等综合推断。
- 燃料量或燃料消耗率：不可用，只能通过动作选择间接表达节能偏好。
- 生存时间或步数：不可用（truncation 不存在，且未提供 episode 长度）。
- 任何与地形或平台形状相关的先验信息：不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_lander
  actuator_type: discrete_thrusters (main + orientation engines)
  contact_structure: two_leg_contact (left/right support)
primary_objectives:
  - reach target pad with near-zero horizontal and vertical offset
  - establish stable contact with both legs on the pad
  - maintain near-horizontal orientation
secondary_objectives:
  - minimize engine usage (fuel conservation)
  - minimize time-to-contact
main_failure_risks:
  - overshooting or undershooting the pad due to excessive thrust
  - landing too fast, causing crash or bounce
  - tipping over (one leg contact or large body angle)
  - drifting out of viewport laterally
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_approach_progress
  purpose: 引导 agent 从任意初始状态向目标平台移动，缩小 x,y 距离
  why_required: 这是任务存在的根本动机，没有该职责 agent 不会学习接近目标
  usable_signals: [next_obs[0], next_obs[1]]
  risks: 仅靠距离奖励可能导致高速接近、碰撞反弹或无视角度/接触质量

- role_id: soft_contact_stability
  purpose: 鼓励在目标区域双足着地且速度、角速度足够小，实现稳定停泊
  why_required: 任务明确要求“settle at the pad”，仅位置接近是不够的，必须实际稳定接触
  usable_signals: [next_obs[2], next_obs[3], next_obs[4], next_obs[5], next_obs[6], next_obs[7]]
  risks: 如果设计不当，可能奖励假接触（一腿触地、另一腿悬空）或保留小速度导致滑动

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency_penalty
  condition_to_use: 在 agent 已能较可靠地完成着陆后（训练后期）引入，以避免早期阻碍探索
  usable_signals: [action] （非零动作即为开启引擎，简单惩罚）
  risks: 过早施加可能导致 agent 拒绝采取任何推力，完全不探索接近目标

- role_id: orientation_keeping
  condition_to_use: 可始终使用，但其强度需要调低，以免与接近目标冲突
  usable_signals: [next_obs[4], next_obs[5]]
  risks: 过度强调水平可能造成 agent 拒绝必要的姿态调整

### 10.3 慎用/禁用职责 avoid_roles
- role_id: time_penalty
  reason: 环境未提供时间或步数的显式信号，info 为空且无 truncation，无法直接惩罚耗时。虽可通过恒定负奖励近似，但缺乏依据且容易溢出
  forbidden_or_missing_signals: [episode_length, elapsed_time, truncation_flag]

- role_id: direct_success_bonus
  reason: 无可用的 explicit success flag，终止条件混叠，判别逻辑复杂且危险（可能奖励 crash 式的终态），必须避免。
  forbidden_or_missing_signals: [info["success"]]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_approach_progress | next_obs[0], next_obs[1] | — | bounded_signal (e.g., exp(-||pos||)) | 使用平滑衰减，避免在远离时梯度消失 |
| soft_contact_stability | next_obs[2:8] | explicit_success_flag | logical_anded_indicators, quadratic_penalty on velocities, bounded_reward for dual contact | 需组合 contact 和速度阈值，注意避免惩罚探索阶段的短暂接触 |
| energy_efficiency_penalty | action | fuel_consumption | discrete_action_cost (1 if action!=0 else 0) | 训练初期最好禁用或设极低权重 |
| orientation_keeping | next_obs[4], next_obs[5] | — | bounded_signal on abs(angle) + squared_penalty on angular_velocity | 可用于始终辅助稳定，但权重应弱于位置项 |
| time_penalty | — | episode_length, truncation | — | 排除，无可用信号 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| agent 悬停于远处不做接近 | 平均 x,y 距离不随时间下降 | 提高 goal_approach_progress 权重或衰减尺度，确保远距离仍有梯度 |
| 高速撞击平台后弹飞或 crash | 终止前 y_velocity 极负，出现 crash 终止 | 强化 soft_contact_stability 对速度的惩罚，可引入接近地面时的速度上限逻辑 |
| 单腿着地、翻倒 | left/right contact 不同时为 1，body_angle 较大 | 增加双足同时接触的奖励信号，强化 orientation_keeping |
| 引擎持续全开耗能极高 | 动作序列中零动作占比极低 | 适时激活 energy_efficiency_penalty，并渐增其系数 |
| 在水平方向上漂移出界 | x_position 频繁超过视口限制导致终止 | 在 goal_approach_progress 中加入横向边界软约束，或惩罚大横向偏离 |



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
- best_score_so_far: -103.790

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| action_penalty + landing_reward + progress_reward | 2 | -103.790 | -103.790 | unsolved |
| action_penalty + contact_reward + descent_reward + progress_reward | 1 | -120.100 | -120.100 | unsolved |
| contact_reward + orientation_penalty + progress_reward + vel_damping | 4 | -124.550 | -159.200 | unsolved |

## Previous interventions

- iter 2 (score=-168.020, structure=contact_reward + orientation_penalty + progress_reward + vel_damping): selected_level**：L2: 这里不是单纯的权重问题，vel_damping 的数学条件需要修改才能从“死组件”变为有效信号。因此选择 L2 修改其公式形态。 | selected_intervention: 目标组件：`vel_damping`（将其重定义为“接近地面时的速度惩罚”）
- iter 3 (score=-124.550, structure=contact_reward + orientation_penalty + progress_reward + vel_damping): selected_level**：**L2**（改动数学形态） | selected_intervention**：
- iter 6 (score=-103.790, structure=action_penalty + landing_reward + progress_reward): 历史记录中（通过 `read_past_reflection` 和 `read_reward_code` 检索）**未出现**“将 landing 奖励改为连续的高度‑速度‑姿态质量函数”的尝试，因此本轮修改不是已有语义的重复。 | selected_level: L2** — 改变 landing_reward 的数学形态，从离散双腿接触奖励变为连续的高度/速度/姿态质量函数。历史中未做过此变换，且当前分数离目标（200）差距远超 10%，L1 微调不足以解决稀疏性问题。 | selected_intervention: 将 `landing_reward` 改写为：
- iter 7 (score=-120.100, structure=action_penalty + contact_reward + descent_reward + progress_reward): selected_level**：**L2: 将 `landing_reward` 从“连续状态质量奖励”改为“下降过程奖励 + 接触事件奖励”，从根本上消除静态悬停可利用点。同步删除 `progress_reward` 的健康门控使其重新生效。 | selected_intervention**：

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
