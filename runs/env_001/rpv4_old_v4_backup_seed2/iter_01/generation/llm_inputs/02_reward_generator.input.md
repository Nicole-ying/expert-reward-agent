# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
主体是一个2D类车辆刚体，从视口上方中央附近出发，带有随机初始作用力。任务是**尽快到达并稳定停靠在中央的目标垫上，同时尽可能减少发动机推力消耗**。理想行为是：平滑接近目标、降低线速度、保持垂直姿态（零角度）、以低速和正确朝向使左右两个支撑点同时接触目标垫并固化。次要目标包含节省燃料和/或快速完成，但不应混淆为主目标。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是“到达并稳定在目标垫上”，这是典型的到达型任务；附属有速度、姿态、能耗等约束，但到达仍是主目标，不含多个权重相当且冲突的核心目标。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float64 (默认) 或 float32
- obs[0]: x_position — 相对目标垫的水平坐标 (reward_usable: true)
- obs[1]: y_position — 相对目标垫垫面高度的垂直坐标 (reward_usable: true)
- obs[2]: x_velocity — 水平线速度 (reward_usable: true)
- obs[3]: y_velocity — 垂直线速度 (reward_usable: true)
- obs[4]: body_angle — 车身朝向角 (reward_usable: true)
- obs[5]: angular_velocity — 角速度 (reward_usable: true)
- obs[6]: left_support_contact — 左支撑脚接触标志，接触为1.0 (reward_usable: true)
- obs[7]: right_support_contact — 右支撑脚接触标志，接触为1.0 (reward_usable: true)

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不点火，靠惯性滑行
- action 1: left_orientation_engine — 点燃左姿态发动机（提供转向力矩）
- action 2: main_engine — 点燃主发动机（提供向上/向前的推力）
- action 3: right_orientation_engine — 点燃右姿态发动机（提供相反力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- crash_or_body_contact: 主体发生碰撞或非预期的身体接触（很可能为失败，但不排除目标垫接触也属于body_contact，需结合后续稳定信号区分）
- horizontal_position_outside_viewport: 水平位置超出视口边界（明确失败）
- body_not_awake_or_settled: 主体不再活动或已稳定（speed/angular接近零且可能接触垫）；此条件可能是成功（settled on pad）或失败（settled elsewhere or falling asleep）
- 没有显式截断（truncation）条件提及，但实际环境可能有最大步数限制，目前未在信息中给出。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false (info为空字典)
- explicit_failure_flag_available: false
- allowed_info_fields: （当前step返回的info = {}，无可用字段）
- forbidden_or_uncertain_info_fields: 无任何显式success/failure/termination_reason字段；不可使用info进行reward或终止判定

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs (ndarray) – 当前步观察
- action (int) – 当前步动作
- next_obs (ndarray) – 下一步观察（可用于评估过渡效果）
- info 中仅允许环境卡片明确声明的字段（当前无任何字段被声明，故info不可用）
- training_progress (float) 仅当任务提示明确允许时可使用，当前未允许

禁止使用：
- original_reward（任何掩蔽的官方奖励）
- 未声明的 info 字段
- 未声明的 obs 切片（例如假定的额外维度）
- 任何内置评估指标或环境内部状态

## 7. 可用于奖励函数的信号
- position: x_position, y_position (相对目标垫)  
- velocity: x_velocity, y_velocity  
- orientation: body_angle  
- angular_velocity: angular_velocity  
- contact: left_support_contact, right_support_contact (二值浮点，1.0表示接触)  
- action/engine: action index (可用于动作惩罚，或结合下一步状态推断推力效果)
- other: 无额外传感器

## 8. 不确定或不可用的信号
- 目标垫的确切尺寸或接触阈值未知；仅能通过观测位置和接触信号推断是否真正着陆
- 步数上限或时间惩罚信息未提供
- official reward 的任何组成部分被屏蔽，不可获取
- 训练进度 training_progress 未授权使用
- 无任何 external 目标位置之外的指引信号

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with two downward support legs
  actuator_type: thrusters (one main engine, two orientation engines)
  contact_structure: two point contacts (left and right support), target pad is flat
primary_objectives:
  - reach center of target pad (x≈0, y≈0)
  - settle with near-zero velocity and zero angle
secondary_objectives:
  - minimize engine usage (sparse or weak thrust preferred)
  - fast completion (implicit via episode length, but no explicit time reward given)
main_failure_risks:
  - crashing into ground or obstacle (outside pad)
  - sliding off viewport horizontally
  - failing to reduce velocity, bouncing or never settling
  - settling outside the pad (incorrect location)
  - excessive fuel burn without making progress
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: approach_target  
  purpose: 引导主体向目标垫（x=0, y=0）移动，同时减速，并逐步减小姿态角。  
  why_required: 核心目标；没有该引导则无法收敛到目标区域。  
  usable_signals: [x_position, y_position, x_velocity, y_velocity, body_angle]  
  risks: 权重过高可能导致急停或只关心位置而忽略稳定；需结合 landing 阶段。

- role_id: stable_landing  
  purpose: 在接近目标垫时，鼓励低速下让左右支撑脚同时接触。  
  why_required: 单脚接触或高速接触不符合“settle”要求；必须引入接触与速度联合条件。  
  usable_signals: [x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact, x_position, y_position（用于判定是否在垫上）]  
  risks: 接触信号本身可能提前出现（如蹭到地面），需用位置过滤；阈值设计不好可能导致早停。

### 10.2 条件职责 conditional_roles
- role_id: action_penalty  
  purpose: 鼓励节省燃料，惩罚非必要的发动机使用（特别是主发动机）。  
  condition_to_use: 当 agent 已经学会基本到达和着陆后，引入轻微的动作成本以优化效率；初期可关闭或极弱。  
  usable_signals: [action]  
  risks: 过强的惩罚会阻止 agent 学习使用发动机；可能引起根本不飞行的懒惰策略。

- role_id: time_efficiency (implicit)  
  purpose: 鼓励快速完成，可通过每步小的负奖励实现。  
  condition_to_use: 如果环境无步数限制或你想鼓励短路径，但需保证不压倒其他目标。  
  usable_signals: [无直接信号，需要依赖函数外部步数计数；但环境不提供信息，可用固定的小常数奖励]  
  risks: 若未正确标定可能导致提前 crash 以赚取总奖励。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: official_reward_replica  
  reason: 明确禁止使用 original_reward；任何试图模仿官方 reward 的组件都不可靠且违反规则。  
  forbidden_or_missing_signals: [original_reward]
- role_id: info_based_success  
  reason: info 字典为空，无 success 标记可用。只能从观测推断成功，不得依赖 info。  
  forbidden_or_missing_signals: [info.success, info.failure]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_target | x_position, y_position, x_velocity, y_velocity, body_angle | — | dense_state_signal, bounded_signal, distance_penalty (欧氏距离到目标), velocity_penalty, angle_penalty | 可分阶段：远距离时侧重位置减少距离，靠近时侧重减速 |
| stable_landing | x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact, x_position, y_position | — | conditional_reward (当位置接近0且接触均成立时启用), speed_penalty, angular_penalty | 引入条件 mask，只有实际在目标垫上方/上且双脚接触才给正奖；可对速度和角度的范数施加penalty |
| action_penalty | action | — | action_cost (如主发动机动作 penalize) | 可仅对动作2（主发动机）施加成本，不影响姿态发动机或空动作 |
| time_efficiency | — | step counter in compute_reward (当前接口无) | step_constant (固定小负值) | 需谨慎；如果环境会truncate，每步负奖励有助于快速完成，但容易让 agent 选择 crash 提前结束 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 始终悬停不下降或水平漂移 | 垂直速度持续接近零，y_position 不减小 | 减弱对主发动机的惩罚，或增强高度减少的正奖励 |
| 高速撞击目标垫或地面 | 快速 y_velocity 变为负大值，然后 episode 终止（很可能 crash） | 在接近目标时显著增加速度惩罚；引入高度相关的衰减系数 |
| 到达目标垫附近但无法稳定，来回震荡 | body_angle 和 angular_velocity 持续变化，接触信号跳动 | 加强角度和角速度惩罚；着陆奖励需覆盖连续多个步以保证持续接触 |
| 只使用姿态发动机摇摆而不使用主发动机 | 主发动机频率低，x/y 位置难以下降 | 对“无推进”或仅姿态的动作给予轻微负奖励，或引入 progress 奖励（如高度减少） |
| 过早满足稳定条件（settled但未在垫上） | 终止时 x_position 或 y_position 明显偏离0 | 在 stable_landing role 中严格要求 x≈0, y≈0，且接触成立 |
| 左右支撑接触不平衡，导致倾斜 | 只有左或右接触成立，body_angle 显著 | 着陆奖励要求左右接触同时为1，并惩罚角度 |



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

